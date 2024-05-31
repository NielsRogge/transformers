from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import re
from nltk import edit_distance
import numpy as np

### define variables

MAX_IMAGES_PER_EXAMPLE = 2
MAX_LENGTH = 1024
FINETUNED_REPO_ID = "nielsr/idefics2-dude-demo"
WANDB_PROJECT = "Idefics2-PL"
WANDB_NAME = "demo-run-dude"

### define dataset

dataset = load_dataset("jordyvl/DUDE_subset_100val")

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", size={"longest_edge": 532, "shortest_edge": 378}, do_image_splitting=False)

class Idefics2Dataset(Dataset):
    """
    PyTorch Dataset for Idefics2. This class takes a HuggingFace Dataset as input.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns one item of the dataset.
        """
        sample = self.dataset[idx]

        images = sample["images"][:MAX_IMAGES_PER_EXAMPLE]
        question = sample["question"]
        answer = sample["answer"]

        return images, question, answer
    
# split Hugging Face dataset into train and validation
dataset = dataset["train"].train_test_split(test_size=0.1)

train_dataset = Idefics2Dataset(dataset["train"])
val_dataset = Idefics2Dataset(dataset["test"])

### load model
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
)

lora_config = LoraConfig(
          r=8,
          lora_alpha=8,
          lora_dropout=0.1,
          target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
          use_dora=False,
          init_lora_weights="gaussian",
      )

# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

### define collate functions

image_token_id = processor.tokenizer.additional_special_tokens_ids[processor.tokenizer.additional_special_tokens.index("<image>")]

def train_collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        images_example, question, answer = example

        content = [{"type": "image"} for _ in range(len(images_example))]
        content += [{"type": "text", "text": question}]


        # Create inputs
        messages = [
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ]
            },
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(prompt)
        images.append(images_example)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == model.config.image_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, labels


def eval_collate_fn(examples):

    # we feed the prompt to the model
    images = []
    texts = []
    answers = []
    for example in examples:
        images_example, question, answer = example

        content = [{"type": "image"} for _ in range(len(images_example))]
        content += [{"type": "text", "text": question}]

        messages = [
            {
                "role": "user",
                "content": content,
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images.append(images_example)
        texts.append(text.strip())
        answers.append(answer)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, answers


### Define LightningModule

class Idefics2ModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, pixel_attention_mask, labels = batch

        outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                pixel_attention_mask=pixel_attention_mask,
                                labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, pixel_attention_mask, answers = batch

        # autoregressively generate token IDs
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask,
                                       max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
config = {"max_epochs": 10,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 1,
          "precision": "16-mixed", # we'll use mixed precision
          # "seed":2022, # can be used for reproducibility
          "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True,
}

model_module = Idefics2ModelPLModule(config, processor, model)

### Define callbacks

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(FINETUNED_REPO_ID,
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(FINETUNED_REPO_ID,
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub(FINETUNED_REPO_ID,
                                    commit_message=f"Training done")

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

## Train the model

wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
        accelerator="gpu",
        devices=[0,1],
        max_epochs=config.get("max_epochs"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        precision=config.get("precision"),
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[PushToHubCallback(), early_stop_callback],
)

trainer.fit(model_module)