"""
Fine-tune Idefics2 by overwriting the `Seq2SeqTrainer` class.

First set the CUDA_VISIBLE_DEVICES environment variable to the GPUs you want to use, e.g., `export CUDA_VISIBLE_DEVICES=1,2`.

One can run the script using `python src/transformers/models/idefics2/fine_tune_idefics2.py`.
"""

import json
import random
from typing import Any

import Levenshtein
import numpy as np
import torch
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    Idefics2ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


USE_LORA = False
USE_QLORA = True
PROMPT_LENGTH = 79
MAX_LENGTH = 768

import os

os.environ["WANDB_PROJECT"] = "idefics2"

## Load model

# Three options for training, from the lowest precision training to the highest precision training:
# - QLora
# - Standard Lora
# - Full fine-tuning
if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        quantization_config=bnb_config if USE_QLORA else None,
    )
    # apply PEFT
    lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian",
        )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
else:
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",  # Only available on A100 or H100
    )


## Load dataset
dataset = load_dataset("naver-clova-ix/cord-v2")

## Create PyTorch dataset
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)


class CustomDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        split,
        sort_json_key: bool = True,
    ):
        self.dataset = hf_dataset[split]
        self.split = split
        self.sort_json_key = sort_json_key

        ground_truth_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # some datasets have multiple ground truths available, e.g. DocVQA
                assert isinstance(ground_truth["gt_parses"], list)
                ground_truth_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                ground_truth_jsons = [ground_truth["gt_parse"]]

            ground_truth_token_sequences.append(
                [
                    self.json2token(
                        ground_truth_json,
                        sort_json_key=self.sort_json_key,
                    )
                    for ground_truth_json in ground_truth_jsons  # load json from list of json
                ]
            )

        self.ground_truth_token_sequences = ground_truth_token_sequences

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        rf"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + rf"</s_{k}>"
                    )
                return output
        elif isinstance(obj, list):
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        target_sequence = random.choice(self.ground_truth_token_sequences[idx])  # can be more than one, e.g., DocVQA

        return image, target_sequence


train_dataset = CustomDataset(hf_dataset=dataset, split="train")
eval_dataset = CustomDataset(hf_dataset=dataset, split="validation")

## Define data collators
image_token_id = processor.tokenizer.additional_special_tokens_ids[processor.tokenizer.additional_special_tokens.index("<image>")]

def train_collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image, ground_truth = example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract JSON."},
                    {"type": "image"},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ground_truth}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = image_token_id
    batch["labels"] = labels

    return batch


def eval_collate_fn(examples):
    # we feed the prompt to the model
    images = []
    texts = []
    for example in examples:
        image, ground_truth = example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract JSON."},
                    {"type": "image"},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ground_truth}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = image_token_id
    batch["labels"] = labels

    # finally: limit the input_ids and attention_mask to the prompt length
    batch["input_ids"] = batch["input_ids"][:, :PROMPT_LENGTH]
    batch["attention_mask"] = batch["attention_mask"][:, :PROMPT_LENGTH]

    return batch

## Create PyTorch DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=train_collate_fn,
)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=eval_collate_fn,
)


## Define metrics
def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0


def average_normalized_levenshtein_similarity(ground_truth, predicted_answers):
    assert len(ground_truth) == len(predicted_answers), "Length of ground_truth and predicted_answers must match."

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = ground_truth[i]
        o_q_i = predicted_answers[i]
        if o_q_i == "":
            print("Warning: Skipped an empty prediction.")
            max_score = 0
        else:
            max_score = max(similarity_score(a_ij, o_q_i) for a_ij in a_i)

        total_score += max_score

    return total_score / N


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Strip the prompt from the labels
    labels = labels[:, PROMPT_LENGTH:]

    # Replace -100s used for padding as we can't decode them
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, processor.tokenizer.pad_token_id)
    # Decode back into text, skipping special tokens like padding and image tokens
    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    for pred, target in zip(decoded_preds, decoded_labels):
        print("Prediction:", pred)
        print("Ground truth:", target)
        print("---------------")

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    score = average_normalized_levenshtein_similarity(decoded_labels, decoded_preds)
    result = {"levenshtein": score}

    prediction_lens = [np.count_nonzero(pred != processor.tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


## Define Training Arguments and Trainer

generation_config = GenerationConfig.from_pretrained("HuggingFaceM4/idefics2-8b", max_new_tokens=MAX_LENGTH)

training_args = Seq2SeqTrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    output_dir="idefics2_ft_tutorial",
    eval_strategy="steps",
    eval_steps=1,
    save_total_limit=1,
    fp16=True,
    remove_unused_columns=False,
    report_to="wandb",
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH - PROMPT_LENGTH,
    generation_config=generation_config,
)

# important: we need to disable caching during training
# otherwise the model generates past_key_values which is of type DynamicCache
model.config.use_cache = False


class Idefics2Trainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        return train_dataloader

    def get_eval_dataloader(self, eval_dataset = None):
        return eval_dataloader

trainer = Idefics2Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()