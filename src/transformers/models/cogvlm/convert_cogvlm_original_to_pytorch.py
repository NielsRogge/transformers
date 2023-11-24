# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert CogVLM checkpoints from the original repository.
"""

import argparse

import requests
import torch
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    CLIPImageProcessor,
    CogVLMConfig,
    CogVLMForCausalLM,
    CogVLMProcessor,
    LlamaTokenizer,
)
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


@torch.no_grad()
def convert_cogvlm_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to Transformers design.
    """
    # load original model
    original_model = AutoModelForCausalLM.from_pretrained(
        "THUDM/cogvlm-chat-hf",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        # load_in_4bit=True,
        revision="refs/pr/3",
    )
    original_model.to("cuda:0")

    # verify chat example
    query = "Describe this image"
    image = Image.open(
        requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
    ).convert("RGB")

    tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    inputs = original_model.build_conversation_input_ids(
        tokenizer, query=query, history=[], images=[image]
    )  # chat mode
    original_pixel_values = inputs["images"][0]

    def gather_inputs(inputs, device, use_bfloat16=True):
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(device),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
            "images": [[inputs["images"][0].to(device).to(dtype)]],
        }
        return inputs
    
    original_inputs = gather_inputs(inputs, device="cuda")
    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        outputs = original_model.generate(**original_inputs, **gen_kwargs)
        outputs = outputs[:, original_inputs["input_ids"].shape[1] :]
        print(tokenizer.decode(outputs[0]))

    # load HF model
    # rename in_channels to num_channels for sake of consistency
    original_model.config.vision_config["num_channels"] = original_model.config.vision_config.pop("in_channels")

    print("Original config:", original_model.config)

    config = CogVLMConfig(**original_model.config.to_dict())
    model = CogVLMForCausalLM(config)

    # load state dict
    model.load_state_dict(original_model.state_dict())
    model.to("cuda:1")
    model.eval()

    # create processor
    image_size = original_model.config.vision_config["image_size"]
    image_processor = CLIPImageProcessor(
        size={"height": image_size, "width": image_size},
        do_center_crop=False,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
    )
    processor = CogVLMProcessor(image_processor=image_processor, tokenizer=tokenizer)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to("cuda")

    # verify pixel values
    assert torch.allclose(pixel_values, original_pixel_values.to(pixel_values.device))

    # TODO use processor instead
    inputs = gather_inputs(inputs, device="cuda:1", use_bfloat16=False)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        print("HF outputs:", tokenizer.decode(outputs[0]))

    # with torch.no_grad():
    #      original_logits = original_model({"image": original_pixel_values, "text_input": [""]}).logits
    #      logits = model(pixel_values, input_ids).logits

    # assert original_logits.shape == logits.shape
    # print("First values of original logits:", original_logits[0, :3, :3])
    # print("First values of HF logits:", logits[0, :3, :3])

    # # assert values
    # assert torch.allclose(original_logits.to(logits.device), logits, atol=1e-4)
    # print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        processor.save_pretrained(pytorch_dump_folder_path)
        # model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        processor.push_to_hub(f"nielsr/{model_name}")
        # model.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="THUDM/cogvlm-chat-hf",
        type=str,
        help="Name of the model to convert",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    args = parser.parse_args()

    convert_cogvlm_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
