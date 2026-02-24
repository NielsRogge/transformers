# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Convert official VidEoMT checkpoints from https://huggingface.co/tue-mps/VidEoMT to HF format."""

from __future__ import annotations

import argparse
import re

import torch
from huggingface_hub import hf_hub_download

from transformers import VideomtConfig, VideomtForUniversalSegmentation


MODEL_REPO_ID = "tue-mps/VidEoMT"


def infer_num_attention_heads(checkpoint_filename: str, hidden_size: int) -> int:
    if "vit_small" in checkpoint_filename:
        return 6
    if "vit_base" in checkpoint_filename:
        return 12
    if "vit_large" in checkpoint_filename:
        return 16
    if hidden_size % 64 == 0:
        return hidden_size // 64
    raise ValueError(f"Could not infer num_attention_heads from checkpoint name '{checkpoint_filename}'.")


def infer_videomt_config(
    state_dict: dict[str, torch.Tensor], checkpoint_filename: str, image_size: int, num_frames: int
):
    hidden_size = state_dict["backbone.encoder.backbone.cls_token"].shape[-1]
    num_hidden_layers = (
        max(
            int(match.group(1))
            for key in state_dict
            if (match := re.match(r"backbone\.encoder\.backbone\.blocks\.(\d+)\.norm1\.weight", key))
        )
        + 1
    )

    return VideomtConfig(
        hidden_size=hidden_size,
        num_attention_heads=infer_num_attention_heads(checkpoint_filename, hidden_size),
        intermediate_size=state_dict["backbone.encoder.backbone.blocks.0.mlp.fc1.weight"].shape[0],
        image_size=image_size,
        patch_size=state_dict["backbone.encoder.backbone.patch_embed.proj.weight"].shape[-1],
        num_register_tokens=state_dict["backbone.encoder.backbone.reg_token"].shape[1],
        num_hidden_layers=num_hidden_layers,
        num_queries=state_dict["backbone.q.weight"].shape[0],
        num_blocks=state_dict["backbone.attn_mask_probs"].shape[0],
        num_labels=state_dict["backbone.class_head.weight"].shape[0] - 1,
        num_frames=num_frames,
    )


def convert_state_dict(
    original_state_dict: dict[str, torch.Tensor], config: VideomtConfig
) -> tuple[dict[str, torch.Tensor], set[str]]:
    converted = {}
    consumed_keys = set()

    converted["attn_mask_probs"] = original_state_dict["backbone.attn_mask_probs"]
    consumed_keys.add("backbone.attn_mask_probs")
    converted["embeddings.cls_token"] = original_state_dict["backbone.encoder.backbone.cls_token"]
    consumed_keys.add("backbone.encoder.backbone.cls_token")
    converted["embeddings.register_tokens"] = original_state_dict["backbone.encoder.backbone.reg_token"]
    consumed_keys.add("backbone.encoder.backbone.reg_token")
    converted["embeddings.patch_embeddings.weight"] = original_state_dict[
        "backbone.encoder.backbone.patch_embed.proj.weight"
    ]
    consumed_keys.add("backbone.encoder.backbone.patch_embed.proj.weight")
    converted["embeddings.patch_embeddings.bias"] = original_state_dict[
        "backbone.encoder.backbone.patch_embed.proj.bias"
    ]
    consumed_keys.add("backbone.encoder.backbone.patch_embed.proj.bias")
    converted["layernorm.weight"] = original_state_dict["backbone.encoder.backbone.norm.weight"]
    consumed_keys.add("backbone.encoder.backbone.norm.weight")
    converted["layernorm.bias"] = original_state_dict["backbone.encoder.backbone.norm.bias"]
    consumed_keys.add("backbone.encoder.backbone.norm.bias")
    converted["query.weight"] = original_state_dict["backbone.q.weight"]
    consumed_keys.add("backbone.q.weight")

    converted["class_predictor.weight"] = original_state_dict["backbone.class_head.weight"]
    consumed_keys.add("backbone.class_head.weight")
    converted["class_predictor.bias"] = original_state_dict["backbone.class_head.bias"]
    consumed_keys.add("backbone.class_head.bias")

    converted["mask_head.fc1.weight"] = original_state_dict["backbone.mask_head.0.weight"]
    converted["mask_head.fc1.bias"] = original_state_dict["backbone.mask_head.0.bias"]
    converted["mask_head.fc2.weight"] = original_state_dict["backbone.mask_head.2.weight"]
    converted["mask_head.fc2.bias"] = original_state_dict["backbone.mask_head.2.bias"]
    converted["mask_head.fc3.weight"] = original_state_dict["backbone.mask_head.4.weight"]
    converted["mask_head.fc3.bias"] = original_state_dict["backbone.mask_head.4.bias"]
    consumed_keys.update(
        {
            "backbone.mask_head.0.weight",
            "backbone.mask_head.0.bias",
            "backbone.mask_head.2.weight",
            "backbone.mask_head.2.bias",
            "backbone.mask_head.4.weight",
            "backbone.mask_head.4.bias",
        }
    )

    for idx in range(2):
        converted[f"upscale_block.block.{idx}.conv1.weight"] = original_state_dict[
            f"backbone.upscale.{idx}.conv1.weight"
        ]
        converted[f"upscale_block.block.{idx}.conv1.bias"] = original_state_dict[f"backbone.upscale.{idx}.conv1.bias"]
        converted[f"upscale_block.block.{idx}.conv2.weight"] = original_state_dict[
            f"backbone.upscale.{idx}.conv2.weight"
        ]
        converted[f"upscale_block.block.{idx}.layernorm2d.weight"] = original_state_dict[
            f"backbone.upscale.{idx}.norm.weight"
        ]
        converted[f"upscale_block.block.{idx}.layernorm2d.bias"] = original_state_dict[
            f"backbone.upscale.{idx}.norm.bias"
        ]
        consumed_keys.update(
            {
                f"backbone.upscale.{idx}.conv1.weight",
                f"backbone.upscale.{idx}.conv1.bias",
                f"backbone.upscale.{idx}.conv2.weight",
                f"backbone.upscale.{idx}.norm.weight",
                f"backbone.upscale.{idx}.norm.bias",
            }
        )

    for layer_idx in range(config.num_hidden_layers):
        layer_prefix = f"backbone.encoder.backbone.blocks.{layer_idx}"
        converted[f"layers.{layer_idx}.norm1.weight"] = original_state_dict[f"{layer_prefix}.norm1.weight"]
        converted[f"layers.{layer_idx}.norm1.bias"] = original_state_dict[f"{layer_prefix}.norm1.bias"]
        converted[f"layers.{layer_idx}.norm2.weight"] = original_state_dict[f"{layer_prefix}.norm2.weight"]
        converted[f"layers.{layer_idx}.norm2.bias"] = original_state_dict[f"{layer_prefix}.norm2.bias"]

        qkv_weight = original_state_dict[f"{layer_prefix}.attn.qkv.weight"]
        qkv_bias = original_state_dict[f"{layer_prefix}.attn.qkv.bias"]
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, _k_bias, v_bias = qkv_bias.chunk(3, dim=0)

        converted[f"layers.{layer_idx}.attention.q_proj.weight"] = q_weight
        converted[f"layers.{layer_idx}.attention.q_proj.bias"] = q_bias
        converted[f"layers.{layer_idx}.attention.k_proj.weight"] = k_weight
        converted[f"layers.{layer_idx}.attention.v_proj.weight"] = v_weight
        converted[f"layers.{layer_idx}.attention.v_proj.bias"] = v_bias
        converted[f"layers.{layer_idx}.attention.o_proj.weight"] = original_state_dict[
            f"{layer_prefix}.attn.proj.weight"
        ]
        converted[f"layers.{layer_idx}.attention.o_proj.bias"] = original_state_dict[f"{layer_prefix}.attn.proj.bias"]

        converted[f"layers.{layer_idx}.layer_scale1.lambda1"] = original_state_dict[f"{layer_prefix}.ls1.gamma"]
        converted[f"layers.{layer_idx}.layer_scale2.lambda1"] = original_state_dict[f"{layer_prefix}.ls2.gamma"]

        converted[f"layers.{layer_idx}.mlp.up_proj.weight"] = original_state_dict[f"{layer_prefix}.mlp.fc1.weight"]
        converted[f"layers.{layer_idx}.mlp.up_proj.bias"] = original_state_dict[f"{layer_prefix}.mlp.fc1.bias"]
        converted[f"layers.{layer_idx}.mlp.down_proj.weight"] = original_state_dict[f"{layer_prefix}.mlp.fc2.weight"]
        converted[f"layers.{layer_idx}.mlp.down_proj.bias"] = original_state_dict[f"{layer_prefix}.mlp.fc2.bias"]
        consumed_keys.update(
            {
                f"{layer_prefix}.norm1.weight",
                f"{layer_prefix}.norm1.bias",
                f"{layer_prefix}.norm2.weight",
                f"{layer_prefix}.norm2.bias",
                f"{layer_prefix}.attn.qkv.weight",
                f"{layer_prefix}.attn.qkv.bias",
                f"{layer_prefix}.attn.proj.weight",
                f"{layer_prefix}.attn.proj.bias",
                f"{layer_prefix}.ls1.gamma",
                f"{layer_prefix}.ls2.gamma",
                f"{layer_prefix}.mlp.fc1.weight",
                f"{layer_prefix}.mlp.fc1.bias",
                f"{layer_prefix}.mlp.fc2.weight",
                f"{layer_prefix}.mlp.fc2.bias",
            }
        )

    return converted, consumed_keys


def validate_qkv_split(
    original_state_dict: dict[str, torch.Tensor], converted_state_dict: dict[str, torch.Tensor], config
):
    for layer_idx in range(config.num_hidden_layers):
        source_qkv = original_state_dict[f"backbone.encoder.backbone.blocks.{layer_idx}.attn.qkv.weight"]
        recon_qkv = torch.cat(
            [
                converted_state_dict[f"layers.{layer_idx}.attention.q_proj.weight"],
                converted_state_dict[f"layers.{layer_idx}.attention.k_proj.weight"],
                converted_state_dict[f"layers.{layer_idx}.attention.v_proj.weight"],
            ],
            dim=0,
        )
        if not torch.equal(source_qkv, recon_qkv):
            raise ValueError(f"qkv split mismatch at layer {layer_idx}.")


def convert_checkpoint(
    checkpoint_filename: str, image_size: int, num_frames: int, output_dir: str | None = None
) -> None:
    checkpoint_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=checkpoint_filename)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    original_state_dict = checkpoint.get("model", checkpoint)

    config = infer_videomt_config(
        original_state_dict, checkpoint_filename, image_size=image_size, num_frames=num_frames
    )
    model = VideomtForUniversalSegmentation(config)
    converted_state_dict, consumed_keys = convert_state_dict(original_state_dict, config)
    validate_qkv_split(original_state_dict, converted_state_dict, config)

    load_info = model.load_state_dict(converted_state_dict, strict=False)

    dummy_video = torch.randn(1, num_frames, 3, config.image_size, config.image_size)
    with torch.no_grad():
        outputs = model(pixel_values=dummy_video)

    if (
        not torch.isfinite(outputs.class_queries_logits).all()
        or not torch.isfinite(outputs.masks_queries_logits).all()
    ):
        raise ValueError("Converted model produced non-finite outputs.")

    print(f"checkpoint={checkpoint_filename}")
    print(f"missing_keys={len(load_info.missing_keys)}")
    print(f"unexpected_keys={len(load_info.unexpected_keys)}")
    print(f"class_logits_shape={tuple(outputs.class_queries_logits.shape)}")
    print(f"mask_logits_shape={tuple(outputs.masks_queries_logits.shape)}")

    if load_info.missing_keys:
        print("missing_key_list=")
        for key in load_info.missing_keys:
            print(f"  - {key}")

    if load_info.unexpected_keys:
        print("unexpected_key_list=")
        for key in load_info.unexpected_keys:
            print(f"  - {key}")

    unconverted_source_keys = sorted(set(original_state_dict.keys()) - consumed_keys)
    print(f"unconverted_source_keys={len(unconverted_source_keys)}")
    if unconverted_source_keys:
        print("unconverted_source_key_list=")
        for key in unconverted_source_keys:
            print(f"  - {key}")

    if output_dir is not None:
        model.save_pretrained(output_dir)
        config.save_pretrained(output_dir)
        print(f"saved_to={output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert official VidEoMT checkpoints to HF format.")
    parser.add_argument("--checkpoint-filename", type=str, required=True, help="Filename on tue-mps/VidEoMT")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(
        checkpoint_filename=args.checkpoint_filename,
        image_size=args.image_size,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
