# Copyright 2026 The HuggingFace Inc. team.
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
Convert DEIMv2 checkpoints from the original Github repository to the Hugging Face format.

This script converts a DEIMv2 checkpoint from the original Github repository to the Hugging Face format.
It uses the original DEIMv2 repository to load the checkpoint and convert it to the Hugging Face format.
It also performs a parity check between the original and the Hugging Face model.

URL: https://github.com/Intellindust-AI-Lab/DEIMv2
Original repo path: /Users/nielsrogge/Documents/python_projecten/DEIMv2

Usage:

```bash
python convert_deimv2_to_hf.py --checkpoint_path <checkpoint_path> --original_repo_path <original_repo_path> --pytorch_dump_folder_path <pytorch_dump_folder_path>
```
"""

import argparse
import sys
from pathlib import Path

import torch

from transformers import Deimv2Config, Deimv2ForObjectDetection, Deimv2ImageProcessor, HGNetV2Config


def get_deimv2_atto_config() -> Deimv2Config:
    backbone_config = HGNetV2Config(
        out_indices=[3],
        depths=[3, 3, 3],
        hidden_sizes=[64, 256, 256],
        stem_channels=[3, 16, 16],
        stage_in_channels=[16, 64, 256],
        stage_mid_channels=[16, 32, 64],
        stage_out_channels=[64, 256, 256],
        stage_num_blocks=[1, 1, 1],
        stage_downsample=[False, True, True],
        stage_light_block=[False, False, True],
        stage_kernel_size=[3, 3, 3],
        stage_numb_of_layers=[3, 3, 3],
        use_learnable_affine_block=True,
    )

    return Deimv2Config(
        num_labels=80,
        backbone_config=backbone_config,
        encoder_type="lite",
        encoder_in_channels=[256],
        encoder_hidden_dim=64,
        feat_strides=[16, 32],
        d_model=64,
        decoder_in_channels=[64, 64],
        num_feature_levels=2,
        decoder_n_points=[4, 2],
        decoder_layers=3,
        decoder_ffn_dim=160,
        num_queries=100,
        use_gateway=False,
        share_bbox_head=True,
        share_score_head=False,
        hidden_expansion=0.34,
        depth_mult=0.5,
        layer_scale=1,
        eval_idx=-1,
        activation_function="silu",
        decoder_activation_function="silu",
        encoder_activation_function="gelu",
        anchor_image_size=[320, 320],
        freeze_backbone_batch_norms=False,
    )


def map_backbone_key(original_key: str) -> str | None:
    if original_key.startswith("backbone.stem."):
        new_key = "model.backbone.model.embedder." + original_key[len("backbone.stem.") :]
    elif original_key.startswith("backbone.stages."):
        new_key = "model.backbone.model.encoder.stages." + original_key[len("backbone.stages.") :]
    else:
        return None

    new_key = new_key.replace(".conv.weight", ".convolution.weight")
    new_key = new_key.replace(".bn.", ".normalization.")
    return new_key


def map_key(original_key: str) -> str:
    if original_key.startswith("backbone."):
        mapped = map_backbone_key(original_key)
        if mapped is None:
            raise ValueError(f"Could not map backbone key: {original_key}")
        return mapped

    if original_key.startswith("encoder."):
        return "model." + original_key
    if original_key.startswith("decoder.dec_score_head."):
        return "model.decoder.class_embed." + original_key[len("decoder.dec_score_head.") :]
    if original_key.startswith("decoder.dec_bbox_head."):
        return "model.decoder.bbox_embed." + original_key[len("decoder.dec_bbox_head.") :]
    if original_key.startswith("decoder.enc_score_head."):
        return "model.enc_score_head." + original_key[len("decoder.enc_score_head.") :]
    if original_key.startswith("decoder.enc_bbox_head."):
        return "model.enc_bbox_head." + original_key[len("decoder.enc_bbox_head.") :]
    if original_key.startswith("decoder.denoising_class_embed."):
        return "model.denoising_class_embed." + original_key[len("decoder.denoising_class_embed.") :]
    if original_key.startswith("decoder.query_pos_head."):
        return "model.decoder.query_pos_head." + original_key[len("decoder.query_pos_head.") :]
    if original_key.startswith("decoder.pre_bbox_head."):
        return "model.decoder.pre_bbox_head." + original_key[len("decoder.pre_bbox_head.") :]
    if original_key.startswith("decoder.decoder."):
        return "model.decoder.decoder." + original_key[len("decoder.decoder.") :]
    if original_key == "decoder.up":
        return "model.decoder.up"
    if original_key == "decoder.reg_scale":
        return "model.decoder.reg_scale"
    if original_key in {"decoder.anchors", "decoder.valid_mask"}:
        # These are not part of HF state dict; we set them manually after loading.
        return original_key
    return "model." + original_key


def convert_state_dict(
    original_state_dict: dict[str, torch.Tensor], hf_state_dict: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    converted_state_dict = {}
    skipped = []

    for key, value in original_state_dict.items():
        if key.endswith("num_batches_tracked"):
            continue

        mapped_key = map_key(key)

        if mapped_key in {"decoder.anchors", "decoder.valid_mask"}:
            continue

        if mapped_key not in hf_state_dict:
            skipped.append(key)
            continue

        if hf_state_dict[mapped_key].shape != value.shape:
            skipped.append(key)
            continue

        converted_state_dict[mapped_key] = value

    return converted_state_dict, skipped


def load_original_model(
    original_repo_path: str,
    original_checkpoint_path: str,
):
    original_repo = Path(original_repo_path).resolve()
    if not original_repo.exists():
        raise ValueError(f"Original DEIMv2 repo path does not exist: {original_repo}")

    sys.path.insert(0, str(original_repo))
    import torch.distributed as dist

    # Original HGNetv2 init calls get_rank() unguarded.
    dist.get_rank = lambda: 0

    import engine
    from engine.core import YAMLConfig

    _ = engine  # silence lint about unused import (needed for registry side effects)
    config_path = original_repo / "configs" / "deimv2" / "deimv2_hgnetv2_atto_coco.yml"
    original_cfg = YAMLConfig(str(config_path), HGNetv2={"pretrained": False})
    original_model = original_cfg.model
    checkpoint = torch.load(original_checkpoint_path, map_location="cpu")
    original_model.load_state_dict(checkpoint["model"])
    original_model.eval()
    return original_model


@torch.no_grad()
def run_parity_check(
    hf_model: Deimv2ForObjectDetection,
    original_model,
):
    torch.manual_seed(0)
    dummy_pixel_values = torch.randn(1, 3, 320, 320)
    dummy_pixel_mask = torch.ones((1, 320, 320))

    original_outputs = original_model(dummy_pixel_values)
    hf_outputs = hf_model(pixel_values=dummy_pixel_values, pixel_mask=dummy_pixel_mask)

    logits_diff = (original_outputs["pred_logits"] - hf_outputs.logits).abs()
    boxes_diff = (original_outputs["pred_boxes"] - hf_outputs.pred_boxes).abs()

    print("Parity check on dummy input:")
    print(f"  logits max diff: {logits_diff.max().item():.8f}")
    print(f"  logits mean diff: {logits_diff.mean().item():.8f}")
    print(f"  boxes max diff: {boxes_diff.max().item():.8f}")
    print(f"  boxes mean diff: {boxes_diff.mean().item():.8f}")
    print("  original logits[:1, :3, :3]:")
    print(original_outputs["pred_logits"][0, :3, :3])
    print("  hf logits[:1, :3, :3]:")
    print(hf_outputs.logits[0, :3, :3])
    print("  original boxes[:1, :3, :3]:")
    print(original_outputs["pred_boxes"][0, :3, :3])
    print("  hf boxes[:1, :3, :3]:")
    print(hf_outputs.pred_boxes[0, :3, :3])

    return logits_diff.max().item(), boxes_diff.max().item()


def convert_deimv2_atto_checkpoint(
    checkpoint_path: str,
    pytorch_dump_folder_path: str | None,
    original_repo_path: str | None,
    skip_parity_check: bool,
):
    original_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    config = get_deimv2_atto_config()
    model = Deimv2ForObjectDetection(config)
    model.eval()

    converted_state_dict, skipped = convert_state_dict(original_state_dict, model.state_dict())
    if skipped:
        print(f"Skipped {len(skipped)} keys during conversion (first 20 shown):")
        print(skipped[:20])

    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
    print(f"Missing keys after loading converted state dict: {len(missing_keys)}")
    print(f"Unexpected keys after loading converted state dict: {len(unexpected_keys)}")
    if missing_keys:
        print("First missing keys:", missing_keys[:20])
    if unexpected_keys:
        print("First unexpected keys:", unexpected_keys[:20])

    # Keep anchors/valid_mask from checkpoint for exact parity with original implementation.
    model.model.anchors = original_state_dict["decoder.anchors"]
    model.model.valid_mask = original_state_dict["decoder.valid_mask"]

    if not skip_parity_check:
        if original_repo_path is None:
            raise ValueError("--original_repo_path must be provided when parity check is enabled.")
        original_model = load_original_model(original_repo_path, checkpoint_path)
        logits_max_diff, boxes_max_diff = run_parity_check(model, original_model)
        if logits_max_diff != 0.0 or boxes_max_diff != 0.0:
            raise ValueError(
                f"Parity check failed: logits_max_diff={logits_max_diff}, boxes_max_diff={boxes_max_diff} (expected 0.0)"
            )

    if pytorch_dump_folder_path is not None:
        output_dir = Path(pytorch_dump_folder_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        # `share_bbox_head=True` creates shared tensors across decoder layers. Clone tensors to save cleanly.
        state_dict_to_save = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.save_pretrained(output_dir, state_dict=state_dict_to_save)
        Deimv2ImageProcessor().save_pretrained(output_dir)
        print(f"Saved converted model and processor to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/Users/nielsrogge/Downloads/deimv2_hgnetv2_atto_coco.pth",
        help="Path to the original DEIMv2 ATTO checkpoint (.pth).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        default=None,
        help="Where to save the converted HF model.",
    )
    parser.add_argument(
        "--original_repo_path",
        type=str,
        default="/Users/nielsrogge/Documents/python_projecten/DEIMv2",
        help="Path to the original DEIMv2 repository for parity checks.",
    )
    parser.add_argument(
        "--skip_parity_check",
        action="store_true",
        help="Skip original-vs-HF parity check.",
    )
    args = parser.parse_args()

    convert_deimv2_atto_checkpoint(
        checkpoint_path=args.checkpoint_path,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        original_repo_path=args.original_repo_path,
        skip_parity_check=args.skip_parity_check,
    )
