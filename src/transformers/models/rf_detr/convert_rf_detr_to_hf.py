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
"""Convert RF-DETR checkpoints from the original Roboflow implementation.

This script supports:
1. converting checkpoint keys to the HF RF-DETR implementation,
2. saving model + config,
3. optional numerical parity check against the original implementation on dummy inputs.

It can be run as follows:

```bash
uv run src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --checkpoint_path /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-small.pth --original_repo_path /Users/nielsrogge/Documents/python_projecten/rf-detr --pytorch_dump_folder_path . --verify_with_original
```

Or by model name, downloading the original object-detection checkpoint from
`nielsr/rf-detr-checkpoints`:

```bash
uv run src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --model_name small --pytorch_dump_folder_path ./rf-detr-small-hf
```
"""

import argparse
import importlib
import math
import os
import re
import sys
import types
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

from transformers.models.rf_detr.modeling_rf_detr import RfDetrConfig, RfDetrForObjectDetection


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # backbone
    r"backbone.0.encoder.encoder.embeddings.cls_token":                                             r"model.backbone.backbone.embeddings.cls_token",
    r"backbone.0.encoder.encoder.embeddings.mask_token":                                            r"model.backbone.backbone.embeddings.mask_token",
    r"backbone.0.encoder.encoder.embeddings.position_embeddings":                                   r"model.backbone.backbone.embeddings.position_embeddings",
    r"backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.(weight|bias)":            r"model.backbone.backbone.embeddings.patch_embeddings.projection.\1",
    r"backbone.0.encoder.encoder.layernorm.(weight|bias)":                                          r"model.backbone.backbone.layernorm.\1",
    r"backbone.0.encoder.encoder.encoder.layer.(\d+).norm1.(weight|bias)":                         r"model.backbone.backbone.encoder.layer.\1.norm1.\2",
    r"backbone.0.encoder.encoder.encoder.layer.(\d+).attention.attention.(query|key|value).(weight|bias)": r"model.backbone.backbone.encoder.layer.\1.attention.attention.\2.\3",
    r"backbone.0.encoder.encoder.encoder.layer.(\d+).attention.output.dense.(weight|bias)":        r"model.backbone.backbone.encoder.layer.\1.attention.output.dense.\2",
    r"backbone.0.encoder.encoder.encoder.layer.(\d+).layer_scale1.lambda1":                        r"model.backbone.backbone.encoder.layer.\1.layer_scale1.lambda1",
    r"backbone.0.encoder.encoder.encoder.layer.(\d+).norm2.(weight|bias)":                         r"model.backbone.backbone.encoder.layer.\1.norm2.\2",
    r"backbone.0.encoder.encoder.encoder.layer.(\d+).mlp.fc1.(weight|bias)":                       r"model.backbone.backbone.encoder.layer.\1.mlp.fc1.\2",
    r"backbone.0.encoder.encoder.encoder.layer.(\d+).mlp.fc2.(weight|bias)":                       r"model.backbone.backbone.encoder.layer.\1.mlp.fc2.\2",
    r"backbone.0.encoder.encoder.encoder.layer.(\d+).layer_scale2.lambda1":                        r"model.backbone.backbone.encoder.layer.\1.layer_scale2.lambda1",

    # projector (shared mapping with LW-DETR style)
    r"backbone.0.projector.stages.(\d+).0.cv1.conv.(weight|bias)":                                                      r"model.backbone.projector.scale_layers.\1.projector_layer.conv1.conv.\2",
    r"backbone.0.projector.stages.(\d+).0.cv1.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":           r"model.backbone.projector.scale_layers.\1.projector_layer.conv1.norm.\2",
    r"backbone.0.projector.stages.(\d+).0.cv2.conv.(weight|bias)":                                                      r"model.backbone.projector.scale_layers.\1.projector_layer.conv2.conv.\2",
    r"backbone.0.projector.stages.(\d+).0.cv2.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":           r"model.backbone.projector.scale_layers.\1.projector_layer.conv2.norm.\2",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv1.conv.(weight|bias)":                                              r"model.backbone.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv1.conv.\3",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv1.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":   r"model.backbone.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv1.norm.\3",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv2.conv.(weight|bias)":                                              r"model.backbone.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv2.conv.\3",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv2.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":   r"model.backbone.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv2.norm.\3",
    r"backbone.0.projector.stages.(\d+).1.(weight|bias)":                                                               r"model.backbone.projector.scale_layers.\1.layer_norm.\2",

    # decoder + transformer heads
    r"transformer.decoder.layers.(\d+).self_attn.out_proj.(weight|bias)":               r"model.decoder.layers.\1.self_attn.o_proj.\2",
    r"transformer.decoder.layers.(\d+).norm1.(weight|bias)":                            r"model.decoder.layers.\1.self_attn_layer_norm.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.sampling_offsets.(weight|bias)":      r"model.decoder.layers.\1.cross_attn.sampling_offsets.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.attention_weights.(weight|bias)":     r"model.decoder.layers.\1.cross_attn.attention_weights.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.value_proj.(weight|bias)":            r"model.decoder.layers.\1.cross_attn.value_proj.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.output_proj.(weight|bias)":           r"model.decoder.layers.\1.cross_attn.output_proj.\2",
    r"transformer.decoder.layers.(\d+).norm2.(weight|bias)":                            r"model.decoder.layers.\1.cross_attn_layer_norm.\2",
    r"transformer.decoder.layers.(\d+).linear1.(weight|bias)":                          r"model.decoder.layers.\1.mlp.fc1.\2",
    r"transformer.decoder.layers.(\d+).linear2.(weight|bias)":                          r"model.decoder.layers.\1.mlp.fc2.\2",
    r"transformer.decoder.layers.(\d+).norm3.(weight|bias)":                            r"model.decoder.layers.\1.layer_norm.\2",
    r"transformer.decoder.norm.(weight|bias)":                                          r"model.decoder.layernorm.\1",
    r"transformer.decoder.ref_point_head.layers.(\d+).(weight|bias)":                   r"model.decoder.ref_point_head.layers.\1.\2",
    r"transformer.enc_output.(\d+).(weight|bias)":                                      r"model.enc_output.\1.\2",
    r"transformer.enc_output_norm.(\d+).(weight|bias)":                                 r"model.enc_output_norm.\1.\2",
    r"transformer.enc_out_class_embed.(\d+).(weight|bias)":                             r"model.enc_out_class_embed.\1.\2",
    r"transformer.enc_out_bbox_embed.(\d+).layers.(\d+).(weight|bias)":                 r"model.enc_out_bbox_embed.\1.layers.\2.\3",
    r"refpoint_embed.weight":                                                            r"model.reference_point_embed.weight",
    r"query_feat.weight":                                                                r"model.query_feat.weight",

    # detection heads
    r"class_embed.(weight|bias)":                    r"class_embed.\1",
    r"bbox_embed.layers.(\d+).(weight|bias)":       r"bbox_embed.layers.\1.\2",
}
# fmt: on

DEFAULT_RF_DETR_CHECKPOINT_REPO_ID = "nielsr/rf-detr-checkpoints"
# Object-detection checkpoints only (segmentation checkpoints are intentionally excluded for now).
OBJECT_DETECTION_CHECKPOINT_CANDIDATES = {
    "nano": ["rf-detr-nano.pth"],
    "small": ["rf-detr-small.pth"],
    "medium": ["rf-detr-medium.pth"],
    "large": ["rf-detr-large-2026.pth"],
    "base": ["rf-detr-base.pth"],
    "base-2": ["rf-detr-base-2.pth"],
    "base-o365": ["rf-detr-base-o365.pth"],
}
OBJECT_DETECTION_MODEL_NAME_ALIASES = {
    "nano": {"nano", "rfdetrnano"},
    "small": {"small", "rfdetrsmall"},
    "medium": {"medium", "rfdetrmedium"},
    "large": {"large", "large2026", "rfdetrlarge", "rfdetrlarge2026"},
    "base": {"base", "rfdetrbase"},
    "base-2": {"base2", "rfdetrbase2"},
    "base-o365": {"baseo365", "o365", "rfdetrbaseo365"},
}
OBJECT_DETECTION_MODEL_NAME_PATTERNS = {
    "nano": [r"(?:.*/)?rf[-_]?detr[-_]?nano(?:[-_].*)?\.pth$"],
    "small": [r"(?:.*/)?rf[-_]?detr[-_]?small(?:[-_].*)?\.pth$"],
    "medium": [r"(?:.*/)?rf[-_]?detr[-_]?medium(?:[-_].*)?\.pth$"],
    "large": [r"(?:.*/)?rf[-_]?detr[-_]?large[-_]?2026(?:[-_].*)?\.pth$"],
    "base": [r"(?:.*/)?rf[-_]?detr[-_]?base(?:[-_].*)?\.pth$"],
    "base-2": [r"(?:.*/)?rf[-_]?detr[-_]?base[-_]?2(?:[-_].*)?\.pth$"],
    "base-o365": [r"(?:.*/)?rf[-_]?detr[-_]?base[-_]?o365(?:[-_].*)?\.pth$"],
}
OBJECT_DETECTION_CHECKPOINT_DEFAULT_ARGS = {
    "nano": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 2,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "group_detr": 13,
        "resolution": 384,
        "dinov2_patch_size": 16,
        "dinov2_num_windows": 2,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
    },
    "small": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 3,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "group_detr": 13,
        "resolution": 512,
        "dinov2_patch_size": 16,
        "dinov2_num_windows": 2,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
    },
    "medium": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 4,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "group_detr": 13,
        "resolution": 576,
        "dinov2_patch_size": 16,
        "dinov2_num_windows": 2,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
    },
    "large": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 4,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "group_detr": 13,
        "resolution": 704,
        "dinov2_patch_size": 16,
        "dinov2_num_windows": 2,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
    },
    "base": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [2, 5, 8, 11],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 3,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "group_detr": 13,
        "resolution": 560,
        "dinov2_patch_size": 14,
        "dinov2_num_windows": 4,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
    },
    "base-2": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [2, 5, 8, 11],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 3,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "group_detr": 13,
        "resolution": 560,
        "dinov2_patch_size": 14,
        "dinov2_num_windows": 4,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
    },
    "base-o365": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [2, 5, 8, 11],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 3,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "group_detr": 13,
        "resolution": 560,
        "dinov2_patch_size": 14,
        "dinov2_num_windows": 4,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
    },
}


def convert_old_keys_to_new_keys(state_dict_keys: list[str], key_mapping: dict[str, str]) -> dict[str, str]:
    old_text = "\n".join(state_dict_keys)
    new_text = old_text
    for pattern, replacement in key_mapping.items():
        new_text = re.sub(pattern, replacement, new_text)
    return dict(zip(old_text.split("\n"), new_text.split("\n")))


def read_in_decoder_q_k_v(state_dict: dict[str, torch.Tensor], config: RfDetrConfig) -> dict[str, torch.Tensor]:
    d_model = config.d_model
    for i in range(config.decoder_layers):
        in_proj_weight = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_bias")

        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:d_model, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:d_model]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[d_model : 2 * d_model, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[d_model : 2 * d_model]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-d_model:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-d_model:]
    return state_dict


def get_backbone_projector_sampling_key_mapping(config: RfDetrConfig) -> dict[str, str]:
    key_mapping = {}
    for i, scale in enumerate(config.projector_scale_factors):
        if scale == 2.0:
            if config.backbone_config.hidden_size > 512:
                key_mapping.update(
                    {
                        rf"backbone.0.projector.stages_sampling.{i}.(\d+).0.conv.weight": rf"model.backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.0.conv.weight",
                        rf"backbone.0.projector.stages_sampling.{i}.(\d+).0.bn.(weight|bias|running_mean|running_var|num_batches_tracked)": rf"model.backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.0.norm.\2",
                        rf"backbone.0.projector.stages_sampling.{i}.(\d+).1.(weight|bias)": rf"model.backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.1.\2",
                    }
                )
            else:
                key_mapping.update(
                    {
                        rf"backbone.0.projector.stages_sampling.{i}.(\d+).(\d+).(weight|bias)": rf"model.backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.\2.\3",
                    }
                )
        elif scale == 0.5:
            key_mapping.update(
                {
                    rf"backbone.0.projector.stages_sampling.{i}.(\d+).(\d+).conv.weight": rf"model.backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.\2.conv.weight",
                    rf"backbone.0.projector.stages_sampling.{i}.(\d+).(\d+).bn.(weight|bias|running_mean|running_var|num_batches_tracked)": rf"model.backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.\2.norm.\3",
                }
            )
    return key_mapping


def _with_default(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def _get_checkpoint_arg(checkpoint_args: dict, *names: str, default=None, required: bool = True):
    for name in names:
        if name in checkpoint_args and checkpoint_args[name] is not None:
            return checkpoint_args[name]
    if required:
        joined_names = ", ".join(names)
        raise KeyError(f"None of [{joined_names}] were found in checkpoint args.")
    return default


def _infer_default_repo_id(checkpoint_path: str, checkpoint_args: dict) -> str:
    checkpoint_stem = Path(checkpoint_path).stem

    pretrain_weights = checkpoint_args.get("pretrain_weights")
    if pretrain_weights:
        checkpoint_stem = Path(pretrain_weights).stem

    repo_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", checkpoint_stem).strip("-").lower()
    if not repo_name:
        repo_name = "rf-detr-converted"

    return f"nielsr/{repo_name}"


def _normalize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model_name.lower())


def _resolve_detection_model_name(model_name: str) -> str:
    normalized_model_name = _normalize_model_name(model_name)
    for canonical_name, aliases in OBJECT_DETECTION_MODEL_NAME_ALIASES.items():
        if normalized_model_name == _normalize_model_name(canonical_name) or normalized_model_name in aliases:
            return canonical_name

    available_model_names = ", ".join(sorted(OBJECT_DETECTION_CHECKPOINT_CANDIDATES))
    raise ValueError(
        f"Unsupported RF-DETR object-detection model name: `{model_name}`. "
        f"Supported model names are: {available_model_names}."
    )


def _infer_model_name_from_checkpoint_path(checkpoint_path: str) -> str | None:
    checkpoint_filename = Path(checkpoint_path).name.lower()
    for canonical_name, candidate_filenames in OBJECT_DETECTION_CHECKPOINT_CANDIDATES.items():
        if checkpoint_filename in {name.lower() for name in candidate_filenames}:
            return canonical_name
    return None


def _infer_patch_size_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int | None:
    patch_embed_weight_key = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
    patch_embed_weight = state_dict.get(patch_embed_weight_key)
    if patch_embed_weight is None or patch_embed_weight.ndim < 4:
        return None
    if patch_embed_weight.shape[-1] != patch_embed_weight.shape[-2]:
        return None
    return int(patch_embed_weight.shape[-1])


def _infer_image_size_from_position_embeddings(
    state_dict: dict[str, torch.Tensor], patch_size: int, num_register_tokens: int = 0
) -> int | None:
    position_embeddings_key = "backbone.0.encoder.encoder.embeddings.position_embeddings"
    position_embeddings = state_dict.get(position_embeddings_key)
    if position_embeddings is None or position_embeddings.ndim != 3:
        return None

    num_positions = int(position_embeddings.shape[1]) - 1 - num_register_tokens
    if num_positions <= 0:
        return None

    grid_size = int(math.sqrt(num_positions))
    if grid_size * grid_size != num_positions:
        return None

    return grid_size * patch_size


def _prepare_checkpoint_args(
    checkpoint_args: dict | argparse.Namespace | None,
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: str,
    model_name: str | None = None,
) -> tuple[dict, str | None]:
    resolved_model_name = _resolve_detection_model_name(model_name) if model_name is not None else None
    if resolved_model_name is None:
        resolved_model_name = _infer_model_name_from_checkpoint_path(checkpoint_path)

    if checkpoint_args is None:
        if resolved_model_name is None:
            raise ValueError(
                "Checkpoint did not contain an `args` entry and model name could not be inferred from checkpoint path. "
                "Please pass `--model_name`."
            )
        if resolved_model_name not in OBJECT_DETECTION_CHECKPOINT_DEFAULT_ARGS:
            raise ValueError(
                f"No default conversion args available for `{resolved_model_name}`. "
                "Please pass a checkpoint with embedded `args`."
            )
        return dict(OBJECT_DETECTION_CHECKPOINT_DEFAULT_ARGS[resolved_model_name]), resolved_model_name

    normalized_args = vars(checkpoint_args) if not isinstance(checkpoint_args, dict) else dict(checkpoint_args)
    default_args = OBJECT_DETECTION_CHECKPOINT_DEFAULT_ARGS.get(resolved_model_name, {})

    for key, default_value in default_args.items():
        if normalized_args.get(key) is None:
            normalized_args[key] = default_value

    if normalized_args.get("patch_size") is None and normalized_args.get("dinov2_patch_size") is None:
        inferred_patch_size = _infer_patch_size_from_state_dict(state_dict)
        if inferred_patch_size is not None:
            normalized_args["dinov2_patch_size"] = inferred_patch_size

    if normalized_args.get("num_windows") is None and normalized_args.get("dinov2_num_windows") is None:
        default_num_windows = default_args.get("dinov2_num_windows")
        if default_num_windows is None:
            patch_size = normalized_args.get("patch_size", normalized_args.get("dinov2_patch_size"))
            if patch_size == 14:
                default_num_windows = 4
            elif patch_size == 16:
                default_num_windows = 2
            else:
                default_num_windows = 1
        normalized_args["dinov2_num_windows"] = default_num_windows

    patch_size = normalized_args.get("patch_size", normalized_args.get("dinov2_patch_size"))
    if patch_size is not None:
        inferred_image_size = _infer_image_size_from_position_embeddings(state_dict, patch_size)
        if inferred_image_size is not None:
            normalized_args["resolution"] = inferred_image_size

    return normalized_args, resolved_model_name


def _list_object_detection_checkpoint_files(repo_id: str) -> list[str]:
    repo_files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    return sorted(
        file_name
        for file_name in repo_files
        if file_name.lower().endswith(".pth") and "seg" not in Path(file_name).name.lower()
    )


def _resolve_checkpoint_filename_for_model_name(model_name: str, checkpoint_repo_id: str) -> str:
    canonical_model_name = _resolve_detection_model_name(model_name)
    object_detection_files = _list_object_detection_checkpoint_files(checkpoint_repo_id)
    if not object_detection_files:
        raise ValueError(
            f"No RF-DETR object-detection `.pth` checkpoints were found in Hub repo `{checkpoint_repo_id}`."
        )

    basename_to_repo_file = {Path(file_name).name: file_name for file_name in object_detection_files}
    for preferred_filename in OBJECT_DETECTION_CHECKPOINT_CANDIDATES[canonical_model_name]:
        if preferred_filename in basename_to_repo_file:
            return basename_to_repo_file[preferred_filename]

    matching_files = []
    for file_name in object_detection_files:
        for pattern in OBJECT_DETECTION_MODEL_NAME_PATTERNS[canonical_model_name]:
            if re.fullmatch(pattern, file_name.lower()):
                matching_files.append(file_name)
                break

    if len(matching_files) == 1:
        return matching_files[0]

    if len(matching_files) > 1:
        raise ValueError(
            f"Multiple checkpoint files matched model name `{model_name}` in repo `{checkpoint_repo_id}`: "
            f"{matching_files}. Please pass `--checkpoint_path` explicitly."
        )

    raise ValueError(
        f"Could not find a checkpoint file for model name `{model_name}` in repo `{checkpoint_repo_id}`. "
        f"Available object-detection checkpoint files: {object_detection_files}"
    )


def _resolve_checkpoint_path(
    checkpoint_path: str | None,
    model_name: str | None,
    checkpoint_repo_id: str,
) -> str:
    if checkpoint_path is not None and model_name is not None:
        raise ValueError("Please pass either `--checkpoint_path` or `--model_name`, not both.")
    if checkpoint_path is None and model_name is None:
        raise ValueError("Please pass one of `--checkpoint_path` or `--model_name`.")
    if checkpoint_path is not None:
        return checkpoint_path

    checkpoint_filename = _resolve_checkpoint_filename_for_model_name(model_name, checkpoint_repo_id)
    print(
        f"Downloading original RF-DETR object-detection checkpoint from `{checkpoint_repo_id}`: "
        f"`{checkpoint_filename}`"
    )
    resolved_checkpoint_path = hf_hub_download(
        repo_id=checkpoint_repo_id,
        filename=checkpoint_filename,
        repo_type="model",
    )
    print(f"Downloaded checkpoint to: {resolved_checkpoint_path}")
    return resolved_checkpoint_path


def _patch_original_repo_compatibility():
    # Compatibility shims for upstream RF-DETR against bleeding-edge local Transformers.
    import transformers.pytorch_utils as pu
    import transformers.utils.backbone_utils as old_backbone_utils

    if not hasattr(pu, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            heads = set(heads) - already_pruned_heads
            mask = torch.ones(n_heads, head_size)
            heads = {h - sum(1 if h > h2 else 0 for h2 in already_pruned_heads) for h in heads}
            for head in heads:
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(old_backbone_utils, "get_aligned_output_features_output_indices"):

        def get_aligned_output_features_output_indices(out_features=None, out_indices=None, stage_names=None):
            if stage_names is None:
                stage_names = []
            if out_features is None and out_indices is None:
                out_indices = [len(stage_names) - 1]
                out_features = [stage_names[-1]] if stage_names else []
            elif out_features is None:
                out_features = [stage_names[idx] for idx in out_indices]
            elif out_indices is None:
                out_indices = [stage_names.index(name) for name in out_features]
            return out_features, out_indices

        old_backbone_utils.get_aligned_output_features_output_indices = get_aligned_output_features_output_indices

    if not hasattr(old_backbone_utils.BackboneMixin, "_init_backbone"):
        old_backbone_utils.BackboneMixin._init_backbone = (
            lambda self, config: self._init_transformers_backbone()
            if hasattr(self, "_init_transformers_backbone")
            else None
        )


def build_original_rfdetr_model(
    original_repo_path: str,
    checkpoint_args,
):
    _patch_original_repo_compatibility()

    root = Path(original_repo_path).expanduser().resolve() / "src" / "rfdetr"
    if not root.exists():
        raise ValueError(f"Could not find rfdetr sources at {root}")

    pkg = types.ModuleType("rfdetr")
    pkg.__path__ = [str(root)]
    sys.modules["rfdetr"] = pkg

    # RF-DETR imports peft even when not used.
    if "peft" not in sys.modules:
        peft = types.ModuleType("peft")

        class PeftModel(torch.nn.Module):
            pass

        peft.PeftModel = PeftModel
        sys.modules["peft"] = peft

    original_modeling = importlib.import_module("rfdetr.models.lwdetr")

    args = argparse.Namespace(**checkpoint_args)
    _with_default(args, "vit_encoder_num_layers", max(args.out_feature_indexes))
    _with_default(args, "aux_loss", True)
    _with_default(args, "dropout", 0.0)
    _with_default(args, "dim_feedforward", 2048)
    _with_default(args, "position_embedding", "sine")
    _with_default(args, "force_no_pretrain", False)
    _with_default(args, "rms_norm", False)
    _with_default(args, "use_cls_token", False)
    _with_default(args, "freeze_encoder", False)
    _with_default(args, "pretrained_encoder", None)
    _with_default(args, "window_block_indexes", None)
    _with_default(args, "drop_path", 0.0)
    _with_default(args, "shape", (args.resolution, args.resolution))
    _with_default(args, "backbone_lora", False)
    _with_default(args, "gradient_checkpointing", False)
    _with_default(args, "decoder_norm", "LN")
    _with_default(args, "mask_downsample_ratio", 4)
    _with_default(args, "segmentation_head", False)
    _with_default(args, "device", "cpu")
    _with_default(args, "pretrain_weights", None)
    _with_default(args, "encoder_only", False)
    _with_default(args, "backbone_only", False)
    _with_default(args, "patch_size", getattr(args, "dinov2_patch_size", 14))
    _with_default(args, "num_windows", getattr(args, "dinov2_num_windows", 1))
    _with_default(args, "positional_encoding_size", args.resolution // args.patch_size)

    model = original_modeling.build_model(args).eval()
    return model


def build_rf_detr_config_from_checkpoint(checkpoint_args: dict, num_labels: int | None = None) -> RfDetrConfig:
    encoder_name = _get_checkpoint_arg(checkpoint_args, "encoder")
    if "small" in encoder_name:
        hidden_size = 384
        num_attention_heads = 6
    elif "base" in encoder_name:
        hidden_size = 768
        num_attention_heads = 12
    elif "large" in encoder_name:
        hidden_size = 1024
        num_attention_heads = 16
    else:
        raise ValueError(f"Unsupported encoder in checkpoint args: {encoder_name}")

    out_feature_indexes = _get_checkpoint_arg(checkpoint_args, "out_feature_indexes")
    num_hidden_layers = _get_checkpoint_arg(
        checkpoint_args, "vit_encoder_num_layers", default=max(out_feature_indexes), required=False
    )
    out_feature_indexes_set = set(out_feature_indexes)
    window_block_indexes = [idx for idx in range(num_hidden_layers) if idx not in out_feature_indexes_set]

    patch_size = _get_checkpoint_arg(checkpoint_args, "patch_size", "dinov2_patch_size")
    num_windows = _get_checkpoint_arg(checkpoint_args, "num_windows", "dinov2_num_windows", default=1, required=False)

    backbone_config = {
        "model_type": "rf_detr_windowed_dinov2",
        "image_size": _get_checkpoint_arg(checkpoint_args, "resolution"),
        "patch_size": patch_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "mlp_ratio": 4,
        "out_indices": out_feature_indexes,
        "num_register_tokens": 4 if "registers" in encoder_name else 0,
        "num_windows": num_windows,
        "window_block_indexes": window_block_indexes,
    }

    projector_scale = _get_checkpoint_arg(checkpoint_args, "projector_scale")
    level2scalefactor = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}
    projector_scale_factors = [level2scalefactor[level] for level in projector_scale]

    return RfDetrConfig(
        backbone_config=backbone_config,
        projector_scale_factors=projector_scale_factors,
        d_model=_get_checkpoint_arg(checkpoint_args, "hidden_dim"),
        dropout=checkpoint_args.get("dropout", 0.0),
        decoder_ffn_dim=checkpoint_args.get("dim_feedforward", 2048),
        decoder_n_points=_get_checkpoint_arg(checkpoint_args, "dec_n_points"),
        decoder_layers=_get_checkpoint_arg(checkpoint_args, "dec_layers"),
        decoder_self_attention_heads=_get_checkpoint_arg(checkpoint_args, "sa_nheads"),
        decoder_cross_attention_heads=_get_checkpoint_arg(checkpoint_args, "ca_nheads"),
        num_queries=_get_checkpoint_arg(checkpoint_args, "num_queries"),
        group_detr=_get_checkpoint_arg(checkpoint_args, "group_detr"),
        auxiliary_loss=checkpoint_args.get("aux_loss", True),
        num_labels=num_labels if num_labels is not None else (_get_checkpoint_arg(checkpoint_args, "num_classes") + 1),
    )


@torch.no_grad()
def convert_rf_detr_checkpoint(
    checkpoint_path: str,
    pytorch_dump_folder_path: str,
    model_name: str | None = None,
    original_repo_path: str | None = None,
    verify_with_original: bool = False,
    push_to_hub: bool = False,
    repo_id: str | None = None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    checkpoint_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    checkpoint_args, resolved_model_name = _prepare_checkpoint_args(
        checkpoint_args=checkpoint_args,
        state_dict=state_dict,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
    )
    checkpoint_num_labels = None
    if "class_embed.bias" in state_dict:
        checkpoint_num_labels = state_dict["class_embed.bias"].shape[0]
    elif "class_embed.weight" in state_dict:
        checkpoint_num_labels = state_dict["class_embed.weight"].shape[0]

    if resolved_model_name is not None:
        print(f"Resolved RF-DETR model variant: {resolved_model_name}")

    config = build_rf_detr_config_from_checkpoint(checkpoint_args, num_labels=checkpoint_num_labels)
    model = RfDetrForObjectDetection(config).eval()

    original_state_dict = dict(state_dict)
    state_dict = dict(state_dict)
    state_dict = read_in_decoder_q_k_v(state_dict, config)

    key_mapping = ORIGINAL_TO_CONVERTED_KEY_MAPPING | get_backbone_projector_sampling_key_mapping(config)
    all_keys = list(state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys, key_mapping)

    converted_state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        if new_key == "":
            continue
        converted_state_dict[new_key] = state_dict[key]

    register_tokens_key = "model.backbone.backbone.embeddings.register_tokens"
    if register_tokens_key not in converted_state_dict:
        converted_state_dict[register_tokens_key] = (
            model.model.backbone.backbone.embeddings.register_tokens.detach().clone()
        )

    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    if missing_keys:
        print(f"First missing keys: {missing_keys[:25]}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        print(f"First unexpected keys: {unexpected_keys[:25]}")

    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    config.save_pretrained(pytorch_dump_folder_path)
    print(f"Saved converted model to: {pytorch_dump_folder_path}")

    if push_to_hub:
        repo_id = repo_id or _infer_default_repo_id(checkpoint_path, checkpoint_args)
        model.push_to_hub(repo_id=repo_id)
        print(f"Pushed converted model to Hub: {repo_id}")

    if verify_with_original:
        if original_repo_path is None:
            raise ValueError("`--original_repo_path` is required when `--verify_with_original` is set.")
        original_checkpoint_args = dict(checkpoint_args)
        if "class_embed.bias" in original_state_dict:
            original_checkpoint_args["num_classes"] = original_state_dict["class_embed.bias"].shape[0] - 1
        elif "class_embed.weight" in original_state_dict:
            original_checkpoint_args["num_classes"] = original_state_dict["class_embed.weight"].shape[0] - 1

        original_model = build_original_rfdetr_model(original_repo_path, original_checkpoint_args)
        original_model.load_state_dict(original_state_dict, strict=True)
        original_model.eval()

        image_size = config.backbone_config.image_size
        torch.manual_seed(0)
        pixel_values = torch.randn(1, 3, image_size, image_size)

        original_outputs = original_model(pixel_values)
        hf_outputs = model(pixel_values=pixel_values)

        max_abs_logits_diff = (original_outputs["pred_logits"] - hf_outputs.logits).abs().max().item()
        max_abs_boxes_diff = (original_outputs["pred_boxes"] - hf_outputs.pred_boxes).abs().max().item()

        print(f"max_abs_logits_diff={max_abs_logits_diff:.10f}")
        print(f"max_abs_boxes_diff={max_abs_boxes_diff:.10f}")
        print("original_logits_slice", original_outputs["pred_logits"].flatten()[:8])
        print("hf_logits_slice", hf_outputs.logits.flatten()[:8])
        print("original_boxes_slice", original_outputs["pred_boxes"].flatten()[:8])
        print("hf_boxes_slice", hf_outputs.pred_boxes.flatten()[:8])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to original RF-DETR checkpoint. Mutually exclusive with --model_name.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "RF-DETR object-detection model name to download from the Hub repo specified by "
            "--checkpoint_repo_id (e.g. `nano`, `small`, `medium`, `large`, `base`, `base-2`, "
            "`base-o365`)."
        ),
    )
    parser.add_argument(
        "--checkpoint_repo_id",
        type=str,
        default=DEFAULT_RF_DETR_CHECKPOINT_REPO_ID,
        help=(
            "Hub repo containing original RF-DETR checkpoints used by --model_name. "
            f"Defaults to `{DEFAULT_RF_DETR_CHECKPOINT_REPO_ID}`."
        ),
    )
    parser.add_argument("--pytorch_dump_folder_path", type=str, required=True, help="Output folder for HF model")
    parser.add_argument(
        "--verify_with_original",
        action="store_true",
        help="Run numerical comparison with the original RF-DETR implementation on dummy inputs.",
    )
    parser.add_argument(
        "--original_repo_path",
        type=str,
        default=None,
        help="Path to the rf-detr repository root (required when --verify_with_original is set).",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the converted checkpoint to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Target Hub repo id. Defaults to `nielsr/<checkpoint-name>` when --push_to_hub is enabled.",
    )
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint_path(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        checkpoint_repo_id=args.checkpoint_repo_id,
    )

    convert_rf_detr_checkpoint(
        checkpoint_path=checkpoint_path,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        model_name=args.model_name,
        original_repo_path=args.original_repo_path,
        verify_with_original=args.verify_with_original,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
