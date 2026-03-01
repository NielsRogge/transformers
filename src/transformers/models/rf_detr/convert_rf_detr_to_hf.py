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
2. saving model + config + image processor,
3. optional numerical parity checks against the original implementation on dummy inputs.

It can be run as follows:

```bash
python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --checkpoint_path /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-small.pth --original_repo_path /Users/nielsrogge/Documents/python_projecten/rf-detr --pytorch_dump_folder_path . --verify_with_original
```

Or by model name, downloading the original checkpoint from
`nielsr/rf-detr-checkpoints`:

```bash
python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --model_name small --pytorch_dump_folder_path ./rf-detr-small-hf
```
"""

import argparse
import importlib
import json
import math
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

from transformers.models.rf_detr.image_processing_rf_detr import RfDetrImageProcessor
from transformers.models.rf_detr.image_processing_rf_detr_fast import RfDetrImageProcessorFast
from transformers.models.rf_detr.modeling_rf_detr import (
    RfDetrConfig,
    RfDetrForInstanceSegmentation,
    RfDetrForObjectDetection,
)


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

INSTANCE_SEGMENTATION_TO_CONVERTED_KEY_MAPPING = {
    r"segmentation_head.blocks.(\d+).dwconv.(weight|bias)": r"segmentation_head.blocks.\1.dwconv.\2",
    r"segmentation_head.blocks.(\d+).norm.(weight|bias)": r"segmentation_head.blocks.\1.norm.\2",
    r"segmentation_head.blocks.(\d+).pwconv1.(weight|bias)": r"segmentation_head.blocks.\1.pwconv1.\2",
    r"segmentation_head.blocks.(\d+).gamma": r"segmentation_head.blocks.\1.gamma",
    r"segmentation_head.spatial_features_proj.(weight|bias)": r"segmentation_head.spatial_features_proj.\1",
    r"segmentation_head.query_features_block.norm_in.(weight|bias)": r"segmentation_head.query_features_block.norm_in.\1",
    r"segmentation_head.query_features_block.layers.(\d+).(weight|bias)": r"segmentation_head.query_features_block.layers.\1.\2",
    r"segmentation_head.query_features_block.gamma": r"segmentation_head.query_features_block.gamma",
    r"segmentation_head.query_features_proj.(weight|bias)": r"segmentation_head.query_features_proj.\1",
    r"segmentation_head.bias": r"segmentation_head.bias",
}

MODEL_TASK_OBJECT_DETECTION = "object-detection"
MODEL_TASK_INSTANCE_SEGMENTATION = "instance-segmentation"
LABEL_DATASET_COCO = "coco"
LABEL_DATASET_OBJECT365 = "object365"
LABEL_FILES_DATASET_REPO_ID = "huggingface/label-files"
LABEL_FILES_DATASET_FILENAMES = {
    LABEL_DATASET_COCO: "coco-detection-id2label.json",
    LABEL_DATASET_OBJECT365: "object365-id2label.json",
}
MODEL_NAME_TO_LABEL_DATASET = {
    "nano": LABEL_DATASET_COCO,
    "small": LABEL_DATASET_COCO,
    "medium": LABEL_DATASET_COCO,
    "large": LABEL_DATASET_COCO,
    "base": LABEL_DATASET_COCO,
    "base-2": LABEL_DATASET_COCO,
    "base-o365": LABEL_DATASET_OBJECT365,
    "seg-preview": LABEL_DATASET_COCO,
    "seg-nano": LABEL_DATASET_COCO,
    "seg-small": LABEL_DATASET_COCO,
    "seg-medium": LABEL_DATASET_COCO,
    "seg-large": LABEL_DATASET_COCO,
    "seg-xlarge": LABEL_DATASET_COCO,
    "seg-2xlarge": LABEL_DATASET_COCO,
}

DEFAULT_RF_DETR_CHECKPOINT_REPO_ID = "nielsr/rf-detr-checkpoints"
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
INSTANCE_SEGMENTATION_CHECKPOINT_CANDIDATES = {
    "seg-preview": ["rf-detr-seg-preview.pt"],
    "seg-nano": ["rf-detr-seg-nano.pt"],
    "seg-small": ["rf-detr-seg-small.pt"],
    "seg-medium": ["rf-detr-seg-medium.pt"],
    "seg-large": ["rf-detr-seg-large.pt"],
    "seg-xlarge": ["rf-detr-seg-xlarge.pt"],
    "seg-2xlarge": ["rf-detr-seg-2xlarge.pt", "rf-detr-seg-xxlarge.pt"],
}
INSTANCE_SEGMENTATION_MODEL_NAME_ALIASES = {
    "seg-preview": {"segpreview", "rfdetrsegpreview", "instsegpreview", "instancesegpreview"},
    "seg-nano": {"segnano", "rfdetrsegnano", "instsegnano", "instancesegnano"},
    "seg-small": {"segsmall", "rfdetrsegsmall", "instsegsmall", "instancesegsmall"},
    "seg-medium": {"segmedium", "rfdetrsegmedium", "instsegmedium", "instancesegmedium"},
    "seg-large": {"seglarge", "rfdetrseglarge", "instseglarge", "instanceseglarge"},
    "seg-xlarge": {"segxlarge", "rfdetrsegxlarge", "instsegxlarge", "instancesegxlarge"},
    "seg-2xlarge": {
        "seg2xlarge",
        "segxxlarge",
        "rfdetrseg2xlarge",
        "rfdetrsegxxlarge",
        "instseg2xlarge",
        "instsegxxlarge",
        "instanceseg2xlarge",
        "instancesegxxlarge",
    },
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
INSTANCE_SEGMENTATION_MODEL_NAME_PATTERNS = {
    "seg-preview": [r"(?:.*/)?rf[-_]?detr[-_]?seg[-_]?preview(?:[-_].*)?\.(?:pt|pth)$"],
    "seg-nano": [r"(?:.*/)?rf[-_]?detr[-_]?seg[-_]?nano(?:[-_].*)?\.(?:pt|pth)$"],
    "seg-small": [r"(?:.*/)?rf[-_]?detr[-_]?seg[-_]?small(?:[-_].*)?\.(?:pt|pth)$"],
    "seg-medium": [r"(?:.*/)?rf[-_]?detr[-_]?seg[-_]?medium(?:[-_].*)?\.(?:pt|pth)$"],
    "seg-large": [r"(?:.*/)?rf[-_]?detr[-_]?seg[-_]?large(?:[-_].*)?\.(?:pt|pth)$"],
    "seg-xlarge": [r"(?:.*/)?rf[-_]?detr[-_]?seg[-_]?xlarge(?:[-_].*)?\.(?:pt|pth)$"],
    "seg-2xlarge": [r"(?:.*/)?rf[-_]?detr[-_]?seg[-_]?(?:2xlarge|xxlarge)(?:[-_].*)?\.(?:pt|pth)$"],
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
INSTANCE_SEGMENTATION_CHECKPOINT_DEFAULT_ARGS = {
    "seg-preview": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 4,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 200,
        "num_select": 200,
        "group_detr": 13,
        "resolution": 432,
        "dinov2_patch_size": 12,
        "dinov2_num_windows": 2,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
        "segmentation_head": True,
        "mask_downsample_ratio": 4,
    },
    "seg-nano": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 4,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 100,
        "num_select": 100,
        "group_detr": 13,
        "resolution": 312,
        "dinov2_patch_size": 12,
        "dinov2_num_windows": 1,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
        "segmentation_head": True,
        "mask_downsample_ratio": 4,
    },
    "seg-small": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 4,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 100,
        "group_detr": 13,
        "resolution": 384,
        "dinov2_patch_size": 12,
        "dinov2_num_windows": 2,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
        "segmentation_head": True,
        "mask_downsample_ratio": 4,
    },
    "seg-medium": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 5,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 200,
        "num_select": 200,
        "group_detr": 13,
        "resolution": 432,
        "dinov2_patch_size": 12,
        "dinov2_num_windows": 2,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
        "segmentation_head": True,
        "mask_downsample_ratio": 4,
    },
    "seg-large": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 5,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "num_select": 300,
        "group_detr": 13,
        "resolution": 504,
        "dinov2_patch_size": 12,
        "dinov2_num_windows": 2,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
        "segmentation_head": True,
        "mask_downsample_ratio": 4,
    },
    "seg-xlarge": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 6,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "num_select": 300,
        "group_detr": 13,
        "resolution": 624,
        "dinov2_patch_size": 12,
        "dinov2_num_windows": 2,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
        "segmentation_head": True,
        "mask_downsample_ratio": 4,
    },
    "seg-2xlarge": {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "hidden_dim": 256,
        "dec_n_points": 2,
        "dec_layers": 6,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "num_select": 300,
        "group_detr": 13,
        "resolution": 768,
        "dinov2_patch_size": 12,
        "dinov2_num_windows": 2,
        "vit_encoder_num_layers": 12,
        "aux_loss": True,
        "num_classes": 90,
        "segmentation_head": True,
        "mask_downsample_ratio": 4,
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


def _build_default_id2label(num_labels: int) -> dict[int, str]:
    return {idx: f"LABEL_{idx}" for idx in range(num_labels)}


def _load_id2label_from_label_files(dataset_name: str, num_labels: int) -> dict[int, str]:
    filename = LABEL_FILES_DATASET_FILENAMES[dataset_name]
    try:
        label_file_path = hf_hub_download(
            repo_id=LABEL_FILES_DATASET_REPO_ID,
            filename=filename,
            repo_type="dataset",
        )
        with open(label_file_path, encoding="utf-8") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
    except Exception as error:
        print(
            f"Could not download id2label mapping from `{LABEL_FILES_DATASET_REPO_ID}/{filename}`: {error}. "
            "Falling back to default `LABEL_{id}` labels."
        )
        return _build_default_id2label(num_labels)

    # Keep config labels aligned with the checkpoint head dimension.
    return {idx: id2label.get(idx, f"LABEL_{idx}") for idx in range(num_labels)}


def _resolve_label_dataset_name(
    checkpoint_args: dict,
    resolved_model_name: str | None,
    checkpoint_num_labels: int | None,
) -> str | None:
    if resolved_model_name is not None:
        label_dataset_name = MODEL_NAME_TO_LABEL_DATASET.get(resolved_model_name)
        if label_dataset_name is not None:
            return label_dataset_name

    dataset_file = str(checkpoint_args.get("dataset_file", "")).lower()
    if "o365" in dataset_file or "obj365" in dataset_file or "objects365" in dataset_file:
        return LABEL_DATASET_OBJECT365
    if "coco" in dataset_file:
        return LABEL_DATASET_COCO

    if checkpoint_num_labels == 366:
        return LABEL_DATASET_OBJECT365
    if checkpoint_num_labels == 91:
        return LABEL_DATASET_COCO

    return None


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


def _resolve_model_spec_from_name(model_name: str) -> tuple[str, str]:
    normalized_model_name = _normalize_model_name(model_name)

    for canonical_name, aliases in OBJECT_DETECTION_MODEL_NAME_ALIASES.items():
        if normalized_model_name == _normalize_model_name(canonical_name) or normalized_model_name in aliases:
            return MODEL_TASK_OBJECT_DETECTION, canonical_name

    for canonical_name, aliases in INSTANCE_SEGMENTATION_MODEL_NAME_ALIASES.items():
        if normalized_model_name == _normalize_model_name(canonical_name) or normalized_model_name in aliases:
            return MODEL_TASK_INSTANCE_SEGMENTATION, canonical_name

    available_detection_model_names = ", ".join(sorted(OBJECT_DETECTION_CHECKPOINT_CANDIDATES))
    available_segmentation_model_names = ", ".join(sorted(INSTANCE_SEGMENTATION_CHECKPOINT_CANDIDATES))
    raise ValueError(
        f"Unsupported RF-DETR model name: `{model_name}`. "
        "Supported object-detection model names are: "
        f"{available_detection_model_names}. "
        "Supported instance-segmentation model names are: "
        f"{available_segmentation_model_names}."
    )


def _infer_model_spec_from_checkpoint_path(checkpoint_path: str) -> tuple[str, str] | None:
    checkpoint_filename = Path(checkpoint_path).name.lower()

    for canonical_name, candidate_filenames in OBJECT_DETECTION_CHECKPOINT_CANDIDATES.items():
        if checkpoint_filename in {name.lower() for name in candidate_filenames}:
            return MODEL_TASK_OBJECT_DETECTION, canonical_name

    for canonical_name, candidate_filenames in INSTANCE_SEGMENTATION_CHECKPOINT_CANDIDATES.items():
        if checkpoint_filename in {name.lower() for name in candidate_filenames}:
            return MODEL_TASK_INSTANCE_SEGMENTATION, canonical_name

    for canonical_name, patterns in OBJECT_DETECTION_MODEL_NAME_PATTERNS.items():
        if any(re.fullmatch(pattern, checkpoint_filename) for pattern in patterns):
            return MODEL_TASK_OBJECT_DETECTION, canonical_name

    for canonical_name, patterns in INSTANCE_SEGMENTATION_MODEL_NAME_PATTERNS.items():
        if any(re.fullmatch(pattern, checkpoint_filename) for pattern in patterns):
            return MODEL_TASK_INSTANCE_SEGMENTATION, canonical_name

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
) -> tuple[dict, str | None, str | None]:
    resolved_model_task = None
    resolved_model_name = None
    if model_name is not None:
        resolved_model_task, resolved_model_name = _resolve_model_spec_from_name(model_name)
    if resolved_model_name is None:
        inferred_model_spec = _infer_model_spec_from_checkpoint_path(checkpoint_path)
        if inferred_model_spec is not None:
            resolved_model_task, resolved_model_name = inferred_model_spec

    if resolved_model_task == MODEL_TASK_INSTANCE_SEGMENTATION:
        default_args_by_model_name = INSTANCE_SEGMENTATION_CHECKPOINT_DEFAULT_ARGS
    else:
        default_args_by_model_name = OBJECT_DETECTION_CHECKPOINT_DEFAULT_ARGS

    if checkpoint_args is None:
        if resolved_model_name is None:
            raise ValueError(
                "Checkpoint did not contain an `args` entry and model name could not be inferred from checkpoint path. "
                "Please pass `--model_name`."
            )
        if resolved_model_name not in default_args_by_model_name:
            raise ValueError(
                f"No default conversion args available for `{resolved_model_name}`. "
                "Please pass a checkpoint with embedded `args`."
            )
        return dict(default_args_by_model_name[resolved_model_name]), resolved_model_name, resolved_model_task

    normalized_args = vars(checkpoint_args) if not isinstance(checkpoint_args, dict) else dict(checkpoint_args)
    default_args = default_args_by_model_name.get(resolved_model_name, {})

    for key, default_value in default_args.items():
        if normalized_args.get(key) is None:
            normalized_args[key] = default_value

    query_feat = state_dict.get("query_feat.weight")
    if query_feat is not None and query_feat.ndim >= 1:
        checkpoint_query_count = int(query_feat.shape[0])
        group_detr = int(normalized_args.get("group_detr", 1))
        model_query_count = normalized_args.get("num_queries")
        if model_query_count is None or int(model_query_count) * group_detr != checkpoint_query_count:
            if checkpoint_query_count % group_detr == 0:
                normalized_args["num_queries"] = checkpoint_query_count // group_detr
            else:
                normalized_args["group_detr"] = 1
                normalized_args["num_queries"] = checkpoint_query_count

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

    return normalized_args, resolved_model_name, resolved_model_task


def _list_checkpoint_files(repo_id: str, model_task: str) -> list[str]:
    repo_files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    filtered_checkpoint_files = []
    for file_name in repo_files:
        lower_name = file_name.lower()
        if not (lower_name.endswith(".pth") or lower_name.endswith(".pt")):
            continue
        basename = Path(file_name).name.lower()
        is_segmentation = "seg" in basename
        if model_task == MODEL_TASK_OBJECT_DETECTION and not is_segmentation:
            filtered_checkpoint_files.append(file_name)
        if model_task == MODEL_TASK_INSTANCE_SEGMENTATION and is_segmentation:
            filtered_checkpoint_files.append(file_name)
    return sorted(filtered_checkpoint_files)


def _resolve_checkpoint_filename_for_model_name(model_name: str, checkpoint_repo_id: str) -> str:
    model_task, canonical_model_name = _resolve_model_spec_from_name(model_name)
    checkpoint_files = _list_checkpoint_files(checkpoint_repo_id, model_task)
    if not checkpoint_files:
        raise ValueError(f"No RF-DETR `{model_task}` checkpoints were found in Hub repo `{checkpoint_repo_id}`.")

    if model_task == MODEL_TASK_OBJECT_DETECTION:
        checkpoint_candidates = OBJECT_DETECTION_CHECKPOINT_CANDIDATES
        checkpoint_patterns = OBJECT_DETECTION_MODEL_NAME_PATTERNS
    else:
        checkpoint_candidates = INSTANCE_SEGMENTATION_CHECKPOINT_CANDIDATES
        checkpoint_patterns = INSTANCE_SEGMENTATION_MODEL_NAME_PATTERNS

    basename_to_repo_file = {Path(file_name).name: file_name for file_name in checkpoint_files}
    for preferred_filename in checkpoint_candidates[canonical_model_name]:
        if preferred_filename in basename_to_repo_file:
            return basename_to_repo_file[preferred_filename]

    matching_files = []
    for file_name in checkpoint_files:
        for pattern in checkpoint_patterns[canonical_model_name]:
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
        f"Available `{model_task}` checkpoint files: {checkpoint_files}"
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

    model_task, _ = _resolve_model_spec_from_name(model_name)
    checkpoint_filename = _resolve_checkpoint_filename_for_model_name(model_name, checkpoint_repo_id)
    print(
        f"Downloading original RF-DETR `{model_task}` checkpoint from `{checkpoint_repo_id}`: `{checkpoint_filename}`"
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


def _ensure_original_rfdetr_importable(original_repo_path: str):
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


def build_original_rfdetr_model(
    original_repo_path: str,
    checkpoint_args,
):
    _ensure_original_rfdetr_importable(original_repo_path)
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
    _with_default(args, "layer_norm", True)
    _with_default(args, "two_stage", True)
    _with_default(args, "bbox_reparam", True)
    _with_default(args, "lite_refpoint_refine", True)
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


def build_original_rfdetr_postprocessor(original_repo_path: str, num_select: int):
    _ensure_original_rfdetr_importable(original_repo_path)
    original_modeling = importlib.import_module("rfdetr.models.lwdetr")
    return original_modeling.PostProcess(num_select=num_select).eval()


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
        mask_downsample_ratio=checkpoint_args.get("mask_downsample_ratio", 4),
        segmentation_head=checkpoint_args.get("segmentation_head", False),
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
    checkpoint_args, resolved_model_name, resolved_model_task = _prepare_checkpoint_args(
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
        if resolved_model_task is not None:
            print(f"Resolved RF-DETR model variant: {resolved_model_name} ({resolved_model_task})")
        else:
            print(f"Resolved RF-DETR model variant: {resolved_model_name}")

    config = build_rf_detr_config_from_checkpoint(checkpoint_args, num_labels=checkpoint_num_labels)
    label_dataset_name = _resolve_label_dataset_name(
        checkpoint_args=checkpoint_args,
        resolved_model_name=resolved_model_name,
        checkpoint_num_labels=checkpoint_num_labels,
    )
    if label_dataset_name is not None:
        config.id2label = _load_id2label_from_label_files(label_dataset_name, config.num_labels)
        config.label2id = {label: idx for idx, label in config.id2label.items()}
        print(f"Loaded id2label mapping for dataset `{label_dataset_name}` ({len(config.id2label)} labels).")
    else:
        print(
            "Could not infer dataset for id2label mapping from checkpoint metadata. "
            "Keeping default config label mapping."
        )

    is_instance_segmentation = resolved_model_task == MODEL_TASK_INSTANCE_SEGMENTATION or checkpoint_args.get(
        "segmentation_head", False
    )
    num_top_queries = int(checkpoint_args.get("num_select", config.num_queries))

    if is_instance_segmentation:
        model = RfDetrForInstanceSegmentation(config).eval()
    else:
        model = RfDetrForObjectDetection(config).eval()

    image_size = config.backbone_config.image_size
    image_processor = RfDetrImageProcessor(
        size={"height": image_size, "width": image_size},
        num_top_queries=num_top_queries,
    )
    image_processor_fast = RfDetrImageProcessorFast(
        size={"height": image_size, "width": image_size},
        num_top_queries=num_top_queries,
    )

    original_state_dict = dict(state_dict)
    state_dict = dict(state_dict)
    state_dict = read_in_decoder_q_k_v(state_dict, config)

    key_mapping = ORIGINAL_TO_CONVERTED_KEY_MAPPING | get_backbone_projector_sampling_key_mapping(config)
    if is_instance_segmentation:
        key_mapping = key_mapping | INSTANCE_SEGMENTATION_TO_CONVERTED_KEY_MAPPING
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
    image_processor.save_pretrained(pytorch_dump_folder_path)
    print(f"Saved converted model to: {pytorch_dump_folder_path}")

    if push_to_hub:
        repo_id = repo_id or _infer_default_repo_id(checkpoint_path, checkpoint_args)
        model.push_to_hub(repo_id=repo_id)
        image_processor.push_to_hub(repo_id=repo_id)
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

        import torchvision.transforms.functional as torchvision_transforms

        rng = np.random.default_rng(seed=0)
        dummy_image_height = image_size + 15
        dummy_image_width = image_size - 13 if image_size > 32 else image_size + 3
        dummy_image = rng.integers(0, 256, size=(dummy_image_height, dummy_image_width, 3), dtype=np.uint8)

        original_preprocessed = torchvision_transforms.to_tensor(dummy_image)
        original_preprocessed = torchvision_transforms.normalize(
            original_preprocessed,
            image_processor.image_mean,
            image_processor.image_std,
        )
        original_preprocessed = torchvision_transforms.resize(
            original_preprocessed,
            (image_size, image_size),
            interpolation=image_processor.resample,
        )
        original_preprocessed = original_preprocessed.unsqueeze(0)
        hf_preprocessed = image_processor(images=dummy_image, return_tensors="pt").pixel_values
        hf_preprocessed_fast = image_processor_fast(images=dummy_image, return_tensors="pt").pixel_values

        max_abs_preprocess_diff = (original_preprocessed - hf_preprocessed).abs().max().item()
        max_abs_preprocess_fast_diff = (original_preprocessed - hf_preprocessed_fast).abs().max().item()
        max_abs_slow_fast_preprocess_diff = (hf_preprocessed - hf_preprocessed_fast).abs().max().item()
        print(f"max_abs_preprocess_diff={max_abs_preprocess_diff:.10f}")
        print(f"max_abs_preprocess_fast_diff={max_abs_preprocess_fast_diff:.10f}")
        print(f"max_abs_slow_fast_preprocess_diff={max_abs_slow_fast_preprocess_diff:.10f}")
        print("original_preprocess_slice", original_preprocessed.flatten()[:8])
        print("hf_preprocess_slice", hf_preprocessed.flatten()[:8])
        print("hf_fast_preprocess_slice", hf_preprocessed_fast.flatten()[:8])
        if not torch.equal(original_preprocessed, hf_preprocessed):
            raise AssertionError("Preprocessing mismatch between original RF-DETR and RfDetrImageProcessor.")
        if not torch.allclose(original_preprocessed, hf_preprocessed_fast, atol=1e-6, rtol=0.0):
            raise AssertionError("Preprocessing mismatch between original RF-DETR and RfDetrImageProcessorFast.")
        if not torch.allclose(hf_preprocessed, hf_preprocessed_fast, atol=1e-6, rtol=0.0):
            raise AssertionError("Preprocessing mismatch between RfDetrImageProcessor and RfDetrImageProcessorFast.")

        pixel_values = hf_preprocessed

        original_outputs = original_model(pixel_values)
        hf_outputs = model(pixel_values=pixel_values)

        max_abs_logits_diff = (original_outputs["pred_logits"] - hf_outputs.logits).abs().max().item()
        max_abs_boxes_diff = (original_outputs["pred_boxes"] - hf_outputs.pred_boxes).abs().max().item()

        print(f"max_abs_logits_diff={max_abs_logits_diff:.10f}")
        print(f"max_abs_boxes_diff={max_abs_boxes_diff:.10f}")
        if is_instance_segmentation:
            max_abs_masks_diff = (original_outputs["pred_masks"] - hf_outputs.pred_masks).abs().max().item()
            print(f"max_abs_masks_diff={max_abs_masks_diff:.10f}")
        print("original_logits_slice", original_outputs["pred_logits"].flatten()[:8])
        print("hf_logits_slice", hf_outputs.logits.flatten()[:8])
        print("original_boxes_slice", original_outputs["pred_boxes"].flatten()[:8])
        print("hf_boxes_slice", hf_outputs.pred_boxes.flatten()[:8])
        if is_instance_segmentation:
            print("original_masks_slice", original_outputs["pred_masks"].flatten()[:8])
            print("hf_masks_slice", hf_outputs.pred_masks.flatten()[:8])

        target_sizes = torch.tensor([[dummy_image_height, dummy_image_width]], device=pixel_values.device)
        original_postprocessor = build_original_rfdetr_postprocessor(
            original_repo_path=original_repo_path,
            num_select=num_top_queries,
        )
        original_postprocessed_outputs = original_postprocessor(original_outputs, target_sizes)

        if is_instance_segmentation:
            hf_postprocessed_outputs = image_processor.post_process_instance_segmentation(
                outputs=original_outputs,
                threshold=0.0,
                mask_threshold=0.0,
                target_sizes=target_sizes,
                num_top_queries=num_top_queries,
            )
            hf_postprocessed_outputs_fast = image_processor_fast.post_process_instance_segmentation(
                outputs=original_outputs,
                threshold=0.0,
                mask_threshold=0.0,
                target_sizes=target_sizes,
                num_top_queries=num_top_queries,
            )
        else:
            hf_postprocessed_outputs = image_processor.post_process_object_detection(
                outputs=original_outputs,
                threshold=0.0,
                target_sizes=target_sizes,
                num_top_queries=num_top_queries,
            )
            hf_postprocessed_outputs_fast = image_processor_fast.post_process_object_detection(
                outputs=original_outputs,
                threshold=0.0,
                target_sizes=target_sizes,
                num_top_queries=num_top_queries,
            )

        postprocess_scores_diff = (
            (original_postprocessed_outputs[0]["scores"] - hf_postprocessed_outputs[0]["scores"]).abs().max().item()
        )
        postprocess_boxes_diff = (
            (original_postprocessed_outputs[0]["boxes"] - hf_postprocessed_outputs[0]["boxes"]).abs().max().item()
        )
        postprocess_scores_fast_diff = (
            (original_postprocessed_outputs[0]["scores"] - hf_postprocessed_outputs_fast[0]["scores"])
            .abs()
            .max()
            .item()
        )
        postprocess_boxes_fast_diff = (
            (original_postprocessed_outputs[0]["boxes"] - hf_postprocessed_outputs_fast[0]["boxes"]).abs().max().item()
        )
        postprocess_slow_fast_scores_diff = (
            (hf_postprocessed_outputs[0]["scores"] - hf_postprocessed_outputs_fast[0]["scores"]).abs().max().item()
        )
        postprocess_slow_fast_boxes_diff = (
            (hf_postprocessed_outputs[0]["boxes"] - hf_postprocessed_outputs_fast[0]["boxes"]).abs().max().item()
        )
        postprocess_labels_match = torch.equal(
            original_postprocessed_outputs[0]["labels"], hf_postprocessed_outputs[0]["labels"]
        )
        postprocess_labels_fast_match = torch.equal(
            original_postprocessed_outputs[0]["labels"], hf_postprocessed_outputs_fast[0]["labels"]
        )
        postprocess_slow_fast_labels_match = torch.equal(
            hf_postprocessed_outputs[0]["labels"], hf_postprocessed_outputs_fast[0]["labels"]
        )
        print(f"max_abs_postprocess_scores_diff={postprocess_scores_diff:.10f}")
        print(f"max_abs_postprocess_boxes_diff={postprocess_boxes_diff:.10f}")
        print(f"max_abs_postprocess_scores_fast_diff={postprocess_scores_fast_diff:.10f}")
        print(f"max_abs_postprocess_boxes_fast_diff={postprocess_boxes_fast_diff:.10f}")
        print(f"max_abs_postprocess_slow_fast_scores_diff={postprocess_slow_fast_scores_diff:.10f}")
        print(f"max_abs_postprocess_slow_fast_boxes_diff={postprocess_slow_fast_boxes_diff:.10f}")
        print(f"postprocess_labels_match={postprocess_labels_match}")
        print(f"postprocess_labels_fast_match={postprocess_labels_fast_match}")
        print(f"postprocess_slow_fast_labels_match={postprocess_slow_fast_labels_match}")
        if not torch.allclose(original_postprocessed_outputs[0]["scores"], hf_postprocessed_outputs[0]["scores"]):
            raise AssertionError("Object detection postprocess score mismatch with original RF-DETR implementation.")
        if not torch.allclose(original_postprocessed_outputs[0]["boxes"], hf_postprocessed_outputs[0]["boxes"]):
            raise AssertionError("Object detection postprocess box mismatch with original RF-DETR implementation.")
        if not postprocess_labels_match:
            raise AssertionError("Object detection postprocess labels mismatch with original RF-DETR implementation.")
        if not torch.allclose(original_postprocessed_outputs[0]["scores"], hf_postprocessed_outputs_fast[0]["scores"]):
            raise AssertionError("Object detection postprocess score mismatch with RfDetrImageProcessorFast.")
        if not torch.allclose(original_postprocessed_outputs[0]["boxes"], hf_postprocessed_outputs_fast[0]["boxes"]):
            raise AssertionError("Object detection postprocess box mismatch with RfDetrImageProcessorFast.")
        if not postprocess_labels_fast_match:
            raise AssertionError("Object detection postprocess labels mismatch with RfDetrImageProcessorFast.")
        if not torch.allclose(hf_postprocessed_outputs[0]["scores"], hf_postprocessed_outputs_fast[0]["scores"]):
            raise AssertionError("Postprocess score mismatch between slow and fast RF-DETR image processors.")
        if not torch.allclose(hf_postprocessed_outputs[0]["boxes"], hf_postprocessed_outputs_fast[0]["boxes"]):
            raise AssertionError("Postprocess box mismatch between slow and fast RF-DETR image processors.")
        if not postprocess_slow_fast_labels_match:
            raise AssertionError("Postprocess labels mismatch between slow and fast RF-DETR image processors.")
        if is_instance_segmentation:
            postprocess_masks_match = torch.equal(
                original_postprocessed_outputs[0]["masks"], hf_postprocessed_outputs[0]["masks"]
            )
            postprocess_masks_fast_match = torch.equal(
                original_postprocessed_outputs[0]["masks"], hf_postprocessed_outputs_fast[0]["masks"]
            )
            postprocess_slow_fast_masks_match = torch.equal(
                hf_postprocessed_outputs[0]["masks"], hf_postprocessed_outputs_fast[0]["masks"]
            )
            print(f"postprocess_masks_match={postprocess_masks_match}")
            print(f"postprocess_masks_fast_match={postprocess_masks_fast_match}")
            print(f"postprocess_slow_fast_masks_match={postprocess_slow_fast_masks_match}")
            if not postprocess_masks_match:
                raise AssertionError(
                    "Instance segmentation postprocess masks mismatch with original RF-DETR implementation."
                )
            if not postprocess_masks_fast_match:
                raise AssertionError("Instance segmentation postprocess masks mismatch with RfDetrImageProcessorFast.")
            if not postprocess_slow_fast_masks_match:
                raise AssertionError(
                    "Instance segmentation postprocess masks mismatch between slow and fast RF-DETR image processors."
                )


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
            "RF-DETR model name to download from the Hub repo specified by --checkpoint_repo_id "
            "(e.g. object detection: `nano`, `small`, `medium`, `large`, `base`, `base-2`, `base-o365`; "
            "instance segmentation: `seg-preview`, `seg-nano`, `seg-small`, `seg-medium`, `seg-large`, "
            "`seg-xlarge`, `seg-2xlarge`)."
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
