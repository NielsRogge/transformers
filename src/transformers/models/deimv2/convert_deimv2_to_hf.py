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
Convert DEIMv2 checkpoints from the original repository to 🤗 Transformers.

This converter supports both HGNetv2 and DINOv3 DEIMv2 checkpoints.

Supported HGNetv2 checkpoints:
- hgnetv2_atto
- hgnetv2_femto
- hgnetv2_pico
- hgnetv2_n
- hgnetv2_s
- hgnetv2_m
- hgnetv2_l
- hgnetv2_x

Supported DINOv3 checkpoints:
- dinov3_s
- dinov3_m
- dinov3_l
- dinov3_x

Example:
```bash
python src/transformers/models/deimv2/convert_deimv2_to_hf.py \
  --model_name hgnetv2_atto \
  --checkpoint_path /path/to/deimv2_hgnetv2_atto_coco.pth \
  --original_repo_path /path/to/DEIMv2 \
  --pytorch_dump_folder_path /tmp/deimv2-atto-hf
```
"""

import argparse
import re
import sys
from pathlib import Path

import torch

from transformers import Deimv2Config, Deimv2ForObjectDetection, Deimv2ImageProcessor, DINOv3ViTConfig, HGNetV2Config


DEIMV2_CONFIG_FILES = {
    "hgnetv2_atto": "deimv2_hgnetv2_atto_coco.yml",
    "hgnetv2_femto": "deimv2_hgnetv2_femto_coco.yml",
    "hgnetv2_pico": "deimv2_hgnetv2_pico_coco.yml",
    "hgnetv2_n": "deimv2_hgnetv2_n_coco.yml",
    "hgnetv2_s": "deimv2_hgnetv2_s_coco.yml",
    "hgnetv2_m": "deimv2_hgnetv2_m_coco.yml",
    "hgnetv2_l": "deimv2_hgnetv2_l_coco.yml",
    "hgnetv2_x": "deimv2_hgnetv2_x_coco.yml",
    "dinov3_s": "deimv2_dinov3_s_coco.yml",
    "dinov3_m": "deimv2_dinov3_m_coco.yml",
    "dinov3_l": "deimv2_dinov3_l_coco.yml",
    "dinov3_x": "deimv2_dinov3_x_coco.yml",
}

HGNETV2_MODEL_NAMES = [
    "hgnetv2_atto",
    "hgnetv2_femto",
    "hgnetv2_pico",
    "hgnetv2_n",
    "hgnetv2_s",
    "hgnetv2_m",
    "hgnetv2_l",
    "hgnetv2_x",
]

DINOV3_MODEL_NAMES = ["dinov3_s", "dinov3_m", "dinov3_l", "dinov3_x"]
ALL_MODEL_NAMES = [*HGNETV2_MODEL_NAMES, *DINOV3_MODEL_NAMES]

# Copied from the original DEIMv2 HGNetv2 backbone definitions for consistent config reconstruction.
HGNETV2_ARCH_CONFIGS = {
    "Atto": {
        "stem_channels": [3, 16, 16],
        "stage_config": {
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 256, 1, True, False, 3, 3],
            "stage3": [256, 64, 256, 1, True, True, 3, 3],
        },
    },
    "Femto": {
        "stem_channels": [3, 16, 16],
        "stage_config": {
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 256, 1, True, False, 3, 3],
            "stage3": [256, 64, 512, 1, True, True, 5, 3],
        },
    },
    "Pico": {
        "stem_channels": [3, 16, 16],
        "stage_config": {
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 256, 1, True, False, 3, 3],
            "stage3": [256, 64, 512, 2, True, True, 5, 3],
        },
    },
    "B0": {
        "stem_channels": [3, 16, 16],
        "stage_config": {
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 256, 1, True, False, 3, 3],
            "stage3": [256, 64, 512, 2, True, True, 5, 3],
            "stage4": [512, 128, 1024, 1, True, True, 5, 3],
        },
    },
    "B1": {
        "stem_channels": [3, 24, 32],
        "stage_config": {
            "stage1": [32, 32, 64, 1, False, False, 3, 3],
            "stage2": [64, 48, 256, 1, True, False, 3, 3],
            "stage3": [256, 96, 512, 2, True, True, 5, 3],
            "stage4": [512, 192, 1024, 1, True, True, 5, 3],
        },
    },
    "B2": {
        "stem_channels": [3, 24, 32],
        "stage_config": {
            "stage1": [32, 32, 96, 1, False, False, 3, 4],
            "stage2": [96, 64, 384, 1, True, False, 3, 4],
            "stage3": [384, 128, 768, 3, True, True, 5, 4],
            "stage4": [768, 256, 1536, 1, True, True, 5, 4],
        },
    },
    "B3": {
        "stem_channels": [3, 24, 32],
        "stage_config": {
            "stage1": [32, 32, 128, 1, False, False, 3, 5],
            "stage2": [128, 64, 512, 1, True, False, 3, 5],
            "stage3": [512, 128, 1024, 3, True, True, 5, 5],
            "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
        },
    },
    "B4": {
        "stem_channels": [3, 32, 48],
        "stage_config": {
            "stage1": [48, 48, 128, 1, False, False, 3, 6],
            "stage2": [128, 96, 512, 1, True, False, 3, 6],
            "stage3": [512, 192, 1024, 3, True, True, 5, 6],
            "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
        },
    },
    "B5": {
        "stem_channels": [3, 32, 64],
        "stage_config": {
            "stage1": [64, 64, 128, 1, False, False, 3, 6],
            "stage2": [128, 128, 512, 2, True, False, 3, 6],
            "stage3": [512, 256, 1024, 5, True, True, 5, 6],
            "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
        },
    },
    "B6": {
        "stem_channels": [3, 48, 96],
        "stage_config": {
            "stage1": [96, 96, 192, 2, False, False, 3, 6],
            "stage2": [192, 192, 512, 3, True, False, 3, 6],
            "stage3": [512, 384, 1024, 6, True, True, 5, 6],
            "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
        },
    },
}


DINOV3_VIT_ARCH_CONFIGS = {
    "vit_tiny": {
        "hidden_size": 192,
        "intermediate_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 3,
        "hidden_act": "gelu",
        "use_gated_mlp": False,
        "num_register_tokens": 0,
        "layer_norm_eps": 1e-6,
        "layerscale_value": 1.0,
        "query_bias": True,
        "key_bias": True,
        "value_bias": True,
        "backbone_normalize_intermediate": False,
    },
    "vit_tinyplus": {
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 12,
        "num_attention_heads": 4,
        "hidden_act": "gelu",
        "use_gated_mlp": False,
        "num_register_tokens": 0,
        "layer_norm_eps": 1e-6,
        "layerscale_value": 1.0,
        "query_bias": True,
        "key_bias": True,
        "value_bias": True,
        "backbone_normalize_intermediate": False,
    },
    "dinov3_vits16": {
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "hidden_act": "gelu",
        "use_gated_mlp": False,
        "num_register_tokens": 4,
        "layer_norm_eps": 1e-5,
        "layerscale_value": 1e-5,
        "query_bias": True,
        "key_bias": False,
        "value_bias": True,
        "backbone_normalize_intermediate": True,
    },
    "dinov3_vits16plus": {
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "hidden_act": "silu",
        "use_gated_mlp": True,
        "num_register_tokens": 4,
        "layer_norm_eps": 1e-5,
        "layerscale_value": 1e-5,
        "query_bias": True,
        "key_bias": False,
        "value_bias": True,
        "backbone_normalize_intermediate": True,
    },
}


def _setup_original_repo_imports(original_repo_path: str) -> Path:
    original_repo = Path(original_repo_path).resolve()
    if not original_repo.exists():
        raise ValueError(f"Original DEIMv2 repo path does not exist: {original_repo}")

    if str(original_repo) not in sys.path:
        sys.path.insert(0, str(original_repo))

    import torch.distributed as dist

    # Original HGNetv2 init calls get_rank() unguarded.
    dist.get_rank = lambda: 0
    return original_repo


def _get_decoder_runtime_overrides(model_name: str | None) -> dict[str, bool]:
    model_name = model_name or ""
    return {
        "use_gateway": model_name not in {"hgnetv2_atto", "hgnetv2_femto", "hgnetv2_pico"},
        "share_bbox_head": model_name in {"hgnetv2_atto", "hgnetv2_femto", "hgnetv2_pico"},
    }


def _load_original_yaml_config(original_repo_path: str, config_file: str, model_name: str | None = None):
    original_repo = _setup_original_repo_imports(original_repo_path)
    import engine
    from engine.core import YAMLConfig

    _ = engine  # Keep registry side effects.
    config_path = original_repo / "configs" / "deimv2" / config_file
    if not config_path.exists():
        raise ValueError(f"Could not find config file: {config_path}")

    config_overrides = {"HGNetv2": {"pretrained": False}}
    if model_name is not None:
        config_overrides["DEIMTransformer"] = _get_decoder_runtime_overrides(model_name)
    return YAMLConfig(str(config_path), **config_overrides)


def _build_hgnetv2_backbone_config(hgnet_cfg: dict) -> HGNetV2Config:
    name = hgnet_cfg["name"]
    if name not in HGNETV2_ARCH_CONFIGS:
        raise ValueError(f"Unsupported HGNetv2 variant '{name}'.")

    arch = HGNETV2_ARCH_CONFIGS[name]
    stage_values = list(arch["stage_config"].values())

    stage_in_channels = [value[0] for value in stage_values]
    stage_mid_channels = [value[1] for value in stage_values]
    stage_out_channels = [value[2] for value in stage_values]
    stage_num_blocks = [value[3] for value in stage_values]
    stage_downsample = [value[4] for value in stage_values]
    stage_light_block = [value[5] for value in stage_values]
    stage_kernel_size = [value[6] for value in stage_values]
    stage_numb_of_layers = [value[7] for value in stage_values]

    # Original return_idx is stage-indexed (starting at 0 for stage1).
    # HGNetV2Config out_indices is stage-name indexed (0 is "stem"), so add +1.
    out_indices = [index + 1 for index in hgnet_cfg.get("return_idx", [len(stage_values) - 1])]

    return HGNetV2Config(
        out_indices=out_indices,
        depths=stage_numb_of_layers,
        hidden_sizes=stage_out_channels,
        stem_channels=arch["stem_channels"],
        stage_in_channels=stage_in_channels,
        stage_mid_channels=stage_mid_channels,
        stage_out_channels=stage_out_channels,
        stage_num_blocks=stage_num_blocks,
        stage_downsample=stage_downsample,
        stage_light_block=stage_light_block,
        stage_kernel_size=stage_kernel_size,
        stage_numb_of_layers=stage_numb_of_layers,
        use_learnable_affine_block=hgnet_cfg.get("use_lab", False),
    )


def _build_dinov3_vit_backbone_config(dinov3_cfg: dict, image_size: int = 640) -> tuple[DINOv3ViTConfig, dict]:
    name = dinov3_cfg["name"]
    if name not in DINOV3_VIT_ARCH_CONFIGS:
        raise ValueError(f"Unsupported DINOv3STAs variant '{name}'.")

    arch_cfg = DINOV3_VIT_ARCH_CONFIGS[name]
    if name.startswith("dinov3_"):
        # In the original codepath, these variants instantiate DinoVisionTransformer(name=...),
        # so YAML defaults like embed_dim=192/num_heads=3 are ignored.
        hidden_size = arch_cfg["hidden_size"]
        num_attention_heads = arch_cfg["num_attention_heads"]
    else:
        hidden_size = dinov3_cfg.get("embed_dim", arch_cfg["hidden_size"])
        num_attention_heads = dinov3_cfg.get("num_heads", arch_cfg["num_attention_heads"])

    interaction_indexes = dinov3_cfg.get("interaction_indexes", [3, 7, 11])
    out_indices = [layer_index + 1 for layer_index in interaction_indexes]
    patch_size = dinov3_cfg.get("patch_size", 16)

    backbone_config = DINOv3ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        intermediate_size=arch_cfg["intermediate_size"],
        num_hidden_layers=arch_cfg["num_hidden_layers"],
        num_attention_heads=num_attention_heads,
        hidden_act=arch_cfg["hidden_act"],
        use_gated_mlp=arch_cfg["use_gated_mlp"],
        num_register_tokens=arch_cfg["num_register_tokens"],
        layer_norm_eps=arch_cfg["layer_norm_eps"],
        layerscale_value=arch_cfg["layerscale_value"],
        query_bias=arch_cfg["query_bias"],
        key_bias=arch_cfg["key_bias"],
        value_bias=arch_cfg["value_bias"],
        out_indices=out_indices,
    )

    backbone_kwargs = {
        "backbone_interaction_indexes": interaction_indexes,
        "backbone_conv_inplane": dinov3_cfg.get("conv_inplane", 16),
        "backbone_hidden_dim": dinov3_cfg.get("hidden_dim", backbone_config.hidden_size),
        "backbone_use_sta": dinov3_cfg.get("use_sta", True),
        "backbone_normalize_intermediate": arch_cfg["backbone_normalize_intermediate"],
    }
    return backbone_config, backbone_kwargs


def _build_hf_config_from_yaml(yaml_cfg: dict, model_name: str | None = None) -> Deimv2Config:
    deim_cfg = yaml_cfg["DEIM"]
    backbone_type = deim_cfg.get("backbone", "HGNetv2")
    encoder_type = deim_cfg.get("encoder", "HybridEncoder")

    if encoder_type not in {"LiteEncoder", "HybridEncoder"}:
        raise ValueError(f"Unsupported encoder type '{encoder_type}'.")

    encoder_cfg = yaml_cfg[encoder_type]
    decoder_cfg = yaml_cfg["DEIMTransformer"]
    anchor_image_size = yaml_cfg.get("eval_spatial_size", [640, 640])

    if backbone_type == "HGNetv2":
        hgnet_cfg = yaml_cfg["HGNetv2"]
        backbone_config = _build_hgnetv2_backbone_config(hgnet_cfg)
        backbone_kwargs = {}
        freeze_backbone_batch_norms = hgnet_cfg.get("freeze_norm", False)
    elif backbone_type == "DINOv3STAs":
        dinov3_cfg = yaml_cfg["DINOv3STAs"]
        backbone_config, backbone_kwargs = _build_dinov3_vit_backbone_config(
            dinov3_cfg=dinov3_cfg,
            image_size=anchor_image_size[0],
        )
        freeze_backbone_batch_norms = False
    else:
        raise ValueError(f"Unsupported backbone type '{backbone_type}'.")

    decoder_overrides = _get_decoder_runtime_overrides(model_name)

    return Deimv2Config(
        num_labels=yaml_cfg.get("num_classes", 80),
        backbone_config=backbone_config,
        encoder_type="lite" if encoder_type == "LiteEncoder" else "hybrid",
        encoder_in_channels=encoder_cfg["in_channels"],
        encoder_hidden_dim=encoder_cfg["hidden_dim"],
        encoder_attention_heads=encoder_cfg.get("nhead", 8),
        encoder_ffn_dim=encoder_cfg.get("dim_feedforward", 1024),
        encoder_layers=encoder_cfg.get("num_encoder_layers", 1),
        encode_proj_layers=encoder_cfg.get("use_encoder_idx", [len(encoder_cfg["in_channels"]) - 1]),
        feat_strides=decoder_cfg.get("feat_strides", encoder_cfg.get("feat_strides", [8, 16, 32])),
        d_model=decoder_cfg["hidden_dim"],
        decoder_in_channels=decoder_cfg["feat_channels"],
        num_feature_levels=decoder_cfg.get("num_levels", len(decoder_cfg["feat_channels"])),
        decoder_n_points=decoder_cfg["num_points"],
        decoder_layers=decoder_cfg["num_layers"],
        decoder_ffn_dim=decoder_cfg["dim_feedforward"],
        num_queries=decoder_cfg.get("num_queries", 300),
        num_denoising=decoder_cfg.get("num_denoising", 100),
        label_noise_ratio=decoder_cfg.get("label_noise_ratio", 0.5),
        box_noise_scale=decoder_cfg.get("box_noise_scale", 1.0),
        use_gateway=decoder_overrides["use_gateway"],
        share_bbox_head=decoder_overrides["share_bbox_head"],
        share_score_head=decoder_cfg.get("share_score_head", False),
        hidden_expansion=encoder_cfg.get("expansion", 1.0),
        depth_mult=encoder_cfg.get("depth_mult", 1.0),
        layer_scale=decoder_cfg.get("layer_scale", 1),
        eval_idx=decoder_cfg.get("eval_idx", -1),
        max_num_bins=decoder_cfg.get("reg_max", 32),
        reg_scale=float(decoder_cfg.get("reg_scale", 4.0)),
        activation_function=decoder_cfg.get("activation", "silu"),
        decoder_activation_function=decoder_cfg.get("mlp_act", "silu"),
        encoder_activation_function=encoder_cfg.get("enc_act", "gelu"),
        decoder_method=decoder_cfg.get("cross_attn_method", "default"),
        anchor_image_size=anchor_image_size,
        freeze_backbone_batch_norms=freeze_backbone_batch_norms,
        encoder_version=encoder_cfg.get("version", "deim"),
        encoder_csp_type=encoder_cfg.get("csp_type", "csp"),
        encoder_fuse_op=encoder_cfg.get("fuse_op", "sum"),
        **backbone_kwargs,
    )


def get_deimv2_config(model_name: str, original_repo_path: str) -> tuple[Deimv2Config, str]:
    if model_name not in DEIMV2_CONFIG_FILES:
        supported = ", ".join(DEIMV2_CONFIG_FILES)
        raise ValueError(f"Unknown model_name '{model_name}'. Supported values: {supported}")

    config_file = DEIMV2_CONFIG_FILES[model_name]
    original_yaml_config = _load_original_yaml_config(original_repo_path, config_file, model_name=model_name)
    return _build_hf_config_from_yaml(original_yaml_config.yaml_cfg, model_name=model_name), config_file


def map_backbone_key(original_key: str) -> str | None:
    if original_key.startswith("backbone.stem."):
        new_key = "model.backbone.model.embedder." + original_key[len("backbone.stem.") :]
    elif original_key.startswith("backbone.stages."):
        new_key = "model.backbone.model.encoder.stages." + original_key[len("backbone.stages.") :]
    elif original_key.startswith("backbone.sta."):
        new_key = "model.backbone.sta." + original_key[len("backbone.sta.") :]
    elif original_key.startswith("backbone.convs."):
        new_key = "model.backbone.convs." + original_key[len("backbone.convs.") :]
    elif original_key.startswith("backbone.norms."):
        new_key = "model.backbone.norms." + original_key[len("backbone.norms.") :]
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
    if original_key == "decoder.anchors":
        return "model.anchors"
    if original_key == "decoder.valid_mask":
        return "model.valid_mask"
    return "model." + original_key


def _map_and_store_value(
    mapped_key: str,
    value: torch.Tensor,
    hf_state_dict: dict[str, torch.Tensor],
    converted_state_dict: dict[str, torch.Tensor],
) -> bool:
    if mapped_key not in hf_state_dict:
        return False
    if hf_state_dict[mapped_key].shape != value.shape:
        return False
    converted_state_dict[mapped_key] = value
    return True


def _convert_dinov3_backbone_key(
    original_key: str,
    value: torch.Tensor,
    hf_state_dict: dict[str, torch.Tensor],
    converted_state_dict: dict[str, torch.Tensor],
) -> bool:
    key = original_key[len("backbone.dinov3.") :]
    key = key.removeprefix("_model.")

    qkv_match = re.match(r"^blocks\.(\d+)\.attn\.qkv\.(weight|bias)$", key)
    if qkv_match is not None:
        layer_idx = qkv_match.group(1)
        param_name = qkv_match.group(2)
        q_value, k_value, v_value = torch.chunk(value, 3, dim=0)
        mapped_targets = {
            "q": f"model.backbone.model.layer.{layer_idx}.attention.q_proj.{param_name}",
            "k": f"model.backbone.model.layer.{layer_idx}.attention.k_proj.{param_name}",
            "v": f"model.backbone.model.layer.{layer_idx}.attention.v_proj.{param_name}",
        }

        success_q = _map_and_store_value(mapped_targets["q"], q_value, hf_state_dict, converted_state_dict)
        success_v = _map_and_store_value(mapped_targets["v"], v_value, hf_state_dict, converted_state_dict)
        _ = _map_and_store_value(mapped_targets["k"], k_value, hf_state_dict, converted_state_dict)
        return success_q and success_v

    if key == "rope_embed.periods":
        return _map_and_store_value("model.backbone.rope_periods", value, hf_state_dict, converted_state_dict)

    if key.endswith("attn.qkv.bias_mask"):
        return True

    replacements = {
        r"^cls_token$": "embeddings.cls_token",
        r"^mask_token$": "embeddings.mask_token",
        r"^storage_tokens$": "embeddings.register_tokens",
        r"^patch_embed\.proj\.(.*)$": r"embeddings.patch_embeddings.\1",
        r"^blocks\.(\d+)\.attn\.proj\.(.*)$": r"layer.\1.attention.o_proj.\2",
        r"^blocks\.(\d+)\.ls(\d+)\.gamma$": r"layer.\1.layer_scale\2.lambda1",
        r"^blocks\.(\d+)\.mlp\.fc1\.(.*)$": r"layer.\1.mlp.up_proj.\2",
        r"^blocks\.(\d+)\.mlp\.fc2\.(.*)$": r"layer.\1.mlp.down_proj.\2",
        r"^blocks\.(\d+)\.mlp\.w1\.(.*)$": r"layer.\1.mlp.gate_proj.\2",
        r"^blocks\.(\d+)\.mlp\.w2\.(.*)$": r"layer.\1.mlp.up_proj.\2",
        r"^blocks\.(\d+)\.mlp\.w3\.(.*)$": r"layer.\1.mlp.down_proj.\2",
        r"^blocks\.(\d+)\.norm1\.(.*)$": r"layer.\1.norm1.\2",
        r"^blocks\.(\d+)\.norm2\.(.*)$": r"layer.\1.norm2.\2",
        r"^norm\.(.*)$": r"norm.\1",
    }

    mapped_suffix = None
    for pattern, replacement in replacements.items():
        if re.match(pattern, key):
            mapped_candidate = re.sub(pattern, replacement, key)
            mapped_suffix = mapped_candidate
            break

    if mapped_suffix is None:
        return False

    mapped_key = f"model.backbone.model.{mapped_suffix}"
    if mapped_key.endswith("embeddings.mask_token"):
        value = value.unsqueeze(1)
    return _map_and_store_value(mapped_key, value, hf_state_dict, converted_state_dict)


def convert_state_dict(
    original_state_dict: dict[str, torch.Tensor], hf_state_dict: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    converted_state_dict = {}
    skipped = []

    for key, value in original_state_dict.items():
        if key.endswith("num_batches_tracked"):
            continue

        if key.startswith("backbone.dinov3."):
            if not _convert_dinov3_backbone_key(key, value, hf_state_dict, converted_state_dict):
                skipped.append(key)
            continue

        mapped_key = map_key(key)

        if mapped_key not in hf_state_dict:
            skipped.append(key)
            continue

        if hf_state_dict[mapped_key].shape != value.shape:
            skipped.append(key)
            continue

        converted_state_dict[mapped_key] = value

    return converted_state_dict, skipped


def fill_optional_dinov3_backbone_defaults(
    converted_state_dict: dict[str, torch.Tensor], hf_state_dict: dict[str, torch.Tensor]
) -> None:
    for key, value in hf_state_dict.items():
        if key in converted_state_dict:
            continue
        if (
            key
            in {
                "model.backbone.model.embeddings.mask_token",
                "model.backbone.model.embeddings.register_tokens",
                "model.backbone.model.norm.weight",
                "model.backbone.model.norm.bias",
            }
            or ".layer_scale" in key
        ):
            converted_state_dict[key] = value


def load_original_model(
    original_repo_path: str,
    original_checkpoint_path: str,
    config_file: str,
    model_name: str | None = None,
):
    original_cfg = _load_original_yaml_config(original_repo_path, config_file, model_name=model_name)
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
    image_size = hf_model.config.anchor_image_size if hf_model.config.anchor_image_size is not None else [320, 320]
    dummy_height, dummy_width = image_size
    dummy_pixel_values = torch.randn(1, 3, dummy_height, dummy_width)
    dummy_pixel_mask = torch.ones((1, dummy_height, dummy_width))

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


def convert_single_deimv2_checkpoint(
    model_name: str,
    checkpoint_path: str,
    pytorch_dump_folder_path: str | None,
    original_repo_path: str,
    skip_parity_check: bool,
):
    config, config_file = get_deimv2_config(model_name, original_repo_path)
    original_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    model = Deimv2ForObjectDetection(config)
    model.eval()

    converted_state_dict, skipped = convert_state_dict(original_state_dict, model.state_dict())
    if model_name in DINOV3_MODEL_NAMES:
        fill_optional_dinov3_backbone_defaults(converted_state_dict, model.state_dict())
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

    if not skip_parity_check:
        original_model = load_original_model(original_repo_path, checkpoint_path, config_file, model_name=model_name)
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
        if config.anchor_image_size is not None:
            processor = Deimv2ImageProcessor(
                size={"height": config.anchor_image_size[0], "width": config.anchor_image_size[1]}
            )
        else:
            processor = Deimv2ImageProcessor()
        processor.save_pretrained(output_dir)
        print(f"Saved converted model and processor to {output_dir}")


def convert_all_checkpoints(
    model_names: list[str],
    checkpoint_dir: str,
    pytorch_dump_folder_path: str,
    original_repo_path: str,
    skip_parity_check: bool,
):
    checkpoint_root = Path(checkpoint_dir).resolve()
    output_root = Path(pytorch_dump_folder_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        config_file = DEIMV2_CONFIG_FILES[model_name]
        checkpoint_file = config_file.replace(".yml", ".pth")
        checkpoint_path = checkpoint_root / checkpoint_file
        if not checkpoint_path.exists():
            print(f"[SKIP] Missing checkpoint for {model_name}: {checkpoint_path}")
            continue

        output_path = output_root / model_name
        print(f"[RUN] Converting {model_name} from {checkpoint_path}")
        convert_single_deimv2_checkpoint(
            model_name=model_name,
            checkpoint_path=str(checkpoint_path),
            pytorch_dump_folder_path=str(output_path),
            original_repo_path=original_repo_path,
            skip_parity_check=skip_parity_check,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="hgnetv2_atto",
        choices=[*DEIMV2_CONFIG_FILES.keys(), "all_hgnetv2", "all_dinov3", "all"],
        help="Which DEIMv2 variant to convert.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/Users/nielsrogge/Downloads/deimv2_hgnetv2_atto_coco.pth",
        help="Path to a single DEIMv2 checkpoint (.pth).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing multiple checkpoint files named like deimv2_<variant>_coco.pth.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        default=None,
        help="Where to save the converted HF model (or output root for --model_name all*).",
    )
    parser.add_argument(
        "--original_repo_path",
        type=str,
        default="/Users/nielsrogge/Documents/python_projecten/DEIMv2",
        help="Path to the original DEIMv2 repository.",
    )
    parser.add_argument(
        "--skip_parity_check",
        action="store_true",
        help="Skip original-vs-HF parity check.",
    )
    args = parser.parse_args()

    if args.model_name in {"all_hgnetv2", "all_dinov3", "all"}:
        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint_dir must be provided when --model_name is all_hgnetv2/all_dinov3/all.")
        if args.pytorch_dump_folder_path is None:
            raise ValueError(
                "--pytorch_dump_folder_path must be provided when --model_name is all_hgnetv2/all_dinov3/all."
            )
        if args.model_name == "all_hgnetv2":
            model_names = HGNETV2_MODEL_NAMES
        elif args.model_name == "all_dinov3":
            model_names = DINOV3_MODEL_NAMES
        else:
            model_names = ALL_MODEL_NAMES
        convert_all_checkpoints(
            model_names=model_names,
            checkpoint_dir=args.checkpoint_dir,
            pytorch_dump_folder_path=args.pytorch_dump_folder_path,
            original_repo_path=args.original_repo_path,
            skip_parity_check=args.skip_parity_check,
        )
    else:
        convert_single_deimv2_checkpoint(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            pytorch_dump_folder_path=args.pytorch_dump_folder_path,
            original_repo_path=args.original_repo_path,
            skip_parity_check=args.skip_parity_check,
        )
