# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Utilities to convert DEIMv2 checkpoints to the 🤗 Transformers format.

This script currently focuses on translating the HGNetV2 visual backbone weights that are
shared across the HGNet-based DEIMv2 variants. The end-to-end detection heads are not yet
converted.
"""

import argparse
import importlib.util
import re
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from transformers import HGNetV2Backbone, HGNetV2Config
from transformers.utils import logging


logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


# HGNetV2 architectural metadata that mirrors the original implementation's ``arch_configs``.
# Each entry describes the stem channels and every stage's convolutional hyper-parameters.
HGNETV2_ARCHITECTURES: Dict[str, Dict[str, object]] = {
    "Atto": {
        "stem_channels": [3, 16, 16],
        "stages": OrderedDict(
            {
                "stage1": (16, 16, 64, 1, False, False, 3, 3),
                "stage2": (64, 32, 256, 1, True, False, 3, 3),
                "stage3": (256, 64, 256, 1, True, True, 3, 3),
            }
        ),
    },
    "Femto": {
        "stem_channels": [3, 16, 16],
        "stages": OrderedDict(
            {
                "stage1": (16, 16, 64, 1, False, False, 3, 3),
                "stage2": (64, 32, 256, 1, True, False, 3, 3),
                "stage3": (256, 64, 512, 1, True, True, 5, 3),
            }
        ),
    },
    "Pico": {
        "stem_channels": [3, 16, 16],
        "stages": OrderedDict(
            {
                "stage1": (16, 16, 64, 1, False, False, 3, 3),
                "stage2": (64, 32, 256, 1, True, False, 3, 3),
                "stage3": (256, 64, 512, 2, True, True, 5, 3),
            }
        ),
    },
    "B0": {
        "stem_channels": [3, 16, 16],
        "stages": OrderedDict(
            {
                "stage1": (16, 16, 64, 1, False, False, 3, 3),
                "stage2": (64, 32, 256, 1, True, False, 3, 3),
                "stage3": (256, 64, 512, 2, True, True, 5, 3),
                "stage4": (512, 128, 1024, 1, True, True, 5, 3),
            }
        ),
    },
    "B1": {
        "stem_channels": [3, 24, 32],
        "stages": OrderedDict(
            {
                "stage1": (32, 32, 64, 1, False, False, 3, 3),
                "stage2": (64, 48, 256, 1, True, False, 3, 3),
                "stage3": (256, 96, 512, 2, True, True, 5, 3),
                "stage4": (512, 192, 1024, 1, True, True, 5, 3),
            }
        ),
    },
    "B2": {
        "stem_channels": [3, 24, 32],
        "stages": OrderedDict(
            {
                "stage1": (32, 32, 96, 1, False, False, 3, 4),
                "stage2": (96, 64, 384, 1, True, False, 3, 4),
                "stage3": (384, 128, 768, 3, True, True, 5, 4),
                "stage4": (768, 256, 1536, 1, True, True, 5, 4),
            }
        ),
    },
    "B3": {
        "stem_channels": [3, 24, 32],
        "stages": OrderedDict(
            {
                "stage1": (32, 32, 128, 1, False, False, 3, 5),
                "stage2": (128, 64, 512, 1, True, False, 3, 5),
                "stage3": (512, 128, 1024, 3, True, True, 5, 5),
                "stage4": (1024, 256, 2048, 1, True, True, 5, 5),
            }
        ),
    },
    "B4": {
        "stem_channels": [3, 32, 48],
        "stages": OrderedDict(
            {
                "stage1": (48, 48, 128, 1, False, False, 3, 6),
                "stage2": (128, 96, 512, 1, True, False, 3, 6),
                "stage3": (512, 192, 1024, 3, True, True, 5, 6),
                "stage4": (1024, 384, 2048, 1, True, True, 5, 6),
            }
        ),
    },
    "B5": {
        "stem_channels": [3, 32, 64],
        "stages": OrderedDict(
            {
                "stage1": (64, 64, 128, 1, False, False, 3, 6),
                "stage2": (128, 128, 512, 2, True, False, 3, 6),
                "stage3": (512, 256, 1024, 5, True, True, 5, 6),
                "stage4": (1024, 512, 2048, 2, True, True, 5, 6),
            }
        ),
    },
    "B6": {
        "stem_channels": [3, 48, 96],
        "stages": OrderedDict(
            {
                "stage1": (96, 96, 192, 2, False, False, 3, 6),
                "stage2": (192, 192, 512, 3, True, False, 3, 6),
                "stage3": (512, 384, 1024, 6, True, True, 5, 6),
                "stage4": (1024, 768, 2048, 3, True, True, 5, 6),
            }
        ),
    },
}


# Mapping from the downstream DEIMv2 checkpoint identifiers to the HGNetV2 backbone variants.
DEIMV2_HGNET_BACKBONES: Dict[str, Dict[str, object]] = {
    "deimv2_hgnetv2_atto_coco": {"hgnet_name": "Atto", "return_idx": [2], "use_lab": True},
    "deimv2_hgnetv2_femto_coco": {"hgnet_name": "Femto", "return_idx": [2], "use_lab": True},
    "deimv2_hgnetv2_pico_coco": {"hgnet_name": "Pico", "return_idx": [2], "use_lab": True},
    "deimv2_hgnetv2_n_coco": {"hgnet_name": "B0", "return_idx": [2, 3], "use_lab": True},
    "deimv2_hgnetv2_s_coco": {"hgnet_name": "B0", "return_idx": [1, 2, 3], "use_lab": True},
    "deimv2_hgnetv2_m_coco": {"hgnet_name": "B2", "return_idx": [1, 2, 3], "use_lab": True},
    "deimv2_hgnetv2_l_coco": {"hgnet_name": "B4", "return_idx": [1, 2, 3], "use_lab": False},
    "deimv2_hgnetv2_x_coco": {"hgnet_name": "B5", "return_idx": [1, 2, 3], "use_lab": False},
}


KEY_RENAMING_RULES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"^stem\.stem1\.conv\.weight$"), "embedder.stem1.convolution.weight"),
    (re.compile(r"^stem\.stem1\.bn\.(weight|bias|running_mean|running_var)$"), r"embedder.stem1.normalization.\1"),
    (re.compile(r"^stem\.stem1\.lab\.(scale|bias)$"), r"embedder.stem1.lab.\1"),
    (re.compile(r"^stem\.stem2a\.conv\.weight$"), "embedder.stem2a.convolution.weight"),
    (re.compile(r"^stem\.stem2a\.bn\.(weight|bias|running_mean|running_var)$"), r"embedder.stem2a.normalization.\1"),
    (re.compile(r"^stem\.stem2a\.lab\.(scale|bias)$"), r"embedder.stem2a.lab.\1"),
    (re.compile(r"^stem\.stem2b\.conv\.weight$"), "embedder.stem2b.convolution.weight"),
    (re.compile(r"^stem\.stem2b\.bn\.(weight|bias|running_mean|running_var)$"), r"embedder.stem2b.normalization.\1"),
    (re.compile(r"^stem\.stem2b\.lab\.(scale|bias)$"), r"embedder.stem2b.lab.\1"),
    (re.compile(r"^stem\.stem3\.conv\.weight$"), "embedder.stem3.convolution.weight"),
    (re.compile(r"^stem\.stem3\.bn\.(weight|bias|running_mean|running_var)$"), r"embedder.stem3.normalization.\1"),
    (re.compile(r"^stem\.stem3\.lab\.(scale|bias)$"), r"embedder.stem3.lab.\1"),
    (re.compile(r"^stem\.stem4\.conv\.weight$"), "embedder.stem4.convolution.weight"),
    (re.compile(r"^stem\.stem4\.bn\.(weight|bias|running_mean|running_var)$"), r"embedder.stem4.normalization.\1"),
    (re.compile(r"^stem\.stem4\.lab\.(scale|bias)$"), r"embedder.stem4.lab.\1"),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv\.weight$"),
        r"encoder.stages.\1.blocks.\2.layers.\3.convolution.weight",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.bn\.(weight|bias|running_mean|running_var)$"),
        r"encoder.stages.\1.blocks.\2.layers.\3.normalization.\4",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.lab\.(scale|bias)$"),
        r"encoder.stages.\1.blocks.\2.layers.\3.lab.\4",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv1\.conv\.weight$"),
        r"encoder.stages.\1.blocks.\2.layers.\3.conv1.convolution.weight",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv1\.bn\.(weight|bias|running_mean|running_var)$"),
        r"encoder.stages.\1.blocks.\2.layers.\3.conv1.normalization.\4",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv1\.lab\.(scale|bias)$"),
        r"encoder.stages.\1.blocks.\2.layers.\3.conv1.lab.\4",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv2\.conv\.weight$"),
        r"encoder.stages.\1.blocks.\2.layers.\3.conv2.convolution.weight",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv2\.bn\.(weight|bias|running_mean|running_var)$"),
        r"encoder.stages.\1.blocks.\2.layers.\3.conv2.normalization.\4",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv2\.lab\.(scale|bias)$"),
        r"encoder.stages.\1.blocks.\2.layers.\3.conv2.lab.\4",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.aggregation\.0\.conv\.weight$"),
        r"encoder.stages.\1.blocks.\2.aggregation.0.convolution.weight",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.aggregation\.0\.bn\.(weight|bias|running_mean|running_var)$"),
        r"encoder.stages.\1.blocks.\2.aggregation.0.normalization.\3",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.aggregation\.0\.lab\.(scale|bias)$"),
        r"encoder.stages.\1.blocks.\2.aggregation.0.lab.\3",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.aggregation\.1\.conv\.weight$"),
        r"encoder.stages.\1.blocks.\2.aggregation.1.convolution.weight",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.aggregation\.1\.bn\.(weight|bias|running_mean|running_var)$"),
        r"encoder.stages.\1.blocks.\2.aggregation.1.normalization.\3",
    ),
    (
        re.compile(r"^stages\.(\d+)\.blocks\.(\d+)\.aggregation\.1\.lab\.(scale|bias)$"),
        r"encoder.stages.\1.blocks.\2.aggregation.1.lab.\3",
    ),
    (
        re.compile(r"^stages\.(\d+)\.downsample\.conv\.weight$"),
        r"encoder.stages.\1.downsample.convolution.weight",
    ),
    (
        re.compile(r"^stages\.(\d+)\.downsample\.bn\.(weight|bias|running_mean|running_var)$"),
        r"encoder.stages.\1.downsample.normalization.\2",
    ),
    (
        re.compile(r"^stages\.(\d+)\.downsample\.lab\.(scale|bias)$"),
        r"encoder.stages.\1.downsample.lab.\2",
    ),
)


def get_hgnet_config(variant: str, *, out_indices: Sequence[int], use_lab: bool) -> HGNetV2Config:
    if variant not in HGNETV2_ARCHITECTURES:
        supported = ", ".join(sorted(HGNETV2_ARCHITECTURES))
        raise ValueError(f"Unknown HGNetV2 architecture '{variant}'. Supported values: {supported}.")

    architecture = HGNETV2_ARCHITECTURES[variant]
    stem_channels: List[int] = architecture["stem_channels"]  # type: ignore[assignment]
    stage_entries: OrderedDict[str, Tuple[int, int, int, int, bool, bool, int, int]] = architecture["stages"]  # type: ignore[assignment]

    in_channels: List[int] = []
    mid_channels: List[int] = []
    out_channels: List[int] = []
    num_blocks: List[int] = []
    downsample: List[bool] = []
    light_block: List[bool] = []
    kernel_size: List[int] = []
    layer_num: List[int] = []

    for stage_name, stage_values in stage_entries.items():
        (
            stage_in,
            stage_mid,
            stage_out,
            blocks,
            do_downsample,
            uses_light_block,
            ksize,
            layers,
        ) = stage_values
        in_channels.append(stage_in)
        mid_channels.append(stage_mid)
        out_channels.append(stage_out)
        num_blocks.append(blocks)
        downsample.append(do_downsample)
        light_block.append(uses_light_block)
        kernel_size.append(ksize)
        layer_num.append(layers)

    hidden_sizes = list(out_channels)
    depths = list(num_blocks)

    out_features = [f"stage{index}" for index in range(1, len(stage_entries) + 1)]
    selected_out_features = [out_features[idx - 1] for idx in out_indices]

    config = HGNetV2Config(
        embedding_size=stem_channels[-1],
        stem_channels=stem_channels,
        stage_in_channels=in_channels,
        stage_mid_channels=mid_channels,
        stage_out_channels=out_channels,
        stage_num_blocks=num_blocks,
        stage_downsample=downsample,
        stage_light_block=light_block,
        stage_kernel_size=kernel_size,
        stage_numb_of_layers=layer_num,
        hidden_sizes=hidden_sizes,
        depths=depths,
        out_indices=list(out_indices),
        out_features=selected_out_features,
        use_learnable_affine_block=use_lab,
    )
    return config


def strip_known_prefix(key: str) -> str:
    prefixes = (
        "model.backbone.model.",
        "model.backbone.",
        "backbone.",
        "hgnetv2.",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def is_backbone_parameter(key: str) -> bool:
    base_key = strip_known_prefix(key)
    return base_key.startswith("stem.") or base_key.startswith("stages.")


def extract_backbone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    backbone_state: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if not is_backbone_parameter(key):
            continue
        base_key = strip_known_prefix(key)
        backbone_state[base_key] = tensor

    if not backbone_state:
        raise ValueError(
            "No HGNetV2 backbone weights were found in the provided checkpoint. Ensure the checkpoint contains the "
            "DEIMv2 backbone parameters."
        )

    return backbone_state


def rename_hgnet_keys(state_dict: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    renamed_state_dict: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        base_key = strip_known_prefix(key)
        new_key = None
        for pattern, replacement in KEY_RENAMING_RULES:
            if pattern.match(base_key):
                new_key = pattern.sub(replacement, base_key)
                break
        if new_key is None:
            logger.debug("Skipping key '%s' because no renaming rule matched", key)
            continue
        renamed_state_dict[new_key] = tensor
    return renamed_state_dict


def load_original_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, Mapping):
        if "model" in state and isinstance(state["model"], Mapping):
            state = state["model"]
        elif "state_dict" in state and isinstance(state["state_dict"], Mapping):
            state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a mapping when loading weights from {checkpoint_path}, received {type(state)}")
    return dict(state)


def load_original_hgnetv2_class(original_repo_path: Path):
    backbone_dir = original_repo_path / "engine" / "backbone"
    if not backbone_dir.is_dir():
        raise FileNotFoundError(
            f"Expected to find 'engine/backbone' inside {original_repo_path}, but the directory was not found."
        )

    engine_module = types.ModuleType("engine")
    engine_module.__path__ = [str(original_repo_path / "engine")]
    sys.modules.setdefault("engine", engine_module)

    backbone_module = types.ModuleType("engine.backbone")
    backbone_module.__path__ = [str(backbone_dir)]
    sys.modules.setdefault("engine.backbone", backbone_module)

    common_name = "engine.backbone.common"
    if common_name not in sys.modules:
        common_spec = importlib.util.spec_from_file_location(common_name, backbone_dir / "common.py")
        common_module = importlib.util.module_from_spec(common_spec)
        assert common_spec.loader is not None
        common_spec.loader.exec_module(common_module)
        sys.modules[common_name] = common_module

    core_name = "engine.core"
    if core_name not in sys.modules:
        core_module = types.ModuleType(core_name)

        def register(*args, **kwargs):
            def decorator(cls):
                return cls

            return decorator

        core_module.register = register
        sys.modules[core_name] = core_module

    module_name = "engine.backbone.hgnetv2"
    spec = importlib.util.spec_from_file_location(module_name, backbone_dir / "hgnetv2.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module.HGNetv2


def verify_backbone_conversion(
    *,
    original_model,
    converted_backbone: HGNetV2Backbone,
    feature_indices: Sequence[int],
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    converted_backbone.eval()
    original_model.eval()

    pixel_values = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        original_outputs = original_model(pixel_values)
        converted_outputs = converted_backbone(pixel_values).feature_maps

    if len(original_outputs) != len(converted_outputs):
        raise ValueError(
            f"The number of feature maps differs between original ({len(original_outputs)}) and converted "
            f"({len(converted_outputs)})."
        )

    for idx, (original_feature, converted_feature) in enumerate(zip(original_outputs, converted_outputs)):
        torch.testing.assert_close(converted_feature, original_feature, atol=atol, rtol=rtol)
        logger.info(
            "Feature map %d/%d (HGNet stage index %s) successfully validated with mean abs error %.6f",
            idx + 1,
            len(original_outputs),
            feature_indices[idx],
            (converted_feature - original_feature).abs().mean().item(),
        )


def save_backbone(backbone: HGNetV2Backbone, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(output_dir)
    logger.info("Backbone weights saved to %s", output_dir)


def convert_deimv2_backbone(
    model_name: str,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    original_repo_path: Optional[Path] = None,
    skip_verification: bool = False,
) -> None:
    if model_name not in DEIMV2_HGNET_BACKBONES:
        supported = ", ".join(sorted(DEIMV2_HGNET_BACKBONES))
        raise ValueError(
            f"Model '{model_name}' is not currently supported. Available HGNetV2 variants: {supported}."
        )

    variant_info = DEIMV2_HGNET_BACKBONES[model_name]
    hgnet_name: str = variant_info["hgnet_name"]  # type: ignore[assignment]
    return_indices: Sequence[int] = variant_info["return_idx"]  # type: ignore[assignment]
    use_lab: bool = variant_info["use_lab"]  # type: ignore[assignment]

    out_indices = [index + 1 for index in return_indices]

    config = get_hgnet_config(hgnet_name, out_indices=out_indices, use_lab=use_lab)
    backbone = HGNetV2Backbone(config)

    state_dict = load_original_state_dict(checkpoint_path)
    backbone_state_dict = extract_backbone_state_dict(state_dict)
    renamed_state_dict = rename_hgnet_keys(backbone_state_dict)

    missing, unexpected = backbone.load_state_dict(renamed_state_dict, strict=False)
    if missing:
        preview = ", ".join(list(missing)[:5])
        logger.warning(
            "Missing %d parameter(s) after conversion (showing first 5): %s",
            len(missing),
            preview,
        )
    if unexpected:
        preview = ", ".join(list(unexpected)[:5])
        logger.warning(
            "Encountered %d unexpected key(s) during conversion (showing first 5): %s",
            len(unexpected),
            preview,
        )

    if not skip_verification:
        if original_repo_path is None:
            raise ValueError("The original repository path must be provided to run verification.")
        try:
            HGNetv2 = load_original_hgnetv2_class(original_repo_path)
        except Exception as error:
            raise ImportError(
                "Failed to import the original HGNetv2 implementation. Ensure '--original_repo_path' points to the "
                "DEIMv2 repository root."
            ) from error

        original_model = HGNetv2(
            name=hgnet_name,
            return_idx=list(return_indices),
            use_lab=use_lab,
            freeze_at=-1,
            freeze_norm=False,
            pretrained=False,
        )
        load_result = original_model.load_state_dict(backbone_state_dict, strict=False)
        if load_result.missing_keys:
            preview = ", ".join(load_result.missing_keys[:5])
            logger.warning(
                "Original model load missing %d key(s) (showing first 5): %s",
                len(load_result.missing_keys),
                preview,
            )
        if load_result.unexpected_keys:
            preview = ", ".join(load_result.unexpected_keys[:5])
            logger.warning(
                "Original model load encountered %d unexpected key(s) (showing first 5): %s",
                len(load_result.unexpected_keys),
                preview,
            )

        verify_backbone_conversion(
            original_model=original_model,
            converted_backbone=backbone,
            feature_indices=return_indices,
        )
        logger.info("Backbone verification completed successfully.")

    save_backbone(backbone, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DEIMv2 HGNetV2 backbone checkpoints to 🤗 Transformers format.")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to the original PyTorch checkpoint (.pth).")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=sorted(DEIMV2_HGNET_BACKBONES),
        help="Identifier of the DEIMv2 variant to convert (controls the HGNet configuration).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=Path,
        required=True,
        help="Directory where the converted backbone weights will be written.",
    )
    parser.add_argument(
        "--original_repo_path",
        type=Path,
        default=None,
        help="Path to a local clone of https://github.com/Intellindust-AI-Lab/DEIMv2 for verification.",
    )
    parser.add_argument(
        "--skip_verification",
        action="store_true",
        help="Disable layer-wise verification between the original and 🤗 Transformers backbones.",
    )
    return parser.parse_args()


def main() -> None:
    logging.set_verbosity_info()
    logging.enable_default_handler()
    args = parse_args()

    convert_deimv2_backbone(
        args.model_name,
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        original_repo_path=args.original_repo_path,
        skip_verification=args.skip_verification,
    )


if __name__ == "__main__":
    main()
