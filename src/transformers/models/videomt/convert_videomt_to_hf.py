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
import importlib.util
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn as nn
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


def infer_backbone_model_name(checkpoint_filename: str) -> str:
    if "vit_small" in checkpoint_filename:
        return "vit_small_patch16_dinov3_qkvb"
    if "vit_base" in checkpoint_filename:
        return "vit_base_patch16_dinov3_qkvb"
    if "vit_large" in checkpoint_filename:
        return "vit_large_patch16_dinov3_qkvb"
    raise ValueError(f"Could not infer timm backbone model from checkpoint name '{checkpoint_filename}'.")


def _build_reference_load_dict(
    original_state_dict: dict[str, torch.Tensor], reference_state_dict: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    loadable_reference_state_dict = {}
    skipped_reference_keys = []

    for key, value in original_state_dict.items():
        if not key.startswith("backbone."):
            continue

        stripped_key = key[len("backbone.") :]
        candidate_key = stripped_key

        if candidate_key.endswith(".ls1.gamma"):
            gamma_key = candidate_key.replace(".ls1.gamma", ".gamma_1")
            if gamma_key in reference_state_dict:
                candidate_key = gamma_key
        elif candidate_key.endswith(".ls2.gamma"):
            gamma_key = candidate_key.replace(".ls2.gamma", ".gamma_2")
            if gamma_key in reference_state_dict:
                candidate_key = gamma_key
        elif (
            candidate_key.endswith(".reg_token")
            and candidate_key.replace(".reg_token", ".register_tokens") in reference_state_dict
        ):
            candidate_key = candidate_key.replace(".reg_token", ".register_tokens")

        if candidate_key.endswith(".attn.qkv.bias"):
            base_key = candidate_key[: -len(".qkv.bias")]
            hidden_size = value.shape[0] // 3
            q_bias, _k_bias, v_bias = value.split(hidden_size, dim=0)
            q_bias_key = f"{base_key}.q_bias"
            v_bias_key = f"{base_key}.v_bias"
            if q_bias_key in reference_state_dict and v_bias_key in reference_state_dict:
                loadable_reference_state_dict[q_bias_key] = q_bias
                loadable_reference_state_dict[v_bias_key] = v_bias
                continue

        if candidate_key in reference_state_dict and tuple(reference_state_dict[candidate_key].shape) == tuple(
            value.shape
        ):
            loadable_reference_state_dict[candidate_key] = value
        else:
            skipped_reference_keys.append(stripped_key)

    return loadable_reference_state_dict, skipped_reference_keys


def load_reference_videomt_class(reference_repo_path: Path):
    base_path = reference_repo_path / "videomt" / "modeling" / "backbone"

    detectron2_modeling = types.ModuleType("detectron2.modeling")

    class _Backbone:
        pass

    class _Registry:
        def register(self):
            def _deco(cls):
                return cls

            return _deco

    detectron2_modeling.Backbone = _Backbone
    detectron2_modeling.BACKBONE_REGISTRY = _Registry()
    sys.modules["detectron2.modeling"] = detectron2_modeling

    for module_name in ["scale_block", "vit", "videomt"]:
        spec = importlib.util.spec_from_file_location(
            f"hf_videomt_reference.backbone.{module_name}", base_path / f"{module_name}.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module '{module_name}' from {base_path}.")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "hf_videomt_reference.backbone"
        sys.modules[f"hf_videomt_reference.backbone.{module_name}"] = module
        spec.loader.exec_module(module)

    return sys.modules["hf_videomt_reference.backbone.videomt"].VidEoMT_CLASS


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
    checkpoint_filename: str,
    image_size: int,
    num_frames: int,
    output_dir: str | None = None,
    verify: bool = False,
    reference_repo_path: str | None = None,
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

    if verify:
        verify_ok = verify_conversion_against_github_reference(
            hf_model=model,
            original_state_dict=original_state_dict,
            checkpoint_filename=checkpoint_filename,
            image_size=image_size,
            num_frames=num_frames,
            reference_repo_path=reference_repo_path,
        )
        print(f"verify_ok={verify_ok}")


def verify_conversion_against_github_reference(
    hf_model: VideomtForUniversalSegmentation,
    original_state_dict: dict[str, torch.Tensor],
    checkpoint_filename: str,
    image_size: int,
    num_frames: int,
    reference_repo_path: str | None = None,
) -> bool:
    with tempfile.TemporaryDirectory(prefix="videomt_ref_") as tmp_dir:
        repo_path = Path(reference_repo_path) if reference_repo_path is not None else Path(tmp_dir) / "videomt"
        if reference_repo_path is None:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/tue-mps/videomt", str(repo_path)],
                check=True,
                capture_output=True,
                text=True,
            )

        try:
            import timm
            from timm.layers import pos_embed_sincos

            original_create_model = timm.create_model
            original_apply_keep_indices_nlc = pos_embed_sincos.apply_keep_indices_nlc

            def _create_model_no_pretrained(*args, **kwargs):
                kwargs["pretrained"] = False
                return original_create_model(*args, **kwargs)

            def _safe_apply_keep_indices_nlc(x, pos_embed, keep_indices=None):
                if keep_indices is not None and keep_indices.dtype not in (torch.int32, torch.int64):
                    keep_indices = keep_indices.to(dtype=torch.int64)
                return original_apply_keep_indices_nlc(x, pos_embed, keep_indices)

            timm.create_model = _create_model_no_pretrained
            pos_embed_sincos.apply_keep_indices_nlc = _safe_apply_keep_indices_nlc
            reference_cls = load_reference_videomt_class(repo_path)
        finally:
            if "timm" in locals():
                timm.create_model = original_create_model
                pos_embed_sincos.apply_keep_indices_nlc = original_apply_keep_indices_nlc

        candidate_model_names = [infer_backbone_model_name(checkpoint_filename)]
        if "_qkvb" in candidate_model_names[0]:
            candidate_model_names.append(candidate_model_names[0].replace("_qkvb", ""))
        if "vit_small" in checkpoint_filename:
            candidate_model_names.append("vit_small_patch16_224")
        elif "vit_base" in checkpoint_filename:
            candidate_model_names.append("vit_base_patch16_224")
        elif "vit_large" in checkpoint_filename:
            candidate_model_names.append("vit_large_patch16_224")

        best_result = None
        for candidate_model_name in candidate_model_names:
            try:
                reference_model = reference_cls(
                    img_size=image_size,
                    num_classes=hf_model.config.num_labels,
                    name=candidate_model_name,
                    num_frames=num_frames,
                    num_q=hf_model.config.num_queries,
                    segmenter_blocks=list(
                        range(
                            hf_model.config.num_hidden_layers - hf_model.config.num_blocks,
                            hf_model.config.num_hidden_layers,
                        )
                    ),
                ).eval()

                reference_state_dict = reference_model.state_dict()
                loadable_reference_state_dict, skipped_reference_keys = _build_reference_load_dict(
                    original_state_dict, reference_state_dict
                )

                reference_load_info = reference_model.load_state_dict(loadable_reference_state_dict, strict=False)
                ref_missing = set(reference_load_info.missing_keys)
                ref_unexpected = set(reference_load_info.unexpected_keys)

                # Keep verification deterministic and avoid timm patch-drop index path differences across backbones.
                reference_model.encoder.backbone.patch_drop = nn.Identity()

                with torch.no_grad():
                    dummy_video = torch.randn(1, num_frames, 3, image_size, image_size)
                    hf_outputs = hf_model(pixel_values=dummy_video)
                    reference_outputs = reference_model(dummy_video.reshape(-1, 3, image_size, image_size))

                reference_logits = reference_outputs["pred_logits"].reshape(
                    -1, hf_model.config.num_queries, hf_model.config.num_labels + 1
                )
                reference_masks = (
                    reference_outputs["pred_masks"]
                    .permute(0, 2, 1, 3, 4)
                    .reshape(
                        -1,
                        hf_model.config.num_queries,
                        reference_outputs["pred_masks"].shape[-2],
                        reference_outputs["pred_masks"].shape[-1],
                    )
                )

                logits_diff = (hf_outputs.class_queries_logits - reference_logits).abs().max().item()
                masks_diff = (hf_outputs.masks_queries_logits - reference_masks).abs().max().item()
                score = logits_diff + masks_diff + len(ref_missing) + len(ref_unexpected)

                print(f"reference_model_name={candidate_model_name}")
                print(f"reference_missing_keys={len(ref_missing)}")
                print(f"reference_unexpected_keys={len(ref_unexpected)}")
                print(f"reference_skipped_source_keys={len(skipped_reference_keys)}")
                print(f"candidate_verify_logits_max_abs_diff={logits_diff:.8f}")
                print(f"candidate_verify_masks_max_abs_diff={masks_diff:.8f}")

                if best_result is None or score < best_result["score"]:
                    best_result = {
                        "reference_model": reference_model,
                        "name": candidate_model_name,
                        "missing": ref_missing,
                        "unexpected": ref_unexpected,
                        "skipped": skipped_reference_keys,
                        "logits_diff": logits_diff,
                        "masks_diff": masks_diff,
                        "score": score,
                    }
            except Exception as e:
                print(f"reference_model_name={candidate_model_name}")
                print(f"reference_candidate_error={type(e).__name__}: {e}")

        if best_result is None:
            return False

        reference_model = best_result["reference_model"]
        ref_missing = best_result["missing"]
        ref_unexpected = best_result["unexpected"]
        skipped_reference_keys = best_result["skipped"]

        print(f"selected_reference_model_name={best_result['name']}")
        print(f"reference_missing_keys={len(ref_missing)}")
        print(f"reference_unexpected_keys={len(ref_unexpected)}")
        print(f"reference_skipped_source_keys={len(skipped_reference_keys)}")
        if skipped_reference_keys:
            print("reference_skipped_source_key_list=")
            for key in sorted(skipped_reference_keys):
                print(f"  - {key}")

        for layer_idx in range(hf_model.config.num_hidden_layers):
            hf_qkv = torch.cat(
                [
                    hf_model.state_dict()[f"layers.{layer_idx}.attention.q_proj.weight"],
                    hf_model.state_dict()[f"layers.{layer_idx}.attention.k_proj.weight"],
                    hf_model.state_dict()[f"layers.{layer_idx}.attention.v_proj.weight"],
                ],
                dim=0,
            )
            reference_qkv = reference_model.state_dict()[f"encoder.backbone.blocks.{layer_idx}.attn.qkv.weight"]
            qkv_diff = (hf_qkv - reference_qkv).abs().max().item()
            hf_mlp_up = hf_model.state_dict()[f"layers.{layer_idx}.mlp.up_proj.weight"]
            reference_mlp_up = reference_model.state_dict()[f"encoder.backbone.blocks.{layer_idx}.mlp.fc1.weight"]
            mlp_up_diff = (hf_mlp_up - reference_mlp_up).abs().max().item()

            hf_mlp_down = hf_model.state_dict()[f"layers.{layer_idx}.mlp.down_proj.weight"]
            reference_mlp_down = reference_model.state_dict()[f"encoder.backbone.blocks.{layer_idx}.mlp.fc2.weight"]
            mlp_down_diff = (hf_mlp_down - reference_mlp_down).abs().max().item()

            print(f"verify_layer_{layer_idx}_qkv_weight_max_abs_diff={qkv_diff:.8f}")
            print(f"verify_layer_{layer_idx}_mlp_up_weight_max_abs_diff={mlp_up_diff:.8f}")
            print(f"verify_layer_{layer_idx}_mlp_down_weight_max_abs_diff={mlp_down_diff:.8f}")

        head_class_diff = (
            (hf_model.state_dict()["class_predictor.weight"] - reference_model.state_dict()["class_head.weight"])
            .abs()
            .max()
            .item()
        )
        head_mask_diff = (
            (hf_model.state_dict()["mask_head.fc1.weight"] - reference_model.state_dict()["mask_head.0.weight"])
            .abs()
            .max()
            .item()
        )
        print(f"verify_head_class_weight_max_abs_diff={head_class_diff:.8f}")
        print(f"verify_head_mask_fc1_weight_max_abs_diff={head_mask_diff:.8f}")

        dummy_video = torch.randn(1, num_frames, 3, image_size, image_size)
        with torch.no_grad():
            hf_outputs = hf_model(pixel_values=dummy_video)
            reference_outputs = reference_model(dummy_video.reshape(-1, 3, image_size, image_size))

        reference_logits = reference_outputs["pred_logits"].reshape(
            -1, hf_model.config.num_queries, hf_model.config.num_labels + 1
        )
        reference_masks = (
            reference_outputs["pred_masks"]
            .permute(0, 2, 1, 3, 4)
            .reshape(
                -1,
                hf_model.config.num_queries,
                reference_outputs["pred_masks"].shape[-2],
                reference_outputs["pred_masks"].shape[-1],
            )
        )

        logits_diff = (hf_outputs.class_queries_logits - reference_logits).abs().max().item()
        masks_diff = (hf_outputs.masks_queries_logits - reference_masks).abs().max().item()

        print(f"verify_logits_max_abs_diff={logits_diff:.8f}")
        print(f"verify_masks_max_abs_diff={masks_diff:.8f}")

        outputs_match = torch.allclose(
            hf_outputs.class_queries_logits, reference_logits, atol=1e-4, rtol=1e-4
        ) and torch.allclose(hf_outputs.masks_queries_logits, reference_masks, atol=1e-4, rtol=1e-4)
        return outputs_match and not ref_missing and not ref_unexpected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert official VidEoMT checkpoints to HF format.")
    parser.add_argument("--checkpoint-filename", type=str, required=True, help="Filename on tue-mps/VidEoMT")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--reference-repo-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(
        checkpoint_filename=args.checkpoint_filename,
        image_size=args.image_size,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        verify=args.verify,
        reference_repo_path=args.reference_repo_path,
    )


if __name__ == "__main__":
    main()
