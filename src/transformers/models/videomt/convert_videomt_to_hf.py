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
"""Utilities to validate early VidEoMT layers against the reference implementation.

This first milestone validates the embedding layer only.
"""

from __future__ import annotations

import argparse

import torch

from transformers import EomtDinov3Config, VideomtConfig
from transformers.models.eomt_dinov3.modeling_eomt_dinov3 import EomtDinov3ForUniversalSegmentation
from transformers.models.videomt.modeling_videomt import VideomtEmbeddings, VideomtForUniversalSegmentation


def build_reference_backbone(model_name: str, image_size: int, patch_size: int):
    import timm

    backbone = timm.create_model(
        model_name,
        pretrained=False,
        img_size=image_size,
        patch_size=patch_size,
        num_classes=0,
        dynamic_img_size=True,
    )
    backbone.patch_embed.strict_img_size = False
    backbone.patch_embed.dynamic_img_pad = False
    backbone.dynamic_img_size = True
    backbone.eval()
    return backbone


def copy_embedding_weights(reference_backbone, hf_embeddings: VideomtEmbeddings) -> None:
    hf_embeddings.patch_embeddings.weight.data.copy_(reference_backbone.patch_embed.proj.weight.data)
    hf_embeddings.patch_embeddings.bias.data.copy_(reference_backbone.patch_embed.proj.bias.data)
    hf_embeddings.cls_token.data.copy_(reference_backbone.cls_token.data)
    if hasattr(reference_backbone, "register_tokens") and hf_embeddings.register_tokens.numel() > 0:
        hf_embeddings.register_tokens.data.copy_(reference_backbone.register_tokens.data)


def compare_embedding_layer(model_name: str, num_frames: int, image_size: int, patch_size: int, seed: int = 0) -> None:
    torch.manual_seed(seed)

    reference_backbone = build_reference_backbone(model_name, image_size=image_size, patch_size=patch_size)
    config = VideomtConfig(
        hidden_size=reference_backbone.embed_dim,
        num_attention_heads=reference_backbone.blocks[0].attn.num_heads,
        intermediate_size=reference_backbone.embed_dim * 4,
        image_size=image_size,
        patch_size=patch_size,
        num_register_tokens=reference_backbone.num_reg_tokens,
        num_hidden_layers=len(reference_backbone.blocks),
        num_frames=num_frames,
    )

    hf_embeddings = VideomtEmbeddings(config).eval()
    copy_embedding_weights(reference_backbone, hf_embeddings)

    dummy_video = torch.randn(1, num_frames, 3, image_size, image_size)
    flattened_video = dummy_video.reshape(-1, *dummy_video.shape[2:])

    with torch.no_grad():
        # Reference first-layer output at pure embedding level (patch projection + prefix tokens).
        ref_patches = reference_backbone.patch_embed.proj(flattened_video).flatten(2).transpose(1, 2)
        ref_cls = reference_backbone.cls_token.expand(ref_patches.shape[0], -1, -1)
        if hasattr(reference_backbone, "register_tokens") and reference_backbone.num_reg_tokens > 0:
            ref_reg = reference_backbone.register_tokens.expand(ref_patches.shape[0], -1, -1)
            ref_hidden_states = torch.cat([ref_cls, ref_reg, ref_patches], dim=1)
        else:
            ref_hidden_states = torch.cat([ref_cls, ref_patches], dim=1)

        hf_hidden_states = hf_embeddings(dummy_video)
        hf_hidden_states_from_flattened = hf_embeddings(flattened_video)

    # Validate masked embedding parity for common video mask layout (B, T, N).
    num_patches = (image_size // patch_size) ** 2
    video_mask = torch.zeros(1, num_frames, num_patches, dtype=torch.bool)
    video_mask[:, :, : max(1, num_patches // 8)] = True
    flattened_mask = video_mask.reshape(-1, num_patches)

    with torch.no_grad():
        hf_masked_from_5d = hf_embeddings(dummy_video, bool_masked_pos=video_mask)
        hf_masked_from_4d = hf_embeddings(flattened_video, bool_masked_pos=flattened_mask)

    masked_consistency_diff = (hf_masked_from_5d - hf_masked_from_4d).abs().max().item()

    consistency_diff = (hf_hidden_states - hf_hidden_states_from_flattened).abs().max().item()
    if consistency_diff > 0:
        print(f"hf_4d_5d_consistency_max_abs_diff={consistency_diff:.8f}")

    max_abs_diff = (hf_hidden_states - ref_hidden_states).abs().max().item()
    allclose = torch.allclose(hf_hidden_states, ref_hidden_states, atol=1e-5, rtol=1e-5)

    print(f"reference_shape={tuple(ref_hidden_states.shape)}")
    print(f"hf_4d_5d_consistency_max_abs_diff={consistency_diff:.8f}")
    print(f"hf_shape={tuple(hf_hidden_states.shape)}")
    print(f"hf_masked_4d_5d_consistency_max_abs_diff={masked_consistency_diff:.8f}")
    print(f"max_abs_diff={max_abs_diff:.8f}")
    print(f"allclose={allclose}")

    same_4d_5d = torch.allclose(hf_hidden_states, hf_hidden_states_from_flattened, atol=1e-7, rtol=1e-7)
    same_masked_4d_5d = torch.allclose(hf_masked_from_5d, hf_masked_from_4d, atol=1e-7, rtol=1e-7)

    if not same_4d_5d:
        raise ValueError("HF embeddings differ between 5D video input and flattened 4D frame input.")

    if not same_masked_4d_5d:
        raise ValueError("HF masked embeddings differ between 5D mask layout and flattened 4D layout.")

    bad_video_mask = torch.zeros(1, num_frames, max(1, num_patches - 1), dtype=torch.bool)
    caught_bad_mask_error = False
    try:
        _ = hf_embeddings(dummy_video, bool_masked_pos=bad_video_mask)
    except ValueError:
        caught_bad_mask_error = True

    print(f"hf_bad_mask_shape_error_caught={caught_bad_mask_error}")
    if not caught_bad_mask_error:
        raise ValueError("VideomtEmbeddings should raise a ValueError for invalid bool_masked_pos shape.")

    bad_batch_mask = torch.zeros(flattened_video.shape[0] + 1, num_patches, dtype=torch.bool)
    caught_bad_batch_error = False
    try:
        _ = hf_embeddings(flattened_video, bool_masked_pos=bad_batch_mask)
    except ValueError:
        caught_bad_batch_error = True

    print(f"hf_bad_mask_batch_error_caught={caught_bad_batch_error}")
    if not caught_bad_batch_error:
        raise ValueError("VideomtEmbeddings should raise a ValueError for mismatched bool_masked_pos batch size.")

    non_bool_mask = torch.zeros(flattened_video.shape[0], num_patches, dtype=torch.float32)
    caught_non_bool_mask_error = False
    try:
        _ = hf_embeddings(flattened_video, bool_masked_pos=non_bool_mask)
    except ValueError:
        caught_non_bool_mask_error = True

    print(f"hf_non_bool_mask_error_caught={caught_non_bool_mask_error}")
    if not caught_non_bool_mask_error:
        raise ValueError("VideomtEmbeddings should raise a ValueError for non-bool bool_masked_pos masks.")

    if not allclose:
        raise ValueError("Embedding layer outputs diverge from the reference implementation.")


def compare_hf_all_layers_5d_vs_4d(num_frames: int, seed: int = 0) -> None:
    """Validate hidden-state parity for all Transformer layers between 5D and flattened 4D inputs."""

    torch.manual_seed(seed)

    config = VideomtConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        image_size=32,
        patch_size=16,
        num_register_tokens=2,
        num_queries=8,
        num_blocks=1,
        attention_dropout=0.0,
        hidden_dropout_prob=0.0,
        drop_path_rate=0.0,
    )
    model = VideomtForUniversalSegmentation(config).eval()

    dummy_video = torch.randn(1, num_frames, 3, config.image_size, config.image_size)
    flattened_video = dummy_video.reshape(-1, *dummy_video.shape[2:])

    with torch.no_grad():
        hidden_5d = model.dropout(model.embeddings(dummy_video))
        hidden_4d = model.dropout(model.embeddings(flattened_video))

        pos_5d = model.rope_embeddings(dummy_video.reshape(-1, *dummy_video.shape[2:]).to(hidden_5d.dtype))
        pos_4d = model.rope_embeddings(flattened_video.to(hidden_4d.dtype))

        for layer_idx, layer_module in enumerate(model.layers):
            hidden_5d = layer_module(hidden_5d, position_embeddings=pos_5d)
            hidden_4d = layer_module(hidden_4d, position_embeddings=pos_4d)

            layer_diff = (hidden_5d - hidden_4d).abs().max().item()
            print(f"hf_layer_{layer_idx}_4d_5d_max_abs_diff={layer_diff:.8f}")

            if not torch.allclose(hidden_5d, hidden_4d, atol=1e-6, rtol=1e-6):
                raise ValueError(f"HF layer {layer_idx} hidden states differ between 5D and flattened 4D inputs.")


def compare_hf_forward_5d_vs_4d(num_frames: int, seed: int = 0) -> None:
    """Validate that model-level forward supports 5D video input equivalently to flattened 4D frames."""

    torch.manual_seed(seed)

    config = VideomtConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        image_size=32,
        patch_size=16,
        num_register_tokens=2,
        num_queries=8,
        num_blocks=1,
        num_frames=num_frames,
        attention_dropout=0.0,
        hidden_dropout_prob=0.0,
        drop_path_rate=0.0,
    )
    model = VideomtForUniversalSegmentation(config).eval()

    dummy_video = torch.randn(1, num_frames, 3, config.image_size, config.image_size)
    flattened_video = dummy_video.reshape(-1, *dummy_video.shape[2:])

    with torch.no_grad():
        outputs_5d = model(pixel_values=dummy_video)
        outputs_4d = model(pixel_values=flattened_video)

    logits_diff = (outputs_5d.class_queries_logits - outputs_4d.class_queries_logits).abs().max().item()
    masks_diff = (outputs_5d.masks_queries_logits - outputs_4d.masks_queries_logits).abs().max().item()
    hidden_diff = (outputs_5d.last_hidden_state - outputs_4d.last_hidden_state).abs().max().item()

    print(f"hf_forward_4d_5d_logits_max_abs_diff={logits_diff:.8f}")
    print(f"hf_forward_4d_5d_masks_max_abs_diff={masks_diff:.8f}")
    print(f"hf_forward_4d_5d_hidden_max_abs_diff={hidden_diff:.8f}")

    if not torch.allclose(outputs_5d.class_queries_logits, outputs_4d.class_queries_logits, atol=1e-6, rtol=1e-6):
        raise ValueError("HF class logits differ between 5D and flattened 4D forward passes.")

    if not torch.allclose(outputs_5d.masks_queries_logits, outputs_4d.masks_queries_logits, atol=1e-6, rtol=1e-6):
        raise ValueError("HF mask logits differ between 5D and flattened 4D forward passes.")

    if not torch.allclose(outputs_5d.last_hidden_state, outputs_4d.last_hidden_state, atol=1e-6, rtol=1e-6):
        raise ValueError("HF hidden states differ between 5D and flattened 4D forward passes.")

    caught_patch_offsets_error = False
    patch_offsets = [torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.long)]
    try:
        _ = model(pixel_values=dummy_video, patch_offsets=patch_offsets)
    except ValueError:
        caught_patch_offsets_error = True

    print(f"hf_video_patch_offsets_error_caught={caught_patch_offsets_error}")
    if not caught_patch_offsets_error:
        raise ValueError("VideomtForUniversalSegmentation should raise a ValueError for 5D inputs with patch_offsets.")


def _capture_layer_outputs(model: VideomtForUniversalSegmentation, pixel_values: torch.Tensor) -> list[torch.Tensor]:
    layer_outputs: list[torch.Tensor] = []
    hooks = []

    def _hook(_module, _inputs, output):
        layer_outputs.append(output.detach())

    for layer in model.layers:
        hooks.append(layer.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            _ = model(pixel_values=pixel_values)
    finally:
        for hook in hooks:
            hook.remove()

    return layer_outputs


def compare_hf_query_stage_all_layers_5d_vs_4d(num_frames: int, seed: int = 0) -> None:
    """Validate all layer outputs parity for 5D-vs-4D when query stage starts early (num_blocks=2)."""

    torch.manual_seed(seed)

    config = VideomtConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        image_size=32,
        patch_size=16,
        num_register_tokens=2,
        num_queries=8,
        num_blocks=2,
        num_frames=num_frames,
        attention_dropout=0.0,
        hidden_dropout_prob=0.0,
        drop_path_rate=0.0,
    )
    model = VideomtForUniversalSegmentation(config).eval()

    dummy_video = torch.randn(1, num_frames, 3, config.image_size, config.image_size)
    flattened_video = dummy_video.reshape(-1, *dummy_video.shape[2:])

    outputs_5d = _capture_layer_outputs(model, pixel_values=dummy_video)
    outputs_4d = _capture_layer_outputs(model, pixel_values=flattened_video)

    if len(outputs_5d) != len(outputs_4d):
        raise ValueError(f"Mismatched number of layer outputs: {len(outputs_5d)} vs {len(outputs_4d)}")

    for layer_idx, (layer_out_5d, layer_out_4d) in enumerate(zip(outputs_5d, outputs_4d)):
        layer_diff = (layer_out_5d - layer_out_4d).abs().max().item()
        print(f"hf_query_stage_layer_{layer_idx}_4d_5d_max_abs_diff={layer_diff:.8f}")

        if not torch.allclose(layer_out_5d, layer_out_4d, atol=1e-6, rtol=1e-6):
            raise ValueError(f"HF query-stage layer {layer_idx} outputs differ between 5D and flattened 4D inputs.")


def compare_videomt_full_model_against_reference(num_frames: int, seed: int = 0) -> None:
    """Validate full-model parity between VidEoMT and the original EoMT-DINOv3 implementation."""

    torch.manual_seed(seed)

    common_config_kwargs = {
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 256,
        "image_size": 32,
        "patch_size": 16,
        "num_register_tokens": 2,
        "num_queries": 8,
        "num_blocks": 2,
        "num_frames": num_frames,
        "attention_dropout": 0.0,
        "hidden_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
    }
    hf_config = VideomtConfig(**common_config_kwargs)
    reference_config = EomtDinov3Config(**common_config_kwargs)

    hf_model = VideomtForUniversalSegmentation(hf_config).eval()
    reference_model = EomtDinov3ForUniversalSegmentation(reference_config).eval()
    reference_model.load_state_dict(hf_model.state_dict(), strict=True)

    dummy_video = torch.randn(1, num_frames, 3, hf_config.image_size, hf_config.image_size)
    flattened_video = dummy_video.reshape(-1, *dummy_video.shape[2:])

    with torch.no_grad():
        hf_outputs = hf_model(pixel_values=dummy_video)
        reference_outputs = reference_model(pixel_values=flattened_video)

    logits_diff = (hf_outputs.class_queries_logits - reference_outputs.class_queries_logits).abs().max().item()
    masks_diff = (hf_outputs.masks_queries_logits - reference_outputs.masks_queries_logits).abs().max().item()
    hidden_diff = (hf_outputs.last_hidden_state - reference_outputs.last_hidden_state).abs().max().item()

    print(f"hf_vs_reference_logits_max_abs_diff={logits_diff:.8f}")
    print(f"hf_vs_reference_masks_max_abs_diff={masks_diff:.8f}")
    print(f"hf_vs_reference_hidden_max_abs_diff={hidden_diff:.8f}")

    if not torch.allclose(
        hf_outputs.class_queries_logits, reference_outputs.class_queries_logits, atol=1e-6, rtol=1e-6
    ):
        raise ValueError("VidEoMT class logits differ from reference EoMT-DINOv3 on the same dummy video input.")

    if not torch.allclose(
        hf_outputs.masks_queries_logits, reference_outputs.masks_queries_logits, atol=1e-6, rtol=1e-6
    ):
        raise ValueError("VidEoMT mask logits differ from reference EoMT-DINOv3 on the same dummy video input.")

    if not torch.allclose(hf_outputs.last_hidden_state, reference_outputs.last_hidden_state, atol=1e-6, rtol=1e-6):
        raise ValueError("VidEoMT hidden states differ from reference EoMT-DINOv3 on the same dummy video input.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate VidEoMT embedding parity against the reference model.")
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_224", help="timm model name")
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_embedding_layer(
        model_name=args.model_name,
        num_frames=args.num_frames,
        image_size=args.image_size,
        patch_size=args.patch_size,
        seed=args.seed,
    )
    compare_hf_all_layers_5d_vs_4d(num_frames=args.num_frames, seed=args.seed)
    compare_hf_forward_5d_vs_4d(num_frames=args.num_frames, seed=args.seed)

    for num_frames in sorted({args.num_frames, 3}):
        print(f"hf_query_stage_num_frames={num_frames}")
        compare_hf_query_stage_all_layers_5d_vs_4d(num_frames=num_frames, seed=args.seed)
        compare_videomt_full_model_against_reference(num_frames=num_frames, seed=args.seed)


if __name__ == "__main__":
    main()
