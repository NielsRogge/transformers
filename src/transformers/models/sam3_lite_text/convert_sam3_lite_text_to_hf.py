# Copyright 2026 The HuggingFace Team. All rights reserved.
"""Convert EfficientSAM3 LiteText checkpoints to Hugging Face format."""

import argparse
import json
import os
import sys
from pathlib import Path

import regex as re
import torch
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.errors import HfHubHTTPError

from transformers import Sam3LiteTextConfig, Sam3LiteTextModel
from transformers.models.sam3.convert_sam3_to_hf import convert_old_keys_to_new_keys
from transformers.models.sam3_lite_text.configuration_sam3_lite_text import (
    Sam3LiteTextVisionConfig,
    Sam3LiteTextViTConfig,
)


TEXT_KEY_MAPPING = {
    r"^backbone\.language_backbone\.encoder\.embedding_layer\.": r"text_encoder.token_embedding.",
    r"^backbone\.language_backbone\.encoder\.positional_embedding\.pos_embed\.pos_embed$": r"text_encoder.position_embedding.position_embedding",
    r"^backbone\.language_backbone\.encoder\.final_layer_norm\.": r"text_encoder.final_layer_norm.",
    r"^backbone\.language_backbone\.encoder\.projection_layer$": r"text_encoder.projection",
    r"^backbone\.language_backbone\.projector\.": r"text_projection.",
    # RepMixer blocks (0 and 5)
    r"^backbone\.language_backbone\.encoder\.transformer\.0\.layer_scale$": r"text_encoder.layers.0.layer_scale",
    r"^backbone\.language_backbone\.encoder\.transformer\.0\.token_mixer\.layer_scale$": r"text_encoder.layers.0.token_mixer.layer_scale",
    r"^backbone\.language_backbone\.encoder\.transformer\.0\.token_mixer\.norm\.rbr_skip\.": r"text_encoder.layers.0.token_mixer.norm.rbr_skip.",
    r"^backbone\.language_backbone\.encoder\.transformer\.0\.token_mixer\.mixer\.rbr_skip\.": r"text_encoder.layers.0.token_mixer.mixer.rbr_skip.",
    r"^backbone\.language_backbone\.encoder\.transformer\.0\.token_mixer\.mixer\.rbr_conv\.0\.conv\.": r"text_encoder.layers.0.token_mixer.mixer.rbr_conv.0.0.",
    r"^backbone\.language_backbone\.encoder\.transformer\.0\.token_mixer\.mixer\.rbr_conv\.0\.bn\.": r"text_encoder.layers.0.token_mixer.mixer.rbr_conv.0.1.",
    r"^backbone\.language_backbone\.encoder\.transformer\.0\.convffn\.conv\.conv\.": r"text_encoder.layers.0.convffn.conv.0.",
    r"^backbone\.language_backbone\.encoder\.transformer\.0\.convffn\.conv\.bn\.": r"text_encoder.layers.0.convffn.conv.1.",
    r"^backbone\.language_backbone\.encoder\.transformer\.0\.convffn\.fc1\.": r"text_encoder.layers.0.convffn.fc1.",
    r"^backbone\.language_backbone\.encoder\.transformer\.0\.convffn\.fc2\.": r"text_encoder.layers.0.convffn.fc2.",
    r"^backbone\.language_backbone\.encoder\.transformer\.5\.layer_scale$": r"text_encoder.layers.5.layer_scale",
    r"^backbone\.language_backbone\.encoder\.transformer\.5\.token_mixer\.layer_scale$": r"text_encoder.layers.5.token_mixer.layer_scale",
    r"^backbone\.language_backbone\.encoder\.transformer\.5\.token_mixer\.norm\.rbr_skip\.": r"text_encoder.layers.5.token_mixer.norm.rbr_skip.",
    r"^backbone\.language_backbone\.encoder\.transformer\.5\.token_mixer\.mixer\.rbr_skip\.": r"text_encoder.layers.5.token_mixer.mixer.rbr_skip.",
    r"^backbone\.language_backbone\.encoder\.transformer\.5\.token_mixer\.mixer\.rbr_conv\.0\.conv\.": r"text_encoder.layers.5.token_mixer.mixer.rbr_conv.0.0.",
    r"^backbone\.language_backbone\.encoder\.transformer\.5\.token_mixer\.mixer\.rbr_conv\.0\.bn\.": r"text_encoder.layers.5.token_mixer.mixer.rbr_conv.0.1.",
    r"^backbone\.language_backbone\.encoder\.transformer\.5\.convffn\.conv\.conv\.": r"text_encoder.layers.5.convffn.conv.0.",
    r"^backbone\.language_backbone\.encoder\.transformer\.5\.convffn\.conv\.bn\.": r"text_encoder.layers.5.convffn.conv.1.",
    r"^backbone\.language_backbone\.encoder\.transformer\.5\.convffn\.fc1\.": r"text_encoder.layers.5.convffn.fc1.",
    r"^backbone\.language_backbone\.encoder\.transformer\.5\.convffn\.fc2\.": r"text_encoder.layers.5.convffn.fc2.",
}

for i in range(12):
    TEXT_KEY_MAPPING.update(
        {
            rf"^backbone\.language_backbone\.encoder\.transformer\.{i}\.pre_norm_mha\.0\.": rf"text_encoder.layers.{i}.layer_norm1.",
            rf"^backbone\.language_backbone\.encoder\.transformer\.{i}\.pre_norm_mha\.1\.qkv_proj\.": rf"text_encoder.layers.{i}.attention.in_proj_",
            rf"^backbone\.language_backbone\.encoder\.transformer\.{i}\.pre_norm_mha\.1\.out_proj\.": rf"text_encoder.layers.{i}.attention.out_proj.",
            rf"^backbone\.language_backbone\.encoder\.transformer\.{i}\.pre_norm_ffn\.0\.": rf"text_encoder.layers.{i}.layer_norm2.",
            rf"^backbone\.language_backbone\.encoder\.transformer\.{i}\.pre_norm_ffn\.1\.": rf"text_encoder.layers.{i}.fc1.",
            rf"^backbone\.language_backbone\.encoder\.transformer\.{i}\.pre_norm_ffn\.4\.": rf"text_encoder.layers.{i}.fc2.",
        }
    )


def split_qkv_for_sam3_lite_text(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    vision_keys_to_split = [key for key in state_dict.keys() if ".attention.qkv." in key]
    for key in vision_keys_to_split:
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state_dict[key.replace(".qkv.", ".q_proj.")] = q
        state_dict[key.replace(".qkv.", ".k_proj.")] = k
        state_dict[key.replace(".qkv.", ".v_proj.")] = v

    in_proj_keys_to_split = [key for key in state_dict.keys() if ".in_proj_" in key]
    for key in in_proj_keys_to_split:
        # Keep MobileCLIP text encoder MHA in packed in_proj format (nn.MultiheadAttention layout).
        if key.startswith("text_encoder.layers."):
            continue
        in_proj = state_dict.pop(key)
        q, k, v = torch.chunk(in_proj, 3, dim=0)
        if key.endswith("in_proj_weight"):
            base_key = key.replace("in_proj_weight", "")
            state_dict[base_key + "q_proj.weight"] = q
            state_dict[base_key + "k_proj.weight"] = k
            state_dict[base_key + "v_proj.weight"] = v
        elif key.endswith("in_proj_bias"):
            base_key = key.replace("in_proj_bias", "")
            state_dict[base_key + "q_proj.bias"] = q
            state_dict[base_key + "k_proj.bias"] = k
            state_dict[base_key + "v_proj.bias"] = v

    return state_dict


def summarize_checkpoint_components(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    component_prefixes = {
        "vision_backbone": "detector.backbone.vision_backbone.",
        "text_backbone": "detector.backbone.language_backbone.",
        "geometry_encoder": "detector.geometry_encoder.",
        "detr_encoder": "detector.transformer.encoder.",
        "detr_decoder": "detector.transformer.decoder.",
        "mask_decoder": "detector.segmentation_head.",
    }
    return {
        name: sum(1 for key in state_dict if key.startswith(prefix)) for name, prefix in component_prefixes.items()
    }


def load_original_state_dict(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    return checkpoint


def _convert_text_keys(key: str) -> str:
    new_key = key
    for pat, rep in TEXT_KEY_MAPPING.items():
        new_key = re.sub(pat, rep, new_key)
    return new_key


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = {re.sub(r"^detector\.", "", k): v for k, v in state_dict.items() if k.startswith("detector.")}

    # Convert non-text weights via SAM3 converter.
    non_text = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("backbone.language_backbone.") and not k.startswith("backbone.vision_backbone.sam2_convs.")
    }
    key_mapping = convert_old_keys_to_new_keys(list(non_text.keys()))
    converted = {new_k: non_text[old_k] for old_k, new_k in key_mapping.items()}

    # Convert LiteText (MobileCLIP student) encoder.
    text = {k: v for k, v in state_dict.items() if k.startswith("backbone.language_backbone.")}
    for old_k, tensor in text.items():
        if "num_batches_tracked" in old_k:
            continue
        new_k = _convert_text_keys(old_k)
        if new_k == old_k:
            continue
        converted[new_k] = tensor

    converted = split_qkv_for_sam3_lite_text(converted)
    print(
        "Converted key counts:",
        {
            "vision_encoder": sum(1 for key in converted if key.startswith("vision_encoder.")),
            "text_encoder": sum(1 for key in converted if key.startswith("text_encoder.")),
            "geometry_encoder": sum(1 for key in converted if key.startswith("geometry_encoder.")),
            "detr_encoder": sum(1 for key in converted if key.startswith("detr_encoder.")),
            "detr_decoder": sum(1 for key in converted if key.startswith("detr_decoder.")),
            "mask_decoder": sum(1 for key in converted if key.startswith("mask_decoder.")),
        },
    )

    if "vision_encoder.backbone.embeddings.position_embeddings" in converted:
        pos = converted["vision_encoder.backbone.embeddings.position_embeddings"]
        if pos.shape[1] == 577:
            converted["vision_encoder.backbone.embeddings.position_embeddings"] = pos[:, 1:, :]
    # HF models compute rope table and don't load from checkpoint
    for k in list(converted.keys()):
        if k.endswith("rotary_emb.rope_embeddings"):
            converted.pop(k)
    return converted


def _infer_text_architecture_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int | str]:
    text_prefix = "detector.backbone.language_backbone.encoder."

    hidden_size = state_dict[f"{text_prefix}embedding_layer.weight"].shape[1]
    context_length = state_dict[f"{text_prefix}positional_embedding.pos_embed.pos_embed"].shape[2]

    has_repmixer = any(f"{text_prefix}transformer.0.token_mixer" in key for key in state_dict)
    if has_repmixer:
        model_name = "mct"
        num_hidden_layers = 6
    else:
        model_name = "base"
        layer_ids = {
            int(key.split("transformer.")[1].split(".")[0])
            for key in state_dict
            if f"{text_prefix}transformer." in key and ".pre_norm_mha." in key
        }
        num_hidden_layers = max(layer_ids) + 1

    return {
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 4,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": hidden_size // 64,
        "max_position_embeddings": context_length,
        "projection_dim": hidden_size,
        "model_name": model_name,
    }


def _build_config(state_dict: dict[str, torch.Tensor]) -> Sam3LiteTextConfig:
    text_arch = _infer_text_architecture_from_state_dict(state_dict)
    config = Sam3LiteTextConfig(vision_config=Sam3LiteTextVisionConfig(backbone_config=Sam3LiteTextViTConfig()))
    for key, value in text_arch.items():
        setattr(config.text_config, key, value)
    return config


def _default_hub_model_id(checkpoint_name: str) -> str:
    token = os.environ.get("HF_TOKEN")
    try:
        username = HfApi().whoami(token=token)["name"]
    except (HfHubHTTPError, KeyError) as exc:
        raise ValueError(
            "Could not infer Hub username from HF_TOKEN. Provide --hub_model_id explicitly or set a valid HF_TOKEN."
        ) from exc
    return f"{username}/{checkpoint_name}"


def get_litetext_checkpoint_filenames(repo_id: str) -> list[str]:
    files = list_repo_files(repo_id)
    return sorted([f for f in files if f.startswith("sam3_litetext/") and f.endswith(".pt")])


def convert_all_checkpoints(
    repo_id: str,
    output_dir: str,
    original_repo_path: str | None = None,
    debug_intermediates: bool = False,
    push_to_hub: bool = False,
):
    filenames = get_litetext_checkpoint_filenames(repo_id)
    print(f"Found {len(filenames)} sam3_litetext checkpoints in {repo_id}")

    for filename in filenames:
        ckpt_path = hf_hub_download(repo_id, filename)
        ckpt_name = Path(filename).stem
        ckpt_output = Path(output_dir) / ckpt_name
        print(f"\n=== Converting {filename} -> {ckpt_output} ===")
        try:
            hub_model_id = _default_hub_model_id(ckpt_name) if push_to_hub else None
            convert_checkpoint(
                ckpt_path,
                str(ckpt_output),
                original_repo_path=original_repo_path,
                debug_intermediates=debug_intermediates,
                push_to_hub=push_to_hub,
                hub_model_id=hub_model_id,
            )
            print(f"[OK] {filename}")
        except Exception as exc:
            print(f"[FAILED] {filename}: {exc}")


def _debug_compare_text_intermediates(original, model, input_ids: torch.Tensor):
    with torch.no_grad():
        original_hidden = original.encoder.forward_embedding(input_ids)
        hf_hidden = model.text_encoder.token_embedding(input_ids)
        hf_hidden = hf_hidden + model.text_encoder.position_embedding(input_ids.shape[1]).to(hf_hidden.dtype)
        hf_hidden = model.text_encoder.embedding_dropout(hf_hidden)
        print("Embed max abs diff:", (original_hidden - hf_hidden).abs().max().item())

        for idx, original_layer in enumerate(original.encoder.transformer):
            original_hidden = original_layer(original_hidden, key_padding_mask=None, attn_mask=None)

            if idx in model.text_encoder.repmixer_indexes:
                hf_hidden = (
                    model.text_encoder.layers[idx](hf_hidden.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
                )
            else:
                hf_hidden = model.text_encoder.layers[idx](hf_hidden)

            print(f"Layer {idx} max abs diff:", (original_hidden - hf_hidden).abs().max().item())

        original_hidden = original.encoder.final_layer_norm(original_hidden)
        hf_hidden = model.text_encoder.final_layer_norm(hf_hidden)
        print("Final LN max abs diff:", (original_hidden - hf_hidden).abs().max().item())


def verify_text_outputs(
    model: Sam3LiteTextModel, checkpoint_path: str, repo_path: str, debug_intermediates: bool = False
):
    sys.path.insert(0, os.path.join(repo_path, "sam3"))
    from sam3.model.text_encoder_student import TextStudentEncoder

    original_sd = load_original_state_dict(checkpoint_path)
    text_arch = _infer_text_architecture_from_state_dict(original_sd)
    student_cfg = {
        "dim": text_arch["hidden_size"],
        "model_name": text_arch["model_name"],
        "vocab_size": 49408,
        "n_transformer_layers": text_arch["num_hidden_layers"] - 2
        if text_arch["model_name"] == "mct"
        else text_arch["num_hidden_layers"],
        "n_heads_per_layer": text_arch["num_attention_heads"],
        "ffn_multiplier_per_layer": 4.0,
        "context_length": text_arch["max_position_embeddings"],
        "causal_masking": False,
        "norm_layer": "layer_norm_fp32",
        "no_scale_embedding": False,
        "no_pos_embedding": False,
        "embed_dropout": 0.0,
    }
    original = TextStudentEncoder(student_cfg, context_length=text_arch["max_position_embeddings"], output_dim=256)

    orig_text = {}
    for key, value in original_sd.items():
        if not key.startswith("detector.backbone.language_backbone."):
            continue
        if "num_batches_tracked" in key:
            continue

        base_key = key.replace("detector.backbone.language_backbone.", "")
        base_key = base_key.replace(".qkv_proj.", ".attn.in_proj_")
        base_key = base_key.replace(".out_proj.", ".attn.out_proj.")

        # Original module exposes both `tensor_runner.*` and aliases (`encoder.*`, `projector.*`).
        orig_text[f"tensor_runner.{base_key}"] = value
        if base_key.startswith("encoder."):
            orig_text[base_key] = value
        elif base_key.startswith("projector."):
            orig_text[base_key] = value

    missing, unexpected = original.load_state_dict(orig_text, strict=False)
    print("Original load missing:", len(missing), "unexpected:", len(unexpected))
    original.eval()

    model.eval()
    sequence_length = text_arch["max_position_embeddings"]
    input_ids = torch.tensor([[49406, 320, 1125, 49407] + [0] * (sequence_length - 4)], dtype=torch.long)
    if debug_intermediates:
        _debug_compare_text_intermediates(original, model, input_ids)

    with torch.no_grad():
        # original uses internal tokenizer path; we run internal tensor path for exact dummy ids
        input_embeds = original.encoder.forward_embedding(input_ids)
        original_memory = original.encoder(input_embeds, return_all_tokens=True, input_is_embeddings=True)
        original_memory = original.projector(original_memory)
        hf_out = model.get_text_features(input_ids=input_ids).pooler_output
    print("Original text sample:", original_memory[0, 0, :8])
    print("HF text sample:", hf_out[0, 0, :8])
    print("Max abs diff:", (original_memory - hf_out).abs().max().item())


def convert_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    original_repo_path: str | None = None,
    debug_intermediates: bool = False,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    state_dict = load_original_state_dict(checkpoint_path)
    component_counts = summarize_checkpoint_components(state_dict)
    print("Checkpoint component counts:", component_counts)

    converted = convert_state_dict(state_dict)

    config = _build_config(state_dict)
    model = Sam3LiteTextModel(config)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    print("Missing:", len(missing), "Unexpected:", len(unexpected))
    if missing:
        print("Sample missing:", missing[:20])
    if unexpected:
        print("Sample unexpected:", unexpected[:20])

    if original_repo_path is not None:
        verify_text_outputs(model, checkpoint_path, original_repo_path, debug_intermediates=debug_intermediates)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "pytorch_model.bin")
    with (out / "config.json").open("w") as f:
        json.dump(config.to_dict(), f, indent=2)

    if push_to_hub:
        if hub_model_id is None:
            ckpt_name = Path(checkpoint_path).stem
            hub_model_id = _default_hub_model_id(ckpt_name)
        print(f"Pushing converted checkpoint to Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="Simon7108528/EfficientSAM3")
    parser.add_argument(
        "--filename", type=str, default="sam3_litetext/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt"
    )
    parser.add_argument("--original_repo_path", type=str, default=None)
    parser.add_argument("--debug_intermediates", action="store_true")
    parser.add_argument(
        "--convert_all", action="store_true", help="Convert every LiteText checkpoint in the source repo."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push converted checkpoints to the Hugging Face Hub (defaults to <HF_TOKEN username>/<checkpoint_stem>).",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Explicit Hub destination for single-checkpoint conversion (ignored when --convert_all is used).",
    )
    args = parser.parse_args()

    if args.convert_all:
        convert_all_checkpoints(
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            original_repo_path=args.original_repo_path,
            debug_intermediates=args.debug_intermediates,
            push_to_hub=args.push_to_hub,
        )
    else:
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download(args.repo_id, args.filename)

        convert_checkpoint(
            checkpoint_path,
            args.output_dir,
            args.original_repo_path,
            debug_intermediates=args.debug_intermediates,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )


if __name__ == "__main__":
    main()
