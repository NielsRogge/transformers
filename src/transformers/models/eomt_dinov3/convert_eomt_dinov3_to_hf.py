"""Conversion script for EoMT-DINOv3 checkpoints.

Example
-------
Assuming the delta checkpoint has been downloaded to ``/tmp/eomt_delta.bin`` and the
original EoMT repository is cloned at ``/tmp/eomt`` you can run:

.. code-block:: bash

    HF_TOKEN=your_token_here \
    python -m transformers.models.eomt_dinov3.convert_eomt_dinov3_to_hf \
        /tmp/eomt_delta.bin \
        /tmp/eomt_converted \
        --backbone-repo-id facebook/dinov3-vits16-pretrain-lvd1689m \
        --verify \
        --original-repo-path /tmp/eomt

Make sure the token used above has been granted access to the gated DINOv3 weights.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Tuple

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers
from PIL import Image

from transformers import EomtDinov3Config, EomtDinov3ForUniversalSegmentation, EomtDinov3ImageProcessorFast


CAT_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


DELTA_KEY_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"^network\\.encoder\\.backbone\\.patch_embed\\.cls_token$", "embeddings.cls_token"),
    (r"^network\\.encoder\\.backbone\\.patch_embed\\.register_tokens$", "embeddings.register_tokens"),
    (
        r"^network\\.encoder\\.backbone\\.patch_embed\\.patch_embeddings\\.",
        "embeddings.patch_embeddings.",
    ),
    (r"^network\\.encoder\\.backbone\\.blocks\\.(\\d+)\\.", r"layers.\\1."),
    (r"^network\\.encoder\\.backbone\\.norm\\.", "layernorm."),
    (r"^network\\.q\\.", "query."),
    (r"^network\\.class_head\\.", "class_predictor."),
    (r"^network\\.mask_head\\.0\\.", "mask_head.fc1."),
    (r"^network\\.mask_head\\.2\\.", "mask_head.fc2."),
    (r"^network\\.mask_head\\.4\\.", "mask_head.fc3."),
    (r"^network\\.upscale\\.(\\d+)\\.conv1\\.", r"upscale_block.block.\\1.conv1."),
    (r"^network\\.upscale\\.(\\d+)\\.conv2\\.", r"upscale_block.block.\\1.conv2."),
    (r"^network\\.upscale\\.(\\d+)\\.norm\\.", r"upscale_block.block.\\1.layernorm2d."),
    (r"^network\\.attn_mask_probs$", "attn_mask_probs"),
)

SKIP_KEYS = {
    "network.encoder.pixel_mean",
    "network.encoder.pixel_std",
    "criterion.empty_weight",
}


def _download_file(
    repo_id: str,
    filename: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    headers = build_hf_headers(token=token)
    url = hf_hub_url(repo_id, filename=filename, revision=revision)
    cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix="eomt_dinov3_"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename

    if destination.exists():
        return destination

    with requests.get(url, headers=headers, stream=True) as response:
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            if response.status_code == 401:
                message = (
                    "Failed to download gated weights. "
                    "Please make sure you have been granted access and either set the HF_TOKEN "
                    "environment variable or pass --token."
                )
                raise requests.HTTPError(message) from error
            raise
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)

    return destination


def _load_state_dict_from_repo(
    repo_id: str,
    filename: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    path = _download_file(repo_id, filename, token=token, revision=revision)

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path)

    return torch.load(path, map_location="cpu")


def _rename_delta_key(key: str) -> Tuple[Optional[str], bool]:
    if key in SKIP_KEYS:
        return None, False

    for pattern, replacement in DELTA_KEY_REPLACEMENTS:
        if re.match(pattern, key):
            new_key = re.sub(pattern, replacement, key)
            return new_key, key.startswith("network.encoder.backbone")

    if key.startswith("network.encoder.backbone"):
        raise KeyError(f"Unhandled backbone key: {key}")

    return None, False


def convert_delta_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], set[str]]:
    converted: dict[str, torch.Tensor] = {}
    backbone_keys: set[str] = set()

    for key, value in state_dict.items():
        new_key, is_backbone = _rename_delta_key(key)
        if new_key is None:
            continue

        converted[new_key] = value
        if is_backbone:
            backbone_keys.add(new_key)

    return converted, backbone_keys


def map_dinov3_state_to_eomt(base_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}

    for key, tensor in base_state_dict.items():
        if key.startswith("layer."):
            new_key = key.replace("layer.", "layers.", 1)
        elif key == "norm.weight":
            new_key = "layernorm.weight"
        elif key == "norm.bias":
            new_key = "layernorm.bias"
        else:
            new_key = key

        if new_key == "embeddings.mask_token":
            continue

        mapped[new_key] = tensor

    return mapped


def merge_backbone_weights(
    base_backbone: dict[str, torch.Tensor],
    delta_backbone: dict[str, torch.Tensor],
    backbone_delta_keys: Iterable[str],
) -> dict[str, torch.Tensor]:
    merged = dict(base_backbone)

    for key, value in delta_backbone.items():
        if key in backbone_delta_keys:
            merged[key] = merged[key] + value
        else:
            merged[key] = value

    return merged


def build_eomt_config(
    *,
    base_config: dict[str, object],
    delta_state: dict[str, torch.Tensor],
    image_size: int,
) -> EomtDinov3Config:
    num_queries = delta_state["network.q.weight"].shape[0]
    num_blocks = delta_state["network.attn_mask_probs"].numel()
    num_upscale_blocks = len({int(key.split(".")[2]) for key in delta_state if key.startswith("network.upscale")})
    num_register_tokens = delta_state["network.encoder.backbone.patch_embed.register_tokens"].shape[1]
    num_labels = delta_state["network.class_head.weight"].shape[0] - 1

    config = EomtDinov3Config(
        hidden_size=base_config["hidden_size"],
        num_hidden_layers=base_config["num_hidden_layers"],
        num_attention_heads=base_config["num_attention_heads"],
        intermediate_size=base_config["intermediate_size"],
        hidden_act=base_config["hidden_act"],
        hidden_dropout_prob=0.0,
        initializer_range=base_config["initializer_range"],
        layer_norm_eps=base_config["layer_norm_eps"],
        image_size=image_size,
        patch_size=base_config["patch_size"],
        num_channels=base_config.get("num_channels", 3),
        layerscale_value=base_config.get("layerscale_value", 1.0),
        drop_path_rate=base_config.get("drop_path_rate", 0.0),
        attention_dropout=base_config.get("attention_dropout", 0.0),
        num_upscale_blocks=num_upscale_blocks,
        num_blocks=num_blocks,
        num_queries=num_queries,
        num_register_tokens=num_register_tokens,
        rope_theta=base_config.get("rope_theta", 100.0),
        query_bias=base_config.get("query_bias", True),
        key_bias=base_config.get("key_bias", False),
        value_bias=base_config.get("value_bias", True),
        proj_bias=base_config.get("proj_bias", True),
        mlp_bias=base_config.get("mlp_bias", True),
        use_gated_mlp=base_config.get("use_gated_mlp", False),
        pos_embed_shift=base_config.get("pos_embed_shift"),
        pos_embed_jitter=base_config.get("pos_embed_jitter"),
        pos_embed_rescale=base_config.get("pos_embed_rescale"),
        num_labels=num_labels,
    )

    return config


def load_json_config(repo_id: str, token: Optional[str], revision: Optional[str]) -> dict[str, object]:
    path = _download_file(repo_id, "config.json", token=token, revision=revision)
    with open(path, "r") as handle:
        return json.load(handle)


def convert_checkpoint(
    *,
    delta_state: dict[str, torch.Tensor],
    backbone_repo_id: str,
    token: Optional[str],
    backbone_revision: Optional[str],
    image_size: int,
) -> tuple[EomtDinov3Config, dict[str, torch.Tensor]]:
    base_state = _load_state_dict_from_repo(
        backbone_repo_id,
        filename="model.safetensors",
        token=token,
        revision=backbone_revision,
    )

    base_config = load_json_config(backbone_repo_id, token=token, revision=backbone_revision)

    mapped_base = map_dinov3_state_to_eomt(base_state)
    converted_delta, backbone_delta_keys = convert_delta_state_dict(delta_state)
    merged_state = merge_backbone_weights(mapped_base, converted_delta, backbone_delta_keys)

    config = build_eomt_config(
        base_config=base_config,
        delta_state=delta_state,
        image_size=image_size,
    )

    return config, merged_state


def ensure_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict


def convert_model(
    *,
    delta_path: Path,
    backbone_repo_id: str,
    token: Optional[str],
    backbone_revision: Optional[str],
    image_size: int,
    output_dir: Path,
    safe_serialization: bool,
    verify: bool,
    original_repo_path: Optional[Path],
) -> None:
    raw_delta_state = torch.load(delta_path, map_location="cpu")
    delta_state = ensure_state_dict(raw_delta_state)

    config, merged_state = convert_checkpoint(
        delta_state=delta_state,
        backbone_repo_id=backbone_repo_id,
        token=token,
        backbone_revision=backbone_revision,
        image_size=image_size,
    )

    with init_empty_weights():
        model = EomtDinov3ForUniversalSegmentation(config)

    model.load_state_dict(merged_state, strict=True, assign=True)

    processor = EomtDinov3ImageProcessorFast(
        size={"shortest_edge": image_size, "longest_edge": image_size},
        do_split_image=False,
        do_pad=True,
    )

    if verify:
        verify_conversion(
            hf_model=model,
            processor=processor,
            delta_state=delta_state,
            backbone_repo_id=backbone_repo_id,
            token=token,
            image_size=image_size,
            original_repo_path=original_repo_path,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    processor.save_pretrained(output_dir)


def _prepare_image(processor: EomtDinov3ImageProcessorFast) -> torch.Tensor:
    image = Image.open(requests.get(CAT_URL, stream=True).raw).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs.pixel_values


def _load_original_model(
    *,
    original_repo_path: Path,
    backbone_repo_id: str,
    token: Optional[str],
    image_size: int,
    num_labels: int,
    num_queries: int,
    num_blocks: int,
    delta_state: dict[str, torch.Tensor],
) -> "torch.nn.Module":
    sys.path.insert(0, str(original_repo_path))

    from models.eomt import EoMT
    from models.vit import ViT

    if token is not None:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    encoder = ViT((image_size, image_size), backbone_name=backbone_repo_id, ckpt_path=None)
    model = EoMT(encoder=encoder, num_classes=num_labels, num_q=num_queries, num_blocks=num_blocks)

    state_dict = model.state_dict()
    for key, value in delta_state.items():
        if key in SKIP_KEYS or key.startswith("criterion"):
            continue
        if key.startswith("network.encoder.backbone"):
            state_dict[key] = state_dict[key] + value
        else:
            state_dict[key] = value

    model.load_state_dict(state_dict)
    model.eval()
    return model


def _collect_original_backbone_states(model, pixel_values: torch.Tensor):
    backbone = model.encoder.backbone
    hidden_states = (pixel_values - model.encoder.pixel_mean) / model.encoder.pixel_std

    rope = None
    if hasattr(backbone, "rope_embeddings"):
        rope = backbone.rope_embeddings(hidden_states)

    hidden_states = backbone.patch_embed(hidden_states)
    outputs = []
    mask_logits_list = []
    class_logits_list = []
    attn_mask = None

    for idx, block in enumerate(backbone.blocks):
        if idx == len(backbone.blocks) - model.num_blocks:
            query = model.q.weight[None, :, :].expand(hidden_states.shape[0], -1, -1)
            hidden_states = torch.cat((query, hidden_states), dim=1)

        if idx >= len(backbone.blocks) - model.num_blocks:
            norm_hidden_states = backbone.norm(hidden_states)
            mask_logits, class_logits = model._predict(norm_hidden_states)
            mask_logits_list.append(mask_logits)
            class_logits_list.append(class_logits)
            attn_mask = model._attn_mask(hidden_states, mask_logits, idx)

        attn_module = block.attention if hasattr(block, "attention") else block.attn
        attn_output = model._attn(attn_module, block.norm1(hidden_states), attn_mask, rope=rope)
        if hasattr(block, "layer_scale1"):
            hidden_states = hidden_states + block.layer_scale1(attn_output)
        else:
            hidden_states = hidden_states + block.ls1(attn_output)

        mlp_output = block.mlp(block.norm2(hidden_states))
        if hasattr(block, "layer_scale2"):
            hidden_states = hidden_states + block.layer_scale2(mlp_output)
        else:
            hidden_states = hidden_states + block.ls2(mlp_output)

        outputs.append(hidden_states)

    sequence_output = backbone.norm(hidden_states)
    mask_logits, class_logits = model._predict(sequence_output)
    mask_logits_list.append(mask_logits)
    class_logits_list.append(class_logits)

    return outputs, mask_logits_list, class_logits_list, sequence_output


def _collect_hf_backbone_states(model: EomtDinov3ForUniversalSegmentation, pixel_values: torch.Tensor):
    hidden_states = model.embeddings(pixel_values)
    position_embeddings = model.rope_embeddings(pixel_values)

    outputs = []
    mask_logits_list = []
    class_logits_list = []
    attention_mask = None

    for idx, layer_module in enumerate(model.layers):
        if idx == model.num_hidden_layers - model.config.num_blocks:
            query = model.query.weight[None, :, :].expand(hidden_states.shape[0], -1, -1).to(hidden_states.device)
            hidden_states = torch.cat((query, hidden_states), dim=1)

        if idx >= model.num_hidden_layers - model.config.num_blocks:
            norm_hidden_states = model.layernorm(hidden_states)
            mask_logits, class_logits = model.predict(norm_hidden_states)
            mask_logits_list.append(mask_logits)
            class_logits_list.append(class_logits)

            bool_attention_mask = torch.ones(
                hidden_states.shape[0],
                hidden_states.shape[1],
                hidden_states.shape[1],
                device=hidden_states.device,
                dtype=torch.bool,
            )

            interpolated_logits = torch.nn.functional.interpolate(
                mask_logits,
                size=model.grid_size,
                mode="bilinear",
            ).view(mask_logits.size(0), mask_logits.size(1), -1)

            num_query_tokens = model.config.num_queries
            encoder_start_tokens = num_query_tokens + model.embeddings.num_prefix_tokens
            bool_attention_mask[:, :num_query_tokens, encoder_start_tokens:] = interpolated_logits > 0

            bool_attention_mask = bool_attention_mask[:, None, ...].expand(
                -1, model.config.num_attention_heads, -1, -1
            )

            attention_mask = torch.zeros(
                bool_attention_mask.shape,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).masked_fill(~bool_attention_mask, float("-inf"))

        hidden_states = layer_module(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        outputs.append(hidden_states)

    sequence_output = model.layernorm(hidden_states)
    mask_logits, class_logits = model.predict(sequence_output)
    mask_logits_list.append(mask_logits)
    class_logits_list.append(class_logits)

    return outputs, mask_logits_list, class_logits_list, sequence_output


def _assert_allclose(reference: Iterable[torch.Tensor], actual: Iterable[torch.Tensor], message: str) -> None:
    for idx, (ref_tensor, act_tensor) in enumerate(zip(reference, actual)):
        if not torch.allclose(ref_tensor, act_tensor, atol=1e-4, rtol=1e-4):
            raise ValueError(f"Mismatch in {message} at index {idx}")


def verify_conversion(
    *,
    hf_model: EomtDinov3ForUniversalSegmentation,
    processor: EomtDinov3ImageProcessorFast,
    delta_state: dict[str, torch.Tensor],
    backbone_repo_id: str,
    token: Optional[str],
    image_size: int,
    original_repo_path: Optional[Path],
) -> None:
    if original_repo_path is None:
        raise ValueError("Original repository path is required for verification")

    torch.manual_seed(0)

    pixel_values = _prepare_image(processor)

    original_model = _load_original_model(
        original_repo_path=original_repo_path,
        backbone_repo_id=backbone_repo_id,
        token=token,
        image_size=image_size,
        num_labels=hf_model.config.num_labels,
        num_queries=hf_model.config.num_queries,
        num_blocks=hf_model.config.num_blocks,
        delta_state=delta_state,
    )

    hf_model.eval()

    with torch.no_grad():
        orig_outputs = _collect_original_backbone_states(original_model, pixel_values)
        hf_outputs = _collect_hf_backbone_states(hf_model, pixel_values)

    _assert_allclose(orig_outputs[0], hf_outputs[0], "backbone hidden states")
    _assert_allclose(orig_outputs[1], hf_outputs[1], "mask logits")
    _assert_allclose(orig_outputs[2], hf_outputs[2], "class logits")

    if not torch.allclose(orig_outputs[3], hf_outputs[3], atol=1e-4, rtol=1e-4):
        raise ValueError("Mismatch in final sequence output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert EoMT-DINOv3 checkpoints to 🤗 Transformers format")
    parser.add_argument("delta", type=Path, help="Path to the delta checkpoint (pytorch_model.bin)")
    parser.add_argument("output_dir", type=Path, help="Directory to save the converted model")
    parser.add_argument(
        "--backbone-repo-id",
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="Hugging Face Hub repository id for the base DINOv3 weights",
    )
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--backbone-revision", default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--safe-serialization", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--original-repo-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    convert_model(
        delta_path=args.delta,
        backbone_repo_id=args.backbone_repo_id,
        token=args.token,
        backbone_revision=args.backbone_revision,
        image_size=args.image_size,
        output_dir=args.output_dir,
        safe_serialization=args.safe_serialization,
        verify=args.verify,
        original_repo_path=args.original_repo_path,
    )


if __name__ == "__main__":
    main()

