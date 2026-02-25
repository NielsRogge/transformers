"""Standalone inference + visualization for a converted VidEoMT checkpoint.

Reads frames from a directory, runs the HF model, and saves per-frame
visualizations (colored instance masks overlaid on the original image).

Dependencies beyond a standard `pip install transformers[torch]`:
    pip install pillow numpy

Usage:

    python src/transformers/models/videomt/inference_videomt.py \
        --model-path /path/to/converted/model \
        --frames-dir /path/to/video/frames \
        --output-dir /path/to/output \
        --score-threshold 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import VideomtConfig, VideomtForUniversalSegmentation


# fmt: off
YTVIS_2019_CATEGORIES = [
    {"color": [220, 20, 60],  "id": 1,  "name": "person"},
    {"color": [0, 82, 0],     "id": 2,  "name": "giant_panda"},
    {"color": [119, 11, 32],  "id": 3,  "name": "lizard"},
    {"color": [165, 42, 42],  "id": 4,  "name": "parrot"},
    {"color": [134, 134, 103],"id": 5,  "name": "skateboard"},
    {"color": [0, 0, 142],    "id": 6,  "name": "sedan"},
    {"color": [255, 109, 65], "id": 7,  "name": "ape"},
    {"color": [0, 226, 252],  "id": 8,  "name": "dog"},
    {"color": [5, 121, 0],    "id": 9,  "name": "snake"},
    {"color": [0, 60, 100],   "id": 10, "name": "monkey"},
    {"color": [250, 170, 30], "id": 11, "name": "hand"},
    {"color": [100, 170, 30], "id": 12, "name": "rabbit"},
    {"color": [179, 0, 194],  "id": 13, "name": "duck"},
    {"color": [255, 77, 255], "id": 14, "name": "cat"},
    {"color": [120, 166, 157],"id": 15, "name": "cow"},
    {"color": [73, 77, 174],  "id": 16, "name": "fish"},
    {"color": [0, 80, 100],   "id": 17, "name": "train"},
    {"color": [182, 182, 255],"id": 18, "name": "horse"},
    {"color": [0, 143, 149],  "id": 19, "name": "turtle"},
    {"color": [174, 57, 255], "id": 20, "name": "bear"},
    {"color": [0, 0, 230],    "id": 21, "name": "motorbike"},
    {"color": [72, 0, 118],   "id": 22, "name": "giraffe"},
    {"color": [255, 179, 240],"id": 23, "name": "leopard"},
    {"color": [0, 125, 92],   "id": 24, "name": "fox"},
    {"color": [209, 0, 151],  "id": 25, "name": "deer"},
    {"color": [188, 208, 182],"id": 26, "name": "owl"},
    {"color": [145, 148, 174],"id": 27, "name": "surfboard"},
    {"color": [106, 0, 228],  "id": 28, "name": "airplane"},
    {"color": [0, 0, 70],     "id": 29, "name": "truck"},
    {"color": [199, 100, 0],  "id": 30, "name": "zebra"},
    {"color": [166, 196, 102],"id": 31, "name": "tiger"},
    {"color": [110, 76, 0],   "id": 32, "name": "elephant"},
    {"color": [133, 129, 255],"id": 33, "name": "snowboard"},
    {"color": [0, 0, 192],    "id": 34, "name": "boat"},
    {"color": [183, 130, 88], "id": 35, "name": "shark"},
    {"color": [130, 114, 135],"id": 36, "name": "mouse"},
    {"color": [107, 142, 35], "id": 37, "name": "frog"},
    {"color": [0, 228, 0],    "id": 38, "name": "eagle"},
    {"color": [174, 255, 243],"id": 39, "name": "earless_seal"},
    {"color": [255, 208, 186],"id": 40, "name": "tennis_racket"},
]
# fmt: on

YTVIS_2019_CLASS_NAMES = [c["name"] for c in YTVIS_2019_CATEGORIES]
YTVIS_2019_CLASS_COLORS = [c["color"] for c in YTVIS_2019_CATEGORIES]


def load_frames(frames_dir: str | Path, image_size: int) -> tuple[list[np.ndarray], torch.Tensor]:
    """Load frames from a directory, resize, and build a batched tensor.

    Returns the original-size RGB numpy arrays (for visualization) and a
    ``(1, num_frames, 3, H, W)`` float tensor normalized to [0, 1].
    """
    frames_dir = Path(frames_dir)
    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not paths:
        raise FileNotFoundError(f"No image files found in {frames_dir}")

    originals: list[np.ndarray] = []
    tensors: list[torch.Tensor] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        originals.append(np.array(img))
        resized = img.resize((image_size, image_size), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)

    pixel_values = torch.stack(tensors, dim=0).unsqueeze(0)  # (1, T, 3, H, W)
    return originals, pixel_values


def postprocess(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    original_sizes: list[tuple[int, int]],
    score_threshold: float = 0.3,
    max_instances: int = 10,
) -> list[dict]:
    """Convert raw model outputs into per-frame instance predictions.

    Args:
        class_logits: ``(num_frames, num_queries, num_classes + 1)``
        mask_logits: ``(num_frames, num_queries, H_mask, W_mask)``
        original_sizes: list of ``(H, W)`` for each frame (used for upsampling).
        score_threshold: minimum confidence to keep a prediction.
        max_instances: maximum number of instances to show per frame.

    Returns:
        A list of dicts (one per frame), each containing ``scores``, ``labels``,
        and ``masks`` (binary, at the original resolution).
    """
    num_frames = class_logits.shape[0]

    # (num_frames, num_queries, num_classes) -- drop the null/background class
    scores = class_logits.softmax(dim=-1)[..., :-1]
    per_query_scores, per_query_labels = scores.max(dim=-1)  # (T, Q)

    results = []
    for t in range(num_frames):
        h, w = original_sizes[t]
        frame_masks = F.interpolate(
            mask_logits[t].unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)  # (Q, H, W)
        binary_masks = frame_masks > 0

        mask_scores = (frame_masks.sigmoid().flatten(1) * binary_masks.float().flatten(1)).sum(1) / (
            binary_masks.float().flatten(1).sum(1) + 1e-6
        )
        combined_scores = per_query_scores[t] * mask_scores

        top_k = min(max_instances, (combined_scores > score_threshold).sum().item())
        if top_k == 0:
            results.append({"scores": [], "labels": [], "masks": []})
            continue

        topk_scores, topk_idx = combined_scores.topk(top_k, sorted=True)
        topk_labels = per_query_labels[t][topk_idx]
        topk_masks = binary_masks[topk_idx]

        results.append(
            {
                "scores": topk_scores.tolist(),
                "labels": topk_labels.tolist(),
                "masks": topk_masks.cpu().numpy(),
            }
        )
    return results


def overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    labels: list[int],
    scores: list[float],
    class_names: list[str],
    class_colors: list[list[int]],
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw coloured instance masks and labels on an RGB image."""
    overlay = image.copy()

    for mask, label, score in zip(masks, labels, scores):
        color = np.array(class_colors[label % len(class_colors)], dtype=np.uint8)
        overlay[mask] = (alpha * color + (1 - alpha) * overlay[mask]).astype(np.uint8)

    canvas = Image.fromarray(overlay)
    try:
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, image.shape[0] // 40))
        except OSError:
            font = ImageFont.load_default()

        for idx, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            ty, tx = int(ys.min()), int(xs.min())
            text = f"[{idx}] {class_names[label % len(class_names)]} {score:.2f}"
            color = tuple(class_colors[label % len(class_colors)])
            draw.text((tx, ty), text, fill=color, font=font)
    except ImportError:
        pass

    return np.array(canvas)


def run_inference(
    model_path: str,
    frames_dir: str,
    output_dir: str,
    score_threshold: float = 0.3,
    max_instances: int = 10,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = VideomtConfig.from_pretrained(model_path)
    model = VideomtForUniversalSegmentation.from_pretrained(model_path).to(device).eval()

    originals, pixel_values = load_frames(frames_dir, config.image_size)
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    original_sizes = [(img.shape[0], img.shape[1]) for img in originals]
    per_frame = postprocess(
        outputs.class_queries_logits,
        outputs.masks_queries_logits,
        original_sizes,
        score_threshold=score_threshold,
        max_instances=max_instances,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(
        p for p in Path(frames_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    for idx, (orig, preds, fpath) in enumerate(zip(originals, per_frame, frame_paths)):
        if preds["scores"]:
            vis = overlay_masks(
                orig,
                preds["masks"],
                preds["labels"],
                preds["scores"],
                YTVIS_2019_CLASS_NAMES,
                YTVIS_2019_CLASS_COLORS,
            )
        else:
            vis = orig

        out_path = out / f"{fpath.stem}_vis.png"
        Image.fromarray(vis).save(out_path)
        n = len(preds["scores"])
        print(f"frame={idx}  instances={n}  saved={out_path}")

    print(f"Done. {len(originals)} frames visualized in {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VidEoMT inference and save visualized results.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the converted HF model directory")
    parser.add_argument("--frames-dir", type=str, required=True, help="Directory of video frames (jpg/png)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write visualized frames")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        model_path=args.model_path,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        max_instances=args.max_instances,
        device=args.device,
    )


if __name__ == "__main__":
    main()
