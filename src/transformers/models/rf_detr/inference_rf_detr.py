"""Inference script for RF-DETR object detection model.

Loads nielsr/rf-detr-small, runs on the standard COCO cats image,
prints detections, and saves a visualization with bounding boxes.
"""

import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from transformers import RfDetrForObjectDetection


# COCO 91-class mapping (index = COCO category ID)
COCO_ID2LABEL = {
    0: "N/A",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "N/A",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "N/A",
    27: "backpack",
    28: "umbrella",
    29: "N/A",
    30: "N/A",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    45: "N/A",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    66: "N/A",
    67: "dining table",
    68: "N/A",
    69: "N/A",
    70: "toilet",
    71: "N/A",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "N/A",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabebe",
]


def preprocess(image: Image.Image, target_size: int = 512) -> torch.Tensor:
    """Resize to target_size x target_size and normalize with ImageNet stats."""
    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def post_process(outputs, target_size: tuple[int, int], threshold: float = 0.5):
    """Convert model outputs to detected boxes, scores, and labels.

    Args:
        outputs: RfDetrObjectDetectionOutput with logits and pred_boxes.
        target_size: (height, width) of the original image.
        threshold: minimum confidence score.

    Returns:
        List of dicts with 'scores', 'labels', 'boxes' tensors.
    """
    logits = outputs.logits  # (batch, num_queries, num_classes)
    boxes = outputs.pred_boxes  # (batch, num_queries, 4) in cx, cy, w, h normalized

    probs = logits.sigmoid()
    scores, labels = probs.max(dim=-1)

    # convert from cx, cy, w, h to x1, y1, x2, y2
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    # scale to original image size
    img_h, img_w = target_size
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
    boxes_xyxy = boxes_xyxy * scale

    results = []
    for s, l, b in zip(scores, labels, boxes_xyxy):
        keep = s > threshold
        results.append(
            {
                "scores": s[keep],
                "labels": l[keep],
                "boxes": b[keep],
            }
        )
    return results


def visualize(image: Image.Image, result: dict, id2label: dict, output_path: str):
    """Draw bounding boxes with labels on the image and save."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for idx, (score, label, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        score_val = score.item()
        label_val = label.item()
        x1, y1, x2, y2 = box.tolist()
        class_name = id2label.get(label_val, f"class_{label_val}")
        color = COLORS[idx % len(COLORS)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{class_name}: {score_val:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), text, fill="white", font=font)

    image.save(output_path)
    print(f"Saved visualization to {output_path}")


def main():
    model_id = "nielsr/rf-detr-small"
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    threshold = 0.5
    output_path = "rf_detr_result.png"

    print(f"Loading model: {model_id}")
    model = RfDetrForObjectDetection.from_pretrained(model_id)
    model.eval()

    image_size = model.config.backbone_config.image_size
    print(f"Model expects input size: {image_size}x{image_size}")

    print(f"Downloading image: {url}")
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    original_size = image.size  # (width, height)
    print(f"Original image size: {original_size}")

    pixel_values = preprocess(image, target_size=image_size)
    print(f"Input tensor shape: {pixel_values.shape}")

    print("Running inference...")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    target_size = (original_size[1], original_size[0])  # (height, width)
    results = post_process(outputs, target_size=target_size, threshold=threshold)

    result = results[0]
    print(f"\nDetected {len(result['scores'])} objects (threshold={threshold}):")
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        class_name = COCO_ID2LABEL.get(label.item(), f"class_{label.item()}")
        box_str = [round(c, 2) for c in box.tolist()]
        print(f"  {class_name}: {score.item():.3f} at {box_str}")

    vis_image = image.copy()
    visualize(vis_image, result, COCO_ID2LABEL, output_path)


if __name__ == "__main__":
    main()
