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
"""Video processor class for Videomt."""

from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...utils import is_torch_available
from ...video_processing_utils import BaseVideoProcessor


if is_torch_available():
    import torch
    import torch.nn.functional as F


def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and
    `labels`.

    Args:
        masks (`torch.Tensor`):
            A tensor of shape `(num_queries, height, width)`.
        scores (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        labels (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        object_mask_threshold (`float`):
            A number between 0 and 1 used to binarize the masks.
    Raises:
        `ValueError`: Raised when the first dimension doesn't match in all input tensors.
    Returns:
        `tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep]


def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    original_mask = mask_probs[k] >= mask_threshold
    original_area = original_mask.sum()

    final_mask = mask_k & original_mask
    final_mask_area = final_mask.sum()

    mask_exists = mask_k_area > 0 and original_area > 0 and final_mask_area > 0

    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, final_mask


def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    stuff_classes,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    target_size: tuple[int, int] | None = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.long, device=mask_probs.device) - 1
    segments: list[dict] = []

    mask_probs = mask_probs.sigmoid()
    mask_labels = (pred_scores[:, None, None] * mask_probs).argmax(0)

    current_segment_id = 0
    stuff_memory_list: dict[str, int] = {}

    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()

        mask_exists, final_mask = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if not mask_exists:
            continue

        if stuff_classes and pred_class in stuff_classes:
            if pred_class in stuff_memory_list:
                segmentation[final_mask] = stuff_memory_list[pred_class]
                continue
            else:
                stuff_memory_list[pred_class] = current_segment_id

        segmentation[final_mask] = current_segment_id
        segment_score = round(pred_scores[k].item(), 6)
        segments.append(
            {
                "id": current_segment_id,
                "label_id": pred_class,
                "score": segment_score,
            }
        )
        current_segment_id += 1
    return segmentation, segments


class VideomtVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 640, "width": 640}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False
    model_input_names = ["pixel_values"]

    def preprocess(self, videos, **kwargs):
        batch = super().preprocess(videos, **kwargs)
        batch["pixel_values"] = batch.pop("pixel_values_videos")
        return batch

    def _resize_mask_logits(
        self,
        masks_queries_logits: "torch.Tensor",
        target_sizes: list[tuple[int, int]],
    ) -> list["torch.Tensor"]:
        """Interpolates mask logits to each frame's original resolution."""
        resized = []
        for idx, original_size in enumerate(target_sizes):
            upsampled = F.interpolate(
                masks_queries_logits[idx][None, ...],
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )[0]
            resized.append(upsampled)
        return resized

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
    ) -> list["torch.Tensor"]:
        """
        Converts the output of [`VideomtForUniversalSegmentation`] into semantic segmentation predictions.

        Args:
            outputs ([`VideomtForUniversalSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`):
                List of `(height, width)` tuples corresponding to the requested final size of each prediction.
                Length should match the number of frames in the output.

        Returns:
            `list[torch.Tensor]`: A list of tensors, each of shape `(height, width)`, where each value is the
            predicted class index for the corresponding pixel.
        """
        masks_queries_logits = outputs.masks_queries_logits  # [num_frames, num_queries, height, width]
        class_queries_logits = outputs.class_queries_logits  # [num_frames, num_queries, num_classes+1]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()

        segmentation_logits = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        output_logits = []
        for idx in range(len(segmentation_logits)):
            resized_logits = F.interpolate(
                segmentation_logits[idx].unsqueeze(dim=0),
                size=target_sizes[idx],
                mode="bilinear",
                align_corners=False,
            )
            output_logits.append(resized_logits[0])

        return [logit.argmax(dim=0) for logit in output_logits]

    def post_process_instance_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        Converts the output of [`VideomtForUniversalSegmentation`] into instance segmentation predictions.

        Args:
            outputs ([`VideomtForUniversalSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`):
                List of `(height, width)` tuples corresponding to the requested final size of each prediction.
                Length should match the number of frames in the output.
            threshold (`float`, *optional*, defaults to 0.5):
                Minimum combined score to keep an instance.

        Returns:
            `list[dict]`: A list of dicts (one per frame), each containing:
                - `"segmentation"` -- A `torch.Tensor` of shape `(height, width)` with instance IDs (or -1 for background).
                - `"segments_info"` -- A list of dicts with `"id"`, `"label_id"`, and `"score"` for each instance.
        """
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        mask_probs_batch = self._resize_mask_logits(masks_queries_logits, target_sizes)

        device = masks_queries_logits.device
        num_frames = class_queries_logits.shape[0]
        num_queries = class_queries_logits.shape[-2]

        results = []

        for i in range(num_frames):
            mask_pred = mask_probs_batch[i]
            mask_class = class_queries_logits[i]

            scores, pred_classes = mask_class.softmax(dim=-1)[..., :-1].max(-1)
            pred_masks = (mask_pred > 0).float()

            mask_scores = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores * mask_scores

            segmentation = torch.zeros(target_sizes[i], device=device) - 1

            segments = []
            current_segment_id = 0
            for j in range(num_queries):
                score = pred_scores[j].item()

                if not torch.all(pred_masks[j] == 0) and score >= threshold:
                    segmentation[pred_masks[j] == 1] = current_segment_id
                    segments.append(
                        {
                            "id": current_segment_id,
                            "label_id": pred_classes[j].item(),
                            "score": round(score, 6),
                        }
                    )
                    current_segment_id += 1

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

    def post_process_panoptic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        stuff_classes: list[int] | None = None,
    ) -> list[dict]:
        """
        Converts the output of [`VideomtForUniversalSegmentation`] into panoptic segmentation predictions.

        Args:
            outputs ([`VideomtForUniversalSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`):
                List of `(height, width)` tuples corresponding to the requested final size of each prediction.
                Length should match the number of frames in the output.
            threshold (`float`, *optional*, defaults to 0.8):
                Minimum score to keep a predicted segment.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing mask probabilities.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                Overlap threshold to merge masks into a single segment.
            stuff_classes (`list[int]`, *optional*):
                List of class IDs that are "stuff" (amorphous regions). Instances of stuff classes are merged.

        Returns:
            `list[dict]`: A list of dicts (one per frame), each containing:
                - `"segmentation"` -- A `torch.Tensor` of shape `(height, width)` with segment IDs (or -1 for background).
                - `"segments_info"` -- A list of dicts with `"id"`, `"label_id"`, and `"score"` for each segment.
        """
        masks_queries_logits = outputs.masks_queries_logits
        class_queries_logits = outputs.class_queries_logits

        num_frames = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        mask_probs_batch = self._resize_mask_logits(masks_queries_logits, target_sizes)
        pred_scores_batch, pred_labels_batch = class_queries_logits.softmax(dim=-1).max(-1)

        results: list = []

        for i in range(num_frames):
            mask_probs, pred_scores, pred_labels = remove_low_and_no_objects(
                mask_probs_batch[i], pred_scores_batch[i], pred_labels_batch[i], threshold, num_labels
            )

            if mask_probs.shape[0] <= 0:
                height, width = target_sizes[i] if target_sizes is not None else mask_probs.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            segmentation, segments = compute_segments(
                mask_probs=mask_probs,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                stuff_classes=stuff_classes,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                target_size=target_sizes[i] if target_sizes is not None else None,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results


__all__ = ["VideomtVideoProcessor"]
