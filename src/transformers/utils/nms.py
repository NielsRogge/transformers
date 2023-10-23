# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

"""NMS (non-maximum suppression) utilities."""


from . import is_torch_available, is_torchvision_available


if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms


def multiclass_nms(multi_bboxes, multi_scores, score_threshold, iou_threshold, max_num=-1, score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (`torch.Tensor`):
            Shape (N, #class*4) or (N, 4) with N = number of objects.
        multi_scores (`torch.Tensor`):
            Shape (N, #class), where the last column contains scores of the background class, but this will be ignored.
        score_threshold (`float`):
            Bounding box threshold, boxes with scores lower than it will not be considered.
        iou_threshold (`dict`):
            IoU threshold.
        max_num (`int`, *optional*, defaults to -1):
            If there are more than `max_num` bounding boxes after NMS, only top `max_num` will be kept.
        score_factors (`torch.Tensor`, *optional*):
            The factors multiplied to scores before applying NMS.

    Returns:
        `Tuple`: (detections, labels, indices), tensors of shape (k, 5),
            (k), and (k), and indices of boxes to keep. detections are boxes with scores. Labels are 0-based.
    """
    print("Shape of multi_bboxes:", multi_bboxes.shape)

    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # remove low scoring boxes
    valid_mask = scores > score_threshold
    # multiply score_factor after threshold to preserve more bboxes, improves mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[indices], scores[indices], labels[indices]

    if bboxes.numel() == 0:
        detections = torch.cat([bboxes, scores[:, None]], -1)
        return detections, labels, indices

    keep = batched_nms(bboxes, scores, labels, iou_threshold=iou_threshold)
    detections = torch.cat((bboxes[keep], scores[keep].reshape(-1, 1)), dim=1)

    if max_num > 0:
        detections = detections[:max_num]
        keep = keep[:max_num]

    return detections, labels[keep], indices[keep]
