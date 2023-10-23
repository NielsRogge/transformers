# coding=utf-8
# Copyright 2023 The Hugging Face Team.
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

import unittest

import torch

from transformers import is_torchvision_available
from transformers.testing_utils import require_torchvision
from transformers.utils import multiclass_nms


if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms


class NMSTester:
    def __init__(
        self, parent, iou_threshold=0.5, split_threshold=10000, score_threshold=0.7, class_agnostic=False, max_num=2
    ):
        self.parent = parent
        self.iou_threshold = iou_threshold
        self.split_threshold = split_threshold
        self.score_threshold = score_threshold
        self.class_agnostic = class_agnostic
        self.max_num = max_num

    def prepare_config_and_inputs(self):
        boxes = torch.tensor([[10, 10, 20, 20], [15, 15, 25, 25], [30, 30, 40, 40]]).float()
        scores = torch.tensor([0.9, 0.8, 0.7])
        idxs = torch.tensor([0, 1, 0])

        return boxes, scores, idxs


@require_torchvision
class NMSTest(unittest.TestCase):
    def setUp(self):
        self.tester = NMSTester(self)

    def test_batched_nms(self):
        boxes, scores, idxs = self.tester.prepare_config_and_inputs()
        iou_threshold = self.tester.iou_threshold

        keep = batched_nms(boxes, scores, idxs, iou_threshold)

        self.assertTrue(len(keep) <= len(boxes))

    def test_multiclass_nms(self):
        multi_bboxes = torch.tensor([[10, 10, 20, 20, 30, 30, 40, 40], [15, 15, 25, 25, 35, 35, 45, 45]]).float()
        multi_scores = torch.tensor([[0.9, 0.8], [0.85, 0.75]])

        iou_threshold = self.tester.iou_threshold
        score_threshold = self.tester.score_threshold
        max_num = self.tester.max_num

        detections, labels, indices = multiclass_nms(
            multi_bboxes, multi_scores, score_threshold, iou_threshold, max_num
        )

        self.assertTrue(len(detections) == len(labels) == len(indices))
