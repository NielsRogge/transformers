# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

from ..rt_detr.image_processing_rt_detr import RTDetrImageProcessor
from ..rt_detr.image_processing_rt_detr_fast import (
    RTDetrFastImageProcessorKwargs,
    RTDetrImageProcessorFast,
)


class Deimv2ImageProcessor(RTDetrImageProcessor):
    pass


class Deimv2FastImageProcessorKwargs(RTDetrFastImageProcessorKwargs):
    pass


class Deimv2ImageProcessorFast(RTDetrImageProcessorFast):
    valid_kwargs = Deimv2FastImageProcessorKwargs


__all__ = [
    "Deimv2FastImageProcessorKwargs",
    "Deimv2ImageProcessor",
    "Deimv2ImageProcessorFast",
]
