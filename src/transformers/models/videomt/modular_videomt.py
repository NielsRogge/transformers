# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from ..eomt_dinov3.configuration_eomt_dinov3 import EomtDinov3Config
from ..eomt_dinov3.modeling_eomt_dinov3 import (
    EomtDinov3Attention,
    EomtDinov3DropPath,
    EomtDinov3Embeddings,
    EomtDinov3ForUniversalSegmentation,
    EomtDinov3ForUniversalSegmentationOutput,
    EomtDinov3GatedMLP,
    EomtDinov3HungarianMatcher,
    EomtDinov3Layer,
    EomtDinov3LayerNorm2d,
    EomtDinov3LayerScale,
    EomtDinov3Loss,
    EomtDinov3MaskHead,
    EomtDinov3MLP,
    EomtDinov3PreTrainedModel,
    EomtDinov3RotaryEmbedding,
    EomtDinov3ScaleBlock,
    EomtDinov3ScaleLayer,
)


class VideomtConfig(EomtDinov3Config):
    pass


class VideomtAttention(EomtDinov3Attention):
    pass


class VideomtEmbeddings(EomtDinov3Embeddings):
    pass


class VideomtDropPath(EomtDinov3DropPath):
    pass


class VideomtMLP(EomtDinov3MLP):
    pass


class VideomtGatedMLP(EomtDinov3GatedMLP):
    pass


class VideomtLayer(EomtDinov3Layer):
    pass


class VideomtLayerScale(EomtDinov3LayerScale):
    pass


class VideomtRotaryEmbedding(EomtDinov3RotaryEmbedding):
    pass


class VideomtHungarianMatcher(EomtDinov3HungarianMatcher):
    pass


class VideomtLoss(EomtDinov3Loss):
    pass


class VideomtForUniversalSegmentationOutput(EomtDinov3ForUniversalSegmentationOutput):
    pass


class VideomtPreTrainedModel(EomtDinov3PreTrainedModel):
    pass


class VideomtLayerNorm2d(EomtDinov3LayerNorm2d):
    pass


class VideomtScaleLayer(EomtDinov3ScaleLayer):
    pass


class VideomtScaleBlock(EomtDinov3ScaleBlock):
    pass


class VideomtMaskHead(EomtDinov3MaskHead):
    pass


class VideomtForUniversalSegmentation(EomtDinov3ForUniversalSegmentation):
    pass


__all__ = [
    "VideomtConfig",
    "VideomtPreTrainedModel",
    "VideomtForUniversalSegmentation",
]
