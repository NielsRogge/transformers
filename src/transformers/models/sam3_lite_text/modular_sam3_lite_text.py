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

from ..sam3.configuration_sam3 import (
    Sam3Config,
    Sam3DETRDecoderConfig,
    Sam3DETREncoderConfig,
    Sam3GeometryEncoderConfig,
    Sam3MaskDecoderConfig,
    Sam3VisionConfig,
    Sam3ViTConfig,
)
from ..sam3.modeling_sam3 import (
    Sam3Attention,
    Sam3DecoderMLP,
    Sam3DetrDecoder,
    Sam3DetrDecoderLayer,
    Sam3DETRDecoderOutput,
    Sam3DetrEncoder,
    Sam3DetrEncoderLayer,
    Sam3DETREncoderOutput,
    Sam3DotProductScoring,
    Sam3FPNLayer,
    Sam3GeometryEncoder,
    Sam3GeometryEncoderLayer,
    Sam3GeometryEncoderOutput,
    Sam3ImageSegmentationOutput,
    Sam3MaskDecoder,
    Sam3MaskDecoderOutput,
    Sam3MaskEmbedder,
    Sam3MLP,
    Sam3Model,
    Sam3PixelDecoder,
    Sam3PreTrainedModel,
    Sam3SinePositionEmbedding,
    Sam3VisionEncoderOutput,
    Sam3VisionModel,
    Sam3VisionNeck,
    Sam3ViTEmbeddings,
    Sam3ViTLayer,
    Sam3ViTLayerScale,
    Sam3ViTModel,
    Sam3ViTPatchEmbeddings,
    Sam3ViTRoPEAttention,
    Sam3ViTRotaryEmbedding,
)


class Sam3LiteTextViTConfig(Sam3ViTConfig):
    pass


class Sam3LiteTextVisionConfig(Sam3VisionConfig):
    pass


class Sam3LiteTextGeometryEncoderConfig(Sam3GeometryEncoderConfig):
    pass


class Sam3LiteTextDETREncoderConfig(Sam3DETREncoderConfig):
    pass


class Sam3LiteTextDETRDecoderConfig(Sam3DETRDecoderConfig):
    pass


class Sam3LiteTextMaskDecoderConfig(Sam3MaskDecoderConfig):
    pass


class Sam3LiteTextConfig(Sam3Config):
    pass


class Sam3LiteTextVisionEncoderOutput(Sam3VisionEncoderOutput):
    pass


class Sam3LiteTextGeometryEncoderOutput(Sam3GeometryEncoderOutput):
    pass


class Sam3LiteTextDETREncoderOutput(Sam3DETREncoderOutput):
    pass


class Sam3LiteTextDETRDecoderOutput(Sam3DETRDecoderOutput):
    pass


class Sam3LiteTextMaskDecoderOutput(Sam3MaskDecoderOutput):
    pass


class Sam3LiteTextImageSegmentationOutput(Sam3ImageSegmentationOutput):
    pass


class Sam3LiteTextMLP(Sam3MLP):
    pass


class Sam3LiteTextAttention(Sam3Attention):
    pass


class Sam3LiteTextViTRotaryEmbedding(Sam3ViTRotaryEmbedding):
    pass


class Sam3LiteTextViTRoPEAttention(Sam3ViTRoPEAttention):
    pass


class Sam3LiteTextViTPatchEmbeddings(Sam3ViTPatchEmbeddings):
    pass


class Sam3LiteTextViTEmbeddings(Sam3ViTEmbeddings):
    pass


class Sam3LiteTextViTLayerScale(Sam3ViTLayerScale):
    pass


class Sam3LiteTextViTLayer(Sam3ViTLayer):
    pass


class Sam3LiteTextPreTrainedModel(Sam3PreTrainedModel):
    pass


class Sam3LiteTextViTModel(Sam3ViTModel):
    pass


class Sam3LiteTextSinePositionEmbedding(Sam3SinePositionEmbedding):
    pass


class Sam3LiteTextFPNLayer(Sam3FPNLayer):
    pass


class Sam3LiteTextVisionNeck(Sam3VisionNeck):
    pass


class Sam3LiteTextVisionModel(Sam3VisionModel):
    pass


class Sam3LiteTextGeometryEncoderLayer(Sam3GeometryEncoderLayer):
    pass


class Sam3LiteTextGeometryEncoder(Sam3GeometryEncoder):
    pass


class Sam3LiteTextDetrEncoderLayer(Sam3DetrEncoderLayer):
    pass


class Sam3LiteTextDetrEncoder(Sam3DetrEncoder):
    pass


class Sam3LiteTextDecoderMLP(Sam3DecoderMLP):
    pass


class Sam3LiteTextDetrDecoderLayer(Sam3DetrDecoderLayer):
    pass


class Sam3LiteTextDetrDecoder(Sam3DetrDecoder):
    pass


class Sam3LiteTextDotProductScoring(Sam3DotProductScoring):
    pass


class Sam3LiteTextMaskEmbedder(Sam3MaskEmbedder):
    pass


class Sam3LiteTextPixelDecoder(Sam3PixelDecoder):
    pass


class Sam3LiteTextMaskDecoder(Sam3MaskDecoder):
    pass


class Sam3LiteTextModel(Sam3Model):
    pass


__all__ = [
    "Sam3LiteTextConfig",
    "Sam3LiteTextViTConfig",
    "Sam3LiteTextVisionConfig",
    "Sam3LiteTextGeometryEncoderConfig",
    "Sam3LiteTextDETREncoderConfig",
    "Sam3LiteTextDETRDecoderConfig",
    "Sam3LiteTextMaskDecoderConfig",
    "Sam3LiteTextModel",
    "Sam3LiteTextVisionModel",
    "Sam3LiteTextViTModel",
    "Sam3LiteTextPreTrainedModel",
]
