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

import torch

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
    model_type = "videomt"


class VideomtAttention(EomtDinov3Attention):
    pass


class VideomtEmbeddings(EomtDinov3Embeddings):
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.Tensor`):
                Input frames as either `(batch_size, num_frames, num_channels, height, width)` or flattened
                `(batch_size * num_frames, num_channels, height, width)`.
            bool_masked_pos (`torch.Tensor`, *optional*):
                Optional mask for patch replacement.
        """

        if pixel_values.ndim == 5:
            batch_size, num_frames, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)

            if bool_masked_pos is not None and bool_masked_pos.ndim >= 3:
                bool_masked_pos = bool_masked_pos.reshape(batch_size * num_frames, -1)
        elif bool_masked_pos is not None and bool_masked_pos.ndim > 2:
            bool_masked_pos = bool_masked_pos.reshape(bool_masked_pos.shape[0], -1)

        if bool_masked_pos is not None:
            if bool_masked_pos.dtype != torch.bool:
                raise ValueError(f"Expected bool_masked_pos dtype to be torch.bool, but got {bool_masked_pos.dtype}.")

            if bool_masked_pos.shape[0] != pixel_values.shape[0]:
                raise ValueError(
                    f"Expected bool_masked_pos batch dimension to match pixel_values batch dimension "
                    f"({pixel_values.shape[0]}), but got {bool_masked_pos.shape[0]}."
                )

            patch_size = self.config.patch_size
            expected_num_patches = (pixel_values.shape[-2] // patch_size) * (pixel_values.shape[-1] // patch_size)
            if bool_masked_pos.shape[-1] != expected_num_patches:
                raise ValueError(
                    f"Expected bool_masked_pos to provide one value per patch ({expected_num_patches}), "
                    f"but got {bool_masked_pos.shape[-1]}."
                )

        return super().forward(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)


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
    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
        patch_offsets: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> VideomtForUniversalSegmentationOutput:
        if pixel_values.ndim == 5:
            batch_size, num_frames, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)

            if mask_labels is not None or class_labels is not None:
                raise ValueError(
                    "Video training labels are not supported yet for `VideomtForUniversalSegmentation`; "
                    "please provide flattened frame batches for training."
                )

            if patch_offsets is not None:
                raise ValueError(
                    "Video-shaped `patch_offsets` are not supported yet for `VideomtForUniversalSegmentation`; "
                    "please provide flattened frame batches with matching patch offsets."
                )

        return super().forward(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            patch_offsets=patch_offsets,
            **kwargs,
        )


__all__ = [
    "VideomtConfig",
    "VideomtPreTrainedModel",
    "VideomtForUniversalSegmentation",
]
