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
from torch import nn

from ..eomt.configuration_eomt import EomtConfig
from ..eomt.modeling_eomt import (
    EomtAttention,
    EomtDropPath,
    EomtEmbeddings,
    EomtForUniversalSegmentation,
    EomtForUniversalSegmentationOutput,
    EomtHungarianMatcher,
    EomtLayer,
    EomtLayerNorm2d,
    EomtLayerScale,
    EomtLoss,
    EomtMaskHead,
    EomtMLP,
    EomtPreTrainedModel,
    EomtScaleBlock,
    EomtScaleLayer,
    EomtSwiGLUFFN,
)


class VideomtConfig(EomtConfig):
    model_type = "videomt"


class VideomtAttention(EomtAttention):
    pass


class VideomtEmbeddings(EomtEmbeddings):
    def __init__(self, config: VideomtConfig):
        super().__init__(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
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

        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            mask_token = self.mask_token.to(embeddings.dtype)
            embeddings = torch.where(bool_masked_pos.unsqueeze(-1), mask_token, embeddings)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        embeddings = embeddings + self.position_embeddings(self.position_ids)
        embeddings = torch.cat([cls_tokens, register_tokens, embeddings], dim=1)
        embeddings = self.dropout(embeddings)
        return embeddings


class VideomtDropPath(EomtDropPath):
    pass


class VideomtMLP(EomtMLP):
    pass


class VideomtGatedMLP(EomtSwiGLUFFN):
    pass


class VideomtLayer(EomtLayer):
    pass


class VideomtLayerScale(EomtLayerScale):
    pass


class VideomtHungarianMatcher(EomtHungarianMatcher):
    pass


class VideomtLoss(EomtLoss):
    pass


class VideomtForUniversalSegmentationOutput(EomtForUniversalSegmentationOutput):
    pass


class VideomtPreTrainedModel(EomtPreTrainedModel):
    pass


class VideomtLayerNorm2d(EomtLayerNorm2d):
    pass


class VideomtScaleLayer(EomtScaleLayer):
    pass


class VideomtScaleBlock(EomtScaleBlock):
    pass


class VideomtMaskHead(EomtMaskHead):
    pass


class VideomtForUniversalSegmentation(EomtForUniversalSegmentation):
    def __init__(self, config: VideomtConfig):
        super().__init__(config)
        self.query_updater = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
        patch_offsets: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> VideomtForUniversalSegmentationOutput:
        if pixel_values.ndim != 5:
            return super().forward(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
                patch_offsets=patch_offsets,
                **kwargs,
            )

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

        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        flat_pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)
        frame_embeddings = self.embeddings(flat_pixel_values).view(batch_size, num_frames, -1, self.config.hidden_size)

        all_masks_queries_logits = []
        all_class_queries_logits = []
        all_last_hidden_states = []
        propagated_query = None

        for frame_idx in range(num_frames):
            hidden_states = frame_embeddings[:, frame_idx]
            attention_mask = None

            for layer_idx, layer_module in enumerate(self.layers):
                if layer_idx == self.num_hidden_layers - self.config.num_blocks:
                    if propagated_query is None:
                        query_tokens = self.query.weight[None, :, :].expand(batch_size, -1, -1)
                    else:
                        query_tokens = self.query_updater(propagated_query) + self.query.weight[None, :, :]
                    hidden_states = torch.cat((query_tokens.to(hidden_states.device), hidden_states), dim=1)

                if layer_idx >= self.num_hidden_layers - self.config.num_blocks and (
                    self.training
                    or self.attn_mask_probs[layer_idx - self.num_hidden_layers + self.config.num_blocks] > 0
                ):
                    norm_hidden_states = self.layernorm(hidden_states)
                    masks_queries_logits, _ = self.predict(norm_hidden_states)

                    attention_mask = torch.ones(
                        hidden_states.shape[0],
                        hidden_states.shape[1],
                        hidden_states.shape[1],
                        device=hidden_states.device,
                        dtype=torch.bool,
                    )

                    interpolated_logits = torch.nn.functional.interpolate(
                        masks_queries_logits, size=self.grid_size, mode="bilinear"
                    )
                    interpolated_logits = interpolated_logits.view(
                        interpolated_logits.size(0), interpolated_logits.size(1), -1
                    )

                    num_query_tokens = self.config.num_queries
                    encoder_start_tokens = num_query_tokens + self.embeddings.num_prefix_tokens
                    attention_mask[:, :num_query_tokens, encoder_start_tokens:] = interpolated_logits > 0

                    attention_mask = self._disable_attention_mask(
                        attention_mask,
                        prob=self.attn_mask_probs[layer_idx - self.num_hidden_layers + self.config.num_blocks],
                        num_query_tokens=num_query_tokens,
                        encoder_start_tokens=encoder_start_tokens,
                        device=attention_mask.device,
                    )

                    attention_mask = attention_mask[:, None, ...].expand(-1, self.config.num_attention_heads, -1, -1)
                    attention_mask = attention_mask.float().masked_fill(~attention_mask, -1e9)

                hidden_states = layer_module(hidden_states, attention_mask)

            sequence_output = self.layernorm(hidden_states)
            masks_queries_logits, class_queries_logits = self.predict(sequence_output)

            all_masks_queries_logits.append(masks_queries_logits)
            all_class_queries_logits.append(class_queries_logits)
            all_last_hidden_states.append(sequence_output)
            propagated_query = sequence_output[:, : self.config.num_queries, :]

        return VideomtForUniversalSegmentationOutput(
            loss=None,
            masks_queries_logits=torch.cat(all_masks_queries_logits, dim=0),
            class_queries_logits=torch.cat(all_class_queries_logits, dim=0),
            last_hidden_state=torch.cat(all_last_hidden_states, dim=0),
            patch_offsets=patch_offsets,
        )


__all__ = [
    "VideomtConfig",
    "VideomtPreTrainedModel",
    "VideomtForUniversalSegmentation",
]
