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

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils.generic import TransformersKwargs
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


@dataclass
class Sam3LiteTextTextEncoderOutput(BaseModelOutputWithPooling):
    pass


class Sam3LiteTextTextPositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings: int, hidden_size: int):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.empty(1, 1, max_position_embeddings, hidden_size))

    def forward(self, seq_len: int) -> torch.Tensor:
        position_embedding = self.position_embedding
        if seq_len != position_embedding.shape[2]:
            position_embedding = F.interpolate(
                position_embedding,
                size=(seq_len, position_embedding.shape[-1]),
                mode="bilinear",
            )
        return position_embedding.reshape(1, seq_len, -1)


class Sam3LiteTextRepMixer(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 11):
        super().__init__()
        self.norm = Sam3LiteTextMobileOneBlock(hidden_size, kernel_size=kernel_size, use_conv_branch=False)
        self.mixer = Sam3LiteTextMobileOneBlock(hidden_size, kernel_size=kernel_size, use_conv_branch=True)
        self.layer_scale = nn.Parameter(1e-5 * torch.ones((hidden_size, 1, 1)), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.layer_scale * (self.mixer(hidden_states) - self.norm(hidden_states))


class Sam3LiteTextMobileOneBlock(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 3, use_conv_branch: bool = True):
        super().__init__()
        self.rbr_skip = nn.BatchNorm2d(hidden_size)
        self.rbr_conv = nn.ModuleList()
        if use_conv_branch:
            self.rbr_conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_size,
                        hidden_size,
                        kernel_size=(1, kernel_size),
                        stride=1,
                        padding=(0, kernel_size // 2),
                        groups=hidden_size,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_size),
                )
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.rbr_skip(hidden_states)
        for branch in self.rbr_conv:
            output = output + branch(hidden_states)
        return output


class Sam3LiteTextRepMixerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.layer_scale = nn.Parameter(1e-5 * torch.ones((hidden_size, 1, 1)), requires_grad=True)
        self.token_mixer = Sam3LiteTextRepMixer(hidden_size, kernel_size=11)
        self.convffn = Sam3LiteTextConvFFN(hidden_size, intermediate_size, kernel_size=11)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.token_mixer(hidden_states)
        return hidden_states + self.layer_scale * self.convffn(hidden_states)


class Sam3LiteTextConvFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                hidden_size,
                hidden_size,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=hidden_size,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_size),
        )
        self.fc1 = nn.Conv2d(hidden_size, intermediate_size, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(intermediate_size, hidden_size, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Sam3LiteTextTextEncoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = False

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Sam3LiteTextTransformerLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Sam3LiteTextTextEncoderAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc2(self.act(self.fc1(hidden_states)))
        return residual + hidden_states, attn_weights


class Sam3LiteTextTextEncoder(nn.Module):
    def __init__(self, config: "Sam3LiteTextConfig"):
        super().__init__()
        text_config = config.text_config
        self.config = text_config
        self.token_embedding = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.position_embedding = Sam3LiteTextTextPositionEmbedding(
            text_config.max_position_embeddings, text_config.hidden_size
        )
        self.embedding_dropout = nn.Dropout(0.0)
        self.model_name = getattr(text_config, "model_name", "mct")
        if self.model_name == "mct":
            num_transformer_layers = text_config.num_hidden_layers - 2
            self.layers = nn.ModuleList(
                [Sam3LiteTextRepMixerBlock(text_config.hidden_size, text_config.intermediate_size)]
                + [Sam3LiteTextTransformerLayer(text_config) for _ in range(num_transformer_layers)]
                + [Sam3LiteTextRepMixerBlock(text_config.hidden_size, text_config.intermediate_size)]
            )
            self.repmixer_indexes = (0, text_config.num_hidden_layers - 1)
        else:
            self.layers = nn.ModuleList(
                [Sam3LiteTextTransformerLayer(text_config) for _ in range(text_config.num_hidden_layers)]
            )
            self.repmixer_indexes = ()
        self.final_layer_norm = nn.LayerNorm(text_config.hidden_size)
        self.projection = nn.Parameter(torch.empty(text_config.hidden_size, text_config.projection_dim))

    def forward(self, input_ids: torch.LongTensor, **kwargs) -> Sam3LiteTextTextEncoderOutput:
        output_attentions = kwargs.pop("output_attentions", getattr(self.config, "output_attentions", False))
        output_hidden_states = kwargs.pop("output_hidden_states", getattr(self.config, "output_hidden_states", False))
        attention_mask = kwargs.pop("attention_mask", None)
        kwargs.pop("return_dict", None)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = self.token_embedding(input_ids)
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states + self.position_embedding(seq_len).to(hidden_states.dtype)
        hidden_states = self.embedding_dropout(hidden_states)

        # Convert 2D attention mask to 4D for attention layers
        if attention_mask is not None:
            attention_mask = create_bidirectional_mask(self.config, hidden_states, attention_mask)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        for idx, layer in enumerate(self.layers):
            if idx in self.repmixer_indexes:
                hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)
                hidden_states = layer(hidden_states)
                hidden_states = hidden_states.squeeze(2).permute(0, 2, 1)
            else:
                hidden_states, attn_weights = layer(hidden_states, attention_mask=attention_mask, **kwargs)
                if output_attentions:
                    all_attentions = all_attentions + (attn_weights,)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_layer_norm(hidden_states)

        pooled = hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device), input_ids.argmax(dim=-1)
        ]
        pooled = pooled @ self.projection
        return Sam3LiteTextTextEncoderOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Sam3LiteTextViTConfig(Sam3ViTConfig):
    pass


class Sam3LiteTextVisionConfig(Sam3VisionConfig):
    def __init__(
        self,
        backbone_config=None,
        fpn_hidden_size=256,
        backbone_feature_sizes=None,
        scale_factors=None,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        if isinstance(backbone_config, dict):
            backbone_config = Sam3LiteTextViTConfig(**{k: v for k, v in backbone_config.items() if k != "model_type"})
        elif backbone_config is None:
            backbone_config = Sam3LiteTextViTConfig()
        super().__init__(
            backbone_config=backbone_config,
            fpn_hidden_size=fpn_hidden_size,
            backbone_feature_sizes=backbone_feature_sizes,
            scale_factors=scale_factors,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
            **kwargs,
        )


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
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Sam3LiteTextTextPositionEmbedding):
            nn.init.normal_(module.position_embedding, std=module.position_embedding.shape[-1] ** -0.5)
        elif isinstance(module, Sam3LiteTextTextEncoder):
            nn.init.normal_(module.projection, std=module.config.hidden_size**-0.5)


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
    def __init__(self, config: "Sam3LiteTextConfig"):
        super().__init__(config)
        self.text_encoder = Sam3LiteTextTextEncoder(config)


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
