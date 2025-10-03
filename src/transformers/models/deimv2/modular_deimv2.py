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

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...utils.backbone_utils import load_backbone
from ..rt_detr.configuration_rt_detr import RTDetrConfig
from ..rt_detr.modeling_rt_detr import (
    MultiScaleDeformableAttention,
    RTDetrConvNormLayer,
    RTDetrCSPRepLayer,
    RTDetrDecoderOutput,
    RTDetrEncoder,
    RTDetrEncoderLayer,
    RTDetrForObjectDetection,
    RTDetrFrozenBatchNorm2d,
    RTDetrHybridEncoder,
    RTDetrMLPPredictionHead,
    RTDetrModel,
    RTDetrModelOutput,
    RTDetrMultiheadAttention,
    RTDetrMultiscaleDeformableAttention,
    RTDetrObjectDetectionOutput,
    RTDetrPreTrainedModel,
    RTDetrRepVggBlock,
    inverse_sigmoid,
    replace_batch_norm,
)


class Deimv2Config(RTDetrConfig):
    def __init__(self, use_decoder_gate: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_decoder_gate = use_decoder_gate


class MultiScaleDeformableAttention(MultiScaleDeformableAttention):
    pass


class Deimv2DecoderOutput(RTDetrDecoderOutput):
    pass


class Deimv2ModelOutput(RTDetrModelOutput):
    pass


class Deimv2ObjectDetectionOutput(RTDetrObjectDetectionOutput):
    pass


class Deimv2FrozenBatchNorm2d(RTDetrFrozenBatchNorm2d):
    pass


class Deimv2SpatialTuningAdapter(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, expansion: float, activation: str = "silu"):
        super().__init__()

        mid_channels = max(int(round(in_channels * expansion)), hidden_dim)

        if activation == "relu":
            activation_layer = nn.ReLU(inplace=True)
        elif activation == "gelu":
            activation_layer = nn.GELU()
        elif activation == "tanh":
            activation_layer = nn.Tanh()
        else:
            activation_layer = nn.SiLU(inplace=True)

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation_layer,
            nn.Conv2d(mid_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.adapter(feature_map)


def deimv2_bias_init_with_prob(prior_prob: float = 0.5) -> float:
    """Initializes bias values according to a target Bernoulli prior probability."""

    if prior_prob <= 0 or prior_prob >= 1:
        raise ValueError("The prior probability used for bias initialization must be between 0 and 1.")

    return float(-math.log((1 - prior_prob) / prior_prob))


class Deimv2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normalized_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return normalized_states * self.weight


class Deimv2SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.constant_(self.w12.bias, 0)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.constant_(self.w3.bias, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_input, linear_input = self.w12(hidden_states).chunk(2, dim=-1)
        activated = F.silu(gate_input) * linear_input
        return self.w3(activated)


class Deimv2Gate(nn.Module):
    def __init__(self, hidden_size: int, use_rmsnorm: bool = True) -> None:
        super().__init__()

        self.gate = nn.Linear(2 * hidden_size, 2 * hidden_size)
        nn.init.constant_(self.gate.weight, 0)
        nn.init.constant_(self.gate.bias, deimv2_bias_init_with_prob())

        if use_rmsnorm:
            self.norm = Deimv2RMSNorm(hidden_size)
        else:
            self.norm = nn.LayerNorm(hidden_size)

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        gate_input = torch.concat([residual, update], dim=-1)
        gate1, gate2 = torch.sigmoid(self.gate(gate_input)).chunk(2, dim=-1)
        return self.norm(gate1 * residual + gate2 * update)


class Deimv2ConvEncoder(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()

        self.config = config
        self.hidden_dim = config.encoder_hidden_dim
        self.hidden_expansion = config.hidden_expansion
        self.activation = config.activation_function

        backbone_config = getattr(config, "backbone_config", None)
        backbone_model_type = getattr(backbone_config, "model_type", None) if backbone_config is not None else None

        self._uses_dinov3 = backbone_model_type == "dinov3_convnext"

        if self._uses_dinov3:
            from ..dinov3_convnext import DINOv3ConvNextModel

            self.backbone = DINOv3ConvNextModel(backbone_config)
            self.backbone_channels = list(backbone_config.hidden_sizes)
        else:
            backbone = load_backbone(config)

            if config.freeze_backbone_batch_norms:
                with torch.no_grad():
                    replace_batch_norm(backbone)

            self.backbone = backbone
            self.backbone_channels = list(self.backbone.channels)

        if len(self.backbone_channels) != len(config.encoder_in_channels):
            if len(self.backbone_channels) > len(config.encoder_in_channels):
                start = len(self.backbone_channels) - len(config.encoder_in_channels)
                self.backbone_channels = self.backbone_channels[start:]
            else:
                self.backbone_channels = list(config.encoder_in_channels)

        self.spatial_adapters = nn.ModuleList(
            Deimv2SpatialTuningAdapter(
                in_channels=in_channels,
                hidden_dim=self.hidden_dim,
                expansion=self.hidden_expansion,
                activation=self.activation,
            )
            for in_channels in self.backbone_channels
        )

        self.intermediate_channel_sizes = [self.hidden_dim] * len(self.spatial_adapters)

    def _prepare_mask(self, pixel_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        mask = pixel_mask[None].float()
        mask = nn.functional.interpolate(mask, size=(height, width), mode="nearest")
        return mask.to(dtype=torch.bool)[0]

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor | None):
        if pixel_mask is None:
            pixel_mask = torch.ones(
                (pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3]),
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            pixel_mask = pixel_mask.to(device=pixel_values.device, dtype=torch.bool)

        outputs = []

        if self._uses_dinov3:
            backbone_outputs = self.backbone(pixel_values, output_hidden_states=True, return_dict=True).hidden_states[
                1:
            ]
            selected_feature_maps = backbone_outputs[-len(self.spatial_adapters) :]
        else:
            selected_feature_maps = self.backbone(pixel_values).feature_maps

        for feature_map, adapter in zip(selected_feature_maps, self.spatial_adapters):
            tuned_feature = adapter(feature_map)
            mask = self._prepare_mask(pixel_mask, tuned_feature.shape[-2], tuned_feature.shape[-1])
            outputs.append((tuned_feature, mask))

        return outputs


class Deimv2ConvNormLayer(RTDetrConvNormLayer):
    pass


class Deimv2EncoderLayer(RTDetrEncoderLayer):
    pass


class Deimv2RepVggBlock(RTDetrRepVggBlock):
    pass


class Deimv2CSPRepLayer(RTDetrCSPRepLayer):
    pass


class Deimv2MultiscaleDeformableAttention(RTDetrMultiscaleDeformableAttention):
    pass


class Deimv2MultiheadAttention(RTDetrMultiheadAttention):
    pass


class Deimv2DecoderLayer(nn.Module):
    def __init__(self, config: Deimv2Config):
        super().__init__()

        self.config = config
        self.dropout = config.dropout

        self.self_attn = Deimv2MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = Deimv2RMSNorm(config.d_model, eps=config.layer_norm_eps)

        self.encoder_attn = Deimv2MultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_dropout = nn.Dropout(config.dropout)
        self.use_gateway = config.use_decoder_gate
        if self.use_gateway:
            self.gateway = Deimv2Gate(config.d_model, use_rmsnorm=True)
            self.encoder_attn_layer_norm = None
        else:
            self.gateway = None
            self.encoder_attn_layer_norm = Deimv2RMSNorm(config.d_model, eps=config.layer_norm_eps)

        hidden_features = max(config.d_model, config.decoder_ffn_dim // 2)
        self.feedforward = Deimv2SwiGLUFFN(config.d_model, hidden_features, config.d_model)
        self.feedforward_dropout = nn.Dropout(config.dropout)
        self.feedforward_layer_norm = Deimv2RMSNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        spatial_shapes_list: Optional[List[Tuple[int, int]]] = None,
        level_start_index: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )

        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        second_residual = hidden_states
        cross_attn_weights = None

        cross_hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        cross_hidden_states = self.encoder_attn_dropout(cross_hidden_states)

        if self.gateway is not None:
            hidden_states = self.gateway(second_residual, cross_hidden_states)
        else:
            hidden_states = self.encoder_attn_layer_norm(second_residual + cross_hidden_states)

        residual = hidden_states
        feedforward_hidden_states = self.feedforward(hidden_states)
        feedforward_hidden_states = self.feedforward_dropout(feedforward_hidden_states)
        hidden_states = residual + feedforward_hidden_states
        hidden_states = self.feedforward_layer_norm(hidden_states.clamp(min=-65504, max=65504))

        outputs: tuple[torch.Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class Deimv2PreTrainedModel(RTDetrPreTrainedModel):
    pass


class Deimv2Encoder(RTDetrEncoder):
    pass


class Deimv2HybridEncoder(RTDetrHybridEncoder):
    pass


class Deimv2Decoder(RTDetrPreTrainedModel):
    def __init__(self, config: Deimv2Config):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([Deimv2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.query_pos_head = Deimv2MLPPredictionHead(config, 4, 2 * config.d_model, config.d_model, num_layers=2)

        self.bbox_embed = None
        self.class_embed = None

        self.post_init()

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        spatial_shapes_list: Optional[List[Tuple[int, int]]] = None,
        level_start_index: Optional[torch.Tensor] = None,
        valid_ratios: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> RTDetrDecoderOutput | tuple[torch.Tensor, ...]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            raise ValueError("Decoder requires `inputs_embeds` to be provided.")
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_logits = ()

        if reference_points is None:
            raise ValueError("Decoder expects `reference_points` to be provided.")
        reference_points = F.sigmoid(reference_points)

        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            position_embeddings = self.query_pos_head(reference_points)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if self.bbox_embed is not None:
                predicted_corners = self.bbox_embed[idx](hidden_states)
                new_reference_points = F.sigmoid(predicted_corners + inverse_sigmoid(reference_points))
                reference_points = new_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (
                (new_reference_points,) if self.bbox_embed is not None else (reference_points,)
            )

            if self.class_embed is not None:
                logits = self.class_embed[idx](hidden_states)
                intermediate_logits += (logits,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        if self.class_embed is not None:
            intermediate_logits = torch.stack(intermediate_logits, dim=1)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_logits,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return Deimv2DecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_logits=intermediate_logits,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class Deimv2MLPPredictionHead(RTDetrMLPPredictionHead):
    pass


class Deimv2Model(RTDetrModel):
    pass


class Deimv2ForObjectDetection(RTDetrForObjectDetection):
    pass


__all__ = [
    "Deimv2Config",
    "Deimv2ForObjectDetection",
    "Deimv2Model",
    "Deimv2PreTrainedModel",
]
