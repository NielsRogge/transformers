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

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...utils import torch_int
from ...utils.backbone_utils import load_backbone
from ..rt_detr.configuration_rt_detr import RTDetrConfig
from ..rt_detr.modeling_rt_detr import (
    MultiScaleDeformableAttention,
    RTDetrConvNormLayer,
    RTDetrCSPRepLayer,
    RTDetrDecoder,
    RTDetrDecoderLayer,
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
    replace_batch_norm,
)


class Deimv2Config(RTDetrConfig):
    pass


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
            backbone_outputs = self.backbone(
                pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[1:]
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


class Deimv2DecoderLayer(RTDetrDecoderLayer):
    pass


class Deimv2PreTrainedModel(RTDetrPreTrainedModel):
    pass


class Deimv2Encoder(RTDetrEncoder):
    pass


class Deimv2HybridEncoder(RTDetrHybridEncoder):
    pass


class Deimv2Decoder(RTDetrDecoder):
    pass


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
