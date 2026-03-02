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

import copy
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..d_fine.configuration_d_fine import DFineConfig
from ..d_fine.modeling_d_fine import (
    DFineDecoderOutput,
    DFineHybridEncoder,
    DFineIntegral,
    DFineMLP,
    DFineMultiscaleDeformableAttention,
    distance2bbox,
    inverse_sigmoid,
    weighting_function,
)
from ..rt_detr.image_processing_rt_detr import RTDetrImageProcessor, RTDetrImageProcessorKwargs
from ..rt_detr.image_processing_rt_detr_fast import RTDetrImageProcessorFast
from ..rt_detr.modeling_rt_detr import (
    RTDetrConvEncoder,
    RTDetrForObjectDetection,
    RTDetrModelOutput,
    RTDetrObjectDetectionOutput,
    RTDetrPreTrainedModel,
    get_contrastive_denoising_training_group,
)


class Deimv2Config(DFineConfig):
    model_type = "deimv2"
    pass


class Deimv2DecoderOutput(DFineDecoderOutput):
    pass


class Deimv2ModelOutput(RTDetrModelOutput):
    pass


class Deimv2ObjectDetectionOutput(RTDetrObjectDetectionOutput):
    pass


class Deimv2MLP(DFineMLP):
    pass


class Deimv2Integral(DFineIntegral):
    pass


class Deimv2LQE(nn.Module):
    def __init__(self, config: Deimv2Config):
        super().__init__()
        self.top_prob_values = config.top_prob_values
        self.max_num_bins = config.max_num_bins
        self.reg_conf = Deimv2MLP(
            4 * (self.top_prob_values + 1),
            config.lqe_hidden_dim,
            1,
            config.lqe_layers,
            act=config.activation_function,
        )

    def forward(self, scores: torch.Tensor, pred_corners: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(batch_size, length, 4, self.max_num_bins + 1), dim=-1)
        prob_topk, _ = prob.topk(self.top_prob_values, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(batch_size, length, -1))
        return scores + quality_score


class Deimv2DFineMultiscaleDeformableAttention(DFineMultiscaleDeformableAttention):
    pass


class Deimv2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._norm(hidden_states.float()).type_as(hidden_states)
        return hidden_states * self.scale


class Deimv2SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_12 = self.w12(hidden_states)
        hidden_states_1, hidden_states_2 = hidden_states_12.chunk(2, dim=-1)
        hidden_states = F.silu(hidden_states_1) * hidden_states_2
        return self.w3(hidden_states)


class Deimv2Gate(nn.Module):
    def __init__(self, hidden_size: int, use_rmsnorm: bool = True):
        super().__init__()
        self.gate = nn.Linear(2 * hidden_size, 2 * hidden_size)
        bias = float(-math.log((1 - 0.5) / 0.5))
        nn.init.constant_(self.gate.bias, bias)
        nn.init.constant_(self.gate.weight, 0)
        self.norm = Deimv2RMSNorm(hidden_size) if use_rmsnorm else nn.LayerNorm(hidden_size)

    def forward(self, hidden_states_1: torch.Tensor, hidden_states_2: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([hidden_states_1, hidden_states_2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate_1, gate_2 = gates.chunk(2, dim=-1)
        return self.norm(gate_1 * hidden_states_1 + gate_2 * hidden_states_2)


class Deimv2ConvNormLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        padding: int | None = None,
        bias: bool = False,
        activation: str | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.act(hidden_states)


class Deimv2GAPFusion(nn.Module):
    def __init__(self, out_channels: int, activation: str | None = None):
        super().__init__()
        self.cv = Deimv2ConvNormLayer(out_channels, out_channels, 1, 1, activation=activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_hidden_states = F.adaptive_avg_pool2d(hidden_states, 1)
        hidden_states = hidden_states + pooled_hidden_states
        return self.cv(hidden_states)


class Deimv2VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "relu"):
        super().__init__()
        self.conv1 = Deimv2ConvNormLayer(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = Deimv2ConvNormLayer(in_channels, out_channels, 1, 1, padding=0)
        self.act = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states) + self.conv2(hidden_states)
        return self.act(hidden_states)


class Deimv2CSPLayer2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Deimv2ConvNormLayer(in_channels, hidden_channels * 2, 1, 1, bias=bias, activation=activation)
        self.bottlenecks = nn.Sequential(
            *[Deimv2VGGBlock(hidden_channels, hidden_channels, activation=activation) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = Deimv2ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = list(self.conv1(hidden_states).chunk(2, 1))
        return self.conv3(hidden_states[0] + self.bottlenecks(hidden_states[1]))


class Deimv2RepNCSPELAN4(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        bottleneck_channels: int,
        num_blocks: int = 3,
        bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        self.c = hidden_channels // 2
        self.cv1 = Deimv2ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, activation=activation)

        self.cv2 = nn.Sequential(
            Deimv2CSPLayer2(self.c, bottleneck_channels, num_blocks, 1, bias=bias, activation=activation),
            Deimv2ConvNormLayer(bottleneck_channels, bottleneck_channels, 3, 1, bias=bias, activation=activation),
        )
        self.cv3 = nn.Sequential(
            Deimv2CSPLayer2(
                bottleneck_channels,
                bottleneck_channels,
                num_blocks,
                1,
                bias=bias,
                activation=activation,
            ),
            Deimv2ConvNormLayer(bottleneck_channels, bottleneck_channels, 3, 1, bias=bias, activation=activation),
        )
        self.cv4 = Deimv2ConvNormLayer(
            hidden_channels + (2 * bottleneck_channels), out_channels, 1, 1, bias=bias, activation=activation
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output_states = list(self.cv1(hidden_states).split((self.c, self.c), 1))
        output_states.extend(module(output_states[-1]) for module in [self.cv2, self.cv3])
        return self.cv4(torch.cat(output_states, 1))


class Deimv2LiteEncoder(nn.Module):
    def __init__(self, config: Deimv2Config):
        super().__init__()
        self.in_channels = config.encoder_in_channels
        self.feat_strides = [config.feat_strides[-1]]
        self.hidden_dim = config.encoder_hidden_dim
        self.out_channels = [self.hidden_dim for _ in range(len(self.in_channels))]
        self.out_strides = self.feat_strides

        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            proj = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(in_channel, self.hidden_dim, kernel_size=1, bias=False)),
                        ("norm", nn.BatchNorm2d(self.hidden_dim)),
                    ]
                )
            )
            self.input_proj.append(proj)

        activation = config.activation_function
        down_sample = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            ACT2CLS[activation](),
        )
        self.down_sample1 = copy.deepcopy(down_sample)
        self.down_sample2 = copy.deepcopy(down_sample)

        self.bi_fusion = Deimv2GAPFusion(self.hidden_dim, activation=activation)

        expansion = config.hidden_expansion
        depth_mult = config.depth_mult
        hidden_channels = self.hidden_dim * 2
        bottleneck_channels = round(expansion * self.hidden_dim // 2)
        num_blocks = round(3 * depth_mult)

        fuse_block = Deimv2RepNCSPELAN4(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            hidden_channels=hidden_channels,
            bottleneck_channels=bottleneck_channels,
            num_blocks=num_blocks,
            activation=activation,
        )
        self.fpn_block = copy.deepcopy(fuse_block)
        self.pan_block = copy.deepcopy(fuse_block)

    def forward(self, features: list[torch.Tensor], **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        if len(features) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} feature maps, got {len(features)}")

        projected_features = [self.input_proj[i](feature) for i, feature in enumerate(features)]
        projected_features.append(self.down_sample1(projected_features[-1]))
        projected_features[-1] = self.bi_fusion(projected_features[-1])

        outputs = []
        fused_features = projected_features[0] + F.interpolate(projected_features[1], scale_factor=2.0, mode="nearest")
        outputs.append(self.fpn_block(fused_features))

        fused_features = projected_features[1] + self.down_sample2(outputs[-1])
        outputs.append(self.pan_block(fused_features))

        return BaseModelOutput(last_hidden_state=outputs, hidden_states=None, attentions=None)


class Deimv2DecoderLayer(nn.Module):
    def __init__(self, config: Deimv2Config, layer_scale: float | None = None):
        super().__init__()
        hidden_size = config.d_model
        decoder_ffn_dim = config.decoder_ffn_dim

        if layer_scale is not None and layer_scale != 1:
            hidden_size = round(layer_scale * hidden_size)
            decoder_ffn_dim = round(layer_scale * decoder_ffn_dim)

        self.hidden_size = hidden_size
        self.dropout = config.dropout
        self.use_gateway = getattr(config, "use_gateway", True)

        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = Deimv2RMSNorm(hidden_size)

        self.cross_attn = Deimv2DFineMultiscaleDeformableAttention(config)
        self.dropout2 = nn.Dropout(config.dropout)

        if self.use_gateway:
            self.gateway = Deimv2Gate(hidden_size, use_rmsnorm=True)
        else:
            self.norm2 = Deimv2RMSNorm(hidden_size)

        self.swish_ffn = Deimv2SwiGLUFFN(hidden_size, decoder_ffn_dim // 2, hidden_size)
        self.dropout4 = nn.Dropout(config.dropout)
        self.norm3 = Deimv2RMSNorm(hidden_size)

    @staticmethod
    def with_pos_embed(hidden_states: torch.Tensor, position_embeddings: torch.Tensor | None) -> torch.Tensor:
        return hidden_states if position_embeddings is None else hidden_states + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        spatial_shapes,
        spatial_shapes_list,
        attn_mask: torch.Tensor | None = None,
        query_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = keys = self.with_pos_embed(hidden_states, query_pos_embed)
        hidden_states_self_attn, _ = self.self_attn(queries, keys, value=hidden_states, attn_mask=attn_mask)
        hidden_states = hidden_states + self.dropout1(hidden_states_self_attn)
        hidden_states = self.norm1(hidden_states)

        hidden_states_cross_attn, _ = self.cross_attn(
            hidden_states=self.with_pos_embed(hidden_states, query_pos_embed),
            reference_points=reference_points,
            encoder_hidden_states=encoder_hidden_states,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
        )

        if self.use_gateway:
            hidden_states = self.gateway(hidden_states, self.dropout2(hidden_states_cross_attn))
        else:
            hidden_states = hidden_states + self.dropout2(hidden_states_cross_attn)
            hidden_states = self.norm2(hidden_states)

        hidden_states_ffn = self.swish_ffn(hidden_states)
        hidden_states = hidden_states + self.dropout4(hidden_states_ffn)
        hidden_states = self.norm3(hidden_states.clamp(min=-65504, max=65504))
        return hidden_states


class Deimv2TransformerDecoder(nn.Module):
    def __init__(
        self, config: Deimv2Config, decoder_layer: Deimv2DecoderLayer, decoder_layer_wide: Deimv2DecoderLayer | None
    ):
        super().__init__()
        self.hidden_dim = config.d_model
        self.num_layers = config.decoder_layers
        self.layer_scale = config.layer_scale
        self.num_head = config.decoder_attention_heads
        self.eval_idx = config.eval_idx if config.eval_idx >= 0 else config.decoder_layers + config.eval_idx
        self.up = nn.Parameter(torch.tensor([config.up]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([config.reg_scale]), requires_grad=False)
        self.reg_max = config.max_num_bins

        base_layers = [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
        if decoder_layer_wide is not None:
            base_layers += [
                copy.deepcopy(decoder_layer_wide) for _ in range(config.decoder_layers - self.eval_idx - 1)
            ]
        else:
            base_layers += [copy.deepcopy(decoder_layer) for _ in range(config.decoder_layers - self.eval_idx - 1)]
        self.layers = nn.ModuleList(base_layers)
        self.lqe_layers = nn.ModuleList([Deimv2LQE(config) for _ in range(config.decoder_layers)])

    def forward(
        self,
        target: torch.Tensor,
        reference_points_unact: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes,
        spatial_shapes_list,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask: torch.Tensor | None = None,
    ):
        hidden_states = target
        hidden_states_detached = predicted_corners_undetached = 0

        predicted_bboxes = []
        predicted_logits = []
        predicted_corners_list = []
        predicted_refs = []
        hidden_states_list = []

        project = weighting_function(self.reg_max, up, reg_scale)
        reference_points_detached = torch.sigmoid(reference_points_unact)
        query_pos_embed = query_pos_head(reference_points_detached).clamp(min=-10, max=10)

        for layer_idx, layer_module in enumerate(self.layers):
            reference_points_input = reference_points_detached.unsqueeze(2)

            hidden_states = layer_module(
                hidden_states,
                reference_points_input,
                memory,
                spatial_shapes,
                spatial_shapes_list,
                attn_mask=attn_mask,
                query_pos_embed=query_pos_embed,
            )
            hidden_states_list.append(hidden_states)

            if layer_idx == 0:
                pre_bboxes = torch.sigmoid(pre_bbox_head(hidden_states) + inverse_sigmoid(reference_points_detached))
                pre_scores = score_head[0](hidden_states)
                initial_reference_points = pre_bboxes.detach()

            predicted_corners = (
                bbox_head[layer_idx](hidden_states + hidden_states_detached) + predicted_corners_undetached
            )
            inter_ref_bbox = distance2bbox(initial_reference_points, integral(predicted_corners, project), reg_scale)

            if self.training or layer_idx == self.eval_idx:
                scores = score_head[layer_idx](hidden_states)
                scores = self.lqe_layers[layer_idx](scores, predicted_corners)
                predicted_logits.append(scores)
                predicted_bboxes.append(inter_ref_bbox)
                predicted_corners_list.append(predicted_corners)
                predicted_refs.append(initial_reference_points)

                if not self.training:
                    break

            predicted_corners_undetached = predicted_corners
            reference_points_detached = inter_ref_bbox.detach()
            hidden_states_detached = hidden_states.detach()

        return (
            torch.stack(predicted_bboxes),
            torch.stack(predicted_logits),
            torch.stack(predicted_corners_list),
            torch.stack(predicted_refs),
            pre_bboxes,
            pre_scores,
            torch.stack(hidden_states_list),
        )


class Deimv2Decoder(nn.Module):
    def __init__(self, config: Deimv2Config):
        super().__init__()
        self.config = config
        self.eval_idx = config.eval_idx if config.eval_idx >= 0 else config.decoder_layers + config.eval_idx

        self.up = nn.Parameter(torch.tensor([config.up]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([config.reg_scale]), requires_grad=False)

        self.query_pos_head = Deimv2MLP(4, config.d_model, config.d_model, 3, act=config.decoder_activation_function)
        self.pre_bbox_head = Deimv2MLP(config.d_model, config.d_model, 4, 3, act=config.decoder_activation_function)
        self.integral = Deimv2Integral(config)

        decoder_layer = Deimv2DecoderLayer(config)
        decoder_layer_wide = None
        if config.layer_scale != 1 and self.eval_idx < config.decoder_layers - 1:
            decoder_layer_wide = Deimv2DecoderLayer(config, layer_scale=config.layer_scale)
        self.decoder = Deimv2TransformerDecoder(config, decoder_layer, decoder_layer_wide)

        self.class_embed = None
        self.bbox_embed = None

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        inputs_embeds: torch.Tensor,
        spatial_shapes,
        level_start_index=None,
        spatial_shapes_list=None,
        encoder_attention_mask=None,
        memory_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Deimv2DecoderOutput:
        if self.class_embed is None or self.bbox_embed is None:
            raise ValueError("Decoder class and box heads must be initialized before running forward.")

        (
            out_bboxes,
            out_logits,
            out_corners,
            out_refs,
            _pre_bboxes,
            _pre_logits,
            hidden_states,
        ) = self.decoder(
            target=inputs_embeds,
            reference_points_unact=reference_points,
            memory=encoder_hidden_states,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            bbox_head=self.bbox_embed,
            score_head=self.class_embed,
            query_pos_head=self.query_pos_head,
            pre_bbox_head=self.pre_bbox_head,
            integral=self.integral,
            up=self.up,
            reg_scale=self.reg_scale,
            attn_mask=encoder_attention_mask,
        )

        return Deimv2DecoderOutput(
            last_hidden_state=hidden_states[-1],
            intermediate_hidden_states=hidden_states.permute(1, 0, 2, 3),
            intermediate_logits=out_logits.permute(1, 0, 2, 3),
            intermediate_reference_points=out_bboxes.permute(1, 0, 2, 3),
            intermediate_predicted_corners=out_corners.permute(1, 0, 2, 3),
            initial_reference_points=out_refs.permute(1, 0, 2, 3),
        )


class Deimv2PreTrainedModel(RTDetrPreTrainedModel):
    config: Deimv2Config
    base_model_prefix = "model"
    _no_split_modules = [r"Deimv2DecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Deimv2RMSNorm):
            nn.init.ones_(module.scale)
        if isinstance(module, Deimv2SwiGLUFFN):
            nn.init.xavier_uniform_(module.w12.weight)
            nn.init.zeros_(module.w12.bias)
            nn.init.xavier_uniform_(module.w3.weight)
            nn.init.zeros_(module.w3.bias)
        if isinstance(module, Deimv2LQE):
            nn.init.constant_(module.reg_conf.layers[-1].bias, 0)
            nn.init.constant_(module.reg_conf.layers[-1].weight, 0)


class Deimv2ConvEncoder(RTDetrConvEncoder):
    pass


class Deimv2HybridEncoder(DFineHybridEncoder):
    pass


class Deimv2Model(Deimv2PreTrainedModel):
    def __init__(self, config: Deimv2Config):
        super().__init__(config)

        self.backbone = Deimv2ConvEncoder(config)
        num_backbone_outs = len(self.backbone.intermediate_channel_sizes)

        # DEIMv2 encoders own their projection layers.
        self.encoder_input_proj = nn.ModuleList([nn.Identity() for _ in range(num_backbone_outs)])

        if getattr(config, "encoder_type", "hybrid") == "lite":
            self.encoder = Deimv2LiteEncoder(config)
        else:
            self.encoder = Deimv2HybridEncoder(config)

        if config.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                config.num_labels + 1, config.d_model, padding_idx=config.num_labels
            )

        if config.learn_initial_query:
            self.weight_embedding = nn.Embedding(config.num_queries, config.d_model)

        self.enc_output = nn.Identity()
        self.enc_score_head = nn.Linear(config.d_model, config.num_labels)
        self.enc_bbox_head = Deimv2MLP(config.d_model, config.d_model, 4, 3, act=config.decoder_activation_function)

        if config.anchor_image_size:
            self.anchors, self.valid_mask = self.generate_anchors(dtype=self.dtype)

        num_backbone_outs = len(config.decoder_in_channels)
        decoder_input_proj = []
        in_channels = config.decoder_in_channels[-1]
        for _ in range(num_backbone_outs):
            if config.d_model == config.decoder_in_channels[-1]:
                decoder_input_proj.append(nn.Identity())
            else:
                decoder_input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=1, bias=False),
                        nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                    )
                )
        for _ in range(config.num_feature_levels - num_backbone_outs):
            if config.d_model == config.decoder_in_channels[-1]:
                decoder_input_proj.append(nn.Identity())
            else:
                decoder_input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                    )
                )
                in_channels = config.d_model
        self.decoder_input_proj = nn.ModuleList(decoder_input_proj)

        self.decoder = Deimv2Decoder(config)
        self.post_init()

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)

    def generate_anchors(self, spatial_shapes=None, grid_size=0.05, device="cpu", dtype=torch.float32):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.config.anchor_image_size[0] / stride), int(self.config.anchor_image_size[1] / stride)]
                for stride in self.config.feat_strides
            ]
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, device=device).to(dtype),
                torch.arange(end=width, device=device).to(dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            grid_xy = grid_xy.unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
        eps = 1e-2
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(torch.finfo(dtype).max, dtype=dtype, device=device))
        return anchors, valid_mask

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | Deimv2ModelOutput:
        if pixel_values is None and inputs_embeds is None:
            raise ValueError("You have to specify either pixel_values or inputs_embeds")

        if inputs_embeds is None:
            batch_size, _, height, width = pixel_values.shape
            device = pixel_values.device
            if pixel_mask is None:
                pixel_mask = torch.ones(((batch_size, height, width)), device=device)
            features = self.backbone(pixel_values, pixel_mask)
            projected_features = [self.encoder_input_proj[level](source) for level, (source, _) in enumerate(features)]
        else:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            projected_features = inputs_embeds

        if encoder_outputs is None:
            encoder_outputs = self.encoder(projected_features, **kwargs)
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        sources = []
        for level, source in enumerate(encoder_outputs.last_hidden_state):
            sources.append(self.decoder_input_proj[level](source))

        if self.config.num_feature_levels > len(sources):
            len_sources = len(sources)
            sources.append(self.decoder_input_proj[len_sources](encoder_outputs.last_hidden_state[-1]))
            for i in range(len_sources + 1, self.config.num_feature_levels):
                sources.append(self.decoder_input_proj[i](sources[-1]))

        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
        for level, source in enumerate(sources):
            height, width = source.shape[-2:]
            spatial_shapes[level, 0] = height
            spatial_shapes[level, 1] = width
            spatial_shapes_list.append((height, width))
            source_flatten.append(source.flatten(2).transpose(1, 2))
        source_flatten = torch.cat(source_flatten, 1)

        if self.training and self.config.num_denoising > 0 and labels is not None:
            denoising_class, denoising_bbox_unact, attention_mask, denoising_meta_values = (
                get_contrastive_denoising_training_group(
                    targets=labels,
                    num_classes=self.config.num_labels,
                    num_queries=self.config.num_queries,
                    class_embed=self.denoising_class_embed,
                    num_denoising_queries=self.config.num_denoising,
                    label_noise_ratio=self.config.label_noise_ratio,
                    box_noise_scale=self.config.box_noise_scale,
                )
            )
        else:
            denoising_class, denoising_bbox_unact, attention_mask, denoising_meta_values = None, None, None, None

        dtype = source_flatten.dtype
        if self.training or self.config.anchor_image_size is None:
            anchors, valid_mask = self.generate_anchors(tuple(spatial_shapes_list), device=device, dtype=dtype)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
            anchors, valid_mask = anchors.to(device, dtype), valid_mask.to(device, dtype)

        memory = valid_mask.to(source_flatten.dtype) * source_flatten
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.config.num_queries, dim=1)
        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
        )
        enc_topk_bboxes = torch.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        if self.config.learn_initial_query:
            target = self.weight_embedding.tile([batch_size, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        init_reference_points = reference_points_unact.detach()

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=memory,
            encoder_attention_mask=attention_mask,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            **kwargs,
        )

        return Deimv2ModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_logits=decoder_outputs.intermediate_logits,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            intermediate_predicted_corners=decoder_outputs.intermediate_predicted_corners,
            initial_reference_points=decoder_outputs.initial_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            init_reference_points=init_reference_points,
            enc_topk_logits=enc_topk_logits,
            enc_topk_bboxes=enc_topk_bboxes,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            denoising_meta_values=denoising_meta_values,
        )


class Deimv2ForObjectDetection(RTDetrForObjectDetection):
    _no_split_modules = None

    def __init__(self, config: Deimv2Config):
        Deimv2PreTrainedModel.__init__(self, config)

        self.model = Deimv2Model(config)

        num_predictions = config.decoder_layers
        decoder_eval_idx = config.eval_idx if config.eval_idx >= 0 else config.decoder_layers + config.eval_idx

        decoder_score_head = nn.Linear(config.d_model, config.num_labels)
        if getattr(config, "share_score_head", False):
            class_embed = [decoder_score_head for _ in range(num_predictions)]
        else:
            class_embed = [copy.deepcopy(decoder_score_head) for _ in range(num_predictions)]

        decoder_bbox_head = Deimv2MLP(
            config.d_model,
            config.d_model,
            4 * (config.max_num_bins + 1),
            3,
            act=config.decoder_activation_function,
        )
        bbox_embed = []
        for layer_idx in range(num_predictions):
            if getattr(config, "share_bbox_head", False) and layer_idx <= decoder_eval_idx:
                bbox_embed.append(decoder_bbox_head)
            else:
                bbox_embed.append(copy.deepcopy(decoder_bbox_head))

        self.model.decoder.class_embed = nn.ModuleList(class_embed)
        self.model.decoder.bbox_embed = nn.ModuleList(bbox_embed)

        self.post_init()


class Deimv2ImageProcessorKwargs(RTDetrImageProcessorKwargs):
    pass


class Deimv2ImageProcessor(RTDetrImageProcessor):
    pass


class Deimv2ImageProcessorFast(RTDetrImageProcessorFast):
    pass


__all__ = [
    "Deimv2Config",
    "Deimv2ForObjectDetection",
    "Deimv2Model",
    "Deimv2PreTrainedModel",
    "Deimv2ImageProcessor",
    "Deimv2ImageProcessorFast",
]
