# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import torch
import torch.nn as nn

from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ..auto import AutoConfig
from ..dinov2_with_registers.configuration_dinov2_with_registers import Dinov2WithRegistersConfig
from ..dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersBackbone,
    Dinov2WithRegistersEmbeddings,
    Dinov2WithRegistersLayer,
)
from ..lw_detr.configuration_lw_detr import LwDetrConfig
from ..lw_detr.modeling_lw_detr import (
    LwDetrConvNormLayer,
    LwDetrDecoder,
    LwDetrForObjectDetection,
    LwDetrLayerNorm,
    LwDetrMLPPredictionHead,
    LwDetrModel,
    LwDetrMultiScaleProjector,
    LwDetrPreTrainedModel,
)


class RfDetrWindowedDinov2Config(Dinov2WithRegistersConfig):
    model_type = "rf_detr_windowed_dinov2"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        use_swiglu_ffn=False,
        num_register_tokens=4,
        out_features=None,
        out_indices=None,
        apply_layernorm=True,
        reshape_hidden_states=True,
        num_windows: int = 2,
        window_block_indexes: list[int] | None = None,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            qkv_bias=qkv_bias,
            layerscale_value=layerscale_value,
            drop_path_rate=drop_path_rate,
            use_swiglu_ffn=use_swiglu_ffn,
            num_register_tokens=num_register_tokens,
            out_features=out_features,
            out_indices=out_indices,
            apply_layernorm=apply_layernorm,
            reshape_hidden_states=reshape_hidden_states,
            **kwargs,
        )
        self.num_windows = num_windows
        self.window_block_indexes = (
            list(range(self.num_hidden_layers)) if window_block_indexes is None else window_block_indexes
        )
        self.gradient_checkpointing = gradient_checkpointing


class RfDetrWindowedDinov2Embeddings(Dinov2WithRegistersEmbeddings):
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        if self.config.num_windows > 1:
            num_h_patches = height // self.config.patch_size
            num_w_patches = width // self.config.patch_size
            cls_token_with_pos_embed = embeddings[:, :1]
            pixel_tokens_with_pos_embed = embeddings[:, 1:]
            pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(
                batch_size, num_h_patches, num_w_patches, -1
            )

            num_w_patches_per_window = num_w_patches // self.config.num_windows
            num_h_patches_per_window = num_h_patches // self.config.num_windows
            num_windows = self.config.num_windows

            windowed_pixel_tokens = pixel_tokens_with_pos_embed.reshape(
                batch_size * num_windows,
                num_h_patches_per_window,
                num_windows,
                num_w_patches_per_window,
                -1,
            )
            windowed_pixel_tokens = windowed_pixel_tokens.permute(0, 2, 1, 3, 4)
            windowed_pixel_tokens = windowed_pixel_tokens.reshape(
                batch_size * num_windows**2,
                num_h_patches_per_window * num_w_patches_per_window,
                -1,
            )
            windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(num_windows**2, 1, 1)
            embeddings = torch.cat((windowed_cls_token_with_pos_embed, windowed_pixel_tokens), dim=1)

        if self.register_tokens is not None:
            embeddings = torch.cat(
                (embeddings[:, :1], self.register_tokens.expand(embeddings.shape[0], -1, -1), embeddings[:, 1:]), dim=1
            )

        embeddings = self.dropout(embeddings)
        return embeddings


class RfDetrWindowedDinov2Layer(Dinov2WithRegistersLayer):
    def __init__(self, config: RfDetrWindowedDinov2Config):
        super().__init__(config)
        self.num_windows = config.num_windows

    def forward(
        self,
        hidden_states: torch.Tensor,
        run_full_attention: bool = False,
    ) -> torch.Tensor:
        shortcut = hidden_states

        if run_full_attention and self.num_windows > 1:
            batch_size, hidden_state_length, channels = hidden_states.shape
            num_windows_squared = self.num_windows**2
            hidden_states = hidden_states.view(
                batch_size // num_windows_squared, num_windows_squared * hidden_state_length, channels
            )

        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output = self.attention(hidden_states_norm)

        if run_full_attention and self.num_windows > 1:
            batch_size, hidden_state_length, channels = hidden_states.shape
            num_windows_squared = self.num_windows**2
            self_attention_output = self_attention_output.view(
                batch_size * num_windows_squared,
                hidden_state_length // num_windows_squared,
                channels,
            )

        self_attention_output = self.layer_scale1(self_attention_output)
        hidden_states = self.drop_path(self_attention_output) + shortcut

        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)
        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output


class RfDetrWindowedDinov2Encoder(nn.Module):
    def __init__(self, config: RfDetrWindowedDinov2Config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RfDetrWindowedDinov2Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, output_hidden_states: bool = False) -> BaseModelOutput:
        all_hidden_states = [hidden_states] if output_hidden_states else None

        for layer_idx, layer_module in enumerate(self.layer):
            run_full_attention = layer_idx not in self.config.window_block_indexes
            hidden_states = layer_module(hidden_states, run_full_attention=run_full_attention)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )


class RfDetrWindowedDinov2Backbone(Dinov2WithRegistersBackbone):
    config_class = RfDetrWindowedDinov2Config

    def __init__(self, config: RfDetrWindowedDinov2Config):
        super().__init__(config)
        self.embeddings = RfDetrWindowedDinov2Embeddings(config)
        self.encoder = RfDetrWindowedDinov2Encoder(config)
        self.num_register_tokens = config.num_register_tokens
        self.post_init()

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool | None = None, **kwargs
    ) -> BackboneOutput:
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        embedding_output = self.embeddings(pixel_values)
        outputs = self.encoder(embedding_output, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)

                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1 + self.num_register_tokens :]
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size

                    num_h_patches = height // patch_size
                    num_w_patches = width // patch_size

                    if self.config.num_windows > 1:
                        num_windows_squared = self.config.num_windows**2
                        hidden_batch_size, hidden_state_length, channels = hidden_state.shape
                        num_h_patches_per_window = num_h_patches // self.config.num_windows
                        num_w_patches_per_window = num_w_patches // self.config.num_windows

                        hidden_state = hidden_state.reshape(
                            hidden_batch_size // num_windows_squared,
                            num_windows_squared * hidden_state_length,
                            channels,
                        )
                        hidden_state = hidden_state.reshape(
                            (hidden_batch_size // num_windows_squared) * self.config.num_windows,
                            self.config.num_windows,
                            num_h_patches_per_window,
                            num_w_patches_per_window,
                            channels,
                        )
                        hidden_state = hidden_state.permute(0, 2, 1, 3, 4)

                    hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                feature_maps += (hidden_state,)

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )


class RfDetrConfig(LwDetrConfig):
    model_type = "rf_detr"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config: PreTrainedConfig | dict | None = None,
        projector_scale_factors: list[float] | None = None,
        hidden_expansion=0.5,
        c2f_num_blocks=3,
        activation_function="silu",
        batch_norm_eps=1e-5,
        d_model=256,
        dropout: float = 0.0,
        decoder_ffn_dim=2048,
        decoder_n_points: int = 2,
        decoder_layers: int = 3,
        decoder_self_attention_heads: int = 8,
        decoder_cross_attention_heads: int = 16,
        decoder_activation_function="relu",
        num_queries: int = 300,
        attention_bias=True,
        attention_dropout=0.0,
        activation_dropout=0.0,
        group_detr: int = 13,
        init_std=0.02,
        disable_custom_kernels=True,
        class_cost=2,
        bbox_cost=5,
        giou_cost=2,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        focal_alpha=0.25,
        auxiliary_loss=True,
        **kwargs,
    ):
        if backbone_config is None:
            backbone_config = RfDetrWindowedDinov2Config(
                image_size=512,
                patch_size=16,
                hidden_size=384,
                num_hidden_layers=12,
                num_attention_heads=6,
                mlp_ratio=4,
                out_indices=[3, 6, 9, 12],
                num_register_tokens=0,
                num_windows=2,
                window_block_indexes=[0, 1, 2, 4, 5, 7, 8, 10, 11],
            )
        elif isinstance(backbone_config, dict):
            backbone_config = dict(backbone_config)
            model_type = backbone_config.pop("model_type", "rf_detr_windowed_dinov2")
            if model_type == "rf_detr_windowed_dinov2":
                backbone_config = RfDetrWindowedDinov2Config(**backbone_config)
            else:
                backbone_config = AutoConfig.for_model(model_type, **backbone_config)

        if projector_scale_factors is None:
            projector_scale_factors = [1.0]
        if "num_labels" not in kwargs:
            kwargs["num_labels"] = 91

        super().__init__(
            backbone_config=backbone_config,
            projector_scale_factors=projector_scale_factors,
            hidden_expansion=hidden_expansion,
            c2f_num_blocks=c2f_num_blocks,
            activation_function=activation_function,
            batch_norm_eps=batch_norm_eps,
            d_model=d_model,
            dropout=dropout,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_n_points=decoder_n_points,
            decoder_layers=decoder_layers,
            decoder_self_attention_heads=decoder_self_attention_heads,
            decoder_cross_attention_heads=decoder_cross_attention_heads,
            decoder_activation_function=decoder_activation_function,
            num_queries=num_queries,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            group_detr=group_detr,
            init_std=init_std,
            disable_custom_kernels=disable_custom_kernels,
            class_cost=class_cost,
            bbox_cost=bbox_cost,
            giou_cost=giou_cost,
            mask_loss_coefficient=mask_loss_coefficient,
            dice_loss_coefficient=dice_loss_coefficient,
            bbox_loss_coefficient=bbox_loss_coefficient,
            giou_loss_coefficient=giou_loss_coefficient,
            eos_coefficient=eos_coefficient,
            focal_alpha=focal_alpha,
            auxiliary_loss=auxiliary_loss,
            **kwargs,
        )


class RfDetrPreTrainedModel(LwDetrPreTrainedModel):
    pass


class RfDetrDecoder(LwDetrDecoder):
    pass


class RfDetrMultiScaleProjector(LwDetrMultiScaleProjector):
    pass


class RfDetrLayerNorm(LwDetrLayerNorm):
    pass


class RfDetrConvNormLayer(LwDetrConvNormLayer):
    pass


class RfDetrMLPPredictionHead(LwDetrMLPPredictionHead):
    pass


class RfDetrConvEncoder(nn.Module):
    def __init__(self, config: RfDetrConfig):
        super().__init__()
        self.backbone = RfDetrWindowedDinov2Backbone(config.backbone_config)
        self.projector = RfDetrMultiScaleProjector(config)
        self._replace_projector_norms(self.projector)

    def _replace_projector_norms(self, module: nn.Module):
        for child in module.children():
            if isinstance(child, RfDetrConvNormLayer):
                child.norm = RfDetrLayerNorm(child.conv.out_channels, data_format="channels_first")
            else:
                self._replace_projector_norms(child)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        features = self.backbone(pixel_values).feature_maps
        features = self.projector(features)
        out = []
        for feature_map in features:
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


class RfDetrModel(LwDetrModel):
    config_class = RfDetrConfig

    def __init__(self, config: RfDetrConfig):
        RfDetrPreTrainedModel.__init__(self, config)

        self.backbone = RfDetrConvEncoder(config)

        self.group_detr = config.group_detr
        self.num_queries = config.num_queries
        hidden_dim = config.d_model
        self.reference_point_embed = nn.Embedding(self.num_queries * self.group_detr, 4)
        self.query_feat = nn.Embedding(self.num_queries * self.group_detr, hidden_dim)

        self.decoder = RfDetrDecoder(config)

        self.enc_output = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.group_detr)])
        self.enc_output_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.group_detr)])
        self.enc_out_bbox_embed = nn.ModuleList(
            [RfDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3) for _ in range(self.group_detr)]
        )
        self.enc_out_class_embed = nn.ModuleList(
            [nn.Linear(config.d_model, config.num_labels) for _ in range(self.group_detr)]
        )

        self.post_init()


class RfDetrForObjectDetection(LwDetrForObjectDetection):
    config_class = RfDetrConfig

    def __init__(self, config: RfDetrConfig):
        RfDetrPreTrainedModel.__init__(self, config)

        self.model = RfDetrModel(config)
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = RfDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)

        self.post_init()


__all__ = [
    "RfDetrConfig",
    "RfDetrForObjectDetection",
    "RfDetrModel",
    "RfDetrWindowedDinov2Backbone",
    "RfDetrWindowedDinov2Config",
]
