# coding=utf-8
# Copyright 2023 THUDM and The HuggingFace Team. All rights reserved.
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
""" PyTorch CogVLM model."""

import math
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils.logging import get_logger

from .configuration_cogvlm import CogVLMConfig, CogVLMVisionConfig


if TYPE_CHECKING:
    from transformers.utils import ModelOutput

logger = get_logger(__name__)

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


COGVLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/cogvlm-chat-hf",
    # See all CogVLM models at https://huggingface.co/models?filter=cogvlm
]


class CogVLMPatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def forward(self, images: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class CogVLMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim**-0.5
        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)  # 3, B, L, H, D
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = xops.memory_efficient_attention(
            q,
            k,
            v,
            scale=self.scale,
        )
        output = self.dense(out.view(B, L, -1))
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn_weights = attn_weights.softmax(dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class CogVLMVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class CogVLMTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = CogVLMAttention(config)
        self.mlp = CogVLMVisionMLP(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class CogVLMTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([CogVLMTransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class CogVLMGLU(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class CogVLMVisionModel(nn.Module):
    def __init__(self, config: CogVLMVisionConfig):
        super().__init__()

        self.patch_embedding = CogVLMPatchEmbedding(config)
        self.transformer = CogVLMTransformer(config)
        self.linear_proj = CogVLMGLU(config, in_features=config.hidden_size)
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, images: torch.FloatTensor) -> torch.FloatTensor:
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        return x


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class CogVLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class CogVLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def get_expert_mask(token_type_ids: torch.LongTensor) -> (torch.BoolTensor, torch.BoolTensor):
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1] == VISION_TOKEN_TYPE) & (
        token_type_ids[:, 1:] == VISION_TOKEN_TYPE
    )
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


class CogVLMVisionExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_mlp = CogVLMMLP(config)
        self.vision_mlp = CogVLMMLP(config)

    def forward(self, hidden_states: torch.FloatTensor, token_type_ids: torch.LongTensor):
        output = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)
        output[vision_token_mask] = self.vision_mlp(hidden_states[vision_token_mask])
        output[language_token_mask] = self.language_mlp(hidden_states[language_token_mask])
        return output


def attention_fn(
    query_layer: torch.FloatTensor,
    key_layer: torch.FloatTensor,
    value_layer: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    *,
    scaling_attention_score: bool = True,
    attention_dropout: nn.Module = None,
):
    attention_mask_bool = attention_mask == 0
    is_low_triangle = (attention_mask_bool == torch.ones_like(attention_mask_bool, dtype=torch.float).tril()).all()
    is_full = (attention_mask_bool > 0).all()
    if not (int(torch.__version__.split(".")[0]) >= 2):
        warnings.warn("It's recommended to use torch2.0 or higher.")
    if int(torch.__version__.split(".")[0]) >= 2 and scaling_attention_score and (is_full or is_low_triangle):
        dropout_p = 0.0 if attention_dropout is None or not attention_dropout.training else attention_dropout.p
        return torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attn_mask=None, dropout_p=dropout_p, is_causal=not is_full
        )
    else:
        if scaling_attention_score:
            query_layer = query_layer / math.sqrt(query_layer.shape[-1])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask
        attention_scores = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        if attention_dropout is not None:
            attention_scores = attention_dropout(attention_scores)
        context_layer = torch.matmul(attention_scores, value_layer)
        return context_layer


class CogVLMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class CogVLMVisionExpertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.rotary_emb = CogVLMRotaryEmbedding(self.hidden_size // self.num_heads)
        # from .util import FastRotaryEmbedding
        # self.rotary_emb = FastRotaryEmbedding(dim=self.head_dim, pos_idx_in_fp32=False)
        self.vision_expert_query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.vision_expert_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.language_expert_query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.language_expert_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [B, L, H*HD] into a 4D tensor with size [B H L HD]."""
        new_tensor_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)

        shape = list(hidden_states.shape)
        shape[-1] = shape[-1] * 3
        mixed_raw_layer = torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        mixed_raw_layer[vision_token_mask] = self.vision_expert_query_key_value(hidden_states[vision_token_mask])
        mixed_raw_layer[language_token_mask] = self.language_expert_query_key_value(hidden_states[language_token_mask])

        query_states, key_states, value_states = torch.split(mixed_raw_layer, self.hidden_size, dim=-1)
        query_states = self._transpose_for_scores(query_states)  # B, H, L, HD
        key_states = self._transpose_for_scores(key_states)  # B, H, L, HD
        value_states = self._transpose_for_scores(value_states)  # B, H, L, HD

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        query_states, key_states = self.rotary_emb(
            query_states, key_states, position_ids=position_ids, max_seqlen=position_ids.max() + 1
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        context_layer = attention_fn(
            query_layer=query_states,
            key_layer=key_states,
            value_layer=value_states,
            attention_mask=attention_mask,
            scaling_attention_score=True,
            attention_dropout=None,
        )
        if context_layer.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {context_layer.size()}"
            )
        context_layer = context_layer.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)

        attn_output = torch.empty(context_layer.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        attn_output[vision_token_mask] = self.vision_expert_dense(context_layer[vision_token_mask])
        attn_output[language_token_mask] = self.language_expert_dense(context_layer[language_token_mask])

        if output_attentions:
            warnings.warn("output_attentions is not implemented.")

        return attn_output, None, past_key_value


class CogVLMDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = CogVLMVisionExpertAttention(config=config)
        self.mlp = CogVLMVisionExpertMLP(config)
        self.input_layernorm = CogVLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CogVLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_type_ids=token_type_ids)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs  # type: ignore


class CogVLMPreTrainedModel(PreTrainedModel):
    config_class = CogVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["CogVLMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


def is_empty(images_list: Optional[List[List[torch.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if len(image_list):
            return False
    return True


def build_position_ids(x: torch.BoolTensor, attention_mask: Optional[torch.BoolTensor] = None) -> torch.LongTensor:
    if attention_mask is not None:
        tmp = x.clone()
        tmp[~(attention_mask.bool())] = -1
    else:
        tmp = x.clone()
    # image boi eoi token as LANGUAGE_TOKEN_TYPE
    is_boi_eoi = torch.zeros_like(x, dtype=torch.bool)
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= tmp[:, 0] == VISION_TOKEN_TYPE
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= tmp[:, -1] == VISION_TOKEN_TYPE
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
    # final position ids
    y = torch.zeros_like(x, dtype=torch.long)
    y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | (
        (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    )
    y = y.cumsum(dim=-1)
    return y


class CogVLMModel(CogVLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([CogVLMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = CogVLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.vision = CogVLMVisionModel(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def encode_images(self, images: List[List[torch.Tensor]]) -> torch.Tensor:
        images_list, images = images, []

        images = []
        for image_list in images_list:
            for image in image_list:
                images.append(image)

        images = torch.stack(images)
        images_features = self.vision(images)
        return images_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: List[List[torch.Tensor]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, token_type_ids, position_ids and (attention_mask = None is fine)"""

        if past_key_values is not None:
            pass  # generate mode with past_key_values. the image features are already mapped
        else:
            # not allow for inputs_embeds, because we want to process image feature
            assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"
            if not is_empty(images):  # multi-modality
                assert token_type_ids is not None, "multi-modality requires `token_type_ids`!"
                assert len(input_ids) == len(images), f"{len(input_ids)} {len(images)}"
                inputs_embeds = self.embed_tokens(input_ids)
                images_features = self.encode_images(images)
                images_features = rearrange(images_features, "b n d -> (b n) d")
                images_features = images_features.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                inputs_embeds = inputs_embeds.index_put([token_type_ids == VISION_TOKEN_TYPE], images_features)
            else:  # single-modality
                if token_type_ids is None:
                    token_type_ids = (
                        torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device) * LANGUAGE_TOKEN_TYPE
                    )
                assert not (
                    token_type_ids == VISION_TOKEN_TYPE
                ).any(), f"{(token_type_ids == VISION_TOKEN_TYPE).sum()}"
                inputs_embeds = self.embed_tokens(input_ids)

            if position_ids is None:
                position_ids = build_position_ids(token_type_ids, attention_mask)
            input_ids = None

        return self.llm_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def llm_forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """largely copy from llama forward and adapt for cogvlm with `token_type_ids`"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


class CogVLMForCausalLM(CogVLMPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config):
        super().__init__(config)
        self.model = CogVLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: List[List[torch.Tensor]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            images=images,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)  # type: ignore

    def prepare_inputs_for_generation(
        self,
        input_ids,
        token_type_ids,
        images=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # build position_ids if needed
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            position_ids = build_position_ids(token_type_ids, attention_mask)

        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "token_type_ids": token_type_ids,
                "images": images,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: "ModelOutput",
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            new_token_type_ids = (
                torch.ones(size=(token_type_ids.shape[0], 1), dtype=token_type_ids.dtype, device=token_type_ids.device)
                * LANGUAGE_TOKEN_TYPE
            )
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, new_token_type_ids], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
