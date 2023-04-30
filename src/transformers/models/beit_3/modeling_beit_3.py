import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

from transformers import PreTrainedModel
from transformers.activations import get_activation
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    ImageClassifierOutputWithNoAttention,
    SequenceClassifierOutput,
)

# from .configuration_beit_3 import Beit3Config
from transformers.models.beit_3.configuration_beit_3 import Beit3Config
from transformers.utils import ModelOutput, logging


# from ... import PreTrainedModel
# from ...utils import logging

EVAL_CAPACITY_TOKEN_FRACTION = 0.25
SAMPLE_FRACTION = 0.2
logger = logging.get_logger(__name__)


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class TwoLayerMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        norm_layer,
        norm_input=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Beit3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Beit3Config
    base_model_prefix = "beit3"
    main_input_name = "input_ids"

    # def __init__(self,config):
    #     super(Beit3PreTrainedModel, self).__init__(config)
    #     self._set_gradient_checkpointing(config.is_gradient_checkpointing)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(
            module,
            (
                BEiT3ForVisualReasoning,
                BEiT3ForImageTextRetrieval,
                BEiT3ForVisualQuestionAnswering,
                BEiT3ForImageClassification,
                BEiT3ForCaptioning,
            ),
        ):
            module.beit3.text_embed.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # module.vision_embed.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Encoder):
            module.gradient_checkpointing = value

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, BeitEncoder):
    #         module.gradient_checkpointing = value


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1


class VisionEmbedding(Beit3PreTrainedModel):
    """Image to Patch Embedding"""

    def __init__(self, config):
        super().__init__(config)
        img_size = (config.img_size, config.img_size)
        patch_size = (config.patch_size, config.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(config.in_chans, config.embed_dim, kernel_size=patch_size, stride=patch_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

    def num_position_embeddings(self):
        if self.cls_token is None:
            return self.num_patches
        else:
            return self.num_patches + 1

    def forward(self, x, masked_position=None):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        batch_size, seq_len, _ = x.size()

        if masked_position is not None:
            assert self.mask_token is not None
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            w = masked_position.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class TextEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        self._fill_padding_idx_with_zero()


class PositionalEmbedding(nn.Embedding):
    def forward(
        self,
        x,
        positions=None,
    ):
        if positions is None:
            # being consistent with Fairseq, which starts from 2.
            positions = torch.arange(2, x.size(1) + 2, device=x.device).long().unsqueeze(0)
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class FeedForwardNetwork(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.embed_dim
        self.activation_fn = get_activation(config.activation_fn)
        self.activation_dropout_module = torch.nn.Dropout(config.activation_dropout)
        self.dropout_module = torch.nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(self.embed_dim, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, self.embed_dim)
        self.ffn_layernorm = LayerNorm(config.hidden_size, eps=config.layernorm_eps) if config.subln else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x


class MultiheadAttention(Beit3PreTrainedModel):
    def __init__(
        self,
        config,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
    ):
        super().__init__(config)
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayNetwork(nn.Linear(self.embed_dim, self.embed_dim, bias=True))
        self.v_proj = MultiwayNetwork(nn.Linear(self.embed_dim, self.embed_dim, bias=True))
        self.q_proj = MultiwayNetwork(nn.Linear(self.embed_dim, self.embed_dim, bias=True))
        self.out_proj = MultiwayNetwork(nn.Linear(self.embed_dim, self.embed_dim, bias=True))
        self.inner_attn_ln = (
            MultiwayNetwork(LayerNorm(self.embed_dim, eps=config.layernorm_eps))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(config.attention_dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
    ):
        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(bsz * self.num_heads, -1, self.head_dim)
                prev_value = incremental_state["prev_value"].view(bsz * self.num_heads, -1, self.head_dim)
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            incremental_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if rel_pos is not None:
            rel_pos = rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + rel_pos

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)

        return attn, attn_weights


class EncoderLayer(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.embed_dim
        self.self_attn = MultiheadAttention(
            config,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=config.subln,
        )
        self.self_attn_layer_norm = MultiwayNetwork(LayerNorm(self.embed_dim, eps=config.layernorm_eps))
        self.dropout_module = torch.nn.Dropout(config.dropout)

        self.normalize_before = config.normalize_before
        self.ffn_dim = config.hidden_size

        self.ffn = MultiwayNetwork(
            FeedForwardNetwork(config),
        )
        self.final_layer_norm = MultiwayNetwork(LayerNorm(self.embed_dim, eps=config.layernorm_eps))
        self.alpha = 1.0

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        x,
        encoder_padding_mask,
        attn_mask=None,
        rel_pos=None,
        multiway_split_position=None,
        incremental_state=None,
    ):
        if multiway_split_position is not None:
            # assert self.args.multiway
            self.apply(set_split_position(multiway_split_position))

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            rel_pos=rel_pos,
            incremental_state=incremental_state,
        )
        x = self.dropout_module(x)

        x = self.residual_connection(x, residual)

        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


def init_bert_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        if isinstance(module.q_proj, MultiwayNetwork):
            normal_(module.q_proj.A.weight.data)
            normal_(module.q_proj.B.weight.data)
            normal_(module.k_proj.A.weight.data)
            normal_(module.k_proj.B.weight.data)
            normal_(module.v_proj.A.weight.data)
            normal_(module.v_proj.B.weight.data)
        else:
            normal_(module.q_proj.weight.data)
            normal_(module.k_proj.weight.data)
            normal_(module.v_proj.weight.data)


class Encoder(nn.Module):
    def __init__(
        self,
        config,
        embed_positions=None,
    ):
        # super().__init__(config)
        super().__init__()

        self.dropout_module = torch.nn.Dropout(config.dropout)

        embed_dim = config.embed_dim
        self.embed_positions = embed_positions

        self.layers = nn.ModuleList([])

        for i in range(config.layers):
            self.layers.append(EncoderLayer(config))
        self.num_layers = len(self.layers)
        self.layer_norm = MultiwayNetwork(LayerNorm(embed_dim, eps=config.layernorm_eps))

        self.relative_position = None

        if config.subln:
            init_scale = math.sqrt(math.log(config.layers * 2))
            for name, p in self.named_parameters():
                if "fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name:
                    p.data.mul_(init_scale)

    def forward_embedding(
        self,
        src_tokens,
        token_embedding=None,
        positions=None,
    ):
        x = embed = token_embedding
        if self.embed_positions is not None:
            if src_tokens is not None:
                x = embed + self.embed_positions(src_tokens, positions=positions)
            else:
                x = embed + self.embed_positions(x, positions=positions)
        x = self.dropout_module(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        encoder_padding_mask=None,
        attn_mask=None,
        return_all_hiddens=True,
        token_embeddings=None,
        multiway_split_position=None,
        incremental_state=None,
        positions=None,
    ):
        assert src_tokens is not None or token_embeddings is not None
        if encoder_padding_mask is None:
            if src_tokens is not None:
                encoder_padding_mask = torch.zeros_like(src_tokens, device=src_tokens.device).bool()
            else:
                encoder_padding_mask = torch.zeros(
                    [token_embeddings.size(0), token_embeddings.size(1)],
                    device=token_embeddings.device,
                ).bool()

        if multiway_split_position is not None:
            self.apply(set_split_position(multiway_split_position))

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings, positions)
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        rel_pos_bias = None
        if self.relative_position is not None:
            rel_pos_bias = self.relative_position(batch_size=x.size(0), qlen=x.size(1), klen=x.size(1))

        # incremental_state is not None during inference if we use the bidirectional encoder as a generator as in s2s-ft (https://arxiv.org/abs/2110.13640)
        for idx, layer in enumerate(self.layers):
            x = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if incremental_state is None else None,
                attn_mask=attn_mask,
                rel_pos=rel_pos_bias,
                multiway_split_position=multiway_split_position,
                incremental_state=incremental_state[idx] if incremental_state is not None else None,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": x,
            "encoder_embedding": encoder_embedding,
            "encoder_padding_mask": encoder_padding_mask,
            "encoder_states": encoder_states,
        }


class BEiT3Model(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.args = args
        # assert args.multiway
        # assert args.vocab_size > 0
        # assert not args.share_encoder_input_output_embed
        self.text_embed = TextEmbedding(config.vocab_size, config.embed_dim)
        self.vision_embed = VisionEmbedding(config)
        # being consistent with Fairseq, which starts from 2 for position embedding
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(self.vision_embed.num_position_embeddings() + 2, config.embed_dim),
                PositionalEmbedding(config.max_source_positions, config.embed_dim),
            ],
            dim=1,
        )
        self.encoder = Encoder(
            config,
            embed_positions=embed_positions,
        )
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.text_embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.text_embed = value

    def get_num_layers(self):
        return self.encoder.num_layers

    def forward(
        self,
        textual_tokens=None,
        pixel_values=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
    ):
        assert textual_tokens is not None or pixel_values is not None

        if textual_tokens is None:
            x = self.vision_embed(pixel_values, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif pixel_values is None:
            x = self.text_embed(textual_tokens)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        else:
            x1 = self.vision_embed(pixel_values, vision_masked_position)
            multiway_split_position = x1.size(1)
            x2 = self.text_embed(textual_tokens)
            x = torch.cat([x1, x2], dim=1)

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None
        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
            incremental_state=incremental_state,
            positions=positions,
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out


# class BEiT3Wrapper(nn.Module):
#     def __init__(self, args, **kwargs):
#         super().__init__()
#         self.args = args
#         self.beit3 = BEiT3Model(args)
#         self.apply(self._init_weights)

# def fix_init_weight(self):
#     def rescale(param, layer_id):
#         param.div_(math.sqrt(2.0 * layer_id))
#
#     for layer_id, layer in enumerate(self.blocks):
#         rescale(layer.attn.proj.weight.data, layer_id + 1)
#         rescale(layer.mlp.fc2.weight.data, layer_id + 1)


# @torch.jit.ignore
# def no_weight_decay(self):
#     return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}


class BEiT3ForVisualReasoning(Beit3PreTrainedModel):
    def __init__(self, config):
        super(BEiT3ForVisualReasoning, self).__init__(config)
        embed_dim = config.embed_dim
        self.beit3 = BEiT3Model(config)
        self.head = TwoLayerMLP(
            in_features=embed_dim * 4,
            hidden_features=embed_dim * 2,
            out_features=config.num_labels,
            norm_layer=nn.LayerNorm,
        )
        self.post_init()
        # init_scale = 0.001
        # self.head.apply(self._init_weights)
        # if isinstance(self.head.dense1, nn.Linear):
        #     self.head.dense1.weight.data.mul_(init_scale)
        #     self.head.dense1.bias.data.mul_(init_scale)
        #
        # if isinstance(self.head.dense2, nn.Linear):
        #     self.head.dense2.weight.data.mul_(init_scale)
        #     self.head.dense2.bias.data.mul_(init_scale)

    def forward(
        self,
        input_ids,
        pixel_values1,
        pixel_values2,
        padding_mask,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        bsz = input_ids.size()[0]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_input = torch.cat((pixel_values1, pixel_values2), dim=0)
        language_input = torch.cat((input_ids, input_ids), dim=0)
        padding_mask = torch.cat((padding_mask, padding_mask), dim=0)

        outputs = self.beit3(
            textual_tokens=language_input,
            pixel_values=vision_input,
            text_padding_position=padding_mask,
        )
        x = outputs["encoder_out"]
        multiway_split_position = outputs["multiway_split_position"]

        vision_cls = x[:, 0, :]
        language_cls = x[:, multiway_split_position, :]
        cls_rep = torch.cat((vision_cls, language_cls), dim=-1)
        a, b = torch.split(cls_rep, split_size_or_sections=[bsz, bsz], dim=0)
        cls_rep = torch.cat((a, b), dim=-1)

        logits = self.head(cls_rep)
        reshaped_logits = logits.contiguous()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))

        if not return_dict:
            output = (reshaped_logits,) + (outputs["encoder_states"],) if output_hidden_states else (reshaped_logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs["encoder_states"],
        )


class BEiT3ForImageClassification(Beit3PreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config):
        super(BEiT3ForImageClassification, self).__init__(config)
        embed_dim = config.embed_dim
        self.beit3 = BEiT3Model(config)
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.num_labels = config.num_labels
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Any], ImageClassifierOutputWithNoAttention]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.beit3(textual_tokens=None, pixel_values=pixel_values)
        encoder_out = encoder_outputs["encoder_out"]
        t = encoder_out[:, 1:, :]
        logits = self.classifier(self.fc_norm(t.mean(1)))
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            output = (output + (encoder_outputs["encoder_states"],)) if output_hidden_states else output
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs["encoder_states"],
        )


class BEiT3ForCaptioning(Beit3PreTrainedModel):
    def __init__(self, config):
        super(BEiT3ForCaptioning, self).__init__(config)
        embed_dim = config.embed_dim
        self.beit3 = BEiT3Model(config)
        self.label_smoothing = config.label_smoothing
        self.output = nn.Linear(embed_dim, config.vocab_size)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction="none")
        self.post_init()

    def forward(
        self,
        input_ids,
        pixel_values,
        padding_mask,
        language_masked_pos,
        text_len=None,
        incremental_state=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_len = text_len if text_len is not None else input_ids.size(1)
        image_len = self.beit3.vision_embed.num_position_embeddings()
        max_len = text_len + image_len
        uni_mask = torch.zeros((max_len, max_len), dtype=torch.long, device=input_ids.device)
        i_start, i_end = 0, image_len
        t_start, t_end = image_len, max_len
        # triangle mask for caption to caption
        uni_mask[t_start:t_end, t_start:t_end] = torch.tril(
            torch.ones(text_len, text_len, dtype=torch.long, device=input_ids.device)
        )
        # full attention for caption to image
        uni_mask[t_start:t_end, i_start:i_end] = 1
        # full attention for image to image
        uni_mask[i_start:i_end, i_start:i_end] = 1
        uni_mask = 1 - uni_mask

        if incremental_state is not None:
            for idx in range(self.get_num_layers()):
                if idx not in incremental_state:
                    incremental_state[idx] = {}

        # for incremental decoding
        positions = None
        if pixel_values is None:
            uni_mask = uni_mask[-2:]
            padding_mask = None
            # start position (2 (fairseq starts at 2) + cur_position) is equal to text_len
            positions = (
                torch.arange(text_len, input_ids.size(1) + text_len, device=input_ids.device).long().unsqueeze(0)
            )

        outputs = self.beit3(
            textual_tokens=input_ids,
            pixel_values=pixel_values,
            text_padding_position=padding_mask,
            attn_mask=uni_mask,
            incremental_state=incremental_state,
            positions=positions,
        )
        if pixel_values is not None:
            text_feats = outputs["encoder_out"][:, image_len:]
        else:
            text_feats = outputs["encoder_out"]

        if language_masked_pos is not None:
            text_feats = text_feats[language_masked_pos.bool()]

        logits = self.output(text_feats)

        loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            eps = self.label_smoothing
            n_class = logits.size(1)
            one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = self.log_soft(logits)
            loss = self.kl(log_prb, one_hot).sum(1)

        if not return_dict:
            output = (logits,)
            output = output + (outputs["encoder_states"],) if output_hidden_states else output
            return ((loss.mean(),) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss.mean(),
            logits=logits,
            hidden_states=outputs["encoder_states"],
        )


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BEiT3ForVisualQuestionAnswering(Beit3PreTrainedModel):
    def __init__(self, config):
        super(BEiT3ForVisualQuestionAnswering, self).__init__(config)
        embed_dim = config.embed_dim
        self.num_labels = config.num_labels
        self.beit3 = BEiT3Model(config)
        self.pooler = Pooler(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        self.pooler.apply(self._init_weights)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, config.num_labels),
        )
        # self.head.apply(self._init_weights)
        self.post_init()

    def forward(
        self,
        input_ids,
        pixel_values,
        padding_mask,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[Any], SequenceClassifierOutput]:
        encoder_outputs = self.beit3(
            textual_tokens=input_ids,
            pixel_values=pixel_values,
            text_padding_position=padding_mask,
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = encoder_outputs["encoder_out"]
        cls_rep = self.pooler(x)
        logits = self.classifier(cls_rep)
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            log_softmax = nn.LogSoftmax(dim=-1)
            reshaped_logits = log_softmax(reshaped_logits)
            loss = loss_fct(reshaped_logits, labels.contiguous())
        if not return_dict:
            output = (
                (reshaped_logits,) + (encoder_outputs["encoder_states"],)
                if output_hidden_states
                else (reshaped_logits,)
            )
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=encoder_outputs["encoder_states"],
        )


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class Biet3ImageTextMatchingModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        similarity (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
    """

    loss: Optional[torch.Tensor] = None
    text_hidden: Optional[torch.FloatTensor] = None
    image_hidden: Optional[torch.FloatTensor] = None


class BEiT3ForImageTextRetrieval(Beit3PreTrainedModel):
    def __init__(self, config):
        super(BEiT3ForImageTextRetrieval, self).__init__(config)
        embed_dim = config.embed_dim
        self.beit3 = BEiT3Model(config)
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.language_head.apply(self._init_weights)
        # self.vision_head.apply(self._init_weights)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        padding_mask=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Any], Biet3ImageTextMatchingModelOutput]:
        outputs = self.beit3(
            textual_tokens=None,
            pixel_values=pixel_values,
            text_padding_position=None,
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_out = outputs["encoder_out"]
        vision_cls = self.vision_head(vision_out[:, 0, :])
        vision_cls = F.normalize(vision_cls, dim=-1)

        outputs = self.beit3(
            textual_tokens=input_ids,
            pixel_values=None,
            text_padding_position=padding_mask,
        )
        text_out = outputs["encoder_out"]
        text_cls = self.language_head(text_out[:, 0, :])
        text_cls = F.normalize(text_cls, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(vision_cls, text_cls.t()) * logit_scale
        similarity = clip_loss(logits_per_text)

        if not return_dict:
            outputs = (similarity,)
            return (
                outputs
                + (
                    text_out,
                    vision_out,
                )
                if output_hidden_states
                else outputs
            )

        return Biet3ImageTextMatchingModelOutput(similarity, text_out, vision_out)
