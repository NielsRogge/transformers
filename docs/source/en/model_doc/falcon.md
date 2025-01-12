<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Falcon

## Overview

Falcon is a class of causal decoder-only models built by [TII](https://www.tii.ae/). The largest Falcon checkpoints
have been trained on >=1T tokens of text, with a particular emphasis on the [RefinedWeb](https://arxiv.org/abs/2306.01116)
corpus. They are made available under the Apache 2.0 license.


Falcon's architecture is modern and optimized for inference, with multi-query attention and support for efficient
attention variants like `FlashAttention`. Both 'base' models trained only as causal language models as well as
'instruct' models that have received further fine-tuning are available.


Falcon models are (as of 2023) some of the largest and most powerful open-source language models,
and consistently rank highly in the [OpenLLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

## Converting custom checkpoints 

<Tip>

Falcon models were initially added to the Hugging Face Hub as custom code checkpoints. However, Falcon is now fully
supported in the Transformers library. If you fine-tuned a model from a custom code checkpoint, we recommend converting
your checkpoint to the new in-library format, as this should give significant improvements to stability and
performance, especially for generation, as well as removing the need to use `trust_remote_code=True`!

</Tip>

You can convert custom code checkpoints to full Transformers checkpoints using the `convert_custom_code_checkpoint.py` 
script located in the
[Falcon model directory](https://github.com/huggingface/transformers/tree/main/src/transformers/models/falcon)
of the Transformers library. To use this script, simply call it with 
`python convert_custom_code_checkpoint.py --checkpoint_dir my_model`. This will convert your checkpoint in-place, and
you can immediately load it from the directory afterwards with e.g. `from_pretrained()`. If your model hasn't been
uploaded to the Hub, we recommend making a backup before attempting the conversion, just in case!


## FalconConfig


    This is the configuration class to store the configuration of a [`FalconModel`]. It is used to instantiate a Falcon
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 65024):
            Vocabulary size of the Falcon model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FalconModel`]
        hidden_size (`int`, *optional*, defaults to 4544):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 71):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_ln_in_parallel_attn (`int`, *optional*):
            Set to 2 if separate layer norms are to be used for the MLP and the attention output when using parallel
            attention, otherwise, 1.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models). Only relevant if
            `config.is_decoder=True`.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for MLP layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for attention layers.
        num_kv_heads (`int`, *optional*):
            Number of key-value heads to use per attention layer. If unset, defaults to the same value as
            `num_attention_heads`.
        alibi (`bool`, *optional*, defaults to `False`):
            Whether to use ALiBi positional biases during self-attention.
        new_decoder_architecture (`bool`, *optional*, defaults to `False`):
            Whether to use the new (Falcon-40B) decoder architecture. If `True`, the `multi_query` and `parallel_attn`
            arguments are ignored, as the new decoder always uses parallel attention.
        multi_query (`bool`, *optional*, defaults to `True`):
            Whether to use multi-query attention in the decoder. Ignored when `new_decoder_architecture` is `True`.
        parallel_attn (`bool`, *optional*, defaults to `True`):
            Whether to compute attention in parallel with the feedforward layer. If False, they are consecutive
            instead, as in the original Transformer architecture. Ignored when `new_decoder_architecture` is `True`.
        bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias on Linear layers.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with, when `alibi` is `False`. Pretrained
            Falcon models with RoPE support up to 2048 tokens.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        bos_token_id (`int`, *optional*, defaults to 11):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 11):
            The id of the "end-of-sequence" token.
        ffn_hidden_size (`int`, *optional*):
            The hidden size of the feedforward layer in the Transformer decoder.
            defaults to 4x hidden dim
        activation (`str`, *optional*, defaults to `"gelu"`):
            The activation function used in the feedforward layer.

    Example:

    ```python
    >>> from transformers import FalconModel, FalconConfig

    >>> # Initializing a small (2-layer) Falcon configuration
    >>> configuration = FalconConfig(num_hidden_layers=2)

    >>> # Initializing a model from the small configuration
    >>> model = FalconModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

Methods: all

## FalconModel

The bare Falcon Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FalconConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FalconForCausalLM

The Falcon Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FalconConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FalconForSequenceClassification


    The Falcon Model transformer with a sequence classification head on top (linear layer).

    [`FalconForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FalconConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FalconForTokenClassification


    Falcon Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FalconConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FalconForQuestionAnswering


    The Falcon Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FalconConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward


