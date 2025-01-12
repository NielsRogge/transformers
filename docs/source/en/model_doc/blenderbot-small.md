<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Blenderbot Small

Note that [`BlenderbotSmallModel`] and
[`BlenderbotSmallForConditionalGeneration`] are only used in combination with the checkpoint
[facebook/blenderbot-90M](https://huggingface.co/facebook/blenderbot-90M). Larger Blenderbot checkpoints should
instead be used with [`BlenderbotModel`] and
[`BlenderbotForConditionalGeneration`]

## Overview

The Blender chatbot model was proposed in [Recipes for building an open-domain chatbot](https://arxiv.org/pdf/2004.13637.pdf) Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.

The abstract of the paper is the following:

*Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that
scaling neural models in the number of parameters and the size of the data they are trained on gives improved results,
we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of
skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to
their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent
persona. We show that large scale models can learn these skills when given appropriate training data and choice of
generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models
and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn
dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing
failure cases of our models.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The authors' code can be
found [here](https://github.com/facebookresearch/ParlAI).

## Usage tips

Blenderbot Small is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than 
the left.


## Resources

- [Causal language modeling task guide](../tasks/language_modeling)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## BlenderbotSmallConfig


    This is the configuration class to store the configuration of a [`BlenderbotSmallModel`]. It is used to instantiate
    an BlenderbotSmall model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BlenderbotSmall
    [facebook/blenderbot_small-90M](https://huggingface.co/facebook/blenderbot_small-90M) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the BlenderbotSmall model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`BlenderbotSmallModel`] or [`TFBlenderbotSmallModel`].
        d_model (`int`, *optional*, defaults to 512):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 8):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 8):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models)
        forced_eos_token_id (`int`, *optional*, defaults to 2):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.

    Example:

    ```python
    >>> from transformers import BlenderbotSmallConfig, BlenderbotSmallModel

    >>> # Initializing a BlenderbotSmall facebook/blenderbot_small-90M style configuration
    >>> configuration = BlenderbotSmallConfig()

    >>> # Initializing a model (with random weights) from the facebook/blenderbot_small-90M style configuration
    >>> model = BlenderbotSmallModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## BlenderbotSmallTokenizer


    Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    the superclass for more information regarding methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        merges_file (`str`):
            Path to the merges file.
        bos_token (`str`, *optional*, defaults to `"__start__"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"__end__"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"__unk__"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"__null__"`):
            The token used for padding, for example when batching sequences of different lengths.
        kwargs (*optional*):
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    

Methods: build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## BlenderbotSmallTokenizerFast


    Construct a "fast" BlenderbotSmall tokenizer (backed by HuggingFace's *tokenizers* library).

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    

<frameworkcontent>
<pt>

## BlenderbotSmallModel

The bare BlenderbotSmall Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BlenderbotSmallConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## BlenderbotSmallForConditionalGeneration

The BlenderbotSmall Model with a language modeling head. Can be used for summarization.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BlenderbotSmallConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## BlenderbotSmallForCausalLM

No docstring available for BlenderbotSmallForCausalLM

Methods: forward

</pt>
<tf>

## TFBlenderbotSmallModel

No docstring available for TFBlenderbotSmallModel

Methods: call

## TFBlenderbotSmallForConditionalGeneration

No docstring available for TFBlenderbotSmallForConditionalGeneration

Methods: call

</tf>
<jax>

## FlaxBlenderbotSmallModel

No docstring available for FlaxBlenderbotSmallModel

Methods: __call__
    - encode
    - decode

## FlaxBlenderbotForConditionalGeneration

No docstring available for FlaxBlenderbotSmallForConditionalGeneration

Methods: __call__
    - encode
    - decode

</jax>
</frameworkcontent>
