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

# MEGA

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The MEGA model was proposed in [Mega: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655) by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, and Luke Zettlemoyer.
MEGA proposes a new approach to self-attention with each encoder layer having a multi-headed exponential moving average in addition to a single head of standard dot-product attention, giving the attention mechanism
stronger positional biases. This allows MEGA to perform competitively to Transformers on standard benchmarks including LRA
while also having significantly fewer parameters. MEGA's compute efficiency allows it to scale to very long sequences, making it an
attractive option for long-document NLP tasks.

The abstract from the paper is the following:

 *The design choices in the Transformer attention mechanism, including weak inductive bias and quadratic computational complexity, have limited its application for modeling long sequences. In this paper, we introduce Mega, a simple, theoretically grounded, single-head gated attention mechanism equipped with (exponential) moving average to incorporate inductive bias of position-aware local dependencies into the position-agnostic attention mechanism. We further propose a variant of Mega that offers linear time and space complexity yet yields only minimal quality loss, by efficiently splitting the whole sequence into multiple chunks with fixed length. Extensive experiments on a wide range of sequence modeling benchmarks, including the Long Range Arena, neural machine translation, auto-regressive language modeling, and image and speech classification, show that Mega achieves significant improvements over other sequence models, including variants of Transformers and recent state space models. *

This model was contributed by [mnaylor](https://huggingface.co/mnaylor).
The original code can be found [here](https://github.com/facebookresearch/mega).


## Usage tips

- MEGA can perform quite well with relatively few parameters. See Appendix D in the MEGA paper for examples of architectural specs which perform well in various settings. If using MEGA as a decoder, be sure to set `bidirectional=False` to avoid errors with default bidirectional.
- Mega-chunk is a variant of mega that reduces time and spaces complexity from quadratic to linear. Utilize chunking with MegaConfig.use_chunking and control chunk size with MegaConfig.chunk_size


## Implementation Notes

- The original implementation of MEGA had an inconsistent expectation of attention masks for padding and causal self-attention between the softmax attention and Laplace/squared ReLU method. This implementation addresses that inconsistency.
- The original implementation did not include token type embeddings; this implementation adds support for these, with the option controlled by MegaConfig.add_token_type_embeddings


## MegaConfig


    This is the configuration class to store the configuration of a [`MegaModel`]. It is used to instantiate a Mega
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Mega
    [mnaylor/mega-base-wikitext](https://huggingface.co/mnaylor/mega-base-wikitext) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Mega model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MegaModel`].
        hidden_size (`int`, *optional*, defaults to 128):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the Mega encoder.
        intermediate_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden size (self-attention value projection) within the Mega encoder
        ema_projection_size (`int`, *optional*, defaults to 16):
            Dimensionality of the MegaMultiDimensionDampedEma
        bidirectional (`bool`, *optional*, defaults to `True`):
            Whether the MegaMultiDimensionDampedEma used in Mega's self-attention should work bidirectionally (`True`)
            or unidirectionally (`False`). Bidirectional EMA is incompatible with causal decoding, so this should be
            False if you intend to use the model as a decoder.
        shared_representation_size (`int`, *optional*, defaults to 64):
            Dimensionality of the linear projection for shared representation of self-attention queries and keys
        use_chunking (`bool`, *optional*, defaults to `False`):
            Whether to chunk inputs for linear self-attention complexity (described as Mega-chunk in the paper)
        chunk_size (`int`, *optional*, defaults to -1):
            If `use_chunking` is set to `True`, determines the size of the chunks to apply to the input sequence. If
            chunking is used, input sequences must be padded to a multiple of `chunk_size`
        truncation (`int`, *optional*):
            If specified, the sequence length for which to truncate MegaMultiDimensionDampedEma
        normalize_before_mega (`bool`, *optional*, defaults to `True`):
            Whether to normalize before (`True`) or after (`False`) passing through Mega encoder blocks
        normalization_type (`str`, *optional*, defaults to `"scalenorm"`):
            Type of normalization to use in Mega encoder blocks. Choose one of `"scalenorm"`, `"layernorm"`,
            `"rmsnorm"`, `"batchnorm"`, or `"syncbatchnorm"` (GPU required for syncbatchnorm)
        norm_affine (`bool`, *optional*, defaults to `True`):
            If `True`, applies a parameterized affine transformation to inputs during normalization
        activation (`str`, *optional*, defaults to `"silu"`):
            Activation function to apply within Mega encoder blocks. Choose one of `"silu"`, `"relu"`, `"linear"`,
            `"gelu"`, or `"gelu_accurate"`
        attention_activation (`str`, *optional*, defaults to `"softmax"`):
            Activation function to apply for single-headed self-attention (a la Transformer). Choose one of
            `"softmax"`, `"laplace"`, or `"relu2"`
        dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for EMA self-attention
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        use_feature_dropout (`bool`, *optional*, defaults to `False`):
            Whether to use feature-based (`True`) or standard dropout (`False`)
        use_normalized_ffn (`bool`, *optional*, defaults to `True`):
            Whether to use the normalized feed-forward sub-layer in Mega blocks (`True`) or pass Mega encoder output
            as-is (`False`)
        nffn_hidden_size (`int`, *optional*, defaults to 256):
            If using the normalized feed-forward network (NFFN) layer within Mega (`use_normalized_ffn = True`), this
            is the hidden size of the NFFN
        normalize_before_ffn (`bool`, *optional*, defaults to `True`):
            Whether to normalize before (`True`) or after (`False`) the feed-forward portion of NFFN
        nffn_activation_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the NFFN component.
        max_positions (`int`, *optional*, defaults to 2048):
            The maximum sequence length to use for positional representations. For `"simple"` relative positional bias,
            this is a hard limit on input length; `"rotary"` relative positional bias will extrapolate to longer
            sequences
        add_token_type_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to account for token types in embeddings. Left as optional to maintain compatibility with original
            implementation while adding support for token types.
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`MegaModel`]. Only used if
            `add_token_type_embeddings = True`
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        ema_delta_alpha_range (`float`, *optional*, defaults to 0.2):
            The standard deviation for initializing the delta (damping factor) and alpha (decay factor) parameters in
            MegaMultiDimensionDampedEma.
        ema_beta_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for initializing the beta parameter (expansion matrix) in
            MegaMultiDimensionDampedEma.
        ema_gamma_omega_range (`float`, *optional*, defaults to 1.0):
            The standard deviation for initializing the gamma (projection matrix) and omega (residual weight)
            parameters in MultiDimensionEMA.
        relative_positional_bias (`str`, *optional*, defaults to `"rotary"`):
            Type of relative positional encoding. Choose one of `"rotary"` or `"simple"`. If `"simple"` is selected,
            `max_positions` is used as a limit on input size, while `"rotary"` extrapolates beyond `max_positions`.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        add_lm_hidden_dense_layer (`bool`, *optional*, defaults to `True`):
            Whether to include a hidden layer for projection between encoder outputs and LM heads (`True`) or pass
            hidden states directly to LM head (`False`). Remains optional for compatibility with original
            implementation

    Examples:

    ```python
    >>> from transformers import MegaConfig, MegaModel

    >>> # Initializing a Mega configuration
    >>> configuration = MegaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MegaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## MegaModel

The bare MEGA Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MegaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added after self-attention, following the architecture described in *Mega: Moving Average
    Equipped Gated Attention*_ by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig,
    Jonathan May, and Luke Zettlemoyer

    To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to
    `True` and `bidirectional` set to `False`. To be used in a Seq2Seq model, the model needs to initialized with both
    `is_decoder=True` and `bidirectional=False` argument as well as `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Mega: Moving Average Equipped Gated Attention*: https://arxiv.org/abs/2209.10655

    

Methods: forward

## MegaForCausalLM

MEGA Model with a `language modeling` head on top for CLM fine-tuning.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MegaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MegaForMaskedLM

MEGA Model with a `language modeling` head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MegaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MegaForSequenceClassification


    MEGA Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MegaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MegaForMultipleChoice


    MEGA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MegaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MegaForTokenClassification


    MEGA Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MegaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MegaForQuestionAnswering


    MEGA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MegaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
