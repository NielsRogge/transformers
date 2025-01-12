<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# RecurrentGemma

## Overview

The Recurrent Gemma model was proposed in [RecurrentGemma: Moving Past Transformers for Efficient Open Language Models](https://storage.googleapis.com/deepmind-media/gemma/recurrentgemma-report.pdf) by the Griffin, RLHF and Gemma Teams of Google.

The abstract from the paper is the following:

*We introduce RecurrentGemma, an open language model which uses Google’s novel Griffin architecture. Griffin combines linear recurrences with local attention to achieve excellent performance on language. It has a fixed-sized state, which reduces memory use and enables efficient inference on long sequences. We provide a pre-trained model with 2B non-embedding parameters, and an instruction tuned variant. Both models achieve comparable performance to Gemma-2B despite being trained on fewer tokens.*

Tips:

- The original checkpoints can be converted using the conversion script [`src/transformers/models/recurrent_gemma/convert_recurrent_gemma_weights_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/recurrent_gemma/convert_recurrent_gemma_to_hf.py). 

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ). The original code can be found [here](https://github.com/google-deepmind/recurrentgemma).


## RecurrentGemmaConfig


    This is the configuration class to store the configuration of a [`RecurrentGemmaModel`]. It is used to instantiate a RecurrentGemma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RecurrentGemma-7B.

    e.g. [google/recurrentgemma-2b](https://huggingface.co/google/recurrentgemma-2b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_hidden_layers (`int`, *optional*, defaults to 26):
            The number of hidden layers in the model.
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the RecurrentGemma model. Defines the number of
            different tokens that can be represented by the
            `inputs_ids` passed when calling [`RecurrentGemmaModel`]
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 7680):
            Dimension of the MLP representations.
        num_attention_heads (`int`, *optional*, defaults to 10):
            The number of heads for the attention block and the number of
            heads/blocks for the block-diagonal layers used in the RG-LRU gates.
            This number must divide `hidden_size` and `lru_width`.
        lru_width (`int` or `None`, *optional*):
            Dimension of the hidden representations of the RG-LRU. If `None`
            this will be set to `hidden_size`.
            Whether to scale the output of the embeddings by `sqrt(hidden_size)`.
        attention_window_size (`int`, *optional*, defaults to 2048):
            The size of the attention window used in the attention block.
        conv1d_width (`int`, *optional*, defaults to 4):
            The kernel size of conv1d layers used in the recurrent blocks.
        logits_soft_cap (`float`, *optional*, defaults to 30.0):
            The value at which the logits should be soft-capped to after the transformer and LM-head computation in the Causal LM architecture.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values
            attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        hidden_activation (``str` or `function``, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The hidden activation used in the recurrent block as well as the MLP layer of the decoder layers.
        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
            The partial rotary factor used in the initialization of the rotary embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        block_types (`List[str]`, *optional*, defaults to `('recurrent', 'recurrent', 'attention')`):
            List of aleternating blocks that will be repeated to initialize the `temporal_block` layer.
        attention_dropout (`float`, *optional*, defaults to 0.0): dropout value to use after the attention softmax.
        num_key_value_heads (`16`, *optional*, defaults to 16): Number of key value heads to use GQA.
        attention_bias (`bool`, *optional*, defaults to `False`): whether or not the linear q,k,v of the Attention layer should have bias
        w_init_variance_scale (`float`, *optional*, defaults to 0.01): weight initialization variance.
    ```python
    >>> from transformers import RecurrentGemmaModel, RecurrentGemmaConfig

    >>> # Initializing a RecurrentGemma recurrentgemma-2b style configuration
    >>> configuration = RecurrentGemmaConfig()

    >>> # Initializing a model from the recurrentgemma-2b style configuration
    >>> model = RecurrentGemmaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```


## RecurrentGemmaModel

The bare RecurrentGemma Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RecurrentGemmaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`RecurrentGemmaDecoderLayer`]

    Args:
        config: RecurrentGemmaConfig
    

Methods: forward

## RecurrentGemmaForCausalLM

No docstring available for RecurrentGemmaForCausalLM

Methods: forward

