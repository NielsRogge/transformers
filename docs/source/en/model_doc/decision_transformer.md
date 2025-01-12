<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Decision Transformer

## Overview

The Decision Transformer model was proposed in [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)  
by Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.

The abstract from the paper is the following:

*We introduce a framework that abstracts Reinforcement Learning (RL) as a sequence modeling problem. 
This allows us to draw upon the simplicity and scalability of the Transformer architecture, and associated advances
 in language modeling such as GPT-x and BERT. In particular, we present Decision Transformer, an architecture that 
 casts the problem of RL as conditional sequence modeling. Unlike prior approaches to RL that fit value functions or 
 compute policy gradients, Decision Transformer simply outputs the optimal actions by leveraging a causally masked 
 Transformer. By conditioning an autoregressive model on the desired return (reward), past states, and actions, our 
 Decision Transformer model can generate future actions that achieve the desired return. Despite its simplicity, 
 Decision Transformer matches or exceeds the performance of state-of-the-art model-free offline RL baselines on 
 Atari, OpenAI Gym, and Key-to-Door tasks.*

This version of the model is for tasks where the state is a vector.

This model was contributed by [edbeeching](https://huggingface.co/edbeeching). The original code can be found [here](https://github.com/kzl/decision-transformer).

## DecisionTransformerConfig


    This is the configuration class to store the configuration of a [`DecisionTransformerModel`]. It is used to
    instantiate a Decision Transformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the standard
    DecisionTransformer architecture. Many of the config options are used to instatiate the GPT2 model that is used as
    part of the architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        state_dim (`int`, *optional*, defaults to 17):
            The state size for the RL environment
        act_dim (`int`, *optional*, defaults to 4):
            The size of the output action space
        hidden_size (`int`, *optional*, defaults to 128):
            The size of the hidden layers
        max_ep_len (`int`, *optional*, defaults to 4096):
            The maximum length of an episode in the environment
        action_tanh (`bool`, *optional*, defaults to True):
            Whether to use a tanh activation on action prediction
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DecisionTransformerModel`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_layer (`int`, *optional*, defaults to 3):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. If unset, will default to 4 times `n_embd`.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import DecisionTransformerConfig, DecisionTransformerModel

    >>> # Initializing a DecisionTransformer configuration
    >>> configuration = DecisionTransformerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DecisionTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```


## DecisionTransformerGPT2Model

No docstring available for DecisionTransformerGPT2Model

Methods: forward

## DecisionTransformerModel

The Decision Transformer Model
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~DecisionTransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    

Methods: forward
