<!--Copyright 2024 JetMoe team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# JetMoe

## Overview

**JetMoe-8B** is an 8B Mixture-of-Experts (MoE) language model developed by [Yikang Shen](https://scholar.google.com.hk/citations?user=qff5rRYAAAAJ) and [MyShell](https://myshell.ai/).
JetMoe project aims to provide a LLaMA2-level performance and efficient language model with a limited budget.
To achieve this goal, JetMoe uses a sparsely activated architecture inspired by the [ModuleFormer](https://arxiv.org/abs/2306.04640). 
Each JetMoe block consists of two MoE layers: Mixture of Attention Heads and Mixture of MLP Experts.
Given the input tokens, it activates a subset of its experts to process them.
This sparse activation schema enables JetMoe to achieve much better training throughput than similar size dense models. 
The training throughput of JetMoe-8B is around 100B tokens per day on a cluster of 96 H100 GPUs with a straightforward 3-way pipeline parallelism strategy.

This model was contributed by [Yikang Shen](https://huggingface.co/YikangS).


## JetMoeConfig


    This is the configuration class to store the configuration of a [`JetMoeModel`]. It is used to instantiate a
    JetMoe model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a configuration of the JetMoe-4B.

    [jetmoe/jetmoe-8b](https://huggingface.co/jetmoe/jetmoe-8b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the JetMoe model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`JetMoeModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each key and value in the Transformer encoder.
        kv_channels (`int`, *optional*, defaults to 128):
            Defines the number of channels for the key and value tensors.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP representations.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. JetMoe's attention allows sequence of
            up to 4096 tokens.
        activation_function (`string`, *optional*, defaults to `"silu"`):
            Defines the activation function for MLP experts.
        num_local_experts (`int`, *optional*, defaults to 8):
            Defines the number of experts in the MoE and MoA.
        num_experts_per_tok (`int, *optional*, defaults to 2):
            The number of experts to route per-token and for MoE and MoA.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabeling this will also
            allow the model to output the auxiliary loss.
        aux_loss_coef (`float`, *optional*, defaults to 0.01):
            The coefficient for the auxiliary loss.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import JetMoeModel, JetMoeConfig

    >>> # Initializing a JetMoe 4B style configuration
    >>> configuration = JetMoeConfig()

    >>> # Initializing a model from the JetMoe 4B style configuration
    >>> model = JetMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## JetMoeModel

The bare JetMoe Model outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`JetMoeConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`JetMoeBlock`]

    Args:
        config:
            JetMoeConfig
    

Methods: forward

## JetMoeForCausalLM

No docstring available for JetMoeForCausalLM

Methods: forward

## JetMoeForSequenceClassification


    The JetMoe Model transformer with a sequence classification head on top (linear layer).

    [`JetMoeForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`JetMoeConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
