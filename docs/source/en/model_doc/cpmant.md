<!--Copyright 2022 The HuggingFace Team and The OpenBMB Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPMAnt

## Overview

CPM-Ant is an open-source Chinese pre-trained language model (PLM) with 10B parameters. It is also the first milestone of the live training process of CPM-Live. The training process is cost-effective and environment-friendly. CPM-Ant also achieves promising results with delta tuning on the CUGE benchmark. Besides the full model, we also provide various compressed versions to meet the requirements of different hardware configurations. [See more](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live)

This model was contributed by [OpenBMB](https://huggingface.co/openbmb). The original code can be found [here](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live).

## Resources

- A tutorial on [CPM-Live](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live).

## CpmAntConfig


    This is the configuration class to store the configuration of a [`CpmAntModel`]. It is used to instantiate an
    CPMAnt model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CPMAnt
    [openbmb/cpm-ant-10b](https://huggingface.co/openbmb/cpm-ant-10b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30720):
            Vocabulary size of the CPMAnt model. Defines the number of different tokens that can be represented by the
            `input` passed when calling [`CpmAntModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the encoder layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads in the Transformer encoder.
        dim_head (`int`, *optional*, defaults to 128):
            Dimension of attention heads for each attention layer in the Transformer encoder.
        dim_ff (`int`, *optional*, defaults to 10240):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of layers of the Transformer encoder.
        dropout_p (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder.
        position_bias_num_buckets (`int`, *optional*, defaults to 512):
            The number of position_bias buckets.
        position_bias_max_distance (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        init_std (`float`, *optional*, defaults to 1.0):
            Initialize parameters with std = init_std.
        prompt_types (`int`, *optional*, defaults to 32):
            The type of prompt.
        prompt_length (`int`, *optional*, defaults to 32):
            The length of prompt.
        segment_types (`int`, *optional*, defaults to 32):
            The type of segment.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use cache.

    Example:

    ```python
    >>> from transformers import CpmAntModel, CpmAntConfig

    >>> # Initializing a CPMAnt cpm-ant-10b style configuration
    >>> configuration = CpmAntConfig()

    >>> # Initializing a model from the cpm-ant-10b style configuration
    >>> model = CpmAntModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

Methods: all

## CpmAntTokenizer


    Construct a CPMAnt tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bod_token (`str`, *optional*, defaults to `"<d>"`):
            The beginning of document token.
        eod_token (`str`, *optional*, defaults to `"</d>"`):
            The end of document token.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        line_token (`str`, *optional*, defaults to `"</n>"`):
            The line token.
        space_token (`str`, *optional*, defaults to `"</_>"`):
            The space token.
    

Methods: all

## CpmAntModel

The bare CPMAnt Model outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters
        config ([`~CpmAntConfig`]): Model configuration class with all the parameters of the
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: all
    
## CpmAntForCausalLM


    The CPMAnt Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters
        config ([`~CpmAntConfig`]): Model configuration class with all the parameters of the
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: all