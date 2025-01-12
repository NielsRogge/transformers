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
# Zamba

Zamba is a large language model (LLM) trained by Zyphra, and made available under an Apache 2.0 license. Please see the [Zyphra Hugging Face](https://huggingface.co/collections/zyphra/) repository for model weights.

This model was contributed by [pglo](https://huggingface.co/pglo).


## Model details

Zamba-7B-v1 is a hybrid between state-space models (Specifically [Mamba](https://github.com/state-spaces/mamba)) and transformer, and was trained using next-token prediction. Zamba uses a shared transformer layer after every 6 mamba blocks. It uses the [Mistral v0.1 tokenizer](https://huggingface.co/mistralai/Mistral-7B-v0.1). We came to this architecture after a series of ablations at small scales. Zamba-7B-v1 was pre-trained on 1T tokens of text and code data.

<img src=https://github.com/user-attachments/assets/c2cff209-b901-483c-87aa-774b82a0769f width=30% height=40% />

## Quick start


### Presequities

Zamba requires you use `transformers` version 4.46.0 or higher:
```bash
pip install transformers>=4.45.0
```

In order to run optimized Mamba implementations, you first need to install `mamba-ssm` and `causal-conv1d`:
```bash
pip install mamba-ssm causal-conv1d>=1.2.0
```
You also have to have the model on a CUDA device.

You can run the model not using the optimized Mamba kernels, but it is **not** recommended as it will result in significantly lower latencies. In order to do that, you'll need to specify `use_mamba_kernels=False` when loading the model.


## Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba-7B-v1")
model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba-7B-v1", device_map="auto", torch_dtype=torch.bfloat16)

input_text = "A funny prompt would be "
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```


## Model card

The model cards can be found at:
* [Zamba-7B](MODEL_CARD_ZAMBA-7B-v1.md)


## Issues
For issues with model output, or community discussion, please use the Hugging Face community [forum](https://huggingface.co/zyphra/zamba-7b)


## License

The model weights are open-sourced via an Apache 2.0 license.


## ZambaConfig


    This is the configuration class to store the configuration of a [`ZambaModel`]. It is used to instantiate a
    Zamba model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Zamba-v0.1 model.

    [Zyphra/Zamba-7B-v1](https://huggingface.co/Zyphra/Zamba-7B-v1)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Zamba model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ZambaModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 3712):
            Dimension of the hidden representations.
        attention_hidden_size (`int`, *optional*):
            Dimension of the hidden representations of the inputs to the Attention layer.
        intermediate_size (`int`, *optional*, defaults to 14848):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 76):
            Number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        attention_head_dim (`int`, *optional*):
            Dimension of the attention head in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=None`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf).
        n_mamba_heads (`int`, *optional*, defaults to 2):
            Number of mamba heads for each mamba layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder.
        hidden_mamba_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the mamba layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
            logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
            sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
            significantly.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            This value doesn't have any real effect. The maximum sequence length that this model is intended to be
            used with. It can be used with longer sequences, but performance may degrade.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attn_layer_period (`int`, *optional*, defaults to 6):
            Once in this many layers, we will have a shared attention layer
        attn_layer_offset (`int`, *optional*, defaults to 4):
            Offset of the shared attention layer
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
            `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
            `True` and kernels are not available
        mamba_d_state (`int`, *optional*, defaults to 16):
            The dimension the mamba state space latents
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
        mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj_bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj_bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block

    


## ZambaModel

The bare Zamba Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ZambaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ZambaDecoderLayer`]

    Args:
        config: ZambaConfig
    

Methods: forward


## ZambaForCausalLM

No docstring available for ZambaForCausalLM

Methods: forward


## ZambaForSequenceClassification


    The Zamba Model with a sequence classification head on top (linear layer).

    [`ZambaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ZambaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
