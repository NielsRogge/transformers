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

# FalconMamba

## Overview

The FalconMamba model was proposed by TII UAE (Technology Innovation Institute) in their release.

The abstract from the paper is the following:

*We present FalconMamba, a new base large language model based on the novel Mamba architecture. FalconMamba is trained on 5.8 trillion tokens with carefully selected data mixtures. As a pure Mamba-based model, FalconMamba surpasses leading open-weight models based on Transformers, such as Mistral 7B, Llama3 8B, and Falcon2 11B. It is on par with Gemma 7B and outperforms models with different architecture designs, such as RecurrentGemma 9B. Currently, FalconMamba is the best-performing Mamba model in the literature at this scale, surpassing both existing Mamba and hybrid Mamba-Transformer models.
Due to its architecture, FalconMamba is significantly faster at inference and requires substantially less memory for long sequence generation. Despite recent studies suggesting that hybrid Mamba-Transformer models outperform pure architecture designs, we argue and demonstrate that the pure Mamba design can achieve similar, even superior results compared to the hybrid design. We make the weights of our implementation of FalconMamba publicly available under a permissive license.*

Tips:

- FalconMamba is mostly based on Mamba architecture, the same [tips and best practices](./mamba) would be relevant here.

The model has been trained on approximtely 6T tokens consisting a mixture of many data sources such as RefineWeb, Cosmopedia and Math data.

For more details about the training procedure and the architecture, have a look at [the technical paper of FalconMamba]() (coming soon).

# Usage

Below we demonstrate how to use the model:

```python 
from transformers import FalconMambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
model = FalconMambaForCausalLM.from_pretrained("tiiuae/falcon-mamba-7b")

input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

The architecture is also compatible with `torch.compile` for faster generation:

```python 
from transformers import FalconMambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
model = FalconMambaForCausalLM.from_pretrained("tiiuae/falcon-mamba-7b", torch_dtype=torch.bfloat16).to(0)
model = torch.compile(model)

input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

If you have access to a GPU that is compatible with `bitsandbytes`, you can also quantize the model in 4-bit precision:

```python 
from transformers import FalconMambaForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = FalconMambaForCausalLM.from_pretrained("tiiuae/falcon-mamba-7b", quantization_config=quantization_config)

input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

You can also play with the instruction fine-tuned model:

```python 
from transformers import FalconMambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b-instruct")
model = FalconMambaForCausalLM.from_pretrained("tiiuae/falcon-mamba-7b-instruct")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

## FalconMambaConfig


    This is the configuration class to store the configuration of a [`FalconMambaModel`]. It is used to instantiate a FALCON_MAMBA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the FALCON_MAMBA
    [tiiuae/falcon-mamba-7b](https://huggingface.co/tiiuae/falcon-mamba-7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50280):
            Vocabulary size of the FALCON_MAMBA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FalconMambaModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        state_size (`int`, *optional*, defaults to 16): shape of the state space latents.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end of sentence token in the vocabulary.
        expand (`int`, *optional*, defaults to 2): Expanding factor used to determine the intermediate size.
        conv_kernel (`int`, *optional*, defaults to 4): Size of the convolution kernel.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the mixer block.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.1):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
        time_step_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        time_step_scale (`float`, *optional*, defaults to 1.0):
            Scale used used to scale `dt_proj.bias`.
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj.bias`.
        time_step_init_scheme (`float`, *optional*, defaults to `"random"`):
            Init scheme used for `dt_proj.weight`. Should be one of `["random","uniform"]`
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether or not to rescale `out_proj` weights when initializing.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the cache should be used.
        use_mambapy (`bool`, *optional*, defaults to `False`):
            Determines the fallback strategy during training if the CUDA-based official implementation of FalconMamba is not avaiable. If `True`, the falcon_mamba.py implementation is used. If `False`, the naive and slower implementation is used. Consider switching to the naive version if memory is limited.
        mixer_rms_eps (`float`, *optional*, defaults to 1e-06):
            The RMS norm epsilon value that is used in the Mixer RMS norm for B, C and dt states.
    Example:

    ```python
    >>> from transformers import FalconMambaConfig, FalconMambaModel

    >>> # Initializing a FalconMamba configuration
    >>> configuration = FalconMambaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FalconMambaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## FalconMambaModel

The bare FALCONMAMBA Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FalconMambaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FalconMambaLMHeadModel


    The FALCONMAMBA Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FalconMambaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
