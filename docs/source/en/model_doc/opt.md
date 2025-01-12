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

# OPT

## Overview

The OPT model was proposed in [Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068) by Meta AI.
OPT is a series of open-sourced large causal language models which perform similar in performance to GPT3.

The abstract from the paper is the following:

*Large language models, which are often trained for hundreds of thousands of compute days, have shown remarkable capabilities for zero- and few-shot learning. Given their computational cost, these models are difficult to replicate without significant capital. For the few that are available through APIs, no access is granted to the full model weights, making them difficult to study. We present Open Pre-trained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We show that OPT-175B is comparable to GPT-3, while requiring only 1/7th the carbon footprint to develop. We are also releasing our logbook detailing the infrastructure challenges we faced, along with code for experimenting with all of the released models.*

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ), [Younes Belkada](https://huggingface.co/ybelkada), and [Patrick Von Platen](https://huggingface.co/patrickvonplaten).
The original code can be found [here](https://github.com/facebookresearch/metaseq).

Tips:
- OPT has the same architecture as [`BartDecoder`].
- Contrary to GPT2, OPT adds the EOS token `</s>` to the beginning of every prompt.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with OPT. If you're
interested in submitting a resource to be included here, please feel free to open a Pull Request and we will review it.
The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-generation" />

- A notebook on [fine-tuning OPT with PEFT, bitsandbytes, and Transformers](https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing). 🌎
- A blog post on [decoding strategies with OPT](https://huggingface.co/blog/introducing-csearch#62-example-two---opt).
- [Causal language modeling](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) chapter of the 🤗 Hugging Face Course.
- [`OPTForCausalLM`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFOPTForCausalLM`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxOPTForCausalLM`] is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling).

<PipelineTag pipeline="text-classification" />

- [Text classification task guide](sequence_classification.md)
- [`OPTForSequenceClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb).

<PipelineTag pipeline="question-answering" />

- [`OPTForQuestionAnswering`] is supported by this [question answering example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).
- [Question answering](https://huggingface.co/course/chapter7/7?fw=pt) chapter
  of the 🤗 Hugging Face Course.

⚡️ Inference

- A blog post on [How 🤗 Accelerate runs very large models thanks to PyTorch](https://huggingface.co/blog/accelerate-large-models) with OPT.


## Combining OPT and Flash Attention 2

First, make sure to install the latest version of Flash Attention 2 to include the sliding window attention feature.

```bash
pip install -U flash-attn --no-build-isolation
```

Make also sure that you have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of flash-attn repository. Make also sure to load your model in half-precision (e.g. `torch.float16``)

To load and run a model using Flash Attention 2, refer to the snippet below:

```python
>>> import torch
>>> from transformers import OPTForCausalLM, GPT2Tokenizer
>>> device = "cuda" # the device to load the model onto

>>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
>>> tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

>>> prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "
              "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
              "there?")

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
>>> tokenizer.batch_decode(generated_ids)[0]
'</s>A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived there?\nStatue: I have lived here for about a year.\nHuman: What is your favorite place to eat?\nStatue: I love'
```

### Expected speedups

Below is an expected speedup diagram that compares pure inference time between the native implementation in transformers using `facebook/opt-2.7b` checkpoint and the Flash Attention 2 version of the model using two different sequence lengths.

<div style="text-align: center">
<img src="https://user-images.githubusercontent.com/49240599/281101546-d2fca6d2-ee44-48f3-9534-ba8d5bee4531.png">
</div>

Below is an expected speedup diagram that compares pure inference time between the native implementation in transformers using `facebook/opt-350m` checkpoint and the Flash Attention 2 version of the model using two different sequence lengths.

<div style="text-align: center">
<img src="https://user-images.githubusercontent.com/49240599/281101682-d1144e90-0dbc-46f4-8fc8-c6206cb793c9.png">
</div>


### Using Scaled Dot Product Attention (SDPA)
PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```python
from transformers import OPTForCausalLM
model = OPTForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16, attn_implementation="sdpa")
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (L40S-45GB, PyTorch 2.4.0, OS Debian GNU/Linux 11) using `float16` with
[facebook/opt-350m](https://huggingface.co/facebook/opt-350m), we saw the
following speedups during training and inference.

### Training

|    batch_size |    seq_len |  Time per batch (eager - s)   |    Time per batch (sdpa - s) |  Speedup (%)   |  Eager peak mem (MB)   |    sdpa peak mem (MB) |  Mem saving (%)   |
|--------------:|-----------:|:------------------------------|-----------------------------:|:---------------|:-----------------------|----------------------:|:------------------|
|             1 |        128 | 0.047                         |                        0.037 | 26.360         | 1474.611               |               1474.32 | 0.019             |
|             1 |        256 | 0.046                         |                        0.037 | 24.335         | 1498.541               |               1499.49 | -0.063            |
|             1 |        512 | 0.046                         |                        0.037 | 24.959         | 1973.544               |               1551.35 | 27.215            |
|             1 |       1024 | 0.062                         |                        0.038 | 65.135         | 4867.113               |               1698.35 | 186.578           |
|             1 |       2048 | 0.230                         |                        0.039 | 483.933        | 15662.224              |               2715.75 | 476.718           |
|             2 |        128 | 0.045                         |                        0.037 | 20.455         | 1498.164               |               1499.49 | -0.089            |
|             2 |        256 | 0.046                         |                        0.037 | 24.027         | 1569.367               |               1551.35 | 1.161             |
|             2 |        512 | 0.045                         |                        0.037 | 20.965         | 3257.074               |               1698.35 | 91.778            |
|             2 |       1024 | 0.122                         |                        0.038 | 225.958        | 9054.405               |               2715.75 | 233.403           |
|             2 |       2048 | 0.464                         |                        0.067 | 593.646        | 30572.058              |               4750.55 | 543.548           |
|             4 |        128 | 0.045                         |                        0.037 | 21.918         | 1549.448               |               1551.35 | -0.123            |
|             4 |        256 | 0.044                         |                        0.038 | 18.084         | 2451.768               |               1698.35 | 44.361            |
|             4 |        512 | 0.069                         |                        0.037 | 84.421         | 5833.180               |               2715.75 | 114.791           |
|             4 |       1024 | 0.262                         |                        0.062 | 319.475        | 17427.842              |               4750.55 | 266.860           |
|             4 |       2048 | OOM                           |                        0.062 | Eager OOM      | OOM                    |               4750.55 | Eager OOM         |
|             8 |        128 | 0.044                         |                        0.037 | 18.436         | 2049.115               |               1697.78 | 20.694            |
|             8 |        256 | 0.048                         |                        0.036 | 32.887         | 4222.567               |               2715.75 | 55.484            |
|             8 |        512 | 0.153                         |                        0.06  | 154.862        | 10985.391              |               4750.55 | 131.245           |
|             8 |       1024 | 0.526                         |                        0.122 | 330.697        | 34175.763              |               8821.18 | 287.428           |
|             8 |       2048 | OOM                           |                        0.122 | Eager OOM      | OOM                    |               8821.18 | Eager OOM         |

### Inference

|    batch_size |    seq_len |    Per token latency eager (ms) |    Per token latency SDPA (ms) |    Speedup (%) |    Mem eager (MB) |    Mem BT (MB) |    Mem saved (%) |
|--------------:|-----------:|--------------------------------:|-------------------------------:|---------------:|------------------:|---------------:|-----------------:|
|             1 |        128 |                          11.634 |                          8.647 |         34.546 |           717.676 |        717.674 |            0     |
|             1 |        256 |                          11.593 |                          8.86  |         30.851 |           742.852 |        742.845 |            0.001 |
|             1 |        512 |                          11.515 |                          8.816 |         30.614 |           798.232 |        799.593 |           -0.17  |
|             1 |       1024 |                          11.556 |                          8.915 |         29.628 |           917.265 |        895.538 |            2.426 |
|             2 |        128 |                          12.724 |                         11.002 |         15.659 |           762.434 |        762.431 |            0     |
|             2 |        256 |                          12.704 |                         11.063 |         14.83  |           816.809 |        816.733 |            0.009 |
|             2 |        512 |                          12.757 |                         10.947 |         16.535 |           917.383 |        918.339 |           -0.104 |
|             2 |       1024 |                          13.018 |                         11.018 |         18.147 |          1162.65  |       1114.81  |            4.291 |
|             4 |        128 |                          12.739 |                         10.959 |         16.243 |           856.335 |        856.483 |           -0.017 |
|             4 |        256 |                          12.718 |                         10.837 |         17.355 |           957.298 |        957.674 |           -0.039 |
|             4 |        512 |                          12.813 |                         10.822 |         18.393 |          1158.44  |       1158.45  |           -0.001 |
|             4 |       1024 |                          13.416 |                         11.06  |         21.301 |          1653.42  |       1557.19  |            6.18  |
|             8 |        128 |                          12.763 |                         10.891 |         17.193 |          1036.13  |       1036.51  |           -0.036 |
|             8 |        256 |                          12.89  |                         11.104 |         16.085 |          1236.98  |       1236.87  |            0.01  |
|             8 |        512 |                          13.327 |                         10.939 |         21.836 |          1642.29  |       1641.78  |            0.031 |
|             8 |       1024 |                          15.181 |                         11.175 |         35.848 |          2634.98  |       2443.35  |            7.843 |

## OPTConfig


    This is the configuration class to store the configuration of a [`OPTModel`]. It is used to instantiate a OPT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the OPT
    [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50272):
            Vocabulary size of the OPT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`OPTModel`]
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        ffn_dim (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        do_layer_norm_before (`bool`, *optional*, defaults to `True`):
            Whether to perform layer normalization before the attention block.
        word_embed_proj_dim (`int`, *optional*):
            `word_embed_proj_dim` can be set to down-project word embeddings, *e.g.* `opt-350m`. Defaults to
            `hidden_size`.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more
            details.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        enable_bias (`bool`, *optional*, defaults to `True`):
            Whether or not if the linear layers in the attention blocks should use the bias term.
        layer_norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether or not if the layer norms should have learnable parameters.

    Example:

    ```python
    >>> from transformers import OPTConfig, OPTModel

    >>> # Initializing a OPT facebook/opt-large style configuration
    >>> configuration = OPTConfig()

    >>> # Initializing a model (with random weights) from the facebook/opt-large style configuration
    >>> model = OPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

<frameworkcontent>
<pt>

## OPTModel

The bare OPT Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## OPTForCausalLM

No docstring available for OPTForCausalLM

Methods: forward

## OPTForSequenceClassification


    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
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
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## OPTForQuestionAnswering


    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

</pt>
<tf>

## TFOPTModel

No docstring available for TFOPTModel

Methods: call

## TFOPTForCausalLM

No docstring available for TFOPTForCausalLM

Methods: call

</tf>
<jax>

## FlaxOPTModel

No docstring available for FlaxOPTModel

Methods: __call__

## FlaxOPTForCausalLM

No docstring available for FlaxOPTForCausalLM

Methods: __call__

</jax>
</frameworkcontent>
