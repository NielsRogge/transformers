<!--Copyright 2024 Kyutai and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Helium


## Overview

Helium was proposed in [Announcing Helium-1 Preview](https://kyutai.org/2025/01/13/helium.html) by the Kyutai Team.


Helium-1 preview is a lightweight language model with 2B parameters, targeting edge and mobile devices.
It supports the following languages: English, French, German, Italian, Portuguese, Spanish.

- **Developed by:** Kyutai
- **Model type:** Large Language Model
- **Language(s) (NLP):** English, French, German, Italian, Portuguese, Spanish
- **License:** CC-BY 4.0




## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The model was evaluated on MMLU, TriviaQA, NaturalQuestions, ARC Easy & Challenge, Open Book QA, Common Sense QA, 
Physical Interaction QA, Social Interaction QA, HellaSwag, WinoGrande, Multilingual Knowledge QA, FLORES 200.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

We report accuracy on MMLU, ARC, OBQA, CSQA, PIQA, SIQA, HellaSwag, WinoGrande.
We report exact match on TriviaQA, NQ and MKQA.
We report BLEU on FLORES.

### English Results

| Benchmark | Helium-1 Preview | HF SmolLM2 (1.7B) | Gemma-2 (2.6B) | Llama-3.2 (3B) | Qwen2.5 (1.5B) |
|--------------|--------|--------|--------|--------|--------|
| | | | | | |
| MMLU | 51.2 | 50.4 | 53.1 | 56.6 | 61.0 |
| NQ   | 17.3 | 15.1 | 17.7 | 22.0 | 13.1 |
| TQA  | 47.9 | 45.4 | 49.9 | 53.6 | 35.9 |
| ARC E | 80.9 | 81.8 | 81.1 | 84.6 | 89.7 |
| ARC C | 62.7 | 64.7 | 66.0 | 69.0 | 77.2 |
| OBQA | 63.8 | 61.4 | 64.6 | 68.4 | 73.8 |
| CSQA | 65.6 | 59.0 | 64.4 | 65.4 | 72.4 |
| PIQA | 77.4 | 77.7 | 79.8 | 78.9 | 76.0 |
| SIQA | 64.4 | 57.5 | 61.9 | 63.8 | 68.7 |
| HS | 69.7 | 73.2 | 74.7 | 76.9 | 67.5 |
| WG | 66.5 | 65.6 | 71.2 | 72.0 | 64.8 |
| | | | | | |
| Average | 60.7 | 59.3 | 62.2 | 64.7 | 63.6 |

#### Multilingual Results

| Language | Benchmark | Helium-1 Preview | HF SmolLM2 (1.7B) | Gemma-2 (2.6B) | Llama-3.2 (3B) | Qwen2.5 (1.5B) |
|-----|--------------|--------|--------|--------|--------|--------|
| | | | | | | |
|German| MMLU | 45.6 | 35.3 | 45.0 | 47.5 | 49.5 |
|| ARC C | 56.7 | 38.4 | 54.7 | 58.3 | 60.2 |
|| HS | 53.5 | 33.9 | 53.4 | 53.7 | 42.8 |
|| MKQA | 16.1 | 7.1 | 18.9 | 20.2 | 10.4 |
| | | | | | | |
|Spanish| MMLU | 46.5 | 38.9 | 46.2 | 49.6 | 52.8 |
|| ARC C | 58.3 | 43.2 | 58.8 | 60.0 | 68.1 |
|| HS | 58.6 | 40.8 | 60.5 | 61.1 | 51.4 |
|| MKQA | 16.0 | 7.9 | 18.5 | 20.6 | 10.6 |


## Technical Specifications

### Model Architecture and Objective

| Hyperparameter | Value |
|--------------|--------|
| Layers | 24 |
| Heads  | 20 |
| Model dimension | 2560 |
| MLP dimension | 7040 |
| Context size | 4096 |
| Theta RoPE | 100,000 |

Tips:

- This model was contributed by [Laurent Mazare](https://huggingface.co/lmz)

  
## Usage tips

`Helium` can be found on the [Huggingface Hub](https://huggingface.co/collections/kyutai/helium-1-preview)

In the following, we demonstrate how to use `helium-1-preview` for the inference. 

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> model = AutoModelForCausalLM.from_pretrained("helium-1-preview", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("helium-1-preview")

>>> prompt = "Give me a short introduction to large language model."

>>> messages = [{"role": "user", "content": prompt}]

>>> text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

>>> model_inputs = tokenizer([text], return_tensors="pt").to(device)

>>> generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

>>> generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

>>> response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## HeliumConfig

HeliumConfig

    This is the configuration class to store the configuration of a [`HeliumModel`]. It is used to instantiate an Helium
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Helium 2b model.
    e.g. [kyutai/helium-2b](https://huggingface.co/kyutai/helium-2b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 48000):
            Vocabulary size of the Helium model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HeliumModel`]
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 7040):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 20):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The legacy activation function. It is overwritten by the `hidden_activation`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-08):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 100000.0):
            The base period of the RoPE embeddings.
        pad_token_id (`int`, *optional*, defaults to 3):
            Padding token id.
        eos_token_id (`int` | `list`, *optional*, defaults to 2):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
    ```python
    >>> from transformers import HeliumModel, HeliumConfig
    >>> # Initializing a Helium 2b style configuration
    >>> configuration = HeliumConfig()
    >>> # Initializing a model from the Helium 2b style configuration
    >>> model = HeliumModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## HeliumModel

HeliumModel
The bare Helium Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`HeliumConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`HeliumDecoderLayer`]

    Args:
        config: HeliumConfig
    
    - forward

## HeliumForCausalLM

HeliumForCausalLM
    - forward

## HeliumForSequenceClassification

HeliumForSequenceClassification

    The Helium Model transformer with a sequence classification head on top (linear layer).

    [`HeliumForSequenceClassification`] uses the last token in order to do the classification, as other causal models
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
        config ([`HeliumConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    - forward

## HeliumForTokenClassification

HeliumForTokenClassification

    The Helium Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`HeliumConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    - forward
