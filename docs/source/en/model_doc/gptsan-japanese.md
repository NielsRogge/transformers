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

# GPTSAN-japanese

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The GPTSAN-japanese model was released in the repository by Toshiyuki Sakamoto (tanreinama).

GPTSAN is a Japanese language model using Switch Transformer. It has the same structure as the model introduced as Prefix LM
in the T5 paper, and support both Text Generation and Masked Language Modeling tasks. These basic tasks similarly can
fine-tune for translation or summarization.

### Usage example

The `generate()` method can be used to generate text using GPTSAN-Japanese model.

```python
>>> from transformers import AutoModel, AutoTokenizer
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").cuda()
>>> x_tok = tokenizer("は、", prefix_text="織田信長", return_tensors="pt")
>>> torch.manual_seed(0)
>>> gen_tok = model.generate(x_tok.input_ids.cuda(), token_type_ids=x_tok.token_type_ids.cuda(), max_new_tokens=20)
>>> tokenizer.decode(gen_tok[0])
'織田信長は、2004年に『戦国BASARA』のために、豊臣秀吉'
```

## GPTSAN Features

GPTSAN has some unique features. It has a model structure of Prefix-LM. It works as a shifted Masked Language Model for Prefix Input tokens. Un-prefixed inputs behave like normal generative models.
The Spout vector is a GPTSAN specific input. Spout is pre-trained with random inputs, but you can specify a class of text or an arbitrary vector during fine-tuning. This allows you to indicate the tendency of the generated text.
GPTSAN has a sparse Feed Forward based on Switch-Transformer. You can also add other layers and train them partially. See the original GPTSAN repository for details.

### Prefix-LM Model

GPTSAN has the structure of the model named Prefix-LM in the `T5` paper. (The original GPTSAN repository calls it `hybrid`)
In GPTSAN, the `Prefix` part of Prefix-LM, that is, the input position that can be referenced by both tokens, can be specified with any length.
Arbitrary lengths can also be specified differently for each batch.
This length applies to the text entered in `prefix_text` for the tokenizer.
The tokenizer returns the mask of the `Prefix` part of Prefix-LM as `token_type_ids`.
The model treats the part where `token_type_ids` is 1 as a `Prefix` part, that is, the input can refer to both tokens before and after.

## Usage tips

Specifying the Prefix part is done with a mask passed to self-attention.
When token_type_ids=None or all zero, it is equivalent to regular causal mask

for example:

>>> x_token = tokenizer("ｱｲｳｴ")
input_ids:      | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
token_type_ids: | 1   | 0   | 0 | 0 | 0 | 0 |
prefix_lm_mask:
SOT | 1 0 0 0 0 0 |
SEG | 1 1 0 0 0 0 |
ｱ   | 1 1 1 0 0 0 |
ｲ   | 1 1 1 1 0 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 1 |

>>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
input_ids:      | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
token_type_ids: | 1   | 1 | 1 | 1 | 1 | 0  |
prefix_lm_mask:
SOT | 1 1 1 1 1 0 |
ｱ   | 1 1 1 1 1 0 |
ｲ   | 1 1 1 1 1 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 0 |
SEG | 1 1 1 1 1 1 |

>>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
input_ids:      | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
token_type_ids: | 1   | 1 | 1 | 0   | 0 | 0 |
prefix_lm_mask:
SOT | 1 1 1 0 0 0 |
ｱ   | 1 1 1 0 0 0 |
ｲ   | 1 1 1 0 0 0 |
SEG | 1 1 1 1 0 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 1 |

### Spout Vector

A Spout Vector is a special vector for controlling text generation.
This vector is treated as the first embedding in self-attention to bring extraneous attention to the generated tokens.
In the pre-trained model published from `Tanrei/GPTSAN-japanese`, the Spout Vector is a 128-dimensional vector that passes through 8 fully connected layers in the model and is projected into the space acting as external attention.
The Spout Vector projected by the fully connected layer is split to be passed to all self-attentions.

## GPTSanJapaneseConfig


    This is the configuration class to store the configuration of a [`GPTSanJapaneseModel`]. It is used to instantiate
    a GPTSANJapanese model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPTSANJapanese
    [Tanrei/GPTSAN-japanese](https://huggingface.co/Tanrei/GPTSAN-japanese) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 36000):
            Vocabulary size of the GPTSANJapanese model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`GPTSanJapaneseModel`].
        max_position_embeddings (`int`, *optional*, defaults to 1280):
            The maximum sequence length that this model might ever be used with. Defaults set this to 1280.
        d_model (`int`, *optional*, defaults to 1024):
            Size of the encoder layers and the pooler layer.
        d_ff (`int`, *optional*, defaults to 8192):
            Size of the intermediate feed forward layer in each `SwitchTransformersBlock`.
        d_ext (`int`, *optional*, defaults to 4096):
            Size of the intermediate feed forward layer in each Extra-layers.
        d_spout (`int`, *optional*, defaults to 128):
            Size of the `spout` vector.
        num_switch_layers (`int`, *optional*, defaults to 10):
            Number of layers in the Switch Transformer layer.
        num_ext_layers (`int`, *optional*, defaults to 0):
            Number of layers in the Extra-layers.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_experts (`int`, *optional*, defaults to 16):
            Number of experts for each SwitchTransformer layer.
        expert_capacity (`int`, *optional*, defaults to 128):
            Number of tokens that can be stored in each expert. If set to 1, the model will behave like a regular
            Transformer.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        router_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the router.
        router_jitter_noise (`float`, *optional*, defaults to 0.0):
            Amount of noise to add to the router. Set it to 0.0 during prediction or set small value (usually 1e-2)
            during training.
        router_dtype (`str`, *optional*, default to `"float32"`):
            The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
            *selective precision* discussion in [the paper](https://arxiv.org/abs/2101.03961).
        router_ignore_padding_tokens (`bool`, *optional*, defaults to `False`):
            Whether to ignore padding tokens when routing.
        output_hidden_states (`bool`, *optional*, default to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers.
        initializer_factor (`float`, *optional*, defaults to 0.002):
            A factor for initializing all weight matrices.
        output_router_logits (`bool`, *optional*, default to `False`):
            Whether or not to return the router logits of all experts.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models)
    

## GPTSanJapaneseTokenizer


    This tokenizer is based on GPTNeoXJapaneseTokenizer and has the following modifications
    - Decoding byte0~byte255 tokens correctly
    - Added bagofword token handling
    - Return token_type_ids for Prefix-LM model
    The bagofword token represents a repetition of the previous token and is converted to 3 consecutive tokens when
    decoding In addition, the original Japanese special Sub-Word-Encoding has been released in this repository
    (https://github.com/tanreinama/Japanese-BPEEncoder_V2). The token_type_ids is a mask indicating the prefix input
    position of the Prefix-LM model. To specify a prefix position, specify a prefix input for prefix_text, or specify a
    sentence of the prefix part and the part after it as a text pair of batch input.

    Example:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> # You can confirm both 慶応 and 慶應 are encoded to 17750
    >>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
    [35993, 35998, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

    >>> # Both 慶応 and 慶應 are decoded to 慶応
    >>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
    '吾輩は猫である🐯。実は慶応(慶応)大学出身'
    ```

    Example for Prefix-LM:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["input_ids"]
    [35993, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 35998, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

    >>> # Mask for Prefix-LM inputs
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["token_type_ids"]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ```

    Example for batch encode:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["input_ids"]
    [[35993, 35998, 8640, 25948, 35993, 35998, 30647, 35675, 35999, 35999], [35993, 35998, 10382, 9868, 35993, 35998, 30646, 9459, 30646, 35675]]

    >>> # Mask for Prefix-LM inputs
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["token_type_ids"]
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    >>> # Mask for padding
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["attention_mask"]
    [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    ```

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        emoji_file (`str`):
            File containing the emoji.
        unk_token (`str`, *optional*, defaults to `"<|nottoken|>"`):
            The token used for unknown charactor
        pad_token (`str`, *optional*, defaults to `"<|separator|>"`):
            The token used for padding
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"<|segmenter|>"`):
            A special token to separate token to prefix part and general input part.
        do_clean_text (`bool`, *optional*, defaults to `False`):
            Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.
    

## GPTSanJapaneseModel

The bare GPTSAN-japanese Model transformer outputting raw hidden-states without any specific head on top.

    The [GPTSAN-japanese](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer
    based Japanese language model

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


## GPTSanJapaneseForConditionalGeneration

The bare GPTSAN-japanese Model with a language modeling head.

    The [GPTSAN-japanese](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer
    based Japanese language model

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
