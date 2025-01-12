<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MarianMT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=marian">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-marian-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/opus-mt-zh-en">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

A framework for translation models, using the same models as BART. Translations should be similar, but not identical to output in the test set linked to in each model card.
This model was contributed by [sshleifer](https://huggingface.co/sshleifer).


## Implementation Notes

- Each model is about 298 MB on disk, there are more than 1,000 models.
- The list of supported language pairs can be found [here](https://huggingface.co/Helsinki-NLP).
- Models were originally trained by [Jörg Tiedemann](https://researchportal.helsinki.fi/en/persons/j%C3%B6rg-tiedemann) using the [Marian](https://marian-nmt.github.io/) C++ library, which supports fast training and translation.
- All models are transformer encoder-decoders with 6 layers in each component. Each model's performance is documented
  in a model card.
- The 80 opus models that require BPE preprocessing are not supported.
- The modeling code is the same as [`BartForConditionalGeneration`] with a few minor modifications:

  - static (sinusoid) positional embeddings (`MarianConfig.static_position_embeddings=True`)
  - no layernorm_embedding (`MarianConfig.normalize_embedding=False`)
  - the model starts generating with `pad_token_id` (which has 0 as a token_embedding) as the prefix (Bart uses
    `<s/>`),
- Code to bulk convert models can be found in `convert_marian_to_pytorch.py`.


## Naming

- All model names use the following format: `Helsinki-NLP/opus-mt-{src}-{tgt}`
- The language codes used to name models are inconsistent. Two digit codes can usually be found [here](https://developers.google.com/admin-sdk/directory/v1/languages), three digit codes require googling "language
  code {code}".
- Codes formatted like `es_AR` are usually `code_{region}`. That one is Spanish from Argentina.
- The models were converted in two stages. The first 1000 models use ISO-639-2 codes to identify languages, the second
  group use a combination of ISO-639-5 codes and ISO-639-2 codes.


## Examples

- Since Marian models are smaller than many other translation models available in the library, they can be useful for
  fine-tuning experiments and integration tests.
- [Fine-tune on GPU](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/train_distil_marian_enro.sh)

## Multilingual Models

- All model names use the following format: `Helsinki-NLP/opus-mt-{src}-{tgt}`:
- If a model can output multiple languages, and you should specify a language code by prepending the desired output
  language to the `src_text`.
- You can see a models's supported language codes in its model card, under target constituents, like in [opus-mt-en-roa](https://huggingface.co/Helsinki-NLP/opus-mt-en-roa).
- Note that if a model is only multilingual on the source side, like `Helsinki-NLP/opus-mt-roa-en`, no language
  codes are required.

New multi-lingual models from the [Tatoeba-Challenge repo](https://github.com/Helsinki-NLP/Tatoeba-Challenge)
require 3 character language codes:

```python
>>> from transformers import MarianMTModel, MarianTokenizer

>>> src_text = [
...     ">>fra<< this is a sentence in english that we want to translate to french",
...     ">>por<< This should go to portuguese",
...     ">>esp<< And this to Spanish",
... ]

>>> model_name = "Helsinki-NLP/opus-mt-en-roa"
>>> tokenizer = MarianTokenizer.from_pretrained(model_name)
>>> print(tokenizer.supported_language_codes)
['>>zlm_Latn<<', '>>mfe<<', '>>hat<<', '>>pap<<', '>>ast<<', '>>cat<<', '>>ind<<', '>>glg<<', '>>wln<<', '>>spa<<', '>>fra<<', '>>ron<<', '>>por<<', '>>ita<<', '>>oci<<', '>>arg<<', '>>min<<']

>>> model = MarianMTModel.from_pretrained(model_name)
>>> translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
>>> [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
["c'est une phrase en anglais que nous voulons traduire en français",
 'Isto deve ir para o português.',
 'Y esto al español']
```

Here is the code to see all available pretrained models on the hub:

```python
from huggingface_hub import list_models

model_list = list_models()
org = "Helsinki-NLP"
model_ids = [x.id for x in model_list if x.id.startswith(org)]
suffix = [x.split("/")[1] for x in model_ids]
old_style_multi_models = [f"{org}/{s}" for s in suffix if s != s.lower()]
```

## Old Style Multi-Lingual Models

These are the old style multi-lingual models ported from the OPUS-MT-Train repo: and the members of each language
group:

```python no-style
['Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU',
 'Helsinki-NLP/opus-mt-ROMANCE-en',
 'Helsinki-NLP/opus-mt-SCANDINAVIA-SCANDINAVIA',
 'Helsinki-NLP/opus-mt-de-ZH',
 'Helsinki-NLP/opus-mt-en-CELTIC',
 'Helsinki-NLP/opus-mt-en-ROMANCE',
 'Helsinki-NLP/opus-mt-es-NORWAY',
 'Helsinki-NLP/opus-mt-fi-NORWAY',
 'Helsinki-NLP/opus-mt-fi-ZH',
 'Helsinki-NLP/opus-mt-fi_nb_no_nn_ru_sv_en-SAMI',
 'Helsinki-NLP/opus-mt-sv-NORWAY',
 'Helsinki-NLP/opus-mt-sv-ZH']
GROUP_MEMBERS = {
 'ZH': ['cmn', 'cn', 'yue', 'ze_zh', 'zh_cn', 'zh_CN', 'zh_HK', 'zh_tw', 'zh_TW', 'zh_yue', 'zhs', 'zht', 'zh'],
 'ROMANCE': ['fr', 'fr_BE', 'fr_CA', 'fr_FR', 'wa', 'frp', 'oc', 'ca', 'rm', 'lld', 'fur', 'lij', 'lmo', 'es', 'es_AR', 'es_CL', 'es_CO', 'es_CR', 'es_DO', 'es_EC', 'es_ES', 'es_GT', 'es_HN', 'es_MX', 'es_NI', 'es_PA', 'es_PE', 'es_PR', 'es_SV', 'es_UY', 'es_VE', 'pt', 'pt_br', 'pt_BR', 'pt_PT', 'gl', 'lad', 'an', 'mwl', 'it', 'it_IT', 'co', 'nap', 'scn', 'vec', 'sc', 'ro', 'la'],
 'NORTH_EU': ['de', 'nl', 'fy', 'af', 'da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
 'SCANDINAVIA': ['da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
 'SAMI': ['se', 'sma', 'smj', 'smn', 'sms'],
 'NORWAY': ['nb_NO', 'nb', 'nn_NO', 'nn', 'nog', 'no_nb', 'no'],
 'CELTIC': ['ga', 'cy', 'br', 'gd', 'kw', 'gv']
}
```

Example of translating english to many romance languages, using old-style 2 character language codes


```python
>>> from transformers import MarianMTModel, MarianTokenizer

>>> src_text = [
...     ">>fr<< this is a sentence in english that we want to translate to french",
...     ">>pt<< This should go to portuguese",
...     ">>es<< And this to Spanish",
... ]

>>> model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
>>> tokenizer = MarianTokenizer.from_pretrained(model_name)

>>> model = MarianMTModel.from_pretrained(model_name)
>>> translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
>>> tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
["c'est une phrase en anglais que nous voulons traduire en français", 
 'Isto deve ir para o português.',
 'Y esto al español']
```

## Resources

- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)
- [Causal language modeling task guide](../tasks/language_modeling)

## MarianConfig


    This is the configuration class to store the configuration of a [`MarianModel`]. It is used to instantiate an
    Marian model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Marian
    [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 58101):
            Vocabulary size of the Marian model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MarianModel`] or [`TFMarianModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models)
        forced_eos_token_id (`int`, *optional*, defaults to 0):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.

    Examples:

    ```python
    >>> from transformers import MarianModel, MarianConfig

    >>> # Initializing a Marian Helsinki-NLP/opus-mt-en-de style configuration
    >>> configuration = MarianConfig()

    >>> # Initializing a model from the Helsinki-NLP/opus-mt-en-de style configuration
    >>> model = MarianModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## MarianTokenizer


    Construct a Marian tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        source_spm (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary for the source language.
        target_spm (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary for the target language.
        source_lang (`str`, *optional*):
            A string representing the source language.
        target_lang (`str`, *optional*):
            A string representing the target language.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        model_max_length (`int`, *optional*, defaults to 512):
            The maximum sentence length the model accepts.
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<eop>", "<eod>"]`):
            Additional special tokens used by the tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Examples:

    ```python
    >>> from transformers import MarianForCausalLM, MarianTokenizer

    >>> model = MarianForCausalLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    >>> tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    >>> src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
    >>> tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
    >>> inputs = tokenizer(src_texts, text_target=tgt_texts, return_tensors="pt", padding=True)

    >>> outputs = model(**inputs)  # should work
    ```

Methods: build_inputs_with_special_tokens

<frameworkcontent>
<pt>

## MarianModel

The bare Marian Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MarianConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MarianMTModel

The Marian Model with a language modeling head. Can be used for summarization.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MarianConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MarianForCausalLM

No docstring available for MarianForCausalLM

Methods: forward

</pt>
<tf>

## TFMarianModel

No docstring available for TFMarianModel

Methods: call

## TFMarianMTModel

No docstring available for TFMarianMTModel

Methods: call

</tf>
<jax>

## FlaxMarianModel

No docstring available for FlaxMarianModel

Methods: __call__

## FlaxMarianMTModel

No docstring available for FlaxMarianMTModel

Methods: __call__

</jax>
</frameworkcontent>
