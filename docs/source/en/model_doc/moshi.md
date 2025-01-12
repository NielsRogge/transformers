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

# Moshi

## Overview

The Moshi model was proposed in [Moshi: a speech-text foundation model for real-time dialogue](https://kyutai.org/Moshi.pdf) by Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave and Neil Zeghidour.

Moshi is a speech-text foundation model that casts spoken dialogue as speech-to-speech generation. Starting from a text language model backbone, Moshi generates speech as tokens from the residual quantizer of a neural audio codec, while modeling separately its own speech and that of the user into parallel streams. This allows for the removal of explicit speaker turns, and the modeling of arbitrary conversational dynamics. Moshi also predicts time-aligned text tokens as a prefix to audio tokens. This “Inner Monologue” method significantly improves the linguistic quality of generated speech and provides streaming speech recognition and text-to-speech. As a result, Moshi is the first real-time full-duplex spoken large language model, with a theoretical latency of 160ms, 200ms in practice.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ylacombe/benchmark-comparison/resolve/main/moshi_architecture.png">
</div>

The abstract from the paper is the following:

*We introduce Moshi, a speech-text foundation model and full-duplex spoken dialogue framework. Current systems for spoken dialogue rely on pipelines of independent components, namely voice activity detection, speech recognition, textual dialogue and text-to-speech. Such frameworks cannot emulate the experience of real conversations. First, their complexity induces a latency of several seconds between interactions. Second, text being the intermediate modality for dialogue, non-linguistic information that modifies meaning— such as emotion or non-speech sounds— is lost in the interaction. Finally, they rely on a segmentation into speaker turns, which does not take into account overlapping speech, interruptions and interjections. Moshi solves these independent issues altogether by casting spoken dialogue as speech-to-speech generation. Starting from a text language model backbone, Moshi generates speech as tokens from the residual quantizer of a neural audio codec, while modeling separately its own speech and that of the user into parallel streams. This allows for the removal of explicit speaker turns, and the modeling of arbitrary conversational dynamics. We moreover extend the hierarchical semantic-to-acoustic token generation of previous work to first predict time-aligned text tokens as a prefix to audio tokens. Not only this “Inner Monologue” method significantly improves the linguistic quality of generated speech, but we also illustrate how it can provide streaming speech recognition and text-to-speech. Our resulting model is the first real-time full-duplex spoken large language model, with a theoretical latency of 160ms, 200ms in practice, and is available at github.com/kyutai-labs/moshi.* 

Moshi deals with 3 streams of information:
1. The user's audio
2. Moshi's audio
3. Moshi's textual output

Similarly to [`~MusicgenModel`], audio is represented with audio codebooks, which can be interpreted like tokens. The main difference between text tokens and audio codebooks is that audio codebooks introduce an additional dimension of information.
Text tokens are typically of dim `(batch_size, sequence_length)` but audio tokens are of dim `(batch_size, num_codebooks, sequence_length)`.

Moshi's made of 3 components:

**1. The main decoder (Helium in the paper)**

It corresponds to [`MoshiForCausalLM`]. It is strictly a classic text LLM, that uses an architecture similar to [` ~GemmaForCausalLM`]. In other words, it takes text tokens, embeds them, pass them through the decoder and a language head, to get text logits.

**2. The depth decoder**

On its own, it's also a classic LLM, but this time, instead of generating over the time dimension, it generates over the codebook dimension.

It also means that its context length is `num_codebooks`, thus it can't generate more than `num_codebooks`.

Note that each timestamp - i.e each codebook - gets its own set of Linear Layers and Embeddings.

**3. [`MimiModel`]**

It's the audio encoder from Kyutai, that has recently been integrated to transformers, which is used to "tokenize" audio. It has the same use that [`~EncodecModel`] has in [`~MusicgenModel`].


## Tips:

The original checkpoints can be converted using the conversion script `src/transformers/models/moshi/convert_moshi_transformers.py` 


### How to use the model:

This implementation has two main aims:
1. quickly test model generation by simplifying the original API
2. simplify training. A training guide will come soon, but user contributions are welcomed!

<Tip>

It is designed for intermediate use. We strongly recommend using the original [implementation](https://github.com/kyutai-labs/moshi) to infer the model in real-time streaming.

</Tip>

**1. Model generation**

Moshi is a streaming auto-regressive model with two streams of audio. To put it differently, one audio stream corresponds to what the model said/will say and the other audio stream corresponds to what the user said/will say.

[`MoshiForConditionalGeneration.generate`] thus needs 3 inputs:
1. `input_ids` - corresponding to the text token history
2. `moshi_input_values` or `moshi_audio_codes`- corresponding to the model audio history
3. `user_input_values` or `user_audio_codes` - corresponding to the user audio history

These three inputs must be synchronized. Meaning that their lengths must correspond to the same number of tokens.

You can dynamically use the 3 inputs depending on what you want to test:
1. Simply check the model response to an user prompt - in that case, `input_ids` can be filled with pad tokens and `user_input_values` can be a zero tensor of the same shape than the user prompt.
2. Test more complex behaviour - in that case, you must be careful about how the input tokens are synchronized with the audios.

<Tip>

The original model is synchronized text with audio by padding the text in between each token enunciation.

To follow the example of the following image, `"Hello, I'm Moshi"` could be transformed to `"Hello,<pad><unk>I'm Moshi"`.

</Tip>

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ylacombe/benchmark-comparison/resolve/main/moshi_text_sync.png">
</div>


[`MoshiForConditionalGeneration.generate`] then auto-regressively feeds to itself its own audio stream, but since it doesn't have access to the user input stream while using `transformers`, it will thus **assume that the user is producing blank audio**.



```python 
>>> from datasets import load_dataset, Audio
>>> import torch, math
>>> from transformers import MoshiForConditionalGeneration, AutoFeatureExtractor, AutoTokenizer
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


>>> # prepare user input audio 
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> user_input_values = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(device=device, dtype=dtype)

>>> # prepare moshi input values - we suppose moshi didn't say anything while the user spoke
>>> moshi_input_values = torch.zeros_like(user_input_values.input_values)

>>> # prepare moshi input ids - we suppose moshi didn't say anything while the user spoke
>>> num_tokens = math.ceil(moshi_input_values.shape[-1] * waveform_to_token_ratio)
>>> input_ids = torch.ones((1, num_tokens), device=device, dtype=torch.int64) * tokenizer.encode("<pad>")[0]

>>> # generate 25 new tokens (around 2s of audio)
>>> output = model.generate(input_ids=input_ids, user_input_values=user_input_values.input_values, moshi_input_values=moshi_input_values, max_new_tokens=25)

>>> text_tokens = output.sequences
>>> audio_waveforms = output.audio_sequences
```

**2. Model training**

Most of the work has to be done during data creation/pre-processing, because of the need to align/synchronize streams.

Once it's done, you can simply forward `text_labels` and `audio_labels` to [`MoshiForConditionalGeneration.forward`], alongside the usual inputs, to get the model loss.
 
A training guide will come soon, but user contributions are welcomed!

### How does the model forward the inputs / generate:

1. The input streams are embedded and combined into `inputs_embeds`.

2. `inputs_embeds` is passed through the main decoder, which processes it like a normal LLM would.

3. The main decoder outputs `text logits` but also its `last hidden state` which is called `temporal context` in the paper.

3. The depth decoder switches the dimension on which we forward / generate (codebooks instead of time). It uses the token generated from `text logits`  and the `temporal context` to auto-regressively generate audio codebooks.


This model was contributed by [Yoach Lacombe (ylacombe)](https://huggingface.co/ylacombe).

The original code can be found [here](https://github.com/kyutai-labs/moshi).



## MoshiConfig


    This is the configuration class to store the configuration of a [`MoshiModel`]. It is used to instantiate a
    Moshi model according to the specified arguments, defining the audio encoder, Moshi depth decoder and Moshi decoder
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the Moshiko model,
    e.g. [kmhf/hf-moshiko](https://huggingface.co/kmhf/hf-moshiko)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the MoshiDecoder model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`MoshiDecoder`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the layers and the pooler layer of the main decoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the main decoder block.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `num_attention_heads`.
        audio_vocab_size (`int`, *optional*):
            Vocabulary size of the audio part of model. Defines the number of different tokens that can be
            represented by the `audio_codes` passed when calling the Moshi models.
        max_position_embeddings (`int`, *optional*, defaults to 3000):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        head_dim (`int`, *optional*, defaults to `hidden_size // num_attention_heads`):
            The attention head dimension.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        sliding_window (`int`, *optional*, defaults to 3000):
            Sliding window attention window size. If not specified, will default to `3000`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        ffn_dim (`int`, *optional*, defaults to 22528):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the main decoder block. Must be even.
        rms_norm_eps (`float`, *optional*, defaults to 1e-08):
            The epsilon used by the rms normalization layers.
        num_codebooks (`int`, *optional*, defaults to 8):
            The number of audio codebooks for each audio channels.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:
                - **audio_encoder_config** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the audio encoder config.
                - **depth__config** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the depth decoder config.


    Example:

    ```python
    >>> from transformers import (
    ...     MoshiConfig,
    ...     MoshiForConditionalGeneration,
    ... )

    >>> configuration = MoshiConfig()

    >>> # Initializing a MoshiForConditionalGeneration (with random weights) from the kmhf/hf-moshiko style configuration
    >>> model = MoshiForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("kmhf/hf-moshiko")

    >>> # loading model and config from pretrained folder
    >>> moshi_config = MoshiConfig.from_pretrained("kmhf/hf-moshiko")
    >>> model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko", config=moshi_config)
    ```

## MoshiDepthConfig


    This is the configuration class to store the configuration of a [`MoshiDepthDecoder`]. It is used to instantiate a
    Moshi depth decoder model according to the specified arguments, defining the Moshi depth decoder config.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the MoshiDepthDecoder model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`MoshiDepthDecoder`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer of the depth decoder.
        input_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the input hidden states. Used to connect the main decoder to the depth decoder.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of depth decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the depth decoder block.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `num_attention_heads`.
        audio_vocab_size (`int`, *optional*, defaults to 2048):
            Vocabulary size of the audio part of model. Defines the number of different tokens that can be
            represented by the `audio_codes` passed when calling the Moshi models.
        max_position_embeddings (`int`, *optional*, defaults to 9):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the depth decoder.
        head_dim (`int`, *optional*, defaults to `hidden_size // num_attention_heads`):
            The attention head dimension.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        sliding_window (`int`, *optional*, defaults to 8):
            Sliding window attention window size. If not specified, will default to `8`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        ffn_dim (`int`, *optional*, defaults to 5632):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the depth decoder block. Must be even.
        rms_norm_eps (`float`, *optional*, defaults to 1e-08):
            The epsilon used by the rms normalization layers.
        num_codebooks (`int`, *optional*, defaults to 8):
            The number of audio codebooks for each audio channels.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:
                - **audio_encoder_config** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the audio encoder config.

    Example:

    ```python
    >>> from transformers import (
    ...     MoshiDepthConfig,
    ...     MoshiDepthDecoder,
    ... )

    >>> configuration = MoshiDepthConfig()

    >>> # Initializing a MoshiDepthDecoder (with random weights) from the kmhf/hf-moshiko style configuration
    >>> model = MoshiDepthDecoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## MoshiModel

The bare Moshi Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MoshiConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MoshiDecoderLayer`]

    Args:
        config: MoshiConfig
    

Methods: forward

## MoshiForCausalLM

The Moshi decoder model with a text language modelling head on top. Only usable for text.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MoshiConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MoshiForConditionalGeneration

The original Moshi model with an audio encoder, a Moshi depth decoder and a Moshi decoder, for speech-to-speech.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MoshiConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - generate
    - get_unconditional_inputs
