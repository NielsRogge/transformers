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

# MusicGen

## Overview

The MusicGen model was proposed in the paper [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)
by Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi and Alexandre Défossez.

MusicGen is a single stage auto-regressive Transformer model capable of generating high-quality music samples conditioned
on text descriptions or audio prompts. The text descriptions are passed through a frozen text encoder model to obtain a
sequence of hidden-state representations. MusicGen is then trained to predict discrete audio tokens, or *audio codes*,
conditioned on these hidden-states. These audio tokens are then decoded using an audio compression model, such as EnCodec,
to recover the audio waveform.

Through an efficient token interleaving pattern, MusicGen does not require a self-supervised semantic representation of
the text/audio prompts, thus eliminating the need to cascade multiple models to predict a set of codebooks (e.g.
hierarchically or upsampling). Instead, it is able to generate all the codebooks in a single forward pass.

The abstract from the paper is the following:

*We tackle the task of conditional music generation. We introduce MusicGen, a single Language Model (LM) that operates
over several streams of compressed discrete music representation, i.e., tokens. Unlike prior work, MusicGen is comprised
of a single-stage transformer LM together with efficient token interleaving patterns, which eliminates the need for
cascading several models, e.g., hierarchically or upsampling. Following this approach, we demonstrate how MusicGen
can generate high-quality samples, while being conditioned on textual description or melodic features, allowing better
controls over the generated output. We conduct extensive empirical evaluation, considering both automatic and human
studies, showing the proposed approach is superior to the evaluated baselines on a standard text-to-music benchmark.
Through ablation studies, we shed light over the importance of each of the components comprising MusicGen.*

This model was contributed by [sanchit-gandhi](https://huggingface.co/sanchit-gandhi). The original code can be found
[here](https://github.com/facebookresearch/audiocraft). The pre-trained checkpoints can be found on the
[Hugging Face Hub](https://huggingface.co/models?sort=downloads&search=facebook%2Fmusicgen-).

## Usage tips

- After downloading the original checkpoints from [here](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md#importing--exporting-models) , you can convert them using the **conversion script** available at
`src/transformers/models/musicgen/convert_musicgen_transformers.py` with the following command:

```bash
python src/transformers/models/musicgen/convert_musicgen_transformers.py \
    --checkpoint small --pytorch_dump_folder /output/path --safe_serialization 
```

## Generation

MusicGen is compatible with two generation modes: greedy and sampling. In practice, sampling leads to significantly
better results than greedy, thus we encourage sampling mode to be used where possible. Sampling is enabled by default,
and can be explicitly specified by setting `do_sample=True` in the call to [`MusicgenForConditionalGeneration.generate`],
or by overriding the model's generation config (see below).

Generation is limited by the sinusoidal positional embeddings to 30 second inputs. Meaning, MusicGen cannot generate more
than 30 seconds of audio (1503 tokens), and input audio passed by Audio-Prompted Generation contributes to this limit so,
given an input of 20 seconds of audio, MusicGen cannot generate more than 10 seconds of additional audio.

Transformers supports both mono (1-channel) and stereo (2-channel) variants of MusicGen. The mono channel versions 
generate a single set of codebooks. The stereo versions generate 2 sets of codebooks, 1 for each channel (left/right), 
and each set of codebooks is decoded independently through the audio compression model. The audio streams for each 
channel are combined to give the final stereo output.

### Unconditional Generation

The inputs for unconditional (or 'null') generation can be obtained through the method
[`MusicgenForConditionalGeneration.get_unconditional_inputs`]:

```python
>>> from transformers import MusicgenForConditionalGeneration

>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
>>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)

>>> audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
```

The audio outputs are a three-dimensional Torch tensor of shape `(batch_size, num_channels, sequence_length)`. To listen
to the generated audio samples, you can either play them in an ipynb notebook:

```python
from IPython.display import Audio

sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].numpy(), rate=sampling_rate)
```

Or save them as a `.wav` file using a third-party library, e.g. `scipy`:

```python
>>> import scipy

>>> sampling_rate = model.config.audio_encoder.sampling_rate
>>> scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
```

### Text-Conditional Generation

The model can generate an audio sample conditioned on a text prompt through use of the [`MusicgenProcessor`] to pre-process
the inputs:

```python
>>> from transformers import AutoProcessor, MusicgenForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> inputs = processor(
...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

The `guidance_scale` is used in classifier free guidance (CFG), setting the weighting between the conditional logits
(which are predicted from the text prompts) and the unconditional logits (which are predicted from an unconditional or
'null' prompt). Higher guidance scale encourages the model to generate samples that are more closely linked to the input
prompt, usually at the expense of poorer audio quality. CFG is enabled by setting `guidance_scale > 1`. For best results,
use `guidance_scale=3` (default).

### Audio-Prompted Generation

The same [`MusicgenProcessor`] can be used to pre-process an audio prompt that is used for audio continuation. In the
following example, we load an audio file using the 🤗 Datasets library, which can be pip installed through the command
below:

```bash
pip install --upgrade pip
pip install datasets[audio]
```

```python
>>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
>>> sample = next(iter(dataset))["audio"]

>>> # take the first half of the audio sample
>>> sample["array"] = sample["array"][: len(sample["array"]) // 2]

>>> inputs = processor(
...     audio=sample["array"],
...     sampling_rate=sample["sampling_rate"],
...     text=["80s blues track with groovy saxophone"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

For batched audio-prompted generation, the generated `audio_values` can be post-processed to remove padding by using the
[`MusicgenProcessor`] class:

```python
>>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
>>> sample = next(iter(dataset))["audio"]

>>> # take the first quarter of the audio sample
>>> sample_1 = sample["array"][: len(sample["array"]) // 4]

>>> # take the first half of the audio sample
>>> sample_2 = sample["array"][: len(sample["array"]) // 2]

>>> inputs = processor(
...     audio=[sample_1, sample_2],
...     sampling_rate=sample["sampling_rate"],
...     text=["80s blues track with groovy saxophone", "90s rock song with loud guitars and heavy drums"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

>>> # post-process to remove padding from the batched audio
>>> audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)
```

### Generation Configuration

The default parameters that control the generation process, such as sampling, guidance scale and number of generated 
tokens, can be found in the model's generation config, and updated as desired:

```python
>>> from transformers import MusicgenForConditionalGeneration

>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> # inspect the default generation config
>>> model.generation_config

>>> # increase the guidance scale to 4.0
>>> model.generation_config.guidance_scale = 4.0

>>> # decrease the max length to 256 tokens
>>> model.generation_config.max_length = 256
```

Note that any arguments passed to the generate method will **supersede** those in the generation config, so setting 
`do_sample=False` in the call to generate will supersede the setting of `model.generation_config.do_sample` in the 
generation config.

## Model Structure

The MusicGen model can be de-composed into three distinct stages:
1. Text encoder: maps the text inputs to a sequence of hidden-state representations. The pre-trained MusicGen models use a frozen text encoder from either T5 or Flan-T5
2. MusicGen decoder: a language model (LM) that auto-regressively generates audio tokens (or codes) conditional on the encoder hidden-state representations
3. Audio encoder/decoder: used to encode an audio prompt to use as prompt tokens, and recover the audio waveform from the audio tokens predicted by the decoder

Thus, the MusicGen model can either be used as a standalone decoder model, corresponding to the class [`MusicgenForCausalLM`],
or as a composite model that includes the text encoder and audio encoder/decoder, corresponding to the class
[`MusicgenForConditionalGeneration`]. If only the decoder needs to be loaded from the pre-trained checkpoint, it can be loaded by first 
specifying the correct config, or be accessed through the `.decoder` attribute of the composite model:

```python
>>> from transformers import AutoConfig, MusicgenForCausalLM, MusicgenForConditionalGeneration

>>> # Option 1: get decoder config and pass to `.from_pretrained`
>>> decoder_config = AutoConfig.from_pretrained("facebook/musicgen-small").decoder
>>> decoder = MusicgenForCausalLM.from_pretrained("facebook/musicgen-small", **decoder_config)

>>> # Option 2: load the entire composite model, but only return the decoder
>>> decoder = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").decoder
```

Since the text encoder and audio encoder/decoder models are frozen during training, the MusicGen decoder [`MusicgenForCausalLM`]
can be trained standalone on a dataset of encoder hidden-states and audio codes. For inference, the trained decoder can
be combined with the frozen text encoder and audio encoder/decoders to recover the composite [`MusicgenForConditionalGeneration`]
model.

Tips:
* MusicGen is trained on the 32kHz checkpoint of Encodec. You should ensure you use a compatible version of the Encodec model.
* Sampling mode tends to deliver better results than greedy - you can toggle sampling with the variable `do_sample` in the call to [`MusicgenForConditionalGeneration.generate`]

## MusicgenDecoderConfig


    This is the configuration class to store the configuration of an [`MusicgenDecoder`]. It is used to instantiate a
    MusicGen decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MusicGen
    [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            Vocabulary size of the MusicgenDecoder model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`MusicgenDecoder`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer block.
        ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer block.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, text_encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_factor (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(hidden_size).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models)
        num_codebooks (`int`, *optional*, defaults to 4):
            The number of parallel codebooks forwarded to the model.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether input and output word embeddings should be tied.
        audio_channels (`int`, *optional*, defaults to 1
            Number of channels in the audio data. Either 1 for mono or 2 for stereo. Stereo models generate a separate
            audio stream for the left/right output channels. Mono models generate a single audio stream output.
    

## MusicgenConfig


    This is the configuration class to store the configuration of a [`MusicgenModel`]. It is used to instantiate a
    MusicGen model according to the specified arguments, defining the text encoder, audio encoder and MusicGen decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **text_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the text encoder config.
                - **audio_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the audio encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Example:

    ```python
    >>> from transformers import (
    ...     MusicgenConfig,
    ...     MusicgenDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     MusicgenForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = MusicgenDecoderConfig()

    >>> configuration = MusicgenConfig.from_sub_models_config(
    ...     text_encoder_config, audio_encoder_config, decoder_config
    ... )

    >>> # Initializing a MusicgenForConditionalGeneration (with random weights) from the facebook/musicgen-small style configuration
    >>> model = MusicgenForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("musicgen-model")

    >>> # loading model and config from pretrained folder
    >>> musicgen_config = MusicgenConfig.from_pretrained("musicgen-model")
    >>> model = MusicgenForConditionalGeneration.from_pretrained("musicgen-model", config=musicgen_config)
    ```

## MusicgenProcessor


    Constructs a MusicGen processor which wraps an EnCodec feature extractor and a T5 tokenizer into a single processor
    class.

    [`MusicgenProcessor`] offers all the functionalities of [`EncodecFeatureExtractor`] and [`TTokenizer`]. See
    [`~MusicgenProcessor.__call__`] and [`~MusicgenProcessor.decode`] for more information.

    Args:
        feature_extractor (`EncodecFeatureExtractor`):
            An instance of [`EncodecFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`T5Tokenizer`):
            An instance of [`T5Tokenizer`]. The tokenizer is a required input.
    

## MusicgenModel

The bare Musicgen decoder model outputting raw hidden-states without any specific head on top.

    The Musicgen model was proposed in [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by
    Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez. It is an
    encoder decoder transformer trained on the task of conditional music generation

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MusicgenConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MusicgenForCausalLM

The MusicGen decoder model with a language modelling head on top.

    The Musicgen model was proposed in [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by
    Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez. It is an
    encoder decoder transformer trained on the task of conditional music generation

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MusicgenConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MusicgenForConditionalGeneration

The composite MusicGen model with a text encoder, audio encoder and Musicgen decoder, for music generation tasks with one or both of text and audio prompts.

    The Musicgen model was proposed in [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by
    Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez. It is an
    encoder decoder transformer trained on the task of conditional music generation

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MusicgenConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
