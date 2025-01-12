<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

:warning: Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MusicGen Melody

## Overview

The MusicGen Melody model was proposed in [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi and Alexandre Défossez.

MusicGen Melody is a single stage auto-regressive Transformer model capable of generating high-quality music samples conditioned on text descriptions or audio prompts. The text descriptions are passed through a frozen text encoder model to obtain a sequence of hidden-state representations. MusicGen is then trained to predict discrete audio tokens, or *audio codes*, conditioned on these hidden-states. These audio tokens are then decoded using an audio compression model, such as EnCodec, to recover the audio waveform.

Through an efficient token interleaving pattern, MusicGen does not require a self-supervised semantic representation of the text/audio prompts, thus eliminating the need to cascade multiple models to predict a set of codebooks (e.g. hierarchically or upsampling). Instead, it is able to generate all the codebooks in a single forward pass.

The abstract from the paper is the following:

*We tackle the task of conditional music generation. We introduce MusicGen, a single Language Model (LM) that operates over several streams of compressed discrete music representation, i.e., tokens. Unlike prior work, MusicGen is comprised of a single-stage transformer LM together with efficient token interleaving patterns, which eliminates the need for cascading several models, e.g., hierarchically or upsampling. Following this approach, we demonstrate how MusicGen can generate high-quality samples, while being conditioned on textual description or melodic features, allowing better controls over the generated output. We conduct extensive empirical evaluation, considering both automatic and human studies, showing the proposed approach is superior to the evaluated baselines on a standard text-to-music benchmark. Through ablation studies, we shed light over the importance of each of the components comprising MusicGen.*


This model was contributed by [ylacombe](https://huggingface.co/ylacombe). The original code can be found [here](https://github.com/facebookresearch/audiocraft). The pre-trained checkpoints can be found on the [Hugging Face Hub](https://huggingface.co/models?sort=downloads&search=facebook%2Fmusicgen).


## Difference with [MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen)

There are two key differences with MusicGen:
1. The audio prompt is used here as a conditional signal for the generated audio sample, whereas it's used for audio continuation in [MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen).
2. Conditional text and audio signals are concatenated to the decoder's hidden states instead of being used as a cross-attention signal, as in MusicGen.

## Generation

MusicGen Melody is compatible with two generation modes: greedy and sampling. In practice, sampling leads to significantly better results than greedy, thus we encourage sampling mode to be used where possible. Sampling is enabled by default, and can be explicitly specified by setting `do_sample=True` in the call to [`MusicgenMelodyForConditionalGeneration.generate`], or by overriding the model's generation config (see below).

Transformers supports both mono (1-channel) and stereo (2-channel) variants of MusicGen Melody. The mono channel versions generate a single set of codebooks. The stereo versions generate 2 sets of codebooks, 1 for each channel (left/right), and each set of codebooks is decoded independently through the audio compression model. The audio streams for each channel are combined to give the final stereo output.


#### Audio Conditional Generation

The model can generate an audio sample conditioned on a text and an audio prompt through use of the [`MusicgenMelodyProcessor`] to pre-process the inputs.

In the following examples, we load an audio file using the 🤗 Datasets library, which can be pip installed through the command below:

```
pip install --upgrade pip
pip install datasets[audio]
```

The audio file we are about to use is loaded as follows:
```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
>>> sample = next(iter(dataset))["audio"]
```

The audio prompt should ideally be free of the low-frequency signals usually produced by instruments such as drums and bass. The [Demucs](https://github.com/adefossez/demucs/tree/main) model can be used to separate vocals and other signals from the drums and bass components.

If you wish to use Demucs, you first need to follow the installation steps [here](https://github.com/adefossez/demucs/tree/main?tab=readme-ov-file#for-musicians) before using the following snippet:

```python
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio
import torch


wav = torch.tensor(sample["array"]).to(torch.float32)

demucs = pretrained.get_model('htdemucs')

wav = convert_audio(wav[None], sample["sampling_rate"], demucs.samplerate, demucs.audio_channels)
wav = apply_model(demucs, wav[None])
```

You can then use the following snippet to generate music:

```python
>>> from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

>>> inputs = processor(
...     audio=wav,
...     sampling_rate=demucs.samplerate,
...     text=["80s blues track with groovy saxophone"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

You can also pass the audio signal directly without using Demucs, although the quality of the generation will probably be degraded:

```python
>>> from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

>>> inputs = processor(
...     audio=sample["array"],
...     sampling_rate=sample["sampling_rate"],
...     text=["80s blues track with groovy saxophone"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

The audio outputs are a three-dimensional Torch tensor of shape `(batch_size, num_channels, sequence_length)`. To listen to the generated audio samples, you can either play them in an ipynb notebook:

```python
from IPython.display import Audio

sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].numpy(), rate=sampling_rate)
```

Or save them as a `.wav` file using a third-party library, e.g. `soundfile`:

```python
>>> import soundfile as sf

>>> sampling_rate = model.config.audio_encoder.sampling_rate
>>> sf.write("musicgen_out.wav", audio_values[0].T.numpy(), sampling_rate)
```


### Text-only Conditional Generation

The same [`MusicgenMelodyProcessor`] can be used to pre-process a text-only prompt. 

```python
>>> from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

>>> inputs = processor(
...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

The `guidance_scale` is used in classifier free guidance (CFG), setting the weighting between the conditional logits (which are predicted from the text prompts) and the unconditional logits (which are predicted from an unconditional or 'null' prompt). Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer audio quality. CFG is enabled by setting `guidance_scale > 1`. For best results, use `guidance_scale=3` (default).


You can also generate in batch:

```python
>>> from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

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
```

### Unconditional Generation

The inputs for unconditional (or 'null') generation can be obtained through the method [`MusicgenMelodyProcessor.get_unconditional_inputs`]:

```python
>>> from transformers import MusicgenMelodyForConditionalGeneration, MusicgenMelodyProcessor

>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
>>> unconditional_inputs = MusicgenMelodyProcessor.from_pretrained("facebook/musicgen-melody").get_unconditional_inputs(num_samples=1)

>>> audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
```

### Generation Configuration

The default parameters that control the generation process, such as sampling, guidance scale and number of generated tokens, can be found in the model's generation config, and updated as desired:

```python
>>> from transformers import MusicgenMelodyForConditionalGeneration

>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

>>> # inspect the default generation config
>>> model.generation_config

>>> # increase the guidance scale to 4.0
>>> model.generation_config.guidance_scale = 4.0

>>> # decrease the max length to 256 tokens
>>> model.generation_config.max_length = 256
```

Note that any arguments passed to the generate method will **supersede** those in the generation config, so setting `do_sample=False` in the call to generate will supersede the setting of `model.generation_config.do_sample` in the generation config.

## Model Structure

The MusicGen model can be de-composed into three distinct stages:
1. Text encoder: maps the text inputs to a sequence of hidden-state representations. The pre-trained MusicGen models use a frozen text encoder from either T5 or Flan-T5.
2. MusicGen Melody decoder: a language model (LM) that auto-regressively generates audio tokens (or codes) conditional on the encoder hidden-state representations
3. Audio decoder: used to recover the audio waveform from the audio tokens predicted by the decoder.

Thus, the MusicGen model can either be used as a standalone decoder model, corresponding to the class [`MusicgenMelodyForCausalLM`], or as a composite model that includes the text encoder and audio encoder, corresponding to the class [`MusicgenMelodyForConditionalGeneration`]. If only the decoder needs to be loaded from the pre-trained checkpoint, it can be loaded by first specifying the correct config, or be accessed through the `.decoder` attribute of the composite model:

```python
>>> from transformers import AutoConfig, MusicgenMelodyForCausalLM, MusicgenMelodyForConditionalGeneration

>>> # Option 1: get decoder config and pass to `.from_pretrained`
>>> decoder_config = AutoConfig.from_pretrained("facebook/musicgen-melody").decoder
>>> decoder = MusicgenMelodyForCausalLM.from_pretrained("facebook/musicgen-melody", **decoder_config.to_dict())

>>> # Option 2: load the entire composite model, but only return the decoder
>>> decoder = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody").decoder
```

Since the text encoder and audio encoder models are frozen during training, the MusicGen decoder [`MusicgenMelodyForCausalLM`] can be trained standalone on a dataset of encoder hidden-states and audio codes. For inference, the trained decoder can be combined with the frozen text encoder and audio encoder to recover the composite [`MusicgenMelodyForConditionalGeneration`] model.

## Checkpoint Conversion

- After downloading the original checkpoints from [here](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md#importing--exporting-models), you can convert them using the **conversion script** available at `src/transformers/models/musicgen_melody/convert_musicgen_melody_transformers.py` with the following command:

```bash
python src/transformers/models/musicgen_melody/convert_musicgen_melody_transformers.py \
    --checkpoint="facebook/musicgen-melody" --pytorch_dump_folder /output/path 
```

Tips:
* MusicGen is trained on the 32kHz checkpoint of Encodec. You should ensure you use a compatible version of the Encodec model.
* Sampling mode tends to deliver better results than greedy - you can toggle sampling with the variable `do_sample` in the call to [`MusicgenMelodyForConditionalGeneration.generate`]


## MusicgenMelodyDecoderConfig


    This is the configuration class to store the configuration of an [`MusicgenMelodyDecoder`]. It is used to instantiate a
    Musicgen Melody decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Musicgen Melody
    [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            Vocabulary size of the MusicgenMelodyDecoder model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`MusicgenMelodyDecoder`].
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of decoder layers.
        ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer block.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer block.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models)
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, text_encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        initializer_factor (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(hidden_size).
        num_codebooks (`int`, *optional*, defaults to 4):
            The number of parallel codebooks forwarded to the model.
        audio_channels (`int`, *optional*, defaults to 1):
            Number of audio channels used by the model (either mono or stereo). Stereo models generate a separate
            audio stream for the left/right output channels. Mono models generate a single audio stream output.
        pad_token_id (`int`, *optional*, defaults to 2048): The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to 2048): The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*): The id of the *end-of-sequence* token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`): Whether to tie word embeddings with the text encoder.
    

## MusicgenMelodyProcessor


    Constructs a MusicGen Melody processor which wraps a Wav2Vec2 feature extractor - for raw audio waveform processing - and a T5 tokenizer into a single processor
    class.

    [`MusicgenProcessor`] offers all the functionalities of [`MusicgenMelodyFeatureExtractor`] and [`T5Tokenizer`]. See
    [`~MusicgenProcessor.__call__`] and [`~MusicgenProcessor.decode`] for more information.

    Args:
        feature_extractor (`MusicgenMelodyFeatureExtractor`):
            An instance of [`MusicgenMelodyFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`T5Tokenizer`):
            An instance of [`T5Tokenizer`]. The tokenizer is a required input.
    

Methods: get_unconditional_inputs

## MusicgenMelodyFeatureExtractor


    Constructs a MusicgenMelody feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts chroma features from audio processed by [Demucs](https://github.com/adefossez/demucs/tree/main) or
    directly from raw audio waveform.

    Args:
        feature_size (`int`, *optional*, defaults to 12):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 32000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 4096):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chunks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, *optional*, defaults to 16384):
            Size of the Fourier transform.
        num_chroma (`int`, *optional*, defaults to 12):
            Number of chroma bins to use.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the attention mask. Can be overwritten when calling the feature extractor.

            [What are attention masks?](../glossary#attention-mask)

            <Tip>

            For Whisper models, `attention_mask` should always be passed for batched inference, to avoid subtle
            bugs.

            </Tip>
        stem_indices (`List[int]`, *optional*, defaults to `[3, 2]`):
            Stem channels to extract if demucs outputs are passed.
    

## MusicgenMelodyConfig


    This is the configuration class to store the configuration of a [`MusicgenMelodyModel`]. It is used to instantiate a
    Musicgen Melody model according to the specified arguments, defining the text encoder, audio encoder and Musicgen Melody decoder
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the Musicgen Melody
    [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_chroma (`int`, *optional*, defaults to 12): Number of chroma bins to use.
        chroma_length (`int`, *optional*, defaults to 235):
            Maximum chroma duration if audio is used to condition the model. Corresponds to the maximum duration used during training.
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
    ...     MusicgenMelodyConfig,
    ...     MusicgenMelodyDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     MusicgenMelodyForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = MusicgenMelodyDecoderConfig()

    >>> configuration = MusicgenMelodyConfig.from_sub_models_config(
    ...     text_encoder_config, audio_encoder_config, decoder_config
    ... )

    >>> # Initializing a MusicgenMelodyForConditionalGeneration (with random weights) from the facebook/musicgen-melody style configuration
    >>> model = MusicgenMelodyForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("musicgen_melody-model")

    >>> # loading model and config from pretrained folder
    >>> musicgen_melody_config = MusicgenMelodyConfig.from_pretrained("musicgen_melody-model")
    >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("musicgen_melody-model", config=musicgen_melody_config)
    ```

## MusicgenMelodyModel

The bare MusicgenMelody decoder model outputting raw hidden-states without any specific head on top.

    The Musicgen Melody model was proposed in [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by
    Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez. It is a
    decoder-only transformer trained on the task of conditional music generation.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MusicgenMelodyConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MusicgenMelodyForCausalLM

The Musicgen Melody decoder model with a language modelling head on top.

    The Musicgen Melody model was proposed in [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by
    Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez. It is a
    decoder-only transformer trained on the task of conditional music generation.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MusicgenMelodyConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MusicgenMelodyForConditionalGeneration

The composite Musicgen Melody model with a text and audio conditional models, a MusicgenMelody decoder and an audio encoder, for music generation tasks with one or both of text and audio prompts.

    The Musicgen Melody model was proposed in [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by
    Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez. It is a
    decoder-only transformer trained on the task of conditional music generation.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MusicgenMelodyConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

        text_encoder (`Optional[PreTrainedModel]`, *optional*): Text encoder.
        audio_encoder (`Optional[PreTrainedModel]`, *optional*): Audio code decoder.
        decoder (`Optional[MusicgenMelodyForCausalLM]`, *optional*): MusicGen Melody decoder used to generate audio codes.
    

Methods: forward