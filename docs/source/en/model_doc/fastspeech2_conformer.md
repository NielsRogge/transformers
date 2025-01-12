<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# FastSpeech2Conformer

## Overview

The FastSpeech2Conformer model was proposed with the paper [Recent Developments On Espnet Toolkit Boosted By Conformer](https://arxiv.org/abs/2010.13956) by Pengcheng Guo, Florian Boyer, Xuankai Chang, Tomoki Hayashi, Yosuke Higuchi, Hirofumi Inaguma, Naoyuki Kamo, Chenda Li, Daniel Garcia-Romero, Jiatong Shi, Jing Shi, Shinji Watanabe, Kun Wei, Wangyou Zhang, and Yuekai Zhang.

The abstract from the original FastSpeech2 paper is the following:

*Non-autoregressive text to speech (TTS) models such as FastSpeech (Ren et al., 2019) can synthesize speech significantly faster than previous autoregressive models with comparable quality. The training of FastSpeech model relies on an autoregressive teacher model for duration prediction (to provide more information as input) and knowledge distillation (to simplify the data distribution in output), which can ease the one-to-many mapping problem (i.e., multiple speech variations correspond to the same text) in TTS. However, FastSpeech has several disadvantages: 1) the teacher-student distillation pipeline is complicated and time-consuming, 2) the duration extracted from the teacher model is not accurate enough, and the target mel-spectrograms distilled from teacher model suffer from information loss due to data simplification, both of which limit the voice quality. In this paper, we propose FastSpeech 2, which addresses the issues in FastSpeech and better solves the one-to-many mapping problem in TTS by 1) directly training the model with ground-truth target instead of the simplified output from teacher, and 2) introducing more variation information of speech (e.g., pitch, energy and more accurate duration) as conditional inputs. Specifically, we extract duration, pitch and energy from speech waveform and directly take them as conditional inputs in training and use predicted values in inference. We further design FastSpeech 2s, which is the first attempt to directly generate speech waveform from text in parallel, enjoying the benefit of fully end-to-end inference. Experimental results show that 1) FastSpeech 2 achieves a 3x training speed-up over FastSpeech, and FastSpeech 2s enjoys even faster inference speed; 2) FastSpeech 2 and 2s outperform FastSpeech in voice quality, and FastSpeech 2 can even surpass autoregressive models. Audio samples are available at https://speechresearch.github.io/fastspeech2/.*

This model was contributed by [Connor Henderson](https://huggingface.co/connor-henderson). The original code can be found [here](https://github.com/espnet/espnet/blob/master/espnet2/tts/fastspeech2/fastspeech2.py).


## 🤗 Model Architecture
FastSpeech2's general structure with a Mel-spectrogram decoder was implemented, and the traditional transformer blocks were replaced with conformer blocks as done in the ESPnet library.

#### FastSpeech2 Model Architecture
![FastSpeech2 Model Architecture](https://www.microsoft.com/en-us/research/uploads/prod/2021/04/fastspeech2-1.png)

#### Conformer Blocks
![Conformer Blocks](https://www.researchgate.net/profile/Hirofumi-Inaguma-2/publication/344911155/figure/fig2/AS:951455406108673@1603856054097/An-overview-of-Conformer-block.png)

#### Convolution Module
![Convolution Module](https://d3i71xaburhd42.cloudfront.net/8809d0732f6147d4ad9218c8f9b20227c837a746/2-Figure1-1.png)

## 🤗 Transformers Usage

You can run FastSpeech2Conformer locally with the 🤗 Transformers library.

1. First install the 🤗 [Transformers library](https://github.com/huggingface/transformers), g2p-en:

```bash
pip install --upgrade pip
pip install --upgrade transformers g2p-en
```

2. Run inference via the Transformers modelling code with the model and hifigan separately

```python

from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
import soundfile as sf

tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
inputs = tokenizer("Hello, my dog is cute.", return_tensors="pt")
input_ids = inputs["input_ids"]

model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
output_dict = model(input_ids, return_dict=True)
spectrogram = output_dict["spectrogram"]

hifigan = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")
waveform = hifigan(spectrogram)

sf.write("speech.wav", waveform.squeeze().detach().numpy(), samplerate=22050)
```

3. Run inference via the Transformers modelling code with the model and hifigan combined

```python
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerWithHifiGan
import soundfile as sf

tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
inputs = tokenizer("Hello, my dog is cute.", return_tensors="pt")
input_ids = inputs["input_ids"]

model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan")
output_dict = model(input_ids, return_dict=True)
waveform = output_dict["waveform"]

sf.write("speech.wav", waveform.squeeze().detach().numpy(), samplerate=22050)
```

4. Run inference with a pipeline and specify which vocoder to use
```python
from transformers import pipeline, FastSpeech2ConformerHifiGan
import soundfile as sf

vocoder = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")
synthesiser = pipeline(model="espnet/fastspeech2_conformer", vocoder=vocoder)

speech = synthesiser("Hello, my dog is cooler than you!")

sf.write("speech.wav", speech["audio"].squeeze(), samplerate=speech["sampling_rate"])
```


## FastSpeech2ConformerConfig


    This is the configuration class to store the configuration of a [`FastSpeech2ConformerModel`]. It is used to
    instantiate a FastSpeech2Conformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer [espnet/fastspeech2_conformer](https://huggingface.co/espnet/fastspeech2_conformer)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 384):
            The dimensionality of the hidden layers.
        vocab_size (`int`, *optional*, defaults to 78):
            The size of the vocabulary.
        num_mel_bins (`int`, *optional*, defaults to 80):
            The number of mel filters used in the filter bank.
        encoder_num_attention_heads (`int`, *optional*, defaults to 2):
            The number of attention heads in the encoder.
        encoder_layers (`int`, *optional*, defaults to 4):
            The number of layers in the encoder.
        encoder_linear_units (`int`, *optional*, defaults to 1536):
            The number of units in the linear layer of the encoder.
        decoder_layers (`int`, *optional*, defaults to 4):
            The number of layers in the decoder.
        decoder_num_attention_heads (`int`, *optional*, defaults to 2):
            The number of attention heads in the decoder.
        decoder_linear_units (`int`, *optional*, defaults to 1536):
            The number of units in the linear layer of the decoder.
        speech_decoder_postnet_layers (`int`, *optional*, defaults to 5):
            The number of layers in the post-net of the speech decoder.
        speech_decoder_postnet_units (`int`, *optional*, defaults to 256):
            The number of units in the post-net layers of the speech decoder.
        speech_decoder_postnet_kernel (`int`, *optional*, defaults to 5):
            The kernel size in the post-net of the speech decoder.
        positionwise_conv_kernel_size (`int`, *optional*, defaults to 3):
            The size of the convolution kernel used in the position-wise layer.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Specifies whether to normalize before encoder layers.
        decoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Specifies whether to normalize before decoder layers.
        encoder_concat_after (`bool`, *optional*, defaults to `False`):
            Specifies whether to concatenate after encoder layers.
        decoder_concat_after (`bool`, *optional*, defaults to `False`):
            Specifies whether to concatenate after decoder layers.
        reduction_factor (`int`, *optional*, defaults to 1):
            The factor by which the speech frame rate is reduced.
        speaking_speed (`float`, *optional*, defaults to 1.0):
            The speed of the speech produced.
        use_macaron_style_in_conformer (`bool`, *optional*, defaults to `True`):
            Specifies whether to use macaron style in the conformer.
        use_cnn_in_conformer (`bool`, *optional*, defaults to `True`):
            Specifies whether to use convolutional neural networks in the conformer.
        encoder_kernel_size (`int`, *optional*, defaults to 7):
            The kernel size used in the encoder.
        decoder_kernel_size (`int`, *optional*, defaults to 31):
            The kernel size used in the decoder.
        duration_predictor_layers (`int`, *optional*, defaults to 2):
            The number of layers in the duration predictor.
        duration_predictor_channels (`int`, *optional*, defaults to 256):
            The number of channels in the duration predictor.
        duration_predictor_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size used in the duration predictor.
        energy_predictor_layers (`int`, *optional*, defaults to 2):
            The number of layers in the energy predictor.
        energy_predictor_channels (`int`, *optional*, defaults to 256):
            The number of channels in the energy predictor.
        energy_predictor_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size used in the energy predictor.
        energy_predictor_dropout (`float`, *optional*, defaults to 0.5):
            The dropout rate in the energy predictor.
        energy_embed_kernel_size (`int`, *optional*, defaults to 1):
            The kernel size used in the energy embed layer.
        energy_embed_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate in the energy embed layer.
        stop_gradient_from_energy_predictor (`bool`, *optional*, defaults to `False`):
            Specifies whether to stop gradients from the energy predictor.
        pitch_predictor_layers (`int`, *optional*, defaults to 5):
            The number of layers in the pitch predictor.
        pitch_predictor_channels (`int`, *optional*, defaults to 256):
            The number of channels in the pitch predictor.
        pitch_predictor_kernel_size (`int`, *optional*, defaults to 5):
            The kernel size used in the pitch predictor.
        pitch_predictor_dropout (`float`, *optional*, defaults to 0.5):
            The dropout rate in the pitch predictor.
        pitch_embed_kernel_size (`int`, *optional*, defaults to 1):
            The kernel size used in the pitch embed layer.
        pitch_embed_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate in the pitch embed layer.
        stop_gradient_from_pitch_predictor (`bool`, *optional*, defaults to `True`):
            Specifies whether to stop gradients from the pitch predictor.
        encoder_dropout_rate (`float`, *optional*, defaults to 0.2):
            The dropout rate in the encoder.
        encoder_positional_dropout_rate (`float`, *optional*, defaults to 0.2):
            The positional dropout rate in the encoder.
        encoder_attention_dropout_rate (`float`, *optional*, defaults to 0.2):
            The attention dropout rate in the encoder.
        decoder_dropout_rate (`float`, *optional*, defaults to 0.2):
            The dropout rate in the decoder.
        decoder_positional_dropout_rate (`float`, *optional*, defaults to 0.2):
            The positional dropout rate in the decoder.
        decoder_attention_dropout_rate (`float`, *optional*, defaults to 0.2):
            The attention dropout rate in the decoder.
        duration_predictor_dropout_rate (`float`, *optional*, defaults to 0.2):
            The dropout rate in the duration predictor.
        speech_decoder_postnet_dropout (`float`, *optional*, defaults to 0.5):
            The dropout rate in the speech decoder postnet.
        max_source_positions (`int`, *optional*, defaults to 5000):
            if `"relative"` position embeddings are used, defines the maximum source input positions.
        use_masking (`bool`, *optional*, defaults to `True`):
            Specifies whether to use masking in the model.
        use_weighted_masking (`bool`, *optional*, defaults to `False`):
            Specifies whether to use weighted masking in the model.
        num_speakers (`int`, *optional*):
            Number of speakers. If set to > 1, assume that the speaker ids will be provided as the input and use
            speaker id embedding layer.
        num_languages (`int`, *optional*):
            Number of languages. If set to > 1, assume that the language ids will be provided as the input and use the
            languge id embedding layer.
        speaker_embed_dim (`int`, *optional*):
            Speaker embedding dimension. If set to > 0, assume that speaker_embedding will be provided as the input.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Specifies whether the model is an encoder-decoder.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerConfig

    >>> # Initializing a FastSpeech2Conformer style configuration
    >>> configuration = FastSpeech2ConformerConfig()

    >>> # Initializing a model from the FastSpeech2Conformer style configuration
    >>> model = FastSpeech2ConformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## FastSpeech2ConformerHifiGanConfig


    This is the configuration class to store the configuration of a [`FastSpeech2ConformerHifiGanModel`]. It is used to
    instantiate a FastSpeech2Conformer HiFi-GAN vocoder model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer
    [espnet/fastspeech2_conformer_hifigan](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram.
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 2, 2]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[16, 16, 4, 4]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
            length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
            *upsample_rates*.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
            fusion (MRF) module.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            multi-receptive field fusion (MRF) module.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig

    >>> # Initializing a FastSpeech2ConformerHifiGan configuration
    >>> configuration = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FastSpeech2ConformerHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## FastSpeech2ConformerWithHifiGanConfig


    This is the configuration class to store the configuration of a [`FastSpeech2ConformerWithHifiGan`]. It is used to
    instantiate a `FastSpeech2ConformerWithHifiGanModel` model according to the specified sub-models configurations,
    defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2ConformerModel [espnet/fastspeech2_conformer](https://huggingface.co/espnet/fastspeech2_conformer) and
    FastSpeech2ConformerHifiGan
    [espnet/fastspeech2_conformer_hifigan](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) architectures.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_config (`typing.Dict`, *optional*):
            Configuration of the text-to-speech model.
        vocoder_config (`typing.Dict`, *optional*):
            Configuration of the vocoder model.
    model_config ([`FastSpeech2ConformerConfig`], *optional*):
        Configuration of the text-to-speech model.
    vocoder_config ([`FastSpeech2ConformerHiFiGanConfig`], *optional*):
        Configuration of the vocoder model.

    Example:

    ```python
    >>> from transformers import (
    ...     FastSpeech2ConformerConfig,
    ...     FastSpeech2ConformerHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGan,
    ... )

    >>> # Initializing FastSpeech2ConformerWithHifiGan sub-modules configurations.
    >>> model_config = FastSpeech2ConformerConfig()
    >>> vocoder_config = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a FastSpeech2ConformerWithHifiGan module style configuration
    >>> configuration = FastSpeech2ConformerWithHifiGanConfig(model_config.to_dict(), vocoder_config.to_dict())

    >>> # Initializing a model (with random weights)
    >>> model = FastSpeech2ConformerWithHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    

## FastSpeech2ConformerTokenizer


    Construct a FastSpeech2Conformer tokenizer.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The begin of sequence token. Note that for FastSpeech2, it is the same as the `eos_token`.
        eos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The end of sequence token. Note that for FastSpeech2, it is the same as the `bos_token`.
        pad_token (`str`, *optional*, defaults to `"<blank>"`):
            The token used for padding, for example when batching sequences of different lengths.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        should_strip_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to strip the spaces from the list of tokens.
    

Methods: __call__
    - save_vocabulary
    - decode
    - batch_decode

## FastSpeech2ConformerModel

FastSpeech2Conformer Model.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    FastSpeech 2 module.

    This is a module of FastSpeech 2 described in 'FastSpeech 2: Fast and High-Quality End-to-End Text to Speech'
    https://arxiv.org/abs/2006.04558. Instead of quantized pitch and energy, we use token-averaged value introduced in
    FastPitch: Parallel Text-to-speech with Pitch Prediction. The encoder and decoder are Conformers instead of regular
    Transformers.
    

Methods: forward

## FastSpeech2ConformerHifiGan

HiFi-GAN vocoder.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FastSpeech2ConformerWithHifiGan

The FastSpeech2ConformerModel with a FastSpeech2ConformerHifiGan vocoder head that performs text-to-speech (waveform).
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerWithHifiGanConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
