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

# UnivNet

## Overview

The UnivNet model was proposed in [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889) by Won Jang, Dan Lim, Jaesam Yoon, Bongwan Kin, and Juntae Kim.
The UnivNet model is a generative adversarial network (GAN) trained to synthesize high fidelity speech waveforms. The UnivNet model shared in `transformers` is the *generator*, which maps a conditioning log-mel spectrogram and optional noise sequence to a speech waveform (e.g. a vocoder). Only the generator is required for inference. The *discriminator* used to train the `generator` is not implemented.

The abstract from the paper is the following:

*Most neural vocoders employ band-limited mel-spectrograms to generate waveforms. If full-band spectral features are used as the input, the vocoder can be provided with as much acoustic information as possible. However, in some models employing full-band mel-spectrograms, an over-smoothing problem occurs as part of which non-sharp spectrograms are generated. To address this problem, we propose UnivNet, a neural vocoder that synthesizes high-fidelity waveforms in real time. Inspired by works in the field of voice activity detection, we added a multi-resolution spectrogram discriminator that employs multiple linear spectrogram magnitudes computed using various parameter sets. Using full-band mel-spectrograms as input, we expect to generate high-resolution signals by adding a discriminator that employs spectrograms of multiple resolutions as the input. In an evaluation on a dataset containing information on hundreds of speakers, UnivNet obtained the best objective and subjective results among competing models for both seen and unseen speakers. These results, including the best subjective score for text-to-speech, demonstrate the potential for fast adaptation to new speakers without a need for training from scratch.*

Tips:

- The `noise_sequence` argument for [`UnivNetModel.forward`] should be standard Gaussian noise (such as from `torch.randn`) of shape `([batch_size], noise_length, model.config.model_in_channels)`, where `noise_length` should match the length dimension (dimension 1) of the `input_features` argument. If not supplied, it will be randomly generated; a `torch.Generator` can be supplied to the `generator` argument so that the forward pass can be reproduced. (Note that [`UnivNetFeatureExtractor`] will return generated noise by default, so it shouldn't be necessary to generate `noise_sequence` manually.)
- Padding added by [`UnivNetFeatureExtractor`] can be removed from the [`UnivNetModel`] output through the [`UnivNetFeatureExtractor.batch_decode`] method, as shown in the usage example below.
- Padding the end of each waveform with silence can reduce artifacts at the end of the generated audio sample. This can be done by supplying `pad_end = True` to [`UnivNetFeatureExtractor.__call__`]. See [this issue](https://github.com/seungwonpark/melgan/issues/8) for more details.

Usage Example:

```python
import torch
from scipy.io.wavfile import write
from datasets import Audio, load_dataset

from transformers import UnivNetFeatureExtractor, UnivNetModel

model_id_or_path = "dg845/univnet-dev"
model = UnivNetModel.from_pretrained(model_id_or_path)
feature_extractor = UnivNetFeatureExtractor.from_pretrained(model_id_or_path)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# Resample the audio to the model and feature extractor's sampling rate.
ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
# Pad the end of the converted waveforms to reduce artifacts at the end of the output audio samples.
inputs = feature_extractor(
    ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], pad_end=True, return_tensors="pt"
)

with torch.no_grad():
    audio = model(**inputs)

# Remove the extra padding at the end of the output.
audio = feature_extractor.batch_decode(**audio)[0]
# Convert to wav file
write("sample_audio.wav", feature_extractor.sampling_rate, audio)
```

This model was contributed by [dg845](https://huggingface.co/dg845).
To the best of my knowledge, there is no official code release, but an unofficial implementation can be found at [maum-ai/univnet](https://github.com/maum-ai/univnet) with pretrained checkpoints [here](https://github.com/maum-ai/univnet#pre-trained-model).


## UnivNetConfig


    This is the configuration class to store the configuration of a [`UnivNetModel`]. It is used to instantiate a
    UnivNet vocoder model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the UnivNet
    [dg845/univnet-dev](https://huggingface.co/dg845/univnet-dev) architecture, which corresponds to the 'c32'
    architecture in [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/master/config/default_c32.yaml).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_channels (`int`, *optional*, defaults to 64):
            The number of input channels for the UnivNet residual network. This should correspond to
            `noise_sequence.shape[1]` and the value used in the [`UnivNetFeatureExtractor`] class.
        model_hidden_channels (`int`, *optional*, defaults to 32):
            The number of hidden channels of each residual block in the UnivNet residual network.
        num_mel_bins (`int`, *optional*, defaults to 100):
            The number of frequency bins in the conditioning log-mel spectrogram. This should correspond to the value
            used in the [`UnivNetFeatureExtractor`] class.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 3, 3]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the UnivNet residual
            network. The length of `resblock_kernel_sizes` defines the number of resnet blocks and should match that of
            `resblock_stride_sizes` and `resblock_dilation_sizes`.
        resblock_stride_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 4]`):
            A tuple of integers defining the stride sizes of the 1D convolutional layers in the UnivNet residual
            network. The length of `resblock_stride_sizes` should match that of `resblock_kernel_sizes` and
            `resblock_dilation_sizes`.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 9, 27], [1, 3, 9, 27], [1, 3, 9, 27]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            UnivNet residual network. The length of `resblock_dilation_sizes` should match that of
            `resblock_kernel_sizes` and `resblock_stride_sizes`. The length of each nested list in
            `resblock_dilation_sizes` defines the number of convolutional layers per resnet block.
        kernel_predictor_num_blocks (`int`, *optional*, defaults to 3):
            The number of residual blocks in the kernel predictor network, which calculates the kernel and bias for
            each location variable convolution layer in the UnivNet residual network.
        kernel_predictor_hidden_channels (`int`, *optional*, defaults to 64):
            The number of hidden channels for each residual block in the kernel predictor network.
        kernel_predictor_conv_size (`int`, *optional*, defaults to 3):
            The kernel size of each 1D convolutional layer in the kernel predictor network.
        kernel_predictor_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for each residual block in the kernel predictor network.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.2):
            The angle of the negative slope used by the leaky ReLU activation.

    Example:

    ```python
    >>> from transformers import UnivNetModel, UnivNetConfig

    >>> # Initializing a Tortoise TTS style configuration
    >>> configuration = UnivNetConfig()

    >>> # Initializing a model (with random weights) from the Tortoise TTS style configuration
    >>> model = UnivNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    

## UnivNetFeatureExtractor


    Constructs a UnivNet feature extractor.

    This class extracts log-mel-filter bank features from raw speech using the short time Fourier Transform (STFT). The
    STFT implementation follows that of TacoTron 2 and Hifi-GAN.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value to pad with when applying the padding strategy defined by the `padding` argument to
            [`UnivNetFeatureExtractor.__call__`]. Should correspond to audio silence. The `pad_end` argument to
            `__call__` will also use this padding value.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether to perform Tacotron 2 normalization on the input. Normalizing can help to significantly improve the
            performance for some models.
        num_mel_bins (`int`, *optional*, defaults to 100):
            The number of mel-frequency bins in the extracted spectrogram features. This should match
            `UnivNetModel.config.num_mel_bins`.
        hop_length (`int`, *optional*, defaults to 256):
            The direct number of samples between sliding windows. Otherwise referred to as "shift" in many papers. Note
            that this is different from other audio feature extractors such as [`SpeechT5FeatureExtractor`] which take
            the `hop_length` in ms.
        win_length (`int`, *optional*, defaults to 1024):
            The direct number of samples for each sliding window. Note that this is different from other audio feature
            extractors such as [`SpeechT5FeatureExtractor`] which take the `win_length` in ms.
        win_function (`str`, *optional*, defaults to `"hann_window"`):
            Name for the window function used for windowing, must be accessible via `torch.{win_function}`
        filter_length (`int`, *optional*, defaults to 1024):
            The number of FFT components to use. If `None`, this is determined using
            `transformers.audio_utils.optimal_fft_length`.
        max_length_s (`int`, *optional*, defaults to 10):
            The maximum input lenght of the model in seconds. This is used to pad the audio.
        fmin (`float`, *optional*, defaults to 0.0):
            Minimum mel frequency in Hz.
        fmax (`float`, *optional*):
            Maximum mel frequency in Hz. If not set, defaults to `sampling_rate / 2`.
        mel_floor (`float`, *optional*, defaults to 1e-09):
            Minimum value of mel frequency banks. Note that the way [`UnivNetFeatureExtractor`] uses `mel_floor` is
            different than in [`transformers.audio_utils.spectrogram`].
        center (`bool`, *optional*, defaults to `False`):
            Whether to pad the waveform so that frame `t` is centered around time `t * hop_length`. If `False`, frame
            `t` will start at time `t * hop_length`.
        compression_factor (`float`, *optional*, defaults to 1.0):
            The multiplicative compression factor for dynamic range compression during spectral normalization.
        compression_clip_val (`float`, *optional*, defaults to 1e-05):
            The clip value applied to the waveform before applying dynamic range compression during spectral
            normalization.
        normalize_min (`float`, *optional*, defaults to -11.512925148010254):
            The min value used for Tacotron 2-style linear normalization. The default is the original value from the
            Tacotron 2 implementation.
        normalize_max (`float`, *optional*, defaults to 2.3143386840820312):
            The max value used for Tacotron 2-style linear normalization. The default is the original value from the
            Tacotron 2 implementation.
        model_in_channels (`int`, *optional*, defaults to 64):
            The number of input channels to the [`UnivNetModel`] model. This should match
            `UnivNetModel.config.model_in_channels`.
        pad_end_length (`int`, *optional*, defaults to 10):
            If padding the end of each waveform, the number of spectrogram frames worth of samples to append. The
            number of appended samples will be `pad_end_length * hop_length`.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether or not [`~UnivNetFeatureExtractor.__call__`] should return `attention_mask`.
    

Methods: __call__

## UnivNetModel

UnivNet GAN vocoder.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`UnivNetConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward