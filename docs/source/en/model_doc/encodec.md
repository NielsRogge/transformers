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

# EnCodec

## Overview

The EnCodec neural codec model was proposed in [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) by Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi.

The abstract from the paper is the following:

*We introduce a state-of-the-art real-time, high-fidelity, audio codec leveraging neural networks. It consists in a streaming encoder-decoder architecture with quantized latent space trained in an end-to-end fashion. We simplify and speed-up the training by using a single multiscale spectrogram adversary that efficiently reduces artifacts and produce high-quality samples. We introduce a novel loss balancer mechanism to stabilize training: the weight of a loss now defines the fraction of the overall gradient it should represent, thus decoupling the choice of this hyper-parameter from the typical scale of the loss. Finally, we study how lightweight Transformer models can be used to further compress the obtained representation by up to 40%, while staying faster than real time. We provide a detailed description of the key design choices of the proposed model including: training objective, architectural changes and a study of various perceptual loss functions. We present an extensive subjective evaluation (MUSHRA tests) together with an ablation study for a range of bandwidths and audio domains, including speech, noisy-reverberant speech, and music. Our approach is superior to the baselines methods across all evaluated settings, considering both 24 kHz monophonic and 48 kHz stereophonic audio.*

This model was contributed by [Matthijs](https://huggingface.co/Matthijs), [Patrick Von Platen](https://huggingface.co/patrickvonplaten) and [Arthur Zucker](https://huggingface.co/ArthurZ). 
The original code can be found [here](https://github.com/facebookresearch/encodec).

## Usage example 

Here is a quick example of how to encode and decode an audio using this model:

```python 
>>> from datasets import load_dataset, Audio
>>> from transformers import EncodecModel, AutoProcessor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> model = EncodecModel.from_pretrained("facebook/encodec_24khz")
>>> processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
>>> audio_values = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

## EncodecConfig


    This is the configuration class to store the configuration of an [`EncodecModel`]. It is used to instantiate a
    Encodec model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        target_bandwidths (`List[float]`, *optional*, defaults to `[1.5, 3.0, 6.0, 12.0, 24.0]`):
            The range of diffent bandwiths the model can encode audio with.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        audio_channels (`int`, *optional*, defaults to 1):
            Number of channels in the audio data. Either 1 for mono or 2 for stereo.
        normalize (`bool`, *optional*, defaults to `False`):
            Whether the audio shall be normalized when passed.
        chunk_length_s (`float`, *optional*):
            If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.
        overlap (`float`, *optional*):
            Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following
            formulae : `int((1.0 - self.overlap) * self.chunk_length)`.
        hidden_size (`int`, *optional*, defaults to 128):
            Intermediate representation dimension.
        num_filters (`int`, *optional*, defaults to 32):
            Number of convolution kernels of first `EncodecConv1d` down sampling layer.
        num_residual_layers (`int`,  *optional*, defaults to 1):
            Number of residual layers.
        upsampling_ratios (`Sequence[int]` , *optional*, defaults to `[8, 5, 4, 2]`):
            Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it
            will use the ratios in the reverse order to the ones specified here that must match the decoder order.
        norm_type (`str`, *optional*, defaults to `"weight_norm"`):
            Normalization method. Should be in `["weight_norm", "time_group_norm"]`
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for the initial convolution.
        last_kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for the last convolution layer.
        residual_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for the residual layers.
        dilation_growth_rate (`int`, *optional*, defaults to 2):
            How much to increase the dilation with each layer.
        use_causal_conv (`bool`, *optional*, defaults to `True`):
            Whether to use fully causal convolution.
        pad_mode (`str`, *optional*, defaults to `"reflect"`):
            Padding mode for the convolutions.
        compress (`int`, *optional*, defaults to 2):
            Reduced dimensionality in residual branches (from Demucs v3).
        num_lstm_layers (`int`, *optional*, defaults to 2):
            Number of LSTM layers at the end of the encoder.
        trim_right_ratio (`float`, *optional*, defaults to 1.0):
            Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If
            equal to 1.0, it means that all the trimming is done at the right.
        codebook_size (`int`, *optional*, defaults to 1024):
            Number of discret codes that make up VQVAE.
        codebook_dim (`int`, *optional*):
            Dimension of the codebook vectors. If not defined, uses `hidden_size`.
        use_conv_shortcut (`bool`, *optional*, defaults to `True`):
            Whether to use a convolutional layer as the 'skip' connection in the `EncodecResnetBlock` block. If False,
            an identity function will be used, giving a generic residual connection.

    Example:

    ```python
    >>> from transformers import EncodecModel, EncodecConfig

    >>> # Initializing a "facebook/encodec_24khz" style configuration
    >>> configuration = EncodecConfig()

    >>> # Initializing a model (with random weights) from the "facebook/encodec_24khz" style configuration
    >>> model = EncodecModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## EncodecFeatureExtractor


    Constructs an EnCodec feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Instantiating a feature extractor with the defaults will yield a similar configuration to that of the
    [facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz) architecture.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        chunk_length_s (`float`, *optional*):
            If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.
        overlap (`float`, *optional*):
            Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following
            formulae : `int((1.0 - self.overlap) * self.chunk_length)`.
    

Methods: __call__

## EncodecModel

The EnCodec neural audio codec model.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`EncodecConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: decode
    - encode
    - forward
