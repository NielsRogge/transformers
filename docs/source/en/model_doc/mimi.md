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

# Mimi

## Overview

The Mimi model was proposed in [Moshi: a speech-text foundation model for real-time dialogue](https://kyutai.org/Moshi.pdf) by Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave and Neil Zeghidour. Mimi is a high-fidelity audio codec model developed by the Kyutai team, that combines semantic and acoustic information into audio tokens running at 12Hz and a bitrate of 1.1kbps. In other words, it can be used to map audio waveforms into “audio tokens”, known as “codebooks”.

The abstract from the paper is the following:

*We introduce Moshi, a speech-text foundation model and full-duplex spoken dialogue framework. Current systems for spoken dialogue rely on pipelines of independent components, namely voice activity detection, speech recognition, textual dialogue and text-to-speech. Such frameworks cannot emulate the experience of real conversations. First, their complexity induces a latency of several seconds between interactions. Second, text being the intermediate modality for dialogue, non-linguistic information that modifies meaning— such as emotion or non-speech sounds— is lost in the interaction. Finally, they rely on a segmentation into speaker turns, which does not take into account overlapping speech, interruptions and interjections. Moshi solves these independent issues altogether by casting spoken dialogue as speech-to-speech generation. Starting from a text language model backbone, Moshi generates speech as tokens from the residual quantizer of a neural audio codec, while modeling separately its own speech and that of the user into parallel streams. This allows for the removal of explicit speaker turns, and the modeling of arbitrary conversational dynamics. We moreover extend the hierarchical semantic-to-acoustic token generation of previous work to first predict time-aligned text tokens as a prefix to audio tokens. Not only this “Inner Monologue” method significantly improves the linguistic quality of generated speech, but we also illustrate how it can provide streaming speech recognition and text-to-speech. Our resulting model is the first real-time full-duplex spoken large language model, with a theoretical latency of 160ms, 200ms in practice, and is available at github.com/kyutai-labs/moshi.* 

Its architecture is based on [Encodec](model_doc/encodec) with several major differences:
* it uses a much lower frame-rate.
* it uses additional transformers for encoding and decoding for better latent contextualization
* it uses a different quantization scheme: one codebook is dedicated to semantic projection.

## Usage example 

Here is a quick example of how to encode and decode an audio using this model:

```python 
>>> from datasets import load_dataset, Audio
>>> from transformers import MimiModel, AutoFeatureExtractor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> # load model and feature extractor
>>> model = MimiModel.from_pretrained("kyutai/mimi")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

>>> # load audio sample
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
>>> audio_values = model.decode(encoder_outputs.audio_codes, inputs["padding_mask"])[0]
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

This model was contributed by [Yoach Lacombe (ylacombe)](https://huggingface.co/ylacombe).
The original code can be found [here](https://github.com/kyutai-labs/moshi).


## MimiConfig


    This is the configuration class to store the configuration of an [`MimiModel`]. It is used to instantiate a
    Mimi model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [kyutai/mimi](https://huggingface.co/kyutai/mimi) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        frame_rate (`float`, *optional*, defaults to 12.5):
            Framerate of the model.
        audio_channels (`int`, *optional*, defaults to 1):
            Number of channels in the audio data. Either 1 for mono or 2 for stereo.
        hidden_size (`int`, *optional*, defaults to 512):
            Intermediate representation dimension.
        num_filters (`int`, *optional*, defaults to 64):
            Number of convolution kernels of first `MimiConv1d` down sampling layer.
        num_residual_layers (`int`,  *optional*, defaults to 1):
            Number of residual layers.
        upsampling_ratios (`Sequence[int]`, *optional*):
            Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it
            will use the ratios in the reverse order to the ones specified here that must match the decoder order.
            If not specified, will defaults to `[8, 6, 5, 4]`
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for the initial convolution.
        last_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for the last convolution layer.
        residual_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for the residual layers.
        dilation_growth_rate (`int`, *optional*, defaults to 2):
            How much to increase the dilation with each layer.
        use_causal_conv (`bool`, *optional*, defaults to `True`):
            Whether to use fully causal convolution.
        pad_mode (`str`, *optional*, defaults to `"constant"`):
            Padding mode for the convolutions.
        compress (`int`, *optional*, defaults to 2):
            Reduced dimensionality in residual branches.
        trim_right_ratio (`float`, *optional*, defaults to 1.0):
            Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If
            equal to 1.0, it means that all the trimming is done at the right.
        codebook_size (`int`, *optional*, defaults to 2048):
            Number of discret codes in each codebooks.
        codebook_dim (`int`, *optional*, defaults to 256):
            Dimension of the unquantized codebook vectors. If not defined, uses `hidden_size`.
        num_quantizers (`int`, *optional*, defaults to 32):
            Number of quantizer channels, or codebooks, in the quantizer.
        use_conv_shortcut (`bool`, *optional*, defaults to `False`):
            Whether to use a convolutional layer as the 'skip' connection in the `MimiResnetBlock` block. If False,
            an identity function will be used, giving a generic residual connection.
        vector_quantization_hidden_dimension (`int`, *optional*, defaults to 256):
            Intermediate representation dimension in the residual vector quantization space.
        num_semantic_quantizers (`int`, *optional*, defaults to 1):
            Number of semantic quantizer channels, or codebooks, in the semantic quantizer. Must be lower than `num_quantizers`.
        upsample_groups (`int`, *optional*, defaults to 512):
            If `frame_rate!=encodec_frame_rate`, indicates the number of groups used in the upsampling operation to go from one rate to another.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the Transformer models.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        head_dim (`int`, *optional*, defaults to `hidden_size // num_attention_heads`):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8000):
            The maximum sequence length that this model might ever be used with. Mimi's sliding window attention
            allows sequence of up to 8000 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the LayerNorm normalization layers.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        sliding_window (`int`, *optional*, defaults to 250):
            Sliding window attention window size. If not specified, will default to `250`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layer_scale_initial_scale (`float`, *optional*, defaults to 0.01):
            Initiale scale of the residual rescaling operation done in the Transformer models.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
    Example:

    ```python
    >>> from transformers import MimiModel, MimiConfig

    >>> # Initializing a "kyutai/mimi" style configuration
    >>> configuration = MimiConfig()

    >>> # Initializing a model (with random weights) from the "kyutai/mimi" style configuration
    >>> model = MimiModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## MimiModel

The Mimi neural audio codec model.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MimiConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: decode
    - encode
    - forward