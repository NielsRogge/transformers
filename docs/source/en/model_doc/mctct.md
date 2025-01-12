<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# M-CTC-T

<Tip warning={true}>

This model is in maintenance mode only, so we won't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

</Tip>

## Overview

The M-CTC-T model was proposed in [Pseudo-Labeling For Massively Multilingual Speech Recognition](https://arxiv.org/abs/2111.00161) by Loren Lugosch, Tatiana Likhomanenko, Gabriel Synnaeve, and Ronan Collobert. The model is a 1B-param transformer encoder, with a CTC head over 8065 character labels and a language identification head over 60 language ID labels. It is trained on Common Voice (version 6.1, December 2020 release) and VoxPopuli. After training on Common Voice and VoxPopuli, the model is trained on Common Voice only. The labels are unnormalized character-level transcripts (punctuation and capitalization are not removed). The model takes as input Mel filterbank features from a 16Khz audio signal.

The abstract from the paper is the following:

*Semi-supervised learning through pseudo-labeling has become a staple of state-of-the-art monolingual
speech recognition systems. In this work, we extend pseudo-labeling to massively multilingual speech
recognition with 60 languages. We propose a simple pseudo-labeling recipe that works well even
with low-resource languages: train a supervised multilingual model, fine-tune it with semi-supervised
learning on a target language, generate pseudo-labels for that language, and train a final model using
pseudo-labels for all languages, either from scratch or by fine-tuning. Experiments on the labeled
Common Voice and unlabeled VoxPopuli datasets show that our recipe can yield a model with better
performance for many languages that also transfers well to LibriSpeech.*

This model was contributed by [cwkeam](https://huggingface.co/cwkeam). The original code can be found [here](https://github.com/flashlight/wav2letter/tree/main/recipes/mling_pl).

## Usage tips

The PyTorch version of this model is only available in torch 1.9 and higher.

## Resources

- [Automatic speech recognition task guide](../tasks/asr)

## MCTCTConfig


    This is the configuration class to store the configuration of a [`MCTCTModel`]. It is used to instantiate an
    M-CTC-T model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the M-CTC-T
    [speechbrain/m-ctc-t-large](https://huggingface.co/speechbrain/m-ctc-t-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 8065):
            Vocabulary size of the M-CTC-T model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MCTCTModel`].
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of hidden layers in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_head_dim (`int`, *optional*, defaults to 384):
            Dimensions of each attention head for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 920):
            The maximum sequence length that this model might ever be used with (after log-mel spectrogram extraction).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        layerdrop (`float`, *optional*, defaults to 0.3):
            The probability of dropping an encoder layer during training. The default 0.3 value is used in the original
            implementation.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.3):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.3):
            The dropout ratio for the attention probabilities.
        pad_token_id (`int`, *optional*, defaults to 1):
            The tokenizer index of the pad token.
        bos_token_id (`int`, *optional*, defaults to 0):
            The tokenizer index of the bos token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The tokenizer index of the eos token.
        conv_glu_dim (`int`, *optional*, defaults to 1):
            The dimension of the output of the `Conv1dSubsampler` layer in which GLU is applied on. Though the original
            Flashlight code uses the value of 2, here it's adapted to 1 due to transposition differences.
        conv_dropout (`int`, *optional*, defaults to 0.3):
            The probability of randomly dropping the `Conv1dSubsampler` layer during training.
        num_conv_layers (`int`, *optional*, defaults to 1):
            Number of convolution layers before applying transformer encoder layers.
        conv_kernel (`Sequence[int]`, *optional*, defaults to `(7,)`):
            The kernel size of the 1D convolution applied before transformer layers. `len(conv_kernel)` must be equal
            to `num_conv_layers`.
        conv_stride (`Sequence[int]`, *optional*, defaults to `(3,)`):
            The stride length of the 1D convolution applied before transformer layers. `len(conv_stride)` must be equal
            to `num_conv_layers`.
        input_feat_per_channel (`int`, *optional*, defaults to 80):
            Feature dimensions of the channels of the input to the Conv1D layer.
        input_channels (`int`, *optional*, defaults to 1):
            Number of input channels of the input to the Conv1D layer.
        conv_channels (`List[int]`, *optional*):
            Channel sizes of intermediate Conv1D layers.
        ctc_loss_reduction (`str`, *optional*, defaults to `"sum"`):
            Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
            instance of [`MCTCTForCTC`].
        ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
            Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
            occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
            of [`MCTCTForCTC`].

    Example:

    ```python
    >>> from transformers import MCTCTConfig, MCTCTModel

    >>> # Initializing a M-CTC-T mctct-large style configuration
    >>> configuration = MCTCTConfig()

    >>> # Initializing a model (with random weights) from the mctct-large style configuration
    >>> model = MCTCTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## MCTCTFeatureExtractor


    Constructs a M-CTC-T feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods. This
    code has been adapted from Flashlight's C++ code. For more information about the implementation, one can refer to
    this [notebook](https://colab.research.google.com/drive/1GLtINkkhzms-IsdcGy_-tVCkv0qNF-Gt#scrollTo=pMCRGMmUC_an)
    that takes the user step-by-step in the implementation.

    Args:
        feature_size (`int`, defaults to 80):
            The feature dimension of the extracted features. This is the number of mel_frequency
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, defaults to 0.0):
            The value that is used to fill the padding values.
        hop_length (`int`, defaults to 10):
            Number of audio samples between windows. Otherwise referred to as "shift" in many papers.
        win_length (`int`, defaults to 25):
            Number of ms per window
        win_function (`str`, defaults to `"hamming_window"`):
            Name for the window function used for windowing, must be accessible via `torch.{win_function}`
        frame_signal_scale (`float`, defaults to 32768.0):
            Constant multiplied in creating the frames before applying DFT.
        preemphasis_coeff (`float`, defaults to 0.97):
            Constant multiplied in applying Pre-emphasis before DFT.
        mel_floor (`float` defaults to 1.0):
            Minimum value of mel frequency banks.
        normalize_means (`bool`, *optional*, defaults to `True`):
            Whether or not to zero-mean normalize the extracted features.
        normalize_vars (`bool`, *optional*, defaults to `True`):
            Whether or not to unit-variance normalize the extracted features.
    

Methods: __call__

## MCTCTProcessor


    Constructs a MCTCT processor which wraps a MCTCT feature extractor and a MCTCT tokenizer into a single processor.

    [`MCTCTProcessor`] offers all the functionalities of [`MCTCTFeatureExtractor`] and [`AutoTokenizer`]. See the
    [`~MCTCTProcessor.__call__`] and [`~MCTCTProcessor.decode`] for more information.

    Args:
        feature_extractor (`MCTCTFeatureExtractor`):
            An instance of [`MCTCTFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
    

Methods: __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## MCTCTModel

The bare M-CTC-T Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MCTCTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MCTCTForCTC

MCTCT Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MCTCTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
