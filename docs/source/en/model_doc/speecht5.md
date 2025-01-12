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

# SpeechT5

## Overview

The SpeechT5 model was proposed in [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205) by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.

The abstract from the paper is the following:

*Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder. Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder. Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.*

This model was contributed by [Matthijs](https://huggingface.co/Matthijs). The original code can be found [here](https://github.com/microsoft/SpeechT5).

## SpeechT5Config


    This is the configuration class to store the configuration of a [`SpeechT5Model`]. It is used to instantiate a
    SpeechT5 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the SpeechT5
    [microsoft/speecht5_asr](https://huggingface.co/microsoft/speecht5_asr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 81):
            Vocabulary size of the SpeechT5 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed to the forward method of [`SpeechT5Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer decoder.
        decoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        positional_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the text position encoding layers.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        feat_extract_norm (`str`, *optional*, defaults to `"group"`):
            The norm to be applied to 1D convolutional layers in the speech encoder pre-net. One of `"group"` for group
            normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
            convolutional layers.
        feat_proj_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for output of the speech encoder pre-net.
        feat_extract_activation (`str, `optional`, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the 1D convolutional layers of the feature
            extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        conv_dim (`Tuple[int]` or `List[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
            A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
            speech encoder pre-net. The length of *conv_dim* defines the number of 1D convolutional layers.
        conv_stride (`Tuple[int]` or `List[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
            A tuple of integers defining the stride of each 1D convolutional layer in the speech encoder pre-net. The
            length of *conv_stride* defines the number of convolutional layers and has to match the length of
            *conv_dim*.
        conv_kernel (`Tuple[int]` or `List[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 3, 3)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the speech encoder pre-net.
            The length of *conv_kernel* defines the number of convolutional layers and has to match the length of
            *conv_dim*.
        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether the 1D convolutional layers have a bias.
        num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer.
        apply_spec_augment (`bool`, *optional*, defaults to `True`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the speech encoder pre-net. For
            reference see [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
            Recognition](https://arxiv.org/abs/1904.08779).
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
            procecure generates ''mask_time_prob*len(time_axis)/mask_time_length'' independent masks over the axis. If
            reasoning from the propability of each feature vector to be chosen as the start of the vector span to be
            masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
            actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2),:
            The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
            mask_time_min_masks''
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procecure generates ''mask_feature_prob*len(feature_axis)/mask_time_length'' independent masks over
            the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector
            span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
            may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
            True`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        mask_feature_min_masks (`int`, *optional*, defaults to 0),:
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            step, irrespectively of `mask_feature_prob`. Only relevant if
            ''mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks''
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel features used per input features. Used by the speech decoder pre-net. Should correspond to
            the value used in the [`SpeechT5Processor`] class.
        speech_decoder_prenet_layers (`int`, *optional*, defaults to 2):
            Number of layers in the speech decoder pre-net.
        speech_decoder_prenet_units (`int`, *optional*, defaults to 256):
            Dimensionality of the layers in the speech decoder pre-net.
        speech_decoder_prenet_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability for the speech decoder pre-net layers.
        speaker_embedding_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the *XVector* embedding vectors.
        speech_decoder_postnet_layers (`int`, *optional*, defaults to 5):
            Number of layers in the speech decoder post-net.
        speech_decoder_postnet_units (`int`, *optional*, defaults to 256):
            Dimensionality of the layers in the speech decoder post-net.
        speech_decoder_postnet_kernel (`int`, *optional*, defaults to 5):
            Number of convolutional filter channels in the speech decoder post-net.
        speech_decoder_postnet_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability for the speech decoder post-net layers.
        reduction_factor (`int`, *optional*, defaults to 2):
            Spectrogram length reduction factor for the speech decoder inputs.
        max_speech_positions (`int`, *optional*, defaults to 4000):
            The maximum sequence length of speech features that this model might ever be used with.
        max_text_positions (`int`, *optional*, defaults to 450):
            The maximum sequence length of text features that this model might ever be used with.
        encoder_max_relative_position (`int`, *optional*, defaults to 160):
            Maximum distance for relative position embedding in the encoder.
        use_guided_attention_loss (`bool`, *optional*, defaults to `True`):
            Whether to apply guided attention loss while training the TTS model.
        guided_attention_loss_num_heads (`int`, *optional*, defaults to 2):
            Number of attention heads the guided attention loss will be applied to. Use -1 to apply this loss to all
            attention heads.
        guided_attention_loss_sigma (`float`, *optional*, defaults to 0.4):
            Standard deviation for guided attention loss.
        guided_attention_loss_scale (`float`, *optional*, defaults to 10.0):
            Scaling coefficient for guided attention loss (also known as lambda).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import SpeechT5Model, SpeechT5Config

    >>> # Initializing a "microsoft/speecht5_asr" style configuration
    >>> configuration = SpeechT5Config()

    >>> # Initializing a model (with random weights) from the "microsoft/speecht5_asr" style configuration
    >>> model = SpeechT5Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## SpeechT5HifiGanConfig


    This is the configuration class to store the configuration of a [`SpeechT5HifiGanModel`]. It is used to instantiate
    a SpeechT5 HiFi-GAN vocoder model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the SpeechT5
    [microsoft/speecht5_hifigan](https://huggingface.co/microsoft/speecht5_hifigan) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the output audio will be generated, expressed in hertz (Hz).
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 8, 8]`):
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
    >>> from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig

    >>> # Initializing a "microsoft/speecht5_hifigan" style configuration
    >>> configuration = SpeechT5HifiGanConfig()

    >>> # Initializing a model (with random weights) from the "microsoft/speecht5_hifigan" style configuration
    >>> model = SpeechT5HifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## SpeechT5Tokenizer


    Construct a SpeechT5 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The begin of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        normalize (`bool`, *optional*, defaults to `False`):
            Whether to convert numeric quantities in the text to their spelt-out english counterparts.
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

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    

Methods: __call__
    - save_vocabulary
    - decode
    - batch_decode

## SpeechT5FeatureExtractor


    Constructs a SpeechT5 feature extractor.

    This class can pre-process a raw speech signal by (optionally) normalizing to zero-mean unit-variance, for use by
    the SpeechT5 speech encoder prenet.

    This class can also extract log-mel filter bank features from raw speech, for use by the SpeechT5 speech decoder
    prenet.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models.
        num_mel_bins (`int`, *optional*, defaults to 80):
            The number of mel-frequency bins in the extracted spectrogram features.
        hop_length (`int`, *optional*, defaults to 16):
            Number of ms between windows. Otherwise referred to as "shift" in many papers.
        win_length (`int`, *optional*, defaults to 64):
            Number of ms per window.
        win_function (`str`, *optional*, defaults to `"hann_window"`):
            Name for the window function used for windowing, must be accessible via `torch.{win_function}`
        frame_signal_scale (`float`, *optional*, defaults to 1.0):
            Constant multiplied in creating the frames before applying DFT. This argument is deprecated.
        fmin (`float`, *optional*, defaults to 80):
            Minimum mel frequency in Hz.
        fmax (`float`, *optional*, defaults to 7600):
            Maximum mel frequency in Hz.
        mel_floor (`float`, *optional*, defaults to 1e-10):
            Minimum value of mel frequency banks.
        reduction_factor (`int`, *optional*, defaults to 2):
            Spectrogram length reduction factor. This argument is deprecated.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether or not [`~SpeechT5FeatureExtractor.__call__`] should return `attention_mask`.
    

Methods: __call__

## SpeechT5Processor


    Constructs a SpeechT5 processor which wraps a feature extractor and a tokenizer into a single processor.

    [`SpeechT5Processor`] offers all the functionalities of [`SpeechT5FeatureExtractor`] and [`SpeechT5Tokenizer`]. See
    the docstring of [`~SpeechT5Processor.__call__`] and [`~SpeechT5Processor.decode`] for more information.

    Args:
        feature_extractor (`SpeechT5FeatureExtractor`):
            An instance of [`SpeechT5FeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`SpeechT5Tokenizer`):
            An instance of [`SpeechT5Tokenizer`]. The tokenizer is a required input.
    

Methods: __call__
    - pad
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## SpeechT5Model

The bare SpeechT5 Encoder-Decoder Model outputting raw hidden-states without any specific pre- or post-nets.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        encoder ([`SpeechT5EncoderWithSpeechPrenet`] or [`SpeechT5EncoderWithTextPrenet`] or `None`):
            The Transformer encoder module that applies the appropiate speech or text encoder prenet. If `None`,
            [`SpeechT5EncoderWithoutPrenet`] will be used and the `input_values` are assumed to be hidden states.
        decoder ([`SpeechT5DecoderWithSpeechPrenet`] or [`SpeechT5DecoderWithTextPrenet`] or `None`):
            The Transformer decoder module that applies the appropiate speech or text decoder prenet. If `None`,
            [`SpeechT5DecoderWithoutPrenet`] will be used and the `decoder_input_values` are assumed to be hidden
            states.


Methods: forward

## SpeechT5ForSpeechToText

SpeechT5 Model with a speech encoder and a text decoder.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## SpeechT5ForTextToSpeech

SpeechT5 Model with a text encoder and a speech decoder.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - generate

## SpeechT5ForSpeechToSpeech

SpeechT5 Model with a speech encoder and a speech decoder.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - generate_speech

## SpeechT5HifiGan

HiFi-GAN vocoder.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5HifiGanConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
