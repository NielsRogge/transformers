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

# CLVP

## Overview

The CLVP (Contrastive Language-Voice Pretrained Transformer) model was proposed in [Better speech synthesis through scaling](https://arxiv.org/abs/2305.07243) by James Betker.

The abstract from the paper is the following:

*In recent years, the field of image generation has been revolutionized by the application of autoregressive transformers and DDPMs. These approaches model the process of image generation as a step-wise probabilistic processes and leverage large amounts of compute and data to learn the image distribution. This methodology of improving performance need not be confined to images. This paper describes a way to apply advances in the image generative domain to speech synthesis. The result is TorToise - an expressive, multi-voice text-to-speech system.*


This model was contributed by [Susnato Dhar](https://huggingface.co/susnato).
The original code can be found [here](https://github.com/neonbjb/tortoise-tts).


## Usage tips

1. CLVP is an integral part of the Tortoise TTS model.
2. CLVP can be used to compare different generated speech candidates with the provided text, and the best speech tokens are forwarded to the diffusion model.
3. The use of the [`ClvpModelForConditionalGeneration.generate()`] method is strongly recommended for tortoise usage.
4. Note that the CLVP model expects the audio to be sampled at 22.05 kHz contrary to other audio models which expects 16 kHz. 


## Brief Explanation:

- The [`ClvpTokenizer`] tokenizes the text input, and the [`ClvpFeatureExtractor`] extracts the log mel-spectrogram from the desired audio.
- [`ClvpConditioningEncoder`] takes those text tokens and audio representations and converts them into embeddings conditioned on the text and audio.
- The [`ClvpForCausalLM`] uses those embeddings to generate multiple speech candidates.
- Each speech candidate is passed through the speech encoder ([`ClvpEncoder`]) which converts them into a vector representation, and the text encoder ([`ClvpEncoder`]) converts the text tokens into the same latent space. 
- At the end, we compare each speech vector with the text vector to see which speech vector is most similar to the text vector. 
- [`ClvpModelForConditionalGeneration.generate()`] compresses all of the logic described above into a single method.  


Example :

```python
>>> import datasets
>>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

>>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library).
>>> text = "This is an example text."

>>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
>>> sample = ds[0]["audio"]

>>> # Define processor and model.
>>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
>>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

>>> # Generate processor output and model output.
>>> processor_output = processor(raw_speech=sample["array"], sampling_rate=sample["sampling_rate"], text=text, return_tensors="pt")
>>> generated_output = model.generate(**processor_output)
```


## ClvpConfig


    [`ClvpConfig`] is the configuration class to store the configuration of a [`ClvpModelForConditionalGeneration`]. It
    is used to instantiate a CLVP model according to the specified arguments, defining the text model, speech model and
    decoder model configs. Instantiating a configuration with the defaults will yield a similar configuration to that
    of the CLVP [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize the CLVP text encoder.
        speech_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize CLVP speech encoder.
        decoder_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClvpDecoderConfig`].
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of text and speech projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLVP implementation.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ClvpConfig, ClvpModelForConditionalGeneration

    >>> # Initializing a ClvpConfig with susnato/clvp_dev style configuration
    >>> configuration = ClvpConfig()

    >>> # Initializing a ClvpModelForConditionalGeneration (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpModelForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLVPConfig from a CLVPTextConfig, CLVPSpeechConfig and a CLVPAutoRegressiveConfig
    >>> from transformers import ClvpEncoderConfig, ClvpDecoderConfig

    >>> # Initializing a CLVP text, CLVP speech and CLVP decoder configuration
    >>> config_text = ClvpEncoderConfig()
    >>> config_speech = ClvpEncoderConfig()
    >>> decoder_config = ClvpDecoderConfig()

    >>> config = ClvpConfig.from_sub_model_configs(config_text, config_speech, decoder_config)
    ```

Methods: from_sub_model_configs

## ClvpEncoderConfig


    This is the configuration class to store the configuration of a [`ClvpEncoder`]. It is used to instantiate a CLVP
    text or CLVP speech encoder according to the specified arguments. Instantiating a configuration with the defaults
    will yield a similar configuration to that of the encoder of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size of the CLVP Encoder model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the projection vector.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the feed-forward layers in [`ClvpEncoderMLP`].
        use_rotary_embedding (`bool`, *optional*, defaults to `True`):
            Whether to use rotary_embedding or not.
        use_attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in Query, Key and Value layers during self attention.
        summary_type (`str`, *optional*, defaults to `"mean"`):
            What strategy to use to get pooler_output from the last_hidden_state. `"last"`, `"first"`, `"mean"` and
            `"cls_index"` are supported.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        bos_token_id (`int`, *optional*, defaults to 255):
            Beginning of sequence token id.
        eos_token_id (`int`, *optional*, defaults to 0):
            End of sequence token id.

    Example:

    ```python
    >>> from transformers import ClvpEncoderConfig, ClvpEncoder

    >>> # Initializing a ClvpEncoderConfig with susnato/clvp_dev style configuration
    >>> encoder_configuration = ClvpEncoderConfig()

    >>> # Initializing a ClvpEncoder (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpEncoder(encoder_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## ClvpDecoderConfig


    This is the configuration class to store the configuration of a [`ClvpDecoder`]. It is used to instantiate a CLVP
    Decoder Model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Decoder part of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    The architecture is similar to GPT2.

    Args:
        vocab_size (`int`, *optional*, defaults to 8194):
            Vocabulary size of the model.
        max_position_embeddings (`int`, *optional*, defaults to 608):
            The maximum sequence length of mel tokens that this model might ever be used with. Similar to `n_positions`
            in `GPT2Config`.
        max_text_tokens (`int`, *optional*, defaults to 404):
            The maximum sequence length of text tokens that this model might ever be used with. Similar to
            `n_positions` in `GPT2Config`.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 30):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times `hidden_size`.
        num_mel_attn_blocks (`int`, *optional*, defaults to 6):
            Denotes the number of self attention layers in [`ClvpConditioningEncoder`].
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary.

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio to be used after the projection and activation.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 8192):
            Beginning of sequence token id, used at the start of the generation.
        eos_token_id (`int`, *optional*, defaults to 8193):
            End of sequence token id, used in the method
            [`ClvpModelForConditionalGeneration.fix_speech_decoder_output()`] to correct decoder outputs.
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted mel features. This value is used in [`ClvpConditioningEncoder`].
        use_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in Query, Key and Value layers during self attention.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        decoder_fixing_codes (`list`, *optional*, defaults to `[83, 45, 45, 248]`):
            These values are used in the method `fix_speech_decoder_output` to fix decoder generated outputs.

    Example:

    ```python
    >>> from transformers import ClvpDecoderConfig, ClvpDecoder

    >>> # Initializing a ClvpDecoderConfig with susnato/clvp_dev style configuration
    >>> decoder_configuration = ClvpDecoderConfig()

    >>> # Initializing a ClvpDecoder (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpDecoder(decoder_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## ClvpTokenizer


    Construct a CLVP tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import ClvpTokenizer

    >>> tokenizer = ClvpTokenizer.from_pretrained("susnato/clvp_dev")
    >>> tokenizer("Hello world")["input_ids"]
    [62, 84, 28, 2, 179, 79]

    >>> tokenizer(" Hello world")["input_ids"]
    [2, 62, 84, 28, 2, 179, 79]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"[STOP]"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"[STOP]"`):
            The pad token of the sequence.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (CLVP tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether to add `bos_token` in front of the sequence when add_special_tokens=True.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether to add `eos_token` in end of the sequence when add_special_tokens=True.
    

Methods: save_vocabulary

## ClvpFeatureExtractor


    Constructs a CLVP feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts log-mel-spectrogram features from raw speech using a custom numpy implementation of the `Short
    Time Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 22050):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        default_audio_length (`int`, *optional*, defaults to 6):
            The default length of raw audio in seconds. If `max_length` is not set during `__call__` then it will
            automatically be set to default_audio_length * `self.sampling_rate`.
        hop_length (`int`, *optional*, defaults to 256):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        mel_norms (`list` of length `feature_size`, *optional*):
            If `mel_norms` is provided then it will be used to normalize the log-mel spectrograms along each
            mel-filter.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the attention mask. If left to the default, it will return the attention mask.

            [What are attention masks?](../glossary#attention-mask)
    

Methods: __call__

## ClvpProcessor


    Constructs a CLVP processor which wraps a CLVP Feature Extractor and a CLVP Tokenizer into a single processor.

    [`ClvpProcessor`] offers all the functionalities of [`ClvpFeatureExtractor`] and [`ClvpTokenizer`]. See the
    [`~ClvpProcessor.__call__`], [`~ClvpProcessor.decode`] and [`~ClvpProcessor.batch_decode`] for more information.

    Args:
        feature_extractor (`ClvpFeatureExtractor`):
            An instance of [`ClvpFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`ClvpTokenizer`):
            An instance of [`ClvpTokenizer`]. The tokenizer is a required input.
    

Methods: __call__
    - decode
    - batch_decode

## ClvpModelForConditionalGeneration

The composite CLVP model with a text encoder, speech encoder and speech decoder model.The speech decoder model generates the speech_ids from the text and the text encoder and speech encoder workstogether to filter out the best speech_ids.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ClvpConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - generate
    - get_text_features
    - get_speech_features

## ClvpForCausalLM

The CLVP decoder model with a language modelling head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ClvpConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


## ClvpModel

The bare Clvp decoder model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ClvpConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


## ClvpEncoder


    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`ClvpEncoderLayer`].

    Args:
        config: ClvpConfig
    

## ClvpDecoder


    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ClvpDecoderLayer`]
    

