<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Speech2Text

## Overview

The Speech2Text model was proposed in [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) by Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino. It's a
transformer-based seq2seq (encoder-decoder) model designed for end-to-end Automatic Speech Recognition (ASR) and Speech
Translation (ST). It uses a convolutional downsampler to reduce the length of speech inputs by 3/4th before they are
fed into the encoder. The model is trained with standard autoregressive cross-entropy loss and generates the
transcripts/translations autoregressively. Speech2Text has been fine-tuned on several datasets for ASR and ST:
[LibriSpeech](http://www.openslr.org/12), [CoVoST 2](https://github.com/facebookresearch/covost), [MuST-C](https://ict.fbk.eu/must-c/).

This model was contributed by [valhalla](https://huggingface.co/valhalla). The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text).

## Inference

Speech2Text is a speech model that accepts a float tensor of log-mel filter-bank features extracted from the speech
signal. It's a transformer-based seq2seq model, so the transcripts/translations are generated autoregressively. The
`generate()` method can be used for inference.

The [`Speech2TextFeatureExtractor`] class is responsible for extracting the log-mel filter-bank
features. The [`Speech2TextProcessor`] wraps [`Speech2TextFeatureExtractor`] and
[`Speech2TextTokenizer`] into a single instance to both extract the input features and decode the
predicted token ids.

The feature extractor depends on `torchaudio` and the tokenizer depends on `sentencepiece` so be sure to
install those packages before running the examples. You could either install those as extra speech dependencies with
`pip install transformers"[speech, sentencepiece]"` or install the packages separately with `pip install torchaudio sentencepiece`. Also `torchaudio` requires the development version of the [libsndfile](http://www.mega-nerd.com/libsndfile/) package which can be installed via a system package manager. On Ubuntu it can
be installed as follows: `apt install libsndfile1-dev`

- ASR and Speech Translation

```python
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> transcription
['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel']
```

- Multilingual speech translation

  For multilingual speech translation models, `eos_token_id` is used as the `decoder_start_token_id` and
  the target language id is forced as the first generated token. To force the target language id as the first
  generated token, pass the `forced_bos_token_id` parameter to the `generate()` method. The following
  example shows how to transate English speech to French text using the *facebook/s2t-medium-mustc-multilingual-st*
  checkpoint.

```python
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(
...     inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     forced_bos_token_id=processor.tokenizer.lang_code_to_id["fr"],
... )

>>> translation = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> translation
["(Vidéo) Si M. Kilder est l'apossible des classes moyennes, et nous sommes heureux d'être accueillis dans son évangile."]
```

See the [model hub](https://huggingface.co/models?filter=speech_to_text) to look for Speech2Text checkpoints.

## Speech2TextConfig


    This is the configuration class to store the configuration of a [`Speech2TextModel`]. It is used to instantiate a
    Speech2Text model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Speech2Text
    [facebook/s2t-small-librispeech-asr](https://huggingface.co/facebook/s2t-small-librispeech-asr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 10000):
            Vocabulary size of the Speech2Text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`Speech2TextModel`]
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](https://arxiv.org/abs/1909.11556) for
            more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](https://arxiv.org/abs/1909.11556) for
            more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models).
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is set up as an encoder-decoder architecture for sequence-to-sequence tasks.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        d_model (`int`, *optional*, defaults to 256):
            Dimensionality of the layers and the pooler layer.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        decoder_start_token_id (`int`, *optional*, defaults to 2):
            The initial token ID of the decoder when decoding sequences.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Whether the embeddings are scaled by the square root of `d_model`.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end-of-sequence token.
        max_source_positions (`int`, *optional*, defaults to 6000):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        max_target_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        num_conv_layers (`int`, *optional*, defaults to 2):
            Number of 1D convolutional layers in the conv module.
        conv_kernel_sizes (`Tuple[int]`, *optional*, defaults to `(5, 5)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the conv module. The length
            of `conv_kernel_sizes` has to match `num_conv_layers`.
        conv_channels (`int`, *optional*, defaults to 1024):
            An integer defining the number of output channels of each convolution layers except the final one in the
            conv module.
        input_feat_per_channel (`int`, *optional*, defaults to 80):
            An integer specifying the size of feature vector. This is also the dimensions of log-mel filter-bank
            features.
        input_channels (`int`, *optional*, defaults to 1):
            An integer specifying number of input channels of the input feature vector.

    Example:

    ```python
    >>> from transformers import Speech2TextConfig, Speech2TextModel

    >>> # Initializing a Speech2Text s2t_transformer_s style configuration
    >>> configuration = Speech2TextConfig()

    >>> # Initializing a model (with random weights) from the s2t_transformer_s style configuration
    >>> model = Speech2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## Speech2TextTokenizer


    Construct an Speech2Text tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        spm_file (`str`):
            Path to the [SentencePiece](https://github.com/google/sentencepiece) model file
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        do_upper_case (`bool`, *optional*, defaults to `False`):
           Whether or not to uppercase the output when decoding.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        tgt_lang (`str`, *optional*):
            A string representing the target language.
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

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    

Methods: build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## Speech2TextFeatureExtractor


    Constructs a Speech2Text feature extractor.

    This feature extractor inherits from [`Speech2TextFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using TorchAudio if installed or using numpy
    otherwise, and applies utterance-level cepstral mean and variance normalization to the extracted features.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of Mel-frequency bins.
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding vectors.
        do_ceptral_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to apply utterance-level cepstral mean and variance normalization to extracted features.
        normalize_means (`bool`, *optional*, defaults to `True`):
            Whether or not to zero-mean normalize the extracted features.
        normalize_vars (`bool`, *optional*, defaults to `True`):
            Whether or not to unit-variance normalize the extracted features.
    

Methods: __call__

## Speech2TextProcessor


    Constructs a Speech2Text processor which wraps a Speech2Text feature extractor and a Speech2Text tokenizer into a
    single processor.

    [`Speech2TextProcessor`] offers all the functionalities of [`Speech2TextFeatureExtractor`] and
    [`Speech2TextTokenizer`]. See the [`~Speech2TextProcessor.__call__`] and [`~Speech2TextProcessor.decode`] for more
    information.

    Args:
        feature_extractor (`Speech2TextFeatureExtractor`):
            An instance of [`Speech2TextFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Speech2TextTokenizer`):
            An instance of [`Speech2TextTokenizer`]. The tokenizer is a required input.
    

Methods: __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

<frameworkcontent>
<pt>

## Speech2TextModel

The bare Speech2Text Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Speech2TextConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## Speech2TextForConditionalGeneration

The Speech2Text Model with a language modeling head. Can be used for summarization.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Speech2TextConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

</pt>
<tf>

## TFSpeech2TextModel

No docstring available for TFSpeech2TextModel

Methods: call

## TFSpeech2TextForConditionalGeneration

No docstring available for TFSpeech2TextForConditionalGeneration

Methods: call

</tf>
</frameworkcontent>
