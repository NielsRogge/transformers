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

# Whisper

## Overview

The Whisper model was proposed in [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf) by Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever.

The abstract from the paper is the following:

*We study the capabilities of speech processing systems trained simply to predict large amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual and multitask supervision, the resulting models generalize well to standard benchmarks and are often competitive with prior fully supervised results but in a zeroshot transfer setting without the need for any finetuning. When compared to humans, the models approach their accuracy and robustness. We are releasing models and inference code to serve as a foundation for further work on robust speech processing.*

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ). The Tensorflow version of this model was contributed by [amyeroberts](https://huggingface.co/amyeroberts).
The original code can be found [here](https://github.com/openai/whisper).

## Quick usage

You can run Whisper in less than 4 lines of code and transcribe in less than a minute!

```python
# pip install transformers torch

import torch
from transformers import pipeline

whisper = pipeline("automatic-speech-recognition", "openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda:0")

transcription = whisper("<audio_file.mp3>")

print(transcription["text"])
```

Voila! You can swap the model with any [Whisper checkpoints](https://huggingface.co/models?other=whisper&sort=downloads) on the Hugging Face Hub with the same pipeline based on your needs.

Bonus: You can replace `"cuda"` with `"mps"` to make it seamlessly work on Macs.

## Usage tips

- The model usually performs well without requiring any finetuning.
- The architecture follows a classic encoder-decoder architecture, which means that it relies on the [`~generation.GenerationMixin.generate`] function for inference.
- One can use [`WhisperProcessor`] to prepare audio for the model, and decode the predicted ID's back into text.

- To convert the model and the processor, we recommend using the following:

```bash
python src/transformers/models/whisper/convert_openai_to_hf.py --checkpoint_path "" --pytorch_dump_folder_path "Arthur/whisper-3" --convert_preprocessor True
```
The script will automatically determine all necessary parameters from the OpenAI checkpoint. A `tiktoken` library needs to be installed
to perform the conversion of the OpenAI tokenizer to the `tokenizers` version.

## Inference

Here is a step-by-step guide to transcribing an audio sample using a pre-trained Whisper model:

```python
>>> from datasets import load_dataset
>>> from transformers import WhisperProcessor, WhisperForConditionalGeneration

>>> # Select an audio file and read it:
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> audio_sample = ds[0]["audio"]

>>> # Load the Whisper model in Hugging Face format:
>>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

>>> # Use the model and processor to transcribe the audio:
>>> input_features = processor(
...     audio_sample["array"], sampling_rate=audio_sample["sampling_rate"], return_tensors="pt"
... ).input_features

>>> # Generate token ids
>>> predicted_ids = model.generate(input_features)

>>> # Decode token ids to text
>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

>>> transcription[0]
' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

Whisper is compatible with the following optimisations for both short and long-form generation:
- [PyTorch Scaled Dot Product Attention (SDPA)](../perf_infer_gpu_one#pytorch-scaled-dot-product-attention): flash attention and memory-efficient attention kernels. Enabled by default for `torch>=2.1.1`.
- [Flash Attention 2](../perf_infer_gpu_one#flashattention-2): improved implementation of flash attention through better parallelism and work partitioning. 
- [torch.compile](../llm_optims#static-kv-cache-and-torchcompile): JIT-compile the forward pass to dispatch to efficient fused kernels.

As an example, the following codesnippet enables SDPA and `torch.compile` for up to 5x faster inference:

```python
>>> from datasets import load_dataset
>>> from transformers import WhisperProcessor, WhisperForConditionalGeneration

>>> # Select an audio file and read it:
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> audio_sample = ds[0]["audio"]

>>> # Load the Whisper model with SDPA attention
>>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", attn_implementation="sdpa")

>>> # Enable static cache and compile the forward pass
>>> model.generation_config.cache_implementation = "static"
>>> model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

>>> # Use the model and processor to transcribe the audio:
>>> input_features = processor(
...     audio_sample["array"], sampling_rate=audio_sample["sampling_rate"], return_tensors="pt"
... ).input_features

>>> # Compile the forward pass
>>> for _ in range(2):
>>>     model.generate(input_features)

>>> # Generate token ids using compiled graph (fast!)
>>> predicted_ids = model.generate(input_features)

>>> # Decode token ids to text
>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

>>> transcription[0]
' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

For more details on each optimisation, refer to the documentation linked above.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Whisper. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- [Fine-tune Whisper](https://huggingface.co/blog/fine-tune-whisper) on your own dataset for better downstream performance.
- [Distil-Whisper](https://huggingface.co/distil-whisper): Upto 6x faster, 2x smaller distilled Whisper models for English. We release the [model checkpoints](https://huggingface.co/distil-whisper), and [distillation code](https://github.com/huggingface/distil-whisper).
- A fork with a script to [convert a Whisper model in Hugging Face format to OpenAI format](https://github.com/zuazo-forks/transformers/blob/convert_hf_to_openai/src/transformers/models/whisper/convert_hf_to_openai.py). 🌎
Usage example:
```bash
pip install -U openai-whisper
python convert_hf_to_openai.py \
    --checkpoint openai/whisper-tiny \
    --whisper_dump_path whisper-tiny-openai.pt
```

## WhisperConfig


    This is the configuration class to store the configuration of a [`WhisperModel`]. It is used to instantiate a
    Whisper model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Whisper
    [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 51865):
            Vocabulary size of the Whisper model. Defines the number of different tokens that can be represented by the
            `decoder_input_ids` passed when calling [`WhisperModel`]
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel features used per input features. Should correspond to the value used in the
            `WhisperProcessor` class.
        encoder_layers (`int`, *optional*, defaults to 4):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 4):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_start_token_id (`int`, *optional*, defaults to 50257):
            Corresponds to the "<|startoftranscript|>" token, which is automatically used when no `decoder_input_ids`
            are provided to the `generate` function. It is used to guide the model`s generation process depending on
            the task.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        d_model (`int`, *optional*, defaults to 384):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to False):
            Scale embeddings by diving by sqrt(d_model).
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        max_target_positions (`int`, *optional*, defaults to 448):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        pad_token_id (`int`, *optional*, defaults to 50256):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 50256):
            Begin of stream token id.
        eos_token_id (`int`, *optional*, defaults to 50256):
            End of stream token id.
        suppress_tokens (`List[int]`, *optional*):
            A list containing the non-speech tokens that will be used by the logit processor in the `generate`
            function. NON_SPEECH_TOKENS and NON_SPEECH_TOKENS_MULTI each correspond to the `english-only` and the
            `multilingual` model.
        begin_suppress_tokens (`List[int]`, *optional*, defaults to `[220,50256]`):
            A list containing tokens that will be supressed at the beginning of the sampling process. Initialized as
            the token for `" "` (`blank_token_id`) and the `eos_token_id`
        use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
            Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
            instance of [`WhisperForAudioClassification`].
        classifier_proj_size (`int`, *optional*, defaults to 256):
            Dimensionality of the projection before token mean-pooling for classification. Only relevant when using an
            instance of [`WhisperForAudioClassification`].
        apply_spec_augment (`bool`, *optional*, defaults to `False`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
            [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
            Recognition](https://arxiv.org/abs/1904.08779).
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
            procecure generates `mask_time_prob*len(time_axis)/mask_time_length` independent masks over the axis. If
            reasoning from the propability of each feature vector to be chosen as the start of the vector span to be
            masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
            actual percentage of masked vectors. This is only relevant if `apply_spec_augment == True`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2),:
            The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
            mask_time_min_masks''
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procecure generates `mask_feature_prob*len(feature_axis)/mask_time_length` independent masks over
            the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector
            span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
            may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
            True`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        mask_feature_min_masks (`int`, *optional*, defaults to 0),:
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            step, irrespectively of `mask_feature_prob`. Only relevant if
            `mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks`.
        median_filter_width (`int`, *optional*, defaults to 7):
            Width of the median filter used to smoothen to cross-attention outputs when computing token timestamps.
            Should be an odd number.

    Example:

    ```python
    >>> from transformers import WhisperConfig, WhisperModel

    >>> # Initializing a Whisper tiny style configuration
    >>> configuration = WhisperConfig()

    >>> # Initializing a model (with random weights) from the tiny style configuration
    >>> model = WhisperModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## WhisperTokenizer


    Construct a Whisper tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        normalizer_file (`str`, *optional*):
            Path to the normalizer_file file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token. The `decoder_start_token_id` is used to set the first token as
            `"<|startoftranscript|>"` when generating.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
        language (`str`, *optional*):
            The language of the transcription text. The corresponding language id token is appended to the start of the
            sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
            `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
        task (`str`, *optional*):
            Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
            fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
        predict_timestamps (`bool`, *optional*, defaults to `False`):
            Whether to omit the `<|notimestamps|>` token at the start of the sequence.
    

Methods: set_prefix_tokens
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary
    - batch_decode
    - decode
    - basic_normalize
    - normalize

## WhisperTokenizerFast


    Construct a "fast" Whisper tokenizer (backed by HuggingFace's *tokenizers* library).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        normalizer_file (`str`, *optional*):
            Path to the normalizer_file file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token. The `decoder_start_token_id` is used to set the first token as
            `"<|startoftranscript|>"` when generating.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Whisper tokenizer detect beginning of words by the preceding space).
        language (`str`, *optional*):
            The language of the transcription text. The corresponding language id token is appended to the start of the
            sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
            `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
        task (`str`, *optional*):
            Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
            fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
        predict_timestamps (`bool`, *optional*, defaults to `False`):
            Whether to omit the `<|notimestamps|>` token at the start of the sequence.
    

Methods: set_prefix_tokens
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary
    - batch_decode
    - decode
    - basic_normalize
    - normalize

## WhisperFeatureExtractor


    Constructs a Whisper feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time
    Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    

Methods: __call__

## WhisperProcessor


    Constructs a Whisper processor which wraps a Whisper feature extractor and a Whisper tokenizer into a single
    processor.

    [`WhisperProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`WhisperTokenizer`]. See
    the [`~WhisperProcessor.__call__`] and [`~WhisperProcessor.decode`] for more information.

    Args:
        feature_extractor (`WhisperFeatureExtractor`):
            An instance of [`WhisperFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`WhisperTokenizer`):
            An instance of [`WhisperTokenizer`]. The tokenizer is a required input.
    

Methods: __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

<frameworkcontent>
<pt>

## WhisperModel

The bare Whisper Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`WhisperConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - _mask_input_features

## WhisperForConditionalGeneration

The Whisper Model with a language modeling head. Can be used for automatic speech recognition.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`WhisperConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - generate

## WhisperForCausalLM


    Whisper decoder with a language modeling head on top (linear layer with weights tied to the input embeddings).
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`WhisperConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## WhisperForAudioClassification


    Whisper Encoder Model with a sequence classification head on top (a linear layer over the pooled output) for tasks
    like SUPERB Keyword Spotting.
    
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.


Methods: forward

</pt>
<tf>

## TFWhisperModel

No docstring available for TFWhisperModel

Methods: call

## TFWhisperForConditionalGeneration

No docstring available for TFWhisperForConditionalGeneration

Methods: call

</tf>
<jax>

## FlaxWhisperModel

No docstring available for FlaxWhisperModel

Methods: __call__

## FlaxWhisperForConditionalGeneration

No docstring available for FlaxWhisperForConditionalGeneration

Methods: __call__

## FlaxWhisperForAudioClassification

No docstring available for FlaxWhisperForAudioClassification

Methods: __call__

</jax>
</frameworkcontent>

