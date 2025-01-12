<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SeamlessM4T

## Overview

The SeamlessM4T model was proposed in [SeamlessM4T — Massively Multilingual & Multimodal Machine Translation](https://dl.fbaipublicfiles.com/seamless/seamless_m4t_paper.pdf) by the Seamless Communication team from Meta AI.

This is the **version 1** release of the model. For the updated **version 2** release, refer to the [Seamless M4T v2 docs](https://huggingface.co/docs/transformers/main/model_doc/seamless_m4t_v2).

SeamlessM4T is a collection of models designed to provide high quality translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.

SeamlessM4T enables multiple tasks without relying on separate models:

- Speech-to-speech translation (S2ST)
- Speech-to-text translation (S2TT)
- Text-to-speech translation (T2ST)
- Text-to-text translation (T2TT)
- Automatic speech recognition (ASR)

[`SeamlessM4TModel`] can perform all the above tasks, but each task also has its own dedicated sub-model.

The abstract from the paper is the following:

*What does it take to create the Babel Fish, a tool that can help individuals translate speech between any two languages? While recent breakthroughs in text-based models have pushed machine translation coverage beyond 200 languages, unified speech-to-speech translation models have yet to achieve similar strides. More specifically, conventional speech-to-speech translation systems rely on cascaded systems that perform translation progressively, putting high-performing unified systems out of reach. To address these gaps, we introduce SeamlessM4T, a single model that supports speech-to-speech translation, speech-to-text translation, text-to-speech translation, text-to-text translation, and automatic speech recognition for up to 100 languages. To build this, we used 1 million hours of open speech audio data to learn self-supervised speech representations with w2v-BERT 2.0. Subsequently, we created a multimodal corpus of automatically aligned speech translations. Filtered and combined with human-labeled and pseudo-labeled data, we developed the first multilingual system capable of translating from and into English for both speech and text. On FLEURS, SeamlessM4T sets a new standard for translations into multiple target languages, achieving an improvement of 20% BLEU over the previous SOTA in direct speech-to-text translation. Compared to strong cascaded models, SeamlessM4T improves the quality of into-English translation by 1.3 BLEU points in speech-to-text and by 2.6 ASR-BLEU points in speech-to-speech. Tested for robustness, our system performs better against background noises and speaker variations in speech-to-text tasks compared to the current SOTA model. Critically, we evaluated SeamlessM4T on gender bias and added toxicity to assess translation safety. Finally, all contributions in this work are open-sourced and accessible at https://github.com/facebookresearch/seamless_communication*

## Usage

First, load the processor and a checkpoint of the model:

```python
>>> from transformers import AutoProcessor, SeamlessM4TModel

>>> processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
>>> model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
```

You can seamlessly use this model on text or on audio, to generated either translated text or translated audio.

Here is how to use the processor to process text and audio:

```python
>>> # let's load an audio sample from an Arabic speech corpus
>>> from datasets import load_dataset
>>> dataset = load_dataset("arabic_speech_corpus", split="test", streaming=True)
>>> audio_sample = next(iter(dataset))["audio"]

>>> # now, process it
>>> audio_inputs = processor(audios=audio_sample["array"], return_tensors="pt")

>>> # now, process some English test as well
>>> text_inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt")
```


### Speech

[`SeamlessM4TModel`] can *seamlessly* generate text or speech with few or no changes. Let's target Russian voice translation:

```python
>>> audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
>>> audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
```

With basically the same code, I've translated English text and Arabic speech to Russian speech samples.

### Text

Similarly, you can generate translated text from audio files or from text with the same model. You only have to pass `generate_speech=False` to [`SeamlessM4TModel.generate`].
This time, let's translate to French.

```python 
>>> # from audio
>>> output_tokens = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)
>>> translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

>>> # from text
>>> output_tokens = model.generate(**text_inputs, tgt_lang="fra", generate_speech=False)
>>> translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
```

### Tips


#### 1. Use dedicated models

[`SeamlessM4TModel`] is transformers top level model to generate speech and text, but you can also use dedicated models that perform the task without additional components, thus reducing the memory footprint.
For example, you can replace the audio-to-audio generation snippet with the model dedicated to the S2ST task, the rest is exactly the same code: 

```python
>>> from transformers import SeamlessM4TForSpeechToSpeech
>>> model = SeamlessM4TForSpeechToSpeech.from_pretrained("facebook/hf-seamless-m4t-medium")
```

Or you can replace the text-to-text generation snippet with the model dedicated to the T2TT task, you only have to remove `generate_speech=False`.

```python
>>> from transformers import SeamlessM4TForTextToText
>>> model = SeamlessM4TForTextToText.from_pretrained("facebook/hf-seamless-m4t-medium")
```

Feel free to try out [`SeamlessM4TForSpeechToText`] and [`SeamlessM4TForTextToSpeech`] as well.

#### 2. Change the speaker identity

You have the possibility to change the speaker used for speech synthesis with the `spkr_id` argument. Some `spkr_id` works better than other for some languages!

#### 3. Change the generation strategy

You can use different [generation strategies](./generation_strategies) for speech and text generation, e.g `.generate(input_ids=input_ids, text_num_beams=4, speech_do_sample=True)` which will successively perform beam-search decoding on the text model, and multinomial sampling on the speech model.

#### 4. Generate speech and text at the same time

Use `return_intermediate_token_ids=True` with [`SeamlessM4TModel`] to return both speech and text !

## Model architecture


SeamlessM4T features a versatile architecture that smoothly handles the sequential generation of text and speech. This setup comprises two sequence-to-sequence (seq2seq) models. The first model translates the input modality into translated text, while the second model generates speech tokens, known as "unit tokens," from the translated text.

Each modality has its own dedicated encoder with a unique architecture. Additionally, for speech output, a vocoder inspired by the [HiFi-GAN](https://arxiv.org/abs/2010.05646) architecture is placed on top of the second seq2seq model.

Here's how the generation process works:

- Input text or speech is processed through its specific encoder.
- A decoder creates text tokens in the desired language.
- If speech generation is required, the second seq2seq model, following a standard encoder-decoder structure, generates unit tokens.
- These unit tokens are then passed through the final vocoder to produce the actual speech.


This model was contributed by [ylacombe](https://huggingface.co/ylacombe). The original code can be found [here](https://github.com/facebookresearch/seamless_communication).

## SeamlessM4TModel

The original SeamlessM4T Model transformer which can be used for every tasks available (S2ST, S2TT, T2TT, T2ST).
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

        current_modality (`str`, *optional*, defaults to `"text"`):
            Default modality. Used to initialize the model.
    

Methods: generate


## SeamlessM4TForTextToSpeech

The text-to-speech SeamlessM4T Model transformer which can be used for T2ST.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: generate


## SeamlessM4TForSpeechToSpeech

The speech-to-speech SeamlessM4T Model transformer which can be used for S2ST.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: generate


## SeamlessM4TForTextToText

The text-to-text SeamlessM4T Model transformer which can be used for T2TT.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - generate

## SeamlessM4TForSpeechToText

The speech-to-text SeamlessM4T Model transformer which can be used for S2TT.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - generate

## SeamlessM4TConfig


    This is the configuration class to store the configuration of a [`~SeamlessM4TModel`]. It is used to instantiate an
    SeamlessM4T model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SeamlessM4T
    ["facebook/hf-seamless-m4t-medium"](https://huggingface.co/"facebook/hf-seamless-m4t-medium") architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 256102):
            Vocabulary size of the SeamlessM4T model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`~SeamlessM4TModel`], [`~SeamlessM4TForTextToSpeech`] or
            [`~SeamlessM4TForTextToText`].
        t2u_vocab_size (`int`, *optional*, defaults to 10082):
            Unit vocabulary size of the SeamlessM4T model. Defines the number of different unit tokens that can be
            represented by the `inputs_ids` passed when calling the Text-To-Units sub-model of [`~SeamlessM4TModel`],
            [`~SeamlessM4TForSpeechToSpeech`] or [`~SeamlessM4TForTextToSpeech`].

        > Parameters shared across sub-models

        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the "intermediate" layers in the architecture.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model text encoder and decoder might ever be used with. Typically set
            this to something large just in case (e.g., 512 or 1024 or 2048).
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.
        encoder_layerdrop (`float`, *optional*, defaults to 0.05):
            The LayerDrop probability for the encoders. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.05):
            The LayerDrop probability for the decoders. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the decoder and feed-forward layers. If string,
            `"gelu"`, `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, decoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all attention layers.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all activation layers in the model.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Scale embeddings by diving by sqrt(d_model).

        > Text encoder and text decoder specific parameters

        encoder_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer text encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 8192):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer text encoder.
        decoder_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer text decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 8192):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer text decoder.
        decoder_start_token_id (`int`, *optional*, defaults to 3):
            If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token. Only
            applied in the text decoder.
        max_new_tokens (`int`, *optional*, defaults to 256):
            The maximum numbers of text tokens to generate, ignoring the number of tokens in the prompt.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the _padding_ text token. Only applied to the text-decoder model.
        bos_token_id (`int`, *optional*, defaults to 2):
            The id of the _beginning-of-stream_ text token. Only applied to the text-decoder model.
        eos_token_id (`int`, *optional*, defaults to 3):
            The id of the _end-of-stream_ text token. Only applied to the text-decoder model.

        > Speech encoder specific parameters

        speech_encoder_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer speech encoder.
        speech_encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer speech encoder.
        speech_encoder_intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer speech encoder.
        speech_encoder_hidden_act (`str` or `function`, *optional*, defaults to `"swish"`):
            The non-linear activation function (function or string) in the speech encoder. If string, `"gelu"`,
            `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
        speech_encoder_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all layers in the speech encoder.
        add_adapter (`bool`, *optional*, defaults to `True`):
            Add an adapter layer on top of the speech encoder.
        speech_encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the speech encoder. See the [LayerDrop paper](see
            https://arxiv.org/abs/1909.11556) for more details.
        feature_projection_input_dim (`int`, *optional*, defaults to 160):
            Input dimension of the input feature projection of the speech encoder, i.e the dimension after processing
            input audios with [`SeamlessM4TFeatureExtractor`].
        num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer of the speech encoder.
        num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer of the speech encoder.
        adaptor_kernel_size (`int`, *optional*, defaults to 8):
            Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adaptor_stride (`int`, *optional*, defaults to 8):
            Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adaptor_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all layers in the speech adapter.
        num_adapter_layers (`int`, *optional*, defaults to 1):
            Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is
            True`.
        position_embeddings_type (`str`, *optional*, defaults to `"relative"`):
            Can be specified to `relative` or `rotary` for relative or rotary position embeddings respectively. If left
            `None` no relative position embedding is applied. Only applied to the speech encoder.
        rotary_embedding_base (`int`, *optional*, defaults to 10000):
            If `"rotary"` position embeddings are used, defines the size of the embedding base. Only applied to the
            speech encoder.
        max_source_positions (`int`, *optional*, defaults to 4096):
            if `"relative"` position embeddings are used, defines the maximum source input positions. Only applied to
            the speech encoder.
        conv_depthwise_kernel_size (`int`, *optional*, defaults to 31):
            Kernel size of convolutional depthwise 1D layer in Conformer blocks. Only applied to the speech encoder.

        > Text-To-Unit (t2u) model specific parameters

        t2u_bos_token_id (`int`, *optional*, defaults to 0):
            The id of the _beginning-of-stream_ unit token. Only applied to the text-to-unit seq2seq model.
        t2u_pad_token_id (`int`, *optional*, defaults to 1):
            The id of the _padding_ unit token. Only applied to the text-to-unit seq2seq model.
        t2u_eos_token_id (`int`, *optional*, defaults to 2):
            The id of the _end-of-stream_ unit token. Only applied to the text-to-unit seq2seq model.
        t2u_decoder_start_token_id (`int`, *optional*, defaults to 2):
            If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token. Only
            applied to the text-to-unit seq2seq model.
        t2u_max_new_tokens (`int`, *optional*, defaults to 1024):
            The maximum numbers of unit tokens to generate, ignoring the number of tokens in the prompt. Only applied
            to the text-to-unit seq2seq model.
        t2u_encoder_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer text-to-unit encoder.
        t2u_encoder_ffn_dim (`int`, *optional*, defaults to 8192):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text-to-unit encoder.
        t2u_encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer text-to-unit encoder.
        t2u_decoder_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer text-to-unit decoder.
        t2u_decoder_ffn_dim (`int`, *optional*, defaults to 8192):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text-to-unit decoder.
        t2u_decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer text-to-unit decoder.
        t2u_max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model text-to-unit component might ever be used with. Typically set
            this to something large just in case (e.g., 512 or 1024 or 2048).

         > Hifi-Gan Vocoder specific parameters

        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the output audio will be generated, expressed in hertz (Hz).
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the hifi-gan upsampling network. Applies to the vocoder only.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[5, 4, 4, 2, 2]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the vocoder upsampling network.
            The length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*. Applies to the vocoder only.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[11, 8, 8, 4, 4]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the vocoder upsampling
            network. The length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match
            the length of *upsample_rates*. Applies to the vocoder only.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the vocoder 1D convolutional layers in the multi-receptive
            field fusion (MRF) module. Applies to the vocoder only.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the vocoder dilated 1D convolutional layers in
            the multi-receptive field fusion (MRF) module. Applies to the vocoder only.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation in the vocoder. Applies to the vocoder
            only.
        unit_hifi_gan_vocab_size (`int`, *optional*, defaults to 10000):
            Vocabulary size of the SeamlessM4T vocoder. Defines the number of different unit tokens that can be
            represented by the `inputs_ids` passed when calling the vocoder of [`~SeamlessM4TModel`],
            [`~SeamlessM4TForSpeechToSpeech`] or [`~SeamlessM4TForTextToSpeech`].
        unit_embed_dim (`int`, *optional*, defaults to 1280):
            The projection dimension of the input ids given to the hifi-gan vocoder. Applies to the vocoder only.
        lang_embed_dim (`int`, *optional*, defaults to 256):
            The projection dimension of the target language given to the hifi-gan vocoder. Applies to the vocoder only.
        spkr_embed_dim (`int`, *optional*, defaults to 256):
            The projection dimension of the speaker id given to the hifi-gan vocoder. Applies to the vocoder only.
        vocoder_num_langs (`int`, *optional*, defaults to 36):
            Number of langs supported by the vocoder. Might be different from `t2u_num_langs`.
        vocoder_num_spkrs (`int`, *optional*, defaults to 200):
            Number of speakers supported by the vocoder.
        variance_predictor_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the duration predictor. Applies to the vocoder only.
        var_pred_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability of the duration predictor. Applies to the vocoder only.
        vocoder_offset (`int`, *optional*, defaults to 4):
            Offset the unit token ids by this number to account for symbol tokens. Applies to the vocoder only.

    ```python
    >>> from transformers import SeamlessM4TModel, SeamlessM4TConfig

    >>> # Initializing a SeamlessM4T "facebook/hf-seamless-m4t-medium" style configuration
    >>> configuration = SeamlessM4TConfig()

    >>> # Initializing a model from the "facebook/hf-seamless-m4t-medium" style configuration
    >>> model = SeamlessM4TModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```


## SeamlessM4TTokenizer


    Construct a SeamlessM4T tokenizer.

    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    The tokenization method is `<language code> <tokens> <eos>` for source language documents, and `<eos> <language
    code> <tokens> <eos>` for target language documents.

    Examples:

    ```python
    >>> from transformers import SeamlessM4TTokenizer

    >>> tokenizer = SeamlessM4TTokenizer.from_pretrained(
    ...     "facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="fra"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    ```

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        src_lang (`str`, *optional*, defaults to `"eng"`):
            The language to use as source language for translation.
        tgt_lang (`str`, *optional*, defaults to `"fra"`):
            The language to use as target language for translation.
        sp_model_kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the model initialization.
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional special tokens. Can be used to specify the list of languages that will be
            supported by the tokenizer.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
    

Methods: __call__
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## SeamlessM4TTokenizerFast


    Construct a "fast" SeamlessM4T tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    The tokenization method is `<language code> <tokens> <eos>` for source language documents, and `<eos> <language
    code> <tokens> <eos>` for target language documents.

    Examples:

    ```python
    >>> from transformers import SeamlessM4TTokenizerFast

    >>> tokenizer = SeamlessM4TTokenizerFast.from_pretrained(
    ...     "facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="fra"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    ```

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        src_lang (`str`, *optional*, defaults to `"eng"`):
            The language to use as source language for translation.
        tgt_lang (`str`, *optional*, defaults to `"fra"`):
            The language to use as target language for translation.
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional special tokens.
    

Methods: __call__

## SeamlessM4TFeatureExtractor


    Constructs a SeamlessM4T feature extractor.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of Mel-frequency bins.
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding vectors.
        stride (`int`, *optional*, defaults to 2):
            Stride used to reshape audios from shape (batch_size,num_frames,num_mel_bins) to
            (batch_size,num_frames//stride,num_mel_bins*stride).
    

Methods: __call__

## SeamlessM4TProcessor


    Constructs a SeamlessM4T processor which wraps a SeamlessM4T feature extractor and a SeamlessM4T tokenizer into a
    single processor.

    [`SeamlessM4TProcessor`] offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and
    [`SeamlessM4TTokenizerFast`]. See the [`~SeamlessM4TProcessor.__call__`] and [`~SeamlessM4TProcessor.decode`] for
    more information.

    Args:
        feature_extractor ([`SeamlessM4TFeatureExtractor`]):
            The audio processor is a required input.
        tokenizer ([`SeamlessM4TTokenizerFast`]):
            The tokenizer is a required input.
    

Methods: __call__

## SeamlessM4TCodeHifiGan

Code HiFi-GAN vocoder as described in this [repository](https://github.com/facebookresearch/speech-resynthesis).
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SeamlessM4TConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.



## SeamlessM4THifiGan

No docstring available for SeamlessM4THifiGan

## SeamlessM4TTextToUnitModel

Transformer bare text-to-unit encoder-decoder. The encoder is a [`SeamlessM4TEncoder`] without embeddings and the decoder is a [`SeamlessM4TDecoder`].
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    

## SeamlessM4TTextToUnitForConditionalGeneration

Transformer text-to-unit encoder-decoder with a language model head. The base encoder-decoder model is a [`SeamlessM4TTextToUnit`].
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    


