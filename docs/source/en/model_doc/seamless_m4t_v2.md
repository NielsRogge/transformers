<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SeamlessM4T-v2

## Overview

The SeamlessM4T-v2 model was proposed in [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) by the Seamless Communication team from Meta AI.

SeamlessM4T-v2 is a collection of models designed to provide high quality translation, allowing people from different linguistic communities to communicate effortlessly through speech and text. It is an improvement on the [previous version](https://huggingface.co/docs/transformers/main/model_doc/seamless_m4t). For more details on the differences between v1 and v2, refer to section [Difference with SeamlessM4T-v1](#difference-with-seamlessm4t-v1).

SeamlessM4T-v2 enables multiple tasks without relying on separate models:

- Speech-to-speech translation (S2ST)
- Speech-to-text translation (S2TT)
- Text-to-speech translation (T2ST)
- Text-to-text translation (T2TT)
- Automatic speech recognition (ASR)

[`SeamlessM4Tv2Model`] can perform all the above tasks, but each task also has its own dedicated sub-model.

The abstract from the paper is the following:

*Recent advancements in automatic speech translation have dramatically expanded language coverage, improved multimodal capabilities, and enabled a wide range of tasks and functionalities. That said, large-scale automatic speech translation systems today lack key features that help machine-mediated communication feel seamless when compared to human-to-human dialogue. In this work, we introduce a family of models that enable end-to-end expressive and multilingual translations in a streaming fashion. First, we contribute an improved version of the massively multilingual and multimodal SeamlessM4T model—SeamlessM4T v2. This newer model, incorporating an updated UnitY2 framework, was trained on more low-resource language data. The expanded version of SeamlessAlign adds 114,800 hours of automatically aligned data for a total of 76 languages. SeamlessM4T v2 provides the foundation on which our two newest models, SeamlessExpressive and SeamlessStreaming, are initiated. SeamlessExpressive enables translation that preserves vocal styles and prosody. Compared to previous efforts in expressive speech research, our work addresses certain underexplored aspects of prosody, such as speech rate and pauses, while also preserving the style of one’s voice. As for SeamlessStreaming, our model leverages the Efficient Monotonic Multihead Attention (EMMA) mechanism to generate low-latency target translations without waiting for complete source utterances. As the first of its kind, SeamlessStreaming enables simultaneous speech-to-speech/text translation for multiple source and target languages. To understand the performance of these models, we combined novel and modified versions of existing automatic metrics to evaluate prosody, latency, and robustness. For human evaluations, we adapted existing protocols tailored for measuring the most relevant attributes in the preservation of meaning, naturalness, and expressivity. To ensure that our models can be used safely and responsibly, we implemented the first known red-teaming effort for multimodal machine translation, a system for the detection and mitigation of added toxicity, a systematic evaluation of gender bias, and an inaudible localized watermarking mechanism designed to dampen the impact of deepfakes. Consequently, we bring major components from SeamlessExpressive and SeamlessStreaming together to form Seamless, the first publicly available system that unlocks expressive cross-lingual communication in real-time. In sum, Seamless gives us a pivotal look at the technical foundation needed to turn the Universal Speech Translator from a science fiction concept into a real-world technology. Finally, contributions in this work—including models, code, and a watermark detector—are publicly released and accessible at the link below.*

## Usage

In the following example, we'll load an Arabic audio sample and an English text sample and convert them into Russian speech and French text.

First, load the processor and a checkpoint of the model:

```python
>>> from transformers import AutoProcessor, SeamlessM4Tv2Model

>>> processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
>>> model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
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

>>> # now, process some English text as well
>>> text_inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt")
```


### Speech

[`SeamlessM4Tv2Model`] can *seamlessly* generate text or speech with few or no changes. Let's target Russian voice translation:

```python
>>> audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
>>> audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
```

With basically the same code, I've translated English text and Arabic speech to Russian speech samples.

### Text

Similarly, you can generate translated text from audio files or from text with the same model. You only have to pass `generate_speech=False` to [`SeamlessM4Tv2Model.generate`].
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

[`SeamlessM4Tv2Model`] is transformers top level model to generate speech and text, but you can also use dedicated models that perform the task without additional components, thus reducing the memory footprint.
For example, you can replace the audio-to-audio generation snippet with the model dedicated to the S2ST task, the rest is exactly the same code: 

```python
>>> from transformers import SeamlessM4Tv2ForSpeechToSpeech
>>> model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")
```

Or you can replace the text-to-text generation snippet with the model dedicated to the T2TT task, you only have to remove `generate_speech=False`.

```python
>>> from transformers import SeamlessM4Tv2ForTextToText
>>> model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
```

Feel free to try out [`SeamlessM4Tv2ForSpeechToText`] and [`SeamlessM4Tv2ForTextToSpeech`] as well.

#### 2. Change the speaker identity

You have the possibility to change the speaker used for speech synthesis with the `speaker_id` argument. Some `speaker_id` works better than other for some languages!

#### 3. Change the generation strategy

You can use different [generation strategies](../generation_strategies) for text generation, e.g `.generate(input_ids=input_ids, text_num_beams=4, text_do_sample=True)` which will perform multinomial beam-search decoding on the text model. Note that speech generation only supports greedy - by default - or multinomial sampling, which can be used with e.g. `.generate(..., speech_do_sample=True, speech_temperature=0.6)`.

#### 4. Generate speech and text at the same time

Use `return_intermediate_token_ids=True` with [`SeamlessM4Tv2Model`] to return both speech and text !

## Model architecture

SeamlessM4T-v2 features a versatile architecture that smoothly handles the sequential generation of text and speech. This setup comprises two sequence-to-sequence (seq2seq) models. The first model translates the input modality into translated text, while the second model generates speech tokens, known as "unit tokens," from the translated text.

Each modality has its own dedicated encoder with a unique architecture. Additionally, for speech output, a vocoder inspired by the [HiFi-GAN](https://arxiv.org/abs/2010.05646) architecture is placed on top of the second seq2seq model.

### Difference with SeamlessM4T-v1

The architecture of this new version differs from the first in a few aspects:

#### Improvements on the second-pass model

The second seq2seq model, named text-to-unit model, is now non-auto regressive, meaning that it computes units in a **single forward pass**. This achievement is made possible by:
- the use of **character-level embeddings**, meaning that each character of the predicted translated text has its own embeddings, which are then used to predict the unit tokens.
- the use of an intermediate duration predictor, that predicts speech duration at the **character-level** on the predicted translated text.
- the use of a new text-to-unit decoder mixing convolutions and self-attention to handle longer context.

#### Difference in the speech encoder

The speech encoder, which is used during the first-pass generation process to predict the translated text, differs mainly from the previous speech encoder through these mechanisms:
- the use of chunked attention mask to prevent attention across chunks, ensuring that each position attends only to positions within its own chunk and a fixed number of previous chunks.
- the use of relative position embeddings which only considers distance between sequence elements rather than absolute positions. Please refer to [Self-Attentionwith Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155) for more details.
- the use of a causal depth-wise convolution instead of a non-causal one.

### Generation process

Here's how the generation process works:

- Input text or speech is processed through its specific encoder.
- A decoder creates text tokens in the desired language.
- If speech generation is required, the second seq2seq model, generates unit tokens in an non auto-regressive way.
- These unit tokens are then passed through the final vocoder to produce the actual speech.


This model was contributed by [ylacombe](https://huggingface.co/ylacombe). The original code can be found [here](https://github.com/facebookresearch/seamless_communication).

## SeamlessM4Tv2Model

The original SeamlessM4Tv2 Model transformer which can be used for every tasks available (S2ST, S2TT, T2TT, T2ST).
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4Tv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

        current_modality (`str`, *optional*, defaults to `"text"`):
            Default modality. Used only to initialize the model. It can be set to `"text"` or `"speech"`.
            This will be updated automatically according to the modality passed to the forward and generate passes (`input_ids` for text and `input_features` for audio).
    

Methods: generate


## SeamlessM4Tv2ForTextToSpeech

The text-to-speech SeamlessM4Tv2 Model transformer which can be used for T2ST.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4Tv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: generate


## SeamlessM4Tv2ForSpeechToSpeech

The speech-to-speech SeamlessM4Tv2 Model transformer which can be used for S2ST.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4Tv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: generate


## SeamlessM4Tv2ForTextToText

The text-to-text SeamlessM4Tv2 Model transformer which can be used for T2TT.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4Tv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - generate

## SeamlessM4Tv2ForSpeechToText

The speech-to-text SeamlessM4Tv2 Model transformer which can be used for S2TT.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4Tv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - generate

## SeamlessM4Tv2Config


    This is the configuration class to store the configuration of a [`~SeamlessM4Tv2Model`]. It is used to instantiate
    an SeamlessM4Tv2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SeamlessM4Tv2
    [""](https://huggingface.co/"") architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 256102):
            Vocabulary size of the text modality of the SeamlessM4Tv2 model. Defines the number of different tokens
            that can be represented by the `inputs_ids` passed when calling [`~SeamlessM4Tv2Model`],
            [`~SeamlessM4Tv2ForTextToSpeech`] or [`~SeamlessM4Tv2ForTextToText`].
        t2u_vocab_size (`int`, *optional*, defaults to 10082):
            Unit vocabulary size of the SeamlessM4Tv2 model. Defines the number of different "unit tokens" that can be
            represented by the `inputs_ids` passed when calling the Text-To-Units sub-model of [`~SeamlessM4Tv2Model`],
            [`~SeamlessM4Tv2ForSpeechToSpeech`] or [`~SeamlessM4Tv2ForTextToSpeech`].
        char_vocab_size (`int`, *optional*, defaults to 10943):
            Character vocabulary size of the SeamlessM4Tv2 model. Defines the number of different character tokens that
            can be represented by the `char_inputs_ids` passed when calling the Text-To-Units sub-model of
            [`~SeamlessM4Tv2Model`], [`~SeamlessM4Tv2ForSpeechToSpeech`] or [`~SeamlessM4Tv2ForTextToSpeech`].

        > Parameters shared across sub-models

        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the "intermediate" layers in the architecture.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        max_position_embeddings (`int`, *optional*, defaults to 4096):
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
        adaptor_kernel_size (`int`, *optional*, defaults to 8):
            Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adaptor_stride (`int`, *optional*, defaults to 8):
            Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adaptor_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all layers in the speech adapter.
        num_adapter_layers (`int`, *optional*, defaults to 1):
            Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is
            True`.
        position_embeddings_type (`str`, *optional*, defaults to `"relative_key"`):
            Can be specified to `relative_key`. If left to `None`, no relative position embedding is applied. Only
            applied to the speech encoder. For more information on `"relative_key"`, please refer to [Self-Attention
            with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
        conv_depthwise_kernel_size (`int`, *optional*, defaults to 31):
            Kernel size of convolutional depthwise 1D layer in Conformer blocks. Only applied to the speech encoder.
        left_max_position_embeddings (`int`, *optional*, defaults to 64):
            The left clipping value for relative positions.
        right_max_position_embeddings (`int`, *optional*, defaults to 8):
            The right clipping value for relative positions.
        speech_encoder_chunk_size (`int`, *optional*, defaults to 20000): The size of each attention chunk.
        speech_encoder_left_chunk_num (`int`, *optional*, defaults to 128):
            Number of chunks on the left up to which lookahead is allowed.

        > Text-To-Unit (t2u) model specific parameters

        t2u_bos_token_id (`int`, *optional*, defaults to 0):
            The id of the _beginning-of-stream_ unit token. Only applied to the text-to-unit seq2seq model.
        t2u_pad_token_id (`int`, *optional*, defaults to 1):
            The id of the _padding_ unit token. Only applied to the text-to-unit seq2seq model.
        t2u_eos_token_id (`int`, *optional*, defaults to 2):
            The id of the _end-of-stream_ unit token. Only applied to the text-to-unit seq2seq model.
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
        t2u_max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model text-to-unit component might ever be used with. Typically set
            this to something large just in case (e.g., 512 or 1024 or 2048).
        t2u_variance_predictor_embed_dim (`int`, *optional*, defaults to 1024):
            The projection dimension of the text-to-unit's duration predictor.
        t2u_variance_predictor_hidden_dim (`int`, *optional*, defaults to 256):
            Internal dimension of the text-to-unit's duration predictor.
        t2u_variance_predictor_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the convolutional layers of the text-to-unit's duration predictor.
        t2u_variance_pred_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability of the text-to-unit's duration predictor.

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
            Vocabulary size of the SeamlessM4Tv2 vocoder. Defines the number of different unit tokens that can be
            represented by the `inputs_ids` passed when calling the vocoder of [`~SeamlessM4Tv2Model`],
            [`~SeamlessM4Tv2ForSpeechToSpeech`] or [`~SeamlessM4Tv2ForTextToSpeech`].
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
    >>> from transformers import SeamlessM4Tv2Model, SeamlessM4Tv2Config

    >>> # Initializing a SeamlessM4Tv2 "" style configuration
    >>> configuration = SeamlessM4Tv2Config()

    >>> # Initializing a model from the "" style configuration
    >>> model = SeamlessM4Tv2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
