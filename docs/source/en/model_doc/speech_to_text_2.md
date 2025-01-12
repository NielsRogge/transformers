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

# Speech2Text2

  <Tip warning={true}>

  This model is in maintenance mode only, we don't accept any new PRs changing its code.
  If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
  You can do so by running the following command: `pip install -U transformers==4.40.2`.

  </Tip>

## Overview

The Speech2Text2 model is used together with [Wav2Vec2](wav2vec2) for Speech Translation models proposed in
[Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://arxiv.org/abs/2104.06678) by
Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau.

Speech2Text2 is a *decoder-only* transformer model that can be used with any speech *encoder-only*, such as
[Wav2Vec2](wav2vec2) or [HuBERT](hubert) for Speech-to-Text tasks. Please refer to the
[SpeechEncoderDecoder](speech-encoder-decoder) class on how to combine Speech2Text2 with any speech *encoder-only*
model.

This model was contributed by [Patrick von Platen](https://huggingface.co/patrickvonplaten).

The original code can be found [here](https://github.com/pytorch/fairseq/blob/1f7ef9ed1e1061f8c7f88f8b94c7186834398690/fairseq/models/wav2vec/wav2vec2_asr.py#L266).

## Usage tips

- Speech2Text2 achieves state-of-the-art results on the CoVoST Speech Translation dataset. For more information, see
  the [official models](https://huggingface.co/models?other=speech2text2) .
- Speech2Text2 is always used within the [SpeechEncoderDecoder](speech-encoder-decoder) framework.
- Speech2Text2's tokenizer is based on [fastBPE](https://github.com/glample/fastBPE).

## Inference

Speech2Text2's [`SpeechEncoderDecoderModel`] model accepts raw waveform input values from speech and
makes use of [`~generation.GenerationMixin.generate`] to translate the input speech
autoregressively to the target language.

The [`Wav2Vec2FeatureExtractor`] class is responsible for preprocessing the input speech and
[`Speech2Text2Tokenizer`] decodes the generated target tokens to the target string. The
[`Speech2Text2Processor`] wraps [`Wav2Vec2FeatureExtractor`] and
[`Speech2Text2Tokenizer`] into a single instance to both extract the input features and decode the
predicted token ids.

- Step-by-step Speech Translation

```python
>>> import torch
>>> from transformers import Speech2Text2Processor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
>>> processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")
>>> generated_ids = model.generate(inputs=inputs["input_values"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids)
```

- Speech Translation via Pipelines

  The automatic speech recognition pipeline can also be used to translate speech in just a couple lines of code

```python
>>> from datasets import load_dataset
>>> from transformers import pipeline

>>> librispeech_en = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> asr = pipeline(
...     "automatic-speech-recognition",
...     model="facebook/s2t-wav2vec2-large-en-de",
...     feature_extractor="facebook/s2t-wav2vec2-large-en-de",
... )

>>> translation_de = asr(librispeech_en[0]["file"])
```

See [model hub](https://huggingface.co/models?filter=speech2text2) to look for Speech2Text2 checkpoints.

## Resources

- [Causal language modeling task guide](../tasks/language_modeling)

## Speech2Text2Config


    This is the configuration class to store the configuration of a [`Speech2Text2ForCausalLM`]. It is used to
    instantiate an Speech2Text2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Speech2Text2
    [facebook/s2t-wav2vec2-large-en-de](https://huggingface.co/facebook/s2t-wav2vec2-large-en-de) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the Speech2Text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`Speech2TextModel`]
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the pooler. If string, `"gelu"`, `"relu"`,
            `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        max_target_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).

    Example:

    ```python
    >>> from transformers import Speech2Text2Config, Speech2Text2ForCausalLM

    >>> # Initializing a Speech2Text2 s2t_transformer_s style configuration
    >>> configuration = Speech2Text2Config()

    >>> # Initializing a model (with random weights) from the s2t_transformer_s style configuration
    >>> model = Speech2Text2ForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## Speech2TextTokenizer


    Constructs a Speech2Text2Tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    

Methods: batch_decode
    - decode
    - save_vocabulary

## Speech2Text2Processor


    Constructs a Speech2Text2 processor which wraps a Speech2Text2 feature extractor and a Speech2Text2 tokenizer into
    a single processor.

    [`Speech2Text2Processor`] offers all the functionalities of [`AutoFeatureExtractor`] and [`Speech2Text2Tokenizer`].
    See the [`~Speech2Text2Processor.__call__`] and [`~Speech2Text2Processor.decode`] for more information.

    Args:
        feature_extractor (`AutoFeatureExtractor`):
            An instance of [`AutoFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Speech2Text2Tokenizer`):
            An instance of [`Speech2Text2Tokenizer`]. The tokenizer is a required input.
    

Methods: __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## Speech2Text2ForCausalLM

The Speech2Text2 Decoder with a language modeling head. Can be used as the decoder part of [`EncoderDecoderModel`] and [`SpeechEncoderDecoder`].
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Speech2Text2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
