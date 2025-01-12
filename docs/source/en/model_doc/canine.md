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

# CANINE

## Overview

The CANINE model was proposed in [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language
Representation](https://arxiv.org/abs/2103.06874) by Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting. It's
among the first papers that trains a Transformer without using an explicit tokenization step (such as Byte Pair
Encoding (BPE), WordPiece or SentencePiece). Instead, the model is trained directly at a Unicode character-level.
Training at a character-level inevitably comes with a longer sequence length, which CANINE solves with an efficient
downsampling strategy, before applying a deep Transformer encoder.

The abstract from the paper is the following:

*Pipelined NLP systems have largely been superseded by end-to-end neural modeling, yet nearly all commonly-used models
still require an explicit tokenization step. While recent tokenization approaches based on data-derived subword
lexicons are less brittle than manually engineered tokenizers, these techniques are not equally suited to all
languages, and the use of any fixed vocabulary may limit a model's ability to adapt. In this paper, we present CANINE,
a neural encoder that operates directly on character sequences, without explicit tokenization or vocabulary, and a
pre-training strategy that operates either directly on characters or optionally uses subwords as a soft inductive bias.
To use its finer-grained input effectively and efficiently, CANINE combines downsampling, which reduces the input
sequence length, with a deep transformer stack, which encodes context. CANINE outperforms a comparable mBERT model by
2.8 F1 on TyDi QA, a challenging multilingual benchmark, despite having 28% fewer model parameters.*

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/google-research/language/tree/master/language/canine).

## Usage tips

- CANINE uses no less than 3 Transformer encoders internally: 2 "shallow" encoders (which only consist of a single
  layer) and 1 "deep" encoder (which is a regular BERT encoder). First, a "shallow" encoder is used to contextualize
  the character embeddings, using local attention. Next, after downsampling, a "deep" encoder is applied. Finally,
  after upsampling, a "shallow" encoder is used to create the final character embeddings. Details regarding up- and
  downsampling can be found in the paper.
- CANINE uses a max sequence length of 2048 characters by default. One can use [`CanineTokenizer`]
  to prepare text for the model.
- Classification can be done by placing a linear layer on top of the final hidden state of the special [CLS] token
  (which has a predefined Unicode code point). For token classification tasks however, the downsampled sequence of
  tokens needs to be upsampled again to match the length of the original character sequence (which is 2048). The
  details for this can be found in the paper.

Model checkpoints:

  - [google/canine-c](https://huggingface.co/google/canine-c): Pre-trained with autoregressive character loss,
    12-layer, 768-hidden, 12-heads, 121M parameters (size ~500 MB).
  - [google/canine-s](https://huggingface.co/google/canine-s): Pre-trained with subword loss, 12-layer,
    768-hidden, 12-heads, 121M parameters (size ~500 MB).


## Usage example

CANINE works on raw characters, so it can be used **without a tokenizer**:

```python
>>> from transformers import CanineModel
>>> import torch

>>> model = CanineModel.from_pretrained("google/canine-c")  # model pre-trained with autoregressive character loss

>>> text = "hello world"
>>> # use Python's built-in ord() function to turn each character into its unicode code point id
>>> input_ids = torch.tensor([[ord(char) for char in text]])

>>> outputs = model(input_ids)  # forward pass
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

For batched inference and training, it is however recommended to make use of the tokenizer (to pad/truncate all
sequences to the same length):

```python
>>> from transformers import CanineTokenizer, CanineModel

>>> model = CanineModel.from_pretrained("google/canine-c")
>>> tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

>>> inputs = ["Life is like a box of chocolates.", "You never know what you gonna get."]
>>> encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")

>>> outputs = model(**encoding)  # forward pass
>>> pooled_output = outputs.pooler_output
>>> sequence_output = outputs.last_hidden_state
```

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Multiple choice task guide](../tasks/multiple_choice)

## CanineConfig


    This is the configuration class to store the configuration of a [`CanineModel`]. It is used to instantiate an
    CANINE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CANINE
    [google/canine-s](https://huggingface.co/google/canine-s) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the deep Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoders.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoders.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoders, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 16384):
            The maximum sequence length that this model might ever be used with.
        type_vocab_size (`int`, *optional*, defaults to 16):
            The vocabulary size of the `token_type_ids` passed when calling [`CanineModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 57344):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 57345):
            End of stream token id.
        downsampling_rate (`int`, *optional*, defaults to 4):
            The rate at which to downsample the original character sequence length before applying the deep Transformer
            encoder.
        upsampling_kernel_size (`int`, *optional*, defaults to 4):
            The kernel size (i.e. the number of characters in each window) of the convolutional projection layer when
            projecting back from `hidden_size`*2 to `hidden_size`.
        num_hash_functions (`int`, *optional*, defaults to 8):
            The number of hash functions to use. Each hash function has its own embedding matrix.
        num_hash_buckets (`int`, *optional*, defaults to 16384):
            The number of hash buckets to use.
        local_transformer_stride (`int`, *optional*, defaults to 128):
            The stride of the local attention of the first shallow Transformer encoder. Defaults to 128 for good
            TPU/XLA memory alignment.

    Example:

    ```python
    >>> from transformers import CanineConfig, CanineModel

    >>> # Initializing a CANINE google/canine-s style configuration
    >>> configuration = CanineConfig()

    >>> # Initializing a model (with random weights) from the google/canine-s style configuration
    >>> model = CanineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## CanineTokenizer


    Construct a CANINE tokenizer (i.e. a character splitter). It turns text into a sequence of characters, and then
    converts each character into its Unicode code point.

    [`CanineTokenizer`] inherits from [`PreTrainedTokenizer`].

    Refer to superclass [`PreTrainedTokenizer`] for usage examples and documentation concerning parameters.

    Args:
        model_max_length (`int`, *optional*, defaults to 2048):
                The maximum sentence length the model accepts.
    

Methods: build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences

## CANINE specific outputs

Could not find docstring for models.canine.modeling_canine.CanineModelOutputWithPooling

## CanineModel

The bare CANINE Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CanineConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## CanineForSequenceClassification


    CANINE Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CanineConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## CanineForMultipleChoice


    CANINE Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CanineConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## CanineForTokenClassification


    CANINE Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CanineConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## CanineForQuestionAnswering


    CANINE Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CanineConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
