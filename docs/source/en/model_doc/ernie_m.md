<!--Copyright 2023 The HuggingFace and Baidu Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ErnieM

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The ErnieM model was proposed in [ERNIE-M: Enhanced Multilingual Representation by Aligning
Cross-lingual Semantics with Monolingual Corpora](https://arxiv.org/abs/2012.15674)  by Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun,
Hao Tian, Hua Wu, Haifeng Wang.

The abstract from the paper is the following:

*Recent studies have demonstrated that pre-trained cross-lingual models achieve impressive performance in downstream cross-lingual tasks. This improvement benefits from learning a large amount of monolingual and parallel corpora. Although it is generally acknowledged that parallel corpora are critical for improving the model performance, existing methods are often constrained by the size of parallel corpora, especially for lowresource languages. In this paper, we propose ERNIE-M, a new training method that encourages the model to align the representation of multiple languages with monolingual corpora, to overcome the constraint that the parallel corpus size places on the model performance. Our key insight is to integrate back-translation into the pre-training process. We generate pseudo-parallel sentence pairs on a monolingual corpus to enable the learning of semantic alignments between different languages, thereby enhancing the semantic modeling of cross-lingual models. Experimental results show that ERNIE-M outperforms existing cross-lingual models and delivers new state-of-the-art results in various cross-lingual downstream tasks.*
This model was contributed by [Susnato Dhar](https://huggingface.co/susnato). The original code can be found [here](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie_m).


## Usage tips

- Ernie-M is a BERT-like model so it is a stacked Transformer Encoder.
- Instead of using MaskedLM for pretraining (like BERT) the authors used two novel techniques: `Cross-attention Masked Language Modeling` and `Back-translation Masked Language Modeling`. For now these two LMHead objectives are not implemented here.
- It is a multilingual language model.
- Next Sentence Prediction was not used in pretraining process.

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Multiple choice task guide](../tasks/multiple_choice)

## ErnieMConfig


    This is the configuration class to store the configuration of a [`ErnieMModel`]. It is used to instantiate a
    Ernie-M model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the `Ernie-M`
    [susnato/ernie-m-base_pytorch](https://huggingface.co/susnato/ernie-m-base_pytorch) architecture.


    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 250002):
            Vocabulary size of `inputs_ids` in [`ErnieMModel`]. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling
            [`ErnieMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embedding layer, encoder layers and pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors to feed-forward layers are
            firstly projected from hidden_size to intermediate_size, and then projected back to hidden_size. Typically
            intermediate_size is larger than hidden_size.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the feed-forward layer. `"gelu"`, `"relu"` and any other torch
            supported activation functions are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability used in `MultiHeadAttention` in all encoder layers to drop some attention target.
        max_position_embeddings (`int`, *optional*, defaults to 514):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length
            of an input sequence.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the normal initializer for initializing all weight matrices. The index of padding
            token in the token vocabulary.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        act_dropout (`float`, *optional*, defaults to 0.0):
            This dropout probability is used in `ErnieMEncoderLayer` after activation.

    A normal_initializer initializes weight matrices as normal distributions. See
    `ErnieMPretrainedModel._init_weights()` for how weights are initialized in `ErnieMModel`.
    


## ErnieMTokenizer


    Constructs a Ernie-M tokenizer. It uses the `sentencepiece` tools to cut the words to sub-words.

    Args:
        sentencepiece_model_file (`str`):
            The file path of sentencepiece model.
        vocab_file (`str`, *optional*):
            The file path of the vocabulary.
        do_lower_case (`str`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            A special token representing the `unknown (out-of-vocabulary)` token. An unknown token is set to be
            `unk_token` inorder to be converted to an ID.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            A special token separating two different sentences in the same input.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            A special token used to make arrays of tokens the same size for batching purposes.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            A special token used for sequence classification. It is the last token of the sequence when built with
            special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            A special token representing a masked token. This is the token used in the masked language modeling task
            which the model tries to predict the original unmasked ones.
    

Methods: build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## ErnieMModel

The bare ErnieM Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ErnieMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ErnieMForSequenceClassification

ErnieM Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ErnieMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward


## ErnieMForMultipleChoice

ErnieM Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ErnieMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward


## ErnieMForTokenClassification

ErnieM Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ErnieMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward


## ErnieMForQuestionAnswering

ErnieM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ErnieMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ErnieMForInformationExtraction

ErnieMForInformationExtraction is a Ernie-M Model with two linear layer on top of the hidden-states output to
    compute `start_prob` and `end_prob`, designed for Universal Information Extraction.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ErnieMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
