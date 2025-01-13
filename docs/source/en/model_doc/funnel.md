<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Funnel Transformer

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=funnel">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-funnel-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/funnel-transformer-small">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>


## Overview

The Funnel Transformer model was proposed in the paper [Funnel-Transformer: Filtering out Sequential Redundancy for
Efficient Language Processing](https://arxiv.org/abs/2006.03236). It is a bidirectional transformer model, like
BERT, but with a pooling operation after each block of layers, a bit like in traditional convolutional neural networks
(CNN) in computer vision.

The abstract from the paper is the following:

*With the success of language pretraining, it is highly desirable to develop more efficient architectures of good
scalability that can exploit the abundant unlabeled data at a lower cost. To improve the efficiency, we examine the
much-overlooked redundancy in maintaining a full-length token-level presentation, especially for tasks that only
require a single-vector presentation of the sequence. With this intuition, we propose Funnel-Transformer which
gradually compresses the sequence of hidden states to a shorter one and hence reduces the computation cost. More
importantly, by re-investing the saved FLOPs from length reduction in constructing a deeper or wider model, we further
improve the model capacity. In addition, to perform token-level predictions as required by common pretraining
objectives, Funnel-Transformer is able to recover a deep representation for each token from the reduced hidden sequence
via a decoder. Empirically, with comparable or fewer FLOPs, Funnel-Transformer outperforms the standard Transformer on
a wide variety of sequence-level prediction tasks, including text classification, language understanding, and reading
comprehension.*

This model was contributed by [sgugger](https://huggingface.co/sgugger). The original code can be found [here](https://github.com/laiguokun/Funnel-Transformer).

## Usage tips

- Since Funnel Transformer uses pooling, the sequence length of the hidden states changes after each block of layers. This way, their length is divided by 2, which speeds up the computation of the next hidden states.
  The base model therefore has a final sequence length that is a quarter of the original one. This model can be used
  directly for tasks that just require a sentence summary (like sequence classification or multiple choice). For other
  tasks, the full model is used; this full model has a decoder that upsamples the final hidden states to the same
  sequence length as the input.
- For tasks such as classification, this is not a problem, but for tasks like masked language modeling or token classification, we need a hidden state with the same sequence length as the original input. In those cases, the final hidden states are upsampled to the input sequence length and go through two additional layers. That's why there are two versions of each checkpoint. The version suffixed with “-base” contains only the three blocks, while the version without that suffix contains the three blocks and the upsampling head with its additional layers.
- The Funnel Transformer checkpoints are all available with a full version and a base version. The first ones should be
  used for [`FunnelModel`], [`FunnelForPreTraining`],
  [`FunnelForMaskedLM`], [`FunnelForTokenClassification`] and
  [`FunnelForQuestionAnswering`]. The second ones should be used for
  [`FunnelBaseModel`], [`FunnelForSequenceClassification`] and
  [`FunnelForMultipleChoice`].

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)


## FunnelConfig


    This is the configuration class to store the configuration of a [`FunnelModel`] or a [`TFBertModel`]. It is used to
    instantiate a Funnel Transformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Funnel
    Transformer [funnel-transformer/small](https://huggingface.co/funnel-transformer/small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Funnel transformer. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`FunnelModel`] or [`TFFunnelModel`].
        block_sizes (`List[int]`, *optional*, defaults to `[4, 4, 4]`):
            The sizes of the blocks used in the model.
        block_repeats (`List[int]`, *optional*):
            If passed along, each layer of each block is repeated the number of times indicated.
        num_decoder_layers (`int`, *optional*, defaults to 2):
            The number of layers in the decoder (when not using the base model).
        d_model (`int`, *optional*, defaults to 768):
            Dimensionality of the model's hidden states.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_head (`int`, *optional*, defaults to 64):
            Dimensionality of the model's heads.
        d_inner (`int`, *optional*, defaults to 3072):
            Inner dimension in the feed-forward blocks.
        hidden_act (`str` or `callable`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability used between the two layers of the feed-forward blocks.
        initializer_range (`float`, *optional*, defaults to 0.1):
            The upper bound of the *uniform initializer* for initializing all weight matrices in attention layers.
        initializer_std (`float`, *optional*):
            The standard deviation of the *normal initializer* for initializing the embedding matrix and the weight of
            linear layers. Will default to 1 for the embedding matrix and the value given by Xavier initialization for
            linear layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-09):
            The epsilon used by the layer normalization layers.
        pooling_type (`str`, *optional*, defaults to `"mean"`):
            Possible values are `"mean"` or `"max"`. The way pooling is performed at the beginning of each block.
        attention_type (`str`, *optional*, defaults to `"relative_shift"`):
            Possible values are `"relative_shift"` or `"factorized"`. The former is faster on CPU/GPU while the latter
            is faster on TPU.
        separate_cls (`bool`, *optional*, defaults to `True`):
            Whether or not to separate the cls token when applying pooling.
        truncate_seq (`bool`, *optional*, defaults to `True`):
            When using `separate_cls`, whether or not to truncate the last token when pooling, to avoid getting a
            sequence length that is not a multiple of 2.
        pool_q_only (`bool`, *optional*, defaults to `True`):
            Whether or not to apply the pooling only to the query or to query, key and values for the attention layers.
    

## FunnelTokenizer


    Construct a Funnel Transformer tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"<sep>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
    

Methods: build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FunnelTokenizerFast


    Construct a "fast" Funnel Transformer tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"<sep>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        bos_token (`str`, `optional`, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, `optional`, defaults to `"</s>"`):
            The end of sentence token.
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    

## Funnel specific outputs

[[autodoc]] models.funnel.modeling_funnel.FunnelForPreTrainingOutput

[[autodoc]] models.funnel.modeling_tf_funnel.TFFunnelForPreTrainingOutput

<frameworkcontent>
<pt>

## FunnelBaseModel


    The base Funnel Transformer Model transformer outputting raw hidden-states without upsampling head (also called
    decoder) or any task-specific head on top.
    

    The Funnel Transformer model was proposed in [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FunnelConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FunnelModel

The bare Funnel Transformer Model transformer outputting raw hidden-states without any specific head on top.

    The Funnel Transformer model was proposed in [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FunnelConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FunnelModelForPreTraining

No docstring available for FunnelForPreTraining

Methods: forward

## FunnelForMaskedLM

Funnel Transformer Model with a `language modeling` head on top.

    The Funnel Transformer model was proposed in [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FunnelConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FunnelForSequenceClassification


    Funnel Transformer Model with a sequence classification/regression head on top (two linear layer on top of the
    first timestep of the last hidden state) e.g. for GLUE tasks.
    

    The Funnel Transformer model was proposed in [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FunnelConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FunnelForMultipleChoice


    Funnel Transformer Model with a multiple choice classification head on top (two linear layer on top of the first
    timestep of the last hidden state, and a softmax) e.g. for RocStories/SWAG tasks.
    

    The Funnel Transformer model was proposed in [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FunnelConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FunnelForTokenClassification


    Funnel Transformer Model with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    

    The Funnel Transformer model was proposed in [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FunnelConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FunnelForQuestionAnswering


    Funnel Transformer Model with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    

    The Funnel Transformer model was proposed in [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FunnelConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

</pt>
<tf>

## TFFunnelBaseModel

No docstring available for TFFunnelBaseModel

Methods: call

## TFFunnelModel

No docstring available for TFFunnelModel

Methods: call

## TFFunnelModelForPreTraining

No docstring available for TFFunnelForPreTraining

Methods: call

## TFFunnelForMaskedLM

No docstring available for TFFunnelForMaskedLM

Methods: call

## TFFunnelForSequenceClassification

No docstring available for TFFunnelForSequenceClassification

Methods: call

## TFFunnelForMultipleChoice

No docstring available for TFFunnelForMultipleChoice

Methods: call

## TFFunnelForTokenClassification

No docstring available for TFFunnelForTokenClassification

Methods: call

## TFFunnelForQuestionAnswering

No docstring available for TFFunnelForQuestionAnswering

Methods: call

</tf>
</frameworkcontent>
