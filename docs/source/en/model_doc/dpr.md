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

# DPR

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=dpr">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-dpr-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/dpr-question_encoder-bert-base-multilingual">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

Dense Passage Retrieval (DPR) is a set of tools and models for state-of-the-art open-domain Q&A research. It was
introduced in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) by
Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih.

The abstract from the paper is the following:

*Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional
sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can
be practically implemented using dense representations alone, where embeddings are learned from a small number of
questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets,
our dense retriever outperforms a strong Lucene-BM25 system largely by 9%-19% absolute in terms of top-20 passage
retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA
benchmarks.*

This model was contributed by [lhoestq](https://huggingface.co/lhoestq). The original code can be found [here](https://github.com/facebookresearch/DPR).

## Usage tips

- DPR consists in three models:

    * Question encoder: encode questions as vectors
    * Context encoder: encode contexts as vectors
    * Reader: extract the answer of the questions inside retrieved contexts, along with a relevance score (high if the inferred span actually answers the question).

## DPRConfig


    [`DPRConfig`] is the configuration class to store the configuration of a *DPRModel*.

    This is the configuration class to store the configuration of a [`DPRContextEncoder`], [`DPRQuestionEncoder`], or a
    [`DPRReader`]. It is used to instantiate the components of the DPR model according to the specified arguments,
    defining the model component architectures. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the DPRContextEncoder
    [facebook/dpr-ctx_encoder-single-nq-base](https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base)
    architecture.

    This class is a subclass of [`BertConfig`]. Please check the superclass for the documentation of all kwargs.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the DPR model. Defines the different tokens that can be represented by the *inputs_ids*
            passed to the forward method of [`BertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the *token_type_ids* passed into [`BertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        projection_dim (`int`, *optional*, defaults to 0):
            Dimension of the projection for the context and question encoders. If it is set to zero (default), then no
            projection is done.

    Example:

    ```python
    >>> from transformers import DPRConfig, DPRContextEncoder

    >>> # Initializing a DPR facebook/dpr-ctx_encoder-single-nq-base style configuration
    >>> configuration = DPRConfig()

    >>> # Initializing a model (with random weights) from the facebook/dpr-ctx_encoder-single-nq-base style configuration
    >>> model = DPRContextEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## DPRContextEncoderTokenizer


    Construct a DPRContextEncoder tokenizer.

    [`DPRContextEncoderTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    

## DPRContextEncoderTokenizerFast


    Construct a "fast" DPRContextEncoder tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRContextEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    

## DPRQuestionEncoderTokenizer


    Constructs a DPRQuestionEncoder tokenizer.

    [`DPRQuestionEncoderTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    

## DPRQuestionEncoderTokenizerFast


    Constructs a "fast" DPRQuestionEncoder tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRQuestionEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    

## DPRReaderTokenizer


    Construct a DPRReader tokenizer.

    [`DPRReaderTokenizer`] is almost identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece. The difference is that is has three inputs strings: question, titles and texts that are
    combined to be fed to the [`DPRReader`] model.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    
    Return a dictionary with the token ids of the input strings and other information to give to `.decode_best_spans`.
    It converts the strings of a question and different passages (title and text) in a sequence of IDs (integers),
    using the tokenizer and vocabulary. The resulting `input_ids` is a matrix of size `(n_passages, sequence_length)`
    with the format:

    ```
    [CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>
    ```

    Args:
        questions (`str` or `List[str]`):
            The questions to be encoded. You can specify one question for many passages. In this case, the question
            will be duplicated like `[questions] * n_passages`. Otherwise you have to specify as many questions as in
            `titles` or `texts`.
        titles (`str` or `List[str]`):
            The passages titles to be encoded. This can be a string or a list of strings if there are several passages.
        texts (`str` or `List[str]`):
            The passages texts to be encoded. This can be a string or a list of strings if there are several passages.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
            Activates and controls padding. Accepts the following values:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
            Activates and controls truncation. Accepts the following values:

            - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to
              the maximum acceptable input length for the model if that argument is not provided. This will truncate
              token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch
              of pairs) is provided.
            - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided. This will only truncate the first
              sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided. This will only truncate the
              second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
              greater than the model maximum admissible input size).
        max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        return_attention_mask (`bool`, *optional*):
            Whether or not to return the attention mask. If not set, will return the attention mask according to the
            specific tokenizer's default, defined by the `return_outputs` attribute.

            [What are attention masks?](../glossary#attention-mask)

    Returns:
        `Dict[str, List[List[int]]]`: A dictionary with the following keys:

        - `input_ids`: List of token ids to be fed to a model.
        - `attention_mask`: List of indices specifying which tokens should be attended to by the model.
    

## DPRReaderTokenizerFast


    Constructs a "fast" DPRReader tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRReaderTokenizerFast`] is almost identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece. The difference is that is has three inputs strings: question, titles and texts
    that are combined to be fed to the [`DPRReader`] model.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.

    
    Return a dictionary with the token ids of the input strings and other information to give to `.decode_best_spans`.
    It converts the strings of a question and different passages (title and text) in a sequence of IDs (integers),
    using the tokenizer and vocabulary. The resulting `input_ids` is a matrix of size `(n_passages, sequence_length)`
    with the format:

    [CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>

    Args:
        questions (`str` or `List[str]`):
            The questions to be encoded. You can specify one question for many passages. In this case, the question
            will be duplicated like `[questions] * n_passages`. Otherwise you have to specify as many questions as in
            `titles` or `texts`.
        titles (`str` or `List[str]`):
            The passages titles to be encoded. This can be a string or a list of strings if there are several passages.
        texts (`str` or `List[str]`):
            The passages texts to be encoded. This can be a string or a list of strings if there are several passages.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
            Activates and controls padding. Accepts the following values:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
            Activates and controls truncation. Accepts the following values:

            - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to
              the maximum acceptable input length for the model if that argument is not provided. This will truncate
              token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch
              of pairs) is provided.
            - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided. This will only truncate the first
              sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided. This will only truncate the
              second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
              greater than the model maximum admissible input size).
        max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        return_attention_mask (`bool`, *optional*):
            Whether or not to return the attention mask. If not set, will return the attention mask according to the
            specific tokenizer's default, defined by the `return_outputs` attribute.

            [What are attention masks?](../glossary#attention-mask)

    Return:
        `Dict[str, List[List[int]]]`: A dictionary with the following keys:

        - `input_ids`: List of token ids to be fed to a model.
        - `attention_mask`: List of indices specifying which tokens should be attended to by the model.
    

## DPR specific outputs

models.dpr.modeling_dpr.DPRContextEncoderOutput

    Class for outputs of [`DPRQuestionEncoder`].

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The DPR encoder outputs the *pooler_output* that corresponds to the context representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    

models.dpr.modeling_dpr.DPRQuestionEncoderOutput

    Class for outputs of [`DPRQuestionEncoder`].

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The DPR encoder outputs the *pooler_output* that corresponds to the question representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed questions for nearest neighbors queries with context embeddings.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    

models.dpr.modeling_dpr.DPRReaderOutput

    Class for outputs of [`DPRQuestionEncoder`].

    Args:
        start_logits (`torch.FloatTensor` of shape `(n_passages, sequence_length)`):
            Logits of the start index of the span for each passage.
        end_logits (`torch.FloatTensor` of shape `(n_passages, sequence_length)`):
            Logits of the end index of the span for each passage.
        relevance_logits (`torch.FloatTensor` of shape `(n_passages, )`):
            Outputs of the QA classifier of the DPRReader that corresponds to the scores of each passage to answer the
            question, compared to all the other passages.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    

<frameworkcontent>
<pt>

## DPRContextEncoder

The bare DPRContextEncoder transformer outputting pooler outputs as context representations.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DPRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## DPRQuestionEncoder

The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DPRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## DPRReader

The bare DPRReader transformer outputting span predictions.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DPRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

</pt>
<tf>

## TFDPRContextEncoder

No docstring available for TFDPRContextEncoder

Methods: call

## TFDPRQuestionEncoder

No docstring available for TFDPRQuestionEncoder

Methods: call

## TFDPRReader

No docstring available for TFDPRReader

Methods: call

</tf>
</frameworkcontent>

