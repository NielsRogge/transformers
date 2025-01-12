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

# REALM

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The REALM model was proposed in [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) by Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat and Ming-Wei Chang. It's a
retrieval-augmented language model that firstly retrieves documents from a textual knowledge corpus and then
utilizes retrieved documents to process question answering tasks.

The abstract from the paper is the following:

*Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks
such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network,
requiring ever-larger networks to cover more facts. To capture knowledge in a more modular and interpretable way, we
augment language model pre-training with a latent knowledge retriever, which allows the model to retrieve and attend
over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference. For the
first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language
modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents. We
demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the
challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both
explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous
methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as
interpretability and modularity.*

This model was contributed by [qqaatw](https://huggingface.co/qqaatw). The original code can be found
[here](https://github.com/google-research/language/tree/master/language/realm).

## RealmConfig


    This is the configuration class to store the configuration of

    1. [`RealmEmbedder`]
    2. [`RealmScorer`]
    3. [`RealmKnowledgeAugEncoder`]
    4. [`RealmRetriever`]
    5. [`RealmReader`]
    6. [`RealmForOpenQA`]

    It is used to instantiate an REALM model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the REALM
    [google/realm-cc-news-pretrained-embedder](https://huggingface.co/google/realm-cc-news-pretrained-embedder)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the REALM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RealmEmbedder`], [`RealmScorer`], [`RealmKnowledgeAugEncoder`], or
            [`RealmReader`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        retriever_proj_size (`int`, *optional*, defaults to 128):
            Dimension of the retriever(embedder) projection.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_candidates (`int`, *optional*, defaults to 8):
            Number of candidates inputted to the RealmScorer or RealmKnowledgeAugEncoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`RealmEmbedder`], [`RealmScorer`],
            [`RealmKnowledgeAugEncoder`], or [`RealmReader`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        span_hidden_size (`int`, *optional*, defaults to 256):
            Dimension of the reader's spans.
        max_span_width (`int`, *optional*, defaults to 10):
            Max span width of the reader.
        reader_layer_norm_eps (`float`, *optional*, defaults to 1e-3):
            The epsilon used by the reader's layer normalization layers.
        reader_beam_size (`int`, *optional*, defaults to 5):
            Beam size of the reader.
        reader_seq_len (`int`, *optional*, defaults to 288+32):
            Maximum sequence length of the reader.
        num_block_records (`int`, *optional*, defaults to 13353718):
            Number of block records.
        searcher_beam_size (`int`, *optional*, defaults to 5000):
            Beam size of the searcher. Note that when eval mode is enabled, *searcher_beam_size* will be the same as
            *reader_beam_size*.

    Example:

    ```python
    >>> from transformers import RealmConfig, RealmEmbedder

    >>> # Initializing a REALM realm-cc-news-pretrained-* style configuration
    >>> configuration = RealmConfig()

    >>> # Initializing a model (with random weights) from the google/realm-cc-news-pretrained-embedder style configuration
    >>> model = RealmEmbedder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## RealmTokenizer


    Construct a REALM tokenizer.

    [`RealmTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation splitting and
    wordpiece.

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
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    

Methods: build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary
    - batch_encode_candidates

## RealmTokenizerFast


    Construct a "fast" REALM tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    [`RealmTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    

Methods: batch_encode_candidates

## RealmRetriever

The retriever of REALM outputting the retrieved evidence block and whether the block has answers as well as answer
    positions."

        Parameters:
            block_records (`np.ndarray`):
                A numpy array which cantains evidence texts.
            tokenizer ([`RealmTokenizer`]):
                The tokenizer to encode retrieved texts.
    

## RealmEmbedder

The embedder of REALM outputting projected score that will be used to calculate relevance score.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RealmConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## RealmScorer

The scorer of REALM outputting relevance scores representing the score of document candidates (before softmax).
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RealmConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Args:
        query_embedder ([`RealmEmbedder`]):
            Embedder for input sequences. If not specified, it will use the same embedder as candidate sequences.
    

Methods: forward

## RealmKnowledgeAugEncoder

The knowledge-augmented encoder of REALM outputting masked language model logits and marginal log-likelihood loss.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RealmConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## RealmReader

The reader of REALM.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RealmConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## RealmForOpenQA

`RealmForOpenQA` for end-to-end open domain question answering.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RealmConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: block_embedding_to
    - forward
