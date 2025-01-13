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

# RAG

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=rag">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-rag-blueviolet">
</a>
</div>

## Overview

Retrieval-augmented generation ("RAG") models combine the powers of pretrained dense retrieval (DPR) and
sequence-to-sequence models. RAG models retrieve documents, pass them to a seq2seq model, then marginalize to generate
outputs. The retriever and seq2seq modules are initialized from pretrained models, and fine-tuned jointly, allowing
both retrieval and generation to adapt to downstream tasks.

It is based on the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) by Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela.

The abstract from the paper is the following:

*Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve
state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely
manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind
task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge
remain open research problems. Pre-trained models with a differentiable access mechanism to explicit nonparametric
memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a
general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained
parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a
pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a
pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages
across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our
models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks,
outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation
tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art
parametric-only seq2seq baseline.*

This model was contributed by [ola13](https://huggingface.co/ola13).

## Usage tips

Retrieval-augmented generation ("RAG") models combine the powers of pretrained dense retrieval (DPR) and Seq2Seq models. 
RAG models retrieve docs, pass them to a seq2seq model, then marginalize to generate outputs. The retriever and seq2seq 
modules are initialized from pretrained models, and fine-tuned jointly, allowing both retrieval and generation to adapt 
to downstream tasks.

## RagConfig


    [`RagConfig`] stores the configuration of a *RagModel*. Configuration objects inherit from [`PretrainedConfig`] and
    can be used to control the model outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        title_sep (`str`, *optional*, defaults to  `" / "`):
            Separator inserted between the title and the text of the retrieved document when calling [`RagRetriever`].
        doc_sep (`str`, *optional*, defaults to  `" // "`):
            Separator inserted between the text of the retrieved document and the original input when calling
            [`RagRetriever`].
        n_docs (`int`, *optional*, defaults to 5):
            Number of documents to retrieve.
        max_combined_length (`int`, *optional*, defaults to 300):
            Max length of contextualized input returned by [`~RagRetriever.__call__`].
        retrieval_vector_size (`int`, *optional*, defaults to 768):
            Dimensionality of the document embeddings indexed by [`RagRetriever`].
        retrieval_batch_size (`int`, *optional*, defaults to 8):
            Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated
            [`RagRetriever`].
        dataset (`str`, *optional*, defaults to `"wiki_dpr"`):
            A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and ids
            using `datasets.list_datasets()`).
        dataset_split (`str`, *optional*, defaults to `"train"`)
            Which split of the `dataset` to load.
        index_name (`str`, *optional*, defaults to `"compressed"`)
            The index name of the index associated with the `dataset`. One can choose between `"legacy"`, `"exact"` and
            `"compressed"`.
        index_path (`str`, *optional*)
            The path to the serialized faiss index on disk.
        passages_path (`str`, *optional*):
            A path to text passages compatible with the faiss index. Required if using
            [`~models.rag.retrieval_rag.LegacyIndex`]
        use_dummy_dataset (`bool`, *optional*, defaults to `False`)
            Whether to load a "dummy" variant of the dataset specified by `dataset`.
        label_smoothing (`float`, *optional*, defaults to 0.0):
            Only relevant if `return_loss` is set to `True`. Controls the `epsilon` parameter value for label smoothing
            in the loss calculation. If set to 0, no label smoothing is performed.
        do_marginalize (`bool`, *optional*, defaults to `False`):
            If `True`, the logits are marginalized over all documents by making use of
            `torch.nn.functional.log_softmax`.
        reduce_loss (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce the NLL loss using the `torch.Tensor.sum` operation.
        do_deduplication (`bool`, *optional*, defaults to `True`):
            Whether or not to deduplicate the generations from different context documents for a given input. Has to be
            set to `False` if used while training with distributed backend.
        exclude_bos_score (`bool`, *optional*, defaults to `False`):
            Whether or not to disregard the BOS token when computing the loss.
        output_retrieved(`bool`, *optional*, defaults to `False`):
            If set to `True`, `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
            `context_attention_mask` are returned. See returned tensors for more detail.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        forced_eos_token_id (`int`, *optional*):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.


## RagTokenizer

No docstring available for RagTokenizer

## Rag specific outputs

[[autodoc]] models.rag.modeling_rag.RetrievAugLMMarginOutput

[[autodoc]] models.rag.modeling_rag.RetrievAugLMOutput

## RagRetriever


    Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
    contents, and it formats them to be used with a RagModel.

    Args:
        config ([`RagConfig`]):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which
            `Index` to build. You can load your own custom dataset with `config.index_name="custom"` or use a canonical
            one (default) from the datasets library with `config.index_name="wiki_dpr"` for example.
        question_encoder_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
            generator_tokenizer.
        generator_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for the generator part of the RagModel.
        index ([`~models.rag.retrieval_rag.Index`], optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration

    Examples:

    ```python
    >>> # To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
    >>> from transformers import RagRetriever

    >>> retriever = RagRetriever.from_pretrained(
    ...     "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
    ... )

    >>> # To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
    >>> from transformers import RagRetriever

    >>> dataset = (
    ...     ...
    ... )  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a faiss index
    >>> retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", indexed_dataset=dataset)

    >>> # To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py
    >>> from transformers import RagRetriever

    >>> dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*
    >>> index_path = "path/to/my/index.faiss"  # faiss index saved via *dataset.get_index("embeddings").save(...)*
    >>> retriever = RagRetriever.from_pretrained(
    ...     "facebook/dpr-ctx_encoder-single-nq-base",
    ...     index_name="custom",
    ...     passages_path=dataset_path,
    ...     index_path=index_path,
    ... )

    >>> # To load the legacy index built originally for Rag's paper
    >>> from transformers import RagRetriever

    >>> retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", index_name="legacy")
    ```

<frameworkcontent>
<pt>

## RagModel

   The [`RagModel`] forward method, overrides the `__call__` special method.

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>


    RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward
    pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context
    documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

    The question encoder can be any *autoencoding* model, preferably [`DPRQuestionEncoder`], and the generator can be
    any *seq2seq* model, preferably [`BartForConditionalGeneration`].

    The model can be initialized with a [`RagRetriever`] for end-to-end generation or used in combination with the
    outputs of a retriever in multiple steps---see examples for more details. The model is compatible any
    *autoencoding* model as the `question_encoder` and any *seq2seq* model with language model head as the `generator`.
    It has been tested with [`DPRQuestionEncoder`] as the `question_encoder` and [`BartForConditionalGeneration`] or
    [`T5ForConditionalGeneration`] as the `generator`.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.


    Args:
        config ([`RagConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        question_encoder ([`PreTrainedModel`]):
            An encoder model compatible with the faiss index encapsulated by the `retriever`.
        generator ([`PreTrainedModel`]):
            A seq2seq model used as the generator in the RAG architecture.
        retriever ([`RagRetriever`]):
            A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.


Methods: forward

## RagSequenceForGeneration

   The [`RagSequenceForGeneration`] forward method, overrides the `__call__` special method.

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>

    A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.
    

    RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward
    pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context
    documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

    The question encoder can be any *autoencoding* model, preferably [`DPRQuestionEncoder`], and the generator can be
    any *seq2seq* model, preferably [`BartForConditionalGeneration`].

    The model can be initialized with a [`RagRetriever`] for end-to-end generation or used in combination with the
    outputs of a retriever in multiple steps---see examples for more details. The model is compatible any
    *autoencoding* model as the `question_encoder` and any *seq2seq* model with language model head as the `generator`.
    It has been tested with [`DPRQuestionEncoder`] as the `question_encoder` and [`BartForConditionalGeneration`] or
    [`T5ForConditionalGeneration`] as the `generator`.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.


    Args:
        config ([`RagConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        question_encoder ([`PreTrainedModel`]):
            An encoder model compatible with the faiss index encapsulated by the `retriever`.
        generator ([`PreTrainedModel`]):
            A seq2seq model used as the generator in the RAG architecture.
        retriever ([`RagRetriever`]):
            A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.


Methods: forward
    - generate

## RagTokenForGeneration

   The [`RagTokenForGeneration`] forward method, overrides the `__call__` special method.

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>

    A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.
    

    RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward
    pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context
    documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

    The question encoder can be any *autoencoding* model, preferably [`DPRQuestionEncoder`], and the generator can be
    any *seq2seq* model, preferably [`BartForConditionalGeneration`].

    The model can be initialized with a [`RagRetriever`] for end-to-end generation or used in combination with the
    outputs of a retriever in multiple steps---see examples for more details. The model is compatible any
    *autoencoding* model as the `question_encoder` and any *seq2seq* model with language model head as the `generator`.
    It has been tested with [`DPRQuestionEncoder`] as the `question_encoder` and [`BartForConditionalGeneration`] or
    [`T5ForConditionalGeneration`] as the `generator`.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.


    Args:
        config ([`RagConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        question_encoder ([`PreTrainedModel`]):
            An encoder model compatible with the faiss index encapsulated by the `retriever`.
        generator ([`PreTrainedModel`]):
            A seq2seq model used as the generator in the RAG architecture.
        retriever ([`RagRetriever`]):
            A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.


Methods: forward
    - generate

</pt>
<tf>

## TFRagModel

No docstring available for TFRagModel

Methods: call

## TFRagSequenceForGeneration

No docstring available for TFRagSequenceForGeneration

Methods: call
    - generate

## TFRagTokenForGeneration

No docstring available for TFRagTokenForGeneration

Methods: call
    - generate

</tf>
</frameworkcontent>
