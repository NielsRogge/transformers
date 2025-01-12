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

# FlauBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=flaubert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-flaubert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/flaubert_small_cased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The FlauBERT model was proposed in the paper [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) by Hang Le et al. It's a transformer model pretrained using a masked language
modeling (MLM) objective (like BERT).

The abstract from the paper is the following:

*Language models have become a key step to achieve state-of-the art results in many different Natural Language
Processing (NLP) tasks. Leveraging the huge amount of unlabeled texts nowadays available, they provide an efficient way
to pre-train continuous word representations that can be fine-tuned for a downstream task, along with their
contextualization at the sentence level. This has been widely demonstrated for English using contextualized
representations (Dai and Le, 2015; Peters et al., 2018; Howard and Ruder, 2018; Radford et al., 2018; Devlin et al.,
2019; Yang et al., 2019b). In this paper, we introduce and share FlauBERT, a model learned on a very large and
heterogeneous French corpus. Models of different sizes are trained using the new CNRS (French National Centre for
Scientific Research) Jean Zay supercomputer. We apply our French language models to diverse NLP tasks (text
classification, paraphrasing, natural language inference, parsing, word sense disambiguation) and show that most of the
time they outperform other pretraining approaches. Different versions of FlauBERT as well as a unified evaluation
protocol for the downstream tasks, called FLUE (French Language Understanding Evaluation), are shared to the research
community for further reproducible experiments in French NLP.*

This model was contributed by [formiel](https://huggingface.co/formiel). The original code can be found [here](https://github.com/getalp/Flaubert).

Tips:
- Like RoBERTa, without the sentence ordering prediction (so just trained on the MLM objective).

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## FlaubertConfig


    This is the configuration class to store the configuration of a [`FlaubertModel`] or a [`TFFlaubertModel`]. It is
    used to instantiate a FlauBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the FlauBERT
    [flaubert/flaubert_base_uncased](https://huggingface.co/flaubert/flaubert_base_uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        pre_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply the layer normalization before or after the feed forward layer following the attention in
            each layer (Vaswani et al., Tensor2Tensor for Neural Machine Translation. 2018)
        layerdrop (`float`, *optional*, defaults to 0.0):
            Probability to drop layers during training (Fan et al., Reducing Transformer Depth on Demand with
            Structured Dropout. ICLR 2020)
        vocab_size (`int`, *optional*, defaults to 30145):
            Vocabulary size of the FlauBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`FlaubertModel`] or [`TFFlaubertModel`].
        emb_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention mechanism
        gelu_activation (`bool`, *optional*, defaults to `True`):
            Whether or not to use a *gelu* activation instead of *relu*.
        sinusoidal_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.
        causal (`bool`, *optional*, defaults to `False`):
            Whether or not the model should behave in a causal manner. Causal models use a triangular attention mask in
            order to only attend to the left-side context instead if a bidirectional context.
        asm (`bool`, *optional*, defaults to `False`):
            Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the prediction
            layer.
        n_langs (`int`, *optional*, defaults to 1):
            The number of languages the model handles. Set to 1 for monolingual models.
        use_lang_emb (`bool`, *optional*, defaults to `True`)
            Whether to use language embeddings. Some models use additional language embeddings, see [the multilingual
            models page](http://huggingface.co/transformers/multilingual.html#xlm-language-embeddings) for information
            on how to use them.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        embed_init_std (`float`, *optional*, defaults to 2048^-0.5):
            The standard deviation of the truncated_normal_initializer for initializing the embedding matrices.
        init_std (`int`, *optional*, defaults to 50257):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices except the
            embedding matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        bos_index (`int`, *optional*, defaults to 0):
            The index of the beginning of sentence token in the vocabulary.
        eos_index (`int`, *optional*, defaults to 1):
            The index of the end of sentence token in the vocabulary.
        pad_index (`int`, *optional*, defaults to 2):
            The index of the padding token in the vocabulary.
        unk_index (`int`, *optional*, defaults to 3):
            The index of the unknown token in the vocabulary.
        mask_index (`int`, *optional*, defaults to 5):
            The index of the masking token in the vocabulary.
        is_encoder(`bool`, *optional*, defaults to `True`):
            Whether or not the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.
        summary_type (`string`, *optional*, defaults to "first"):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Used in the sequence classification and multiple choice models.

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Used in the sequence classification and multiple choice models.

            The dropout ratio to be used after the projection and activation.
        start_n_top (`int`, *optional*, defaults to 5):
            Used in the SQuAD evaluation script.
        end_n_top (`int`, *optional*, defaults to 5):
            Used in the SQuAD evaluation script.
        mask_token_id (`int`, *optional*, defaults to 0):
            Model agnostic parameter to identify masked tokens when generating text in an MLM context.
        lang_id (`int`, *optional*, defaults to 1):
            The ID of the language used by the model. This parameter is used when generating text in a given language.
    

## FlaubertTokenizer


    Construct a Flaubert tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization.
    - Normalizing all inputs text.
    - The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like
      "__classify__") to a vocabulary.
    - The argument `do_lowercase` controls lower casing (automatically set for pretrained vocabularies).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Vocabulary file.
        merges_file (`str`):
            Merges file.
        do_lowercase (`bool`, *optional*, defaults to `False`):
            Controls lower casing.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"</s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<special1>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>']`):
            List of additional special tokens.
        lang2id (`Dict[str, int]`, *optional*):
            Dictionary mapping languages string identifiers to their IDs.
        id2lang (`Dict[int, str]`, *optional*):
            Dictionary mapping language IDs to their string identifiers.
    

<frameworkcontent>
<pt>

## FlaubertModel

No docstring available for FlaubertModel

Methods: forward

## FlaubertWithLMHeadModel


    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FlaubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FlaubertForSequenceClassification


    Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FlaubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FlaubertForMultipleChoice


    Flaubert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FlaubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FlaubertForTokenClassification


    Flaubert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FlaubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FlaubertForQuestionAnsweringSimple


    Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FlaubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FlaubertForQuestionAnswering

No docstring available for FlaubertForQuestionAnswering

Methods: forward

</pt>
<tf>

## TFFlaubertModel

No docstring available for TFFlaubertModel

Methods: call

## TFFlaubertWithLMHeadModel

No docstring available for TFFlaubertWithLMHeadModel

Methods: call

## TFFlaubertForSequenceClassification

No docstring available for TFFlaubertForSequenceClassification

Methods: call

## TFFlaubertForMultipleChoice

No docstring available for TFFlaubertForMultipleChoice

Methods: call

## TFFlaubertForTokenClassification

No docstring available for TFFlaubertForTokenClassification

Methods: call

## TFFlaubertForQuestionAnsweringSimple

No docstring available for TFFlaubertForQuestionAnsweringSimple

Methods: call

</tf>
</frameworkcontent>



