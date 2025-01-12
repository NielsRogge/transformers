<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ModernBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=modernbert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-modernbert-blueviolet">
</a>
<a href="https://arxiv.org/abs/2412.13663">
<img alt="Paper page" src="https://img.shields.io/badge/Paper%20page-2412.13663-green">
</a>
</div>

## Overview

The ModernBERT model was proposed in [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663) by Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Galalgher, Raja Bisas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Grifin Adams, Jeremy Howard and Iacopo Poli.

It is a refresh of the traditional encoder architecture, as used in previous models such as [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) and [RoBERTa](https://huggingface.co/docs/transformers/en/model_doc/roberta). 

It builds on BERT and implements many modern architectural improvements which have been developed since its original release, such as:
- [Rotary Positional Embeddings](https://huggingface.co/blog/designing-positional-encoding) to support sequences of up to 8192 tokens.
- [Unpadding](https://arxiv.org/abs/2208.08124) to ensure no compute is wasted on padding tokens, speeding up processing time for batches with mixed-length sequences.
- [GeGLU](https://arxiv.org/abs/2002.05202) Replacing the original MLP layers with GeGLU layers, shown to improve performance.
- [Alternating Attention](https://arxiv.org/abs/2004.05150v2) where most attention layers employ a sliding window of 128 tokens, with Global Attention only used every 3 layers.
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) to speed up processing.
- A model designed following recent [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489), ensuring maximum efficiency across inference GPUs.
- Modern training data scales (2 trillion tokens) and mixtures (including code ande math data)

The abstract from the paper is the following:

*Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and classification tasks with respect to larger decoder-only models. Despite being the workhorse of numerous production pipelines, there have been limited Pareto improvements to BERT since its release. In this paper, we introduce ModernBERT, bringing modern model optimizations to encoder-only models and representing a major Pareto improvement over older encoders. Trained on 2 trillion tokens with a native 8192 sequence length, ModernBERT models exhibit state-of-the-art results on a large pool of evaluations encompassing diverse classification tasks and both single and multi-vector retrieval on different domains (including code). In addition to strong downstream performance, ModernBERT is also the most speed and memory efficient encoder and is designed for inference on common GPUs.*

The original code can be found [here](https://github.com/answerdotai/modernbert).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ModernBert.

<PipelineTag pipeline="text-classification"/>

- A notebook on how to [finetune for General Language Understanding Evaluation (GLUE) with Transformers](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/finetune_modernbert_on_glue.ipynb), also available as a Google Colab [notebook](https://colab.research.google.com/github/AnswerDotAI/ModernBERT/blob/main/examples/finetune_modernbert_on_glue.ipynb). 🌎

<PipelineTag pipeline="sentence-similarity"/>

- A script on how to [finetune for text similarity or information retrieval with Sentence Transformers](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/train_st.py). 🌎
- A script on how to [finetune for information retrieval with PyLate](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/train_pylate.py). 🌎

<PipelineTag pipeline="fill-mask"/>

- [Masked language modeling task guide](../tasks/masked_language_modeling)


## ModernBertConfig


    This is the configuration class to store the configuration of a [`ModernBertModel`]. It is used to instantiate an ModernBert
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ModernBERT-base.
    e.g. [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50368):
            Vocabulary size of the ModernBert model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ModernBertModel`]
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1152):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu"`
            if not specified.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_cutoff_factor (`float`, *optional*, defaults to 2.0):
            The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        norm_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the normalization layers.
        pad_token_id (`int`, *optional*, defaults to 50283):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 50282):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 50281):
            Beginning of stream token id.
        cls_token_id (`int`, *optional*, defaults to 50281):
            Classification token id.
        sep_token_id (`int`, *optional*, defaults to 50282):
            Separation token id.
        global_rope_theta (`float`, *optional*, defaults to 160000.0):
            The base period of the global RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        global_attn_every_n_layers (`int`, *optional*, defaults to 3):
            The number of layers between global attention layers.
        local_attention (`int`, *optional*, defaults to 128):
            The window size for local attention.
        local_rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the local RoPE embeddings.
        embedding_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP layers.
        mlp_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the MLP layers.
        decoder_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the decoder layers.
        classifier_pooling (`str`, *optional*, defaults to `"cls"`):
            The pooling method for the classifier. Should be either `"cls"` or `"mean"`. In local attention layers, the
            CLS token doesn't attend to all tokens on long sequences.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the classifier.
        classifier_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the classifier.
        classifier_activation (`str`, *optional*, defaults to `"gelu"`):
            The activation function for the classifier.
        deterministic_flash_attn (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic flash attention. If `False`, inference will be faster but not deterministic.
        sparse_prediction (`bool`, *optional*, defaults to `False`):
            Whether to use sparse prediction for the masked language model instead of returning the full dense logits.
        sparse_pred_ignore_index (`int`, *optional*, defaults to -100):
            The index to ignore for the sparse prediction.
        reference_compile (`bool`, *optional*):
            Whether to compile the layers of the model which were compiled during pretraining. If `None`, then parts of
            the model will be compiled if 1) `triton` is installed, 2) the model is not on MPS, 3) the model is not
            shared between devices, and 4) the model is not resized after initialization. If `True`, then the model may
            be faster in some scenarios.
        repad_logits_with_grad (`bool`, *optional*, defaults to `False`):
            When True, ModernBertForMaskedLM keeps track of the logits' gradient when repadding for output. This only
            applies when using Flash Attention 2 with passed labels. Otherwise output logits always have a gradient.

    Examples:

    ```python
    >>> from transformers import ModernBertModel, ModernBertConfig

    >>> # Initializing a ModernBert style configuration
    >>> configuration = ModernBertConfig()

    >>> # Initializing a model from the modernbert-base style configuration
    >>> model = ModernBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

<frameworkcontent>
<pt>

## ModernBertModel

The bare ModernBert Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ModernBertConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ModernBertForMaskedLM

The ModernBert Model with a decoder head on top that is used for masked language modeling.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ModernBertConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ModernBertForSequenceClassification

The ModernBert Model with a sequence classification head on top that performs pooling.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ModernBertConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ModernBertForTokenClassification

The ModernBert Model with a token classification head on top, e.g. for Named Entity Recognition (NER) tasks.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ModernBertConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

</pt>
</frameworkcontent>
