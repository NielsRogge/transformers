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

# XLM-ProphetNet

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=xprophetnet">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-xprophetnet-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/xprophetnet-large-wiki100-cased-xglue-ntg">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) and assign
@patrickvonplaten


## Overview

The XLM-ProphetNet model was proposed in [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training,](https://arxiv.org/abs/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei
Zhang, Ming Zhou on 13 Jan, 2020.

XLM-ProphetNet is an encoder-decoder model and can predict n-future tokens for "ngram" language modeling instead of
just the next token. Its architecture is identical to ProhpetNet, but the model was trained on the multi-lingual
"wiki100" Wikipedia dump. XLM-ProphetNet's model architecture and pretraining objective is same as ProphetNet, but XLM-ProphetNet was pre-trained on the cross-lingual dataset XGLUE.

The abstract from the paper is the following:

*In this paper, we present a new sequence-to-sequence pretraining model called ProphetNet, which introduces a novel
self-supervised objective named future n-gram prediction and the proposed n-stream self-attention mechanism. Instead of
the optimization of one-step ahead prediction in traditional sequence-to-sequence model, the ProphetNet is optimized by
n-step ahead prediction which predicts the next n tokens simultaneously based on previous context tokens at each time
step. The future n-gram prediction explicitly encourages the model to plan for the future tokens and prevent
overfitting on strong local correlations. We pre-train ProphetNet using a base scale dataset (16GB) and a large scale
dataset (160GB) respectively. Then we conduct experiments on CNN/DailyMail, Gigaword, and SQuAD 1.1 benchmarks for
abstractive summarization and question generation tasks. Experimental results show that ProphetNet achieves new
state-of-the-art results on all these datasets compared to the models using the same scale pretraining corpus.*

The Authors' code can be found [here](https://github.com/microsoft/ProphetNet).

## Resources

- [Causal language modeling task guide](../tasks/language_modeling)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## XLMProphetNetConfig


    This is the configuration class to store the configuration of a [`XLMProphetNetModel`]. It is used to instantiate a
    XLMProphetNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the XLMProphetNet
    [microsoft/xprophetnet-large-wiki100-cased](https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the ProphetNET model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`XLMProphetNetModel`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        num_encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        num_encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the `intermediate` (often named feed-forward) layer in decoder.
        num_decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        num_decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        add_cross_attention (`bool`, *optional*, defaults to `True`):
            Whether cross-attention layers should be added to the model.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether this is an encoder/decoder model.
        pad_token_id (`int`, *optional*, defaults to 1)
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0)
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2)
            End of stream token id.
        ngram (`int`, *optional*, defaults to 2)
            Number of future tokens to predict. Set to 1 to be same as traditional Language model to predict next first
            token.
        num_buckets (`int`, *optional*, defaults to 32)
            The number of buckets to use for each attention layer. This is for relative position calculation. See the
            [T5 paper](see https://arxiv.org/abs/1910.10683) for more details.
        relative_max_distance (`int`, *optional*, defaults to 128)
            Relative distances greater than this number will be put into the last same bucket. This is for relative
            position calculation. See the [T5 paper](see https://arxiv.org/abs/1910.10683) for more details.
        disable_ngram_loss (`bool`, *optional*, defaults to `False`):
            Whether be trained predicting only the next first token.
        eps (`float`, *optional*, defaults to 0.0):
            Controls the `epsilon` parameter value for label smoothing in the loss calculation. If set to 0, no label
            smoothing is performed.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    

## XLMProphetNetTokenizer


    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    

## XLMProphetNetModel

The bare XLMProphetNet Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
    from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
    file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
    behavior.

    Parameters:
        config ([`XLMProphetNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


## XLMProphetNetEncoder

The standalone encoder part of the XLMProphetNetModel.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
    from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
    file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
    behavior.

    Parameters:
        config ([`XLMProphetNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
        The word embedding parameters. This can be used to initialize [`XLMProphetNetEncoder`] with pre-defined word
        embeddings instead of randomly initialized word embeddings.
    

## XLMProphetNetDecoder

The standalone decoder part of the XLMProphetNetModel.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
    from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
    file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
    behavior.

    Parameters:
        config ([`XLMProphetNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
        The word embedding parameters. This can be used to initialize [`XLMProphetNetEncoder`] with pre-defined word
        embeddings instead of randomly initialized word embeddings.
    

## XLMProphetNetForConditionalGeneration

The XLMProphetNet Model with a language modeling head. Can be used for sequence generation tasks.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
    from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
    file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
    behavior.

    Parameters:
        config ([`XLMProphetNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


## XLMProphetNetForCausalLM

The standalone decoder part of the XLMProphetNetModel with a lm head on top. The model can be used for causal language modeling.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
    from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
    file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
    behavior.

    Parameters:
        config ([`XLMProphetNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

