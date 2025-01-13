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

# Transformer XL

<Tip warning={true}>

This model is in maintenance mode only, so we won't accept any new PRs changing its code. This model was deprecated due to security issues linked to `pickle.load`.

We recommend switching to more recent models for improved security.

In case you would still like to use `TransfoXL` in your experiments, we recommend using the [Hub checkpoint](https://huggingface.co/transfo-xl/transfo-xl-wt103) with a specific revision to ensure you are downloading safe files from the Hub.

You will need to set the environment variable `TRUST_REMOTE_CODE` to `True` in order to allow the
usage of `pickle.load()`:

```python
import os
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel

os.environ["TRUST_REMOTE_CODE"] = "True"

checkpoint = 'transfo-xl/transfo-xl-wt103'
revision = '40a186da79458c9f9de846edfaea79c412137f97'

tokenizer = TransfoXLTokenizer.from_pretrained(checkpoint, revision=revision)
model = TransfoXLLMHeadModel.from_pretrained(checkpoint, revision=revision)
```

If you run into any issues running this model, please reinstall the last version that supported this model: v4.35.0.
You can do so by running the following command: `pip install -U transformers==4.35.0`.

</Tip>

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=transfo-xl">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-transfo--xl-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/transfo-xl-wt103">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The Transformer-XL model was proposed in [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan
Salakhutdinov. It's a causal (uni-directional) transformer with relative positioning (sinusoïdal) embeddings which can
reuse previously computed hidden-states to attend to longer context (memory). This model also uses adaptive softmax
inputs and outputs (tied).

The abstract from the paper is the following:

*Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the
setting of language modeling. We propose a novel neural architecture Transformer-XL that enables learning dependency
beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism and a
novel positional encoding scheme. Our method not only enables capturing longer-term dependency, but also resolves the
context fragmentation problem. As a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450%
longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+
times faster than vanilla Transformers during evaluation. Notably, we improve the state-of-the-art results of
bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on Penn
Treebank (without finetuning). When trained only on WikiText-103, Transformer-XL manages to generate reasonably
coherent, novel text articles with thousands of tokens.*

This model was contributed by [thomwolf](https://huggingface.co/thomwolf). The original code can be found [here](https://github.com/kimiyoung/transformer-xl).

## Usage tips

- Transformer-XL uses relative sinusoidal positional embeddings. Padding can be done on the left or on the right. The
  original implementation trains on SQuAD with padding on the left, therefore the padding defaults are set to left.
- Transformer-XL is one of the few models that has no sequence length limit.
- Same as a regular GPT model, but introduces a recurrence mechanism for two consecutive segments (similar to a regular RNNs with two consecutive inputs). In this context, a segment is a number of consecutive tokens (for instance 512) that may span across multiple documents, and segments are fed in order to the model.
- Basically, the hidden states of the previous segment are concatenated to the current input to compute the attention scores. This allows the model to pay attention to information that was in the previous segment as well as the current one. By stacking multiple attention layers, the receptive field can be increased to multiple previous segments.
- This changes the positional embeddings to positional relative embeddings (as the regular positional embeddings would give the same results in the current input and the current hidden state at a given position) and needs to make some adjustments in the way attention scores are computed.


<Tip warning={true}>

TransformerXL does **not** work with *torch.nn.DataParallel* due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035)

</Tip>

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Causal language modeling task guide](../tasks/language_modeling)

## TransfoXLConfig


    This is the configuration class to store the configuration of a [`TransfoXLModel`] or a [`TFTransfoXLModel`]. It is
    used to instantiate a Transformer-XL model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the TransfoXL
    [transfo-xl/transfo-xl-wt103](https://huggingface.co/transfo-xl/transfo-xl-wt103) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 267735):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`TransfoXLModel`] or [`TFTransfoXLModel`].
        cutoffs (`List[int]`, *optional*, defaults to `[20000, 40000, 200000]`):
            Cutoffs for the adaptive softmax.
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the model's hidden states.
        d_embed (`int`, *optional*, defaults to 1024):
            Dimensionality of the embeddings
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_head (`int`, *optional*, defaults to 64):
            Dimensionality of the model's heads.
        d_inner (`int`, *optional*, defaults to 4096):
            Inner dimension in FF
        div_val (`int`, *optional*, defaults to 4):
            Divident value for adapative input and softmax
        pre_lnorm (`boolean`, *optional*, defaults to `False`):
            Whether or not to apply LayerNorm to the input instead of the output in the blocks.
        n_layer (`int`, *optional*, defaults to 18):
            Number of hidden layers in the Transformer encoder.
        mem_len (`int`, *optional*, defaults to 1600):
            Length of the retained previous heads.
        clamp_len (`int`, *optional*, defaults to 1000):
            Use the same pos embeddings after clamp_len.
        same_length (`boolean`, *optional*, defaults to `True`):
            Whether or not to use the same attn length for all tokens
        proj_share_all_but_first (`boolean`, *optional*, defaults to `True`):
            True to share all but first projs, False not to share.
        attn_type (`int`, *optional*, defaults to 0):
            Attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
        sample_softmax (`int`, *optional*, defaults to -1):
            Number of samples in the sampled softmax.
        adaptive (`boolean`, *optional*, defaults to `True`):
            Whether or not to use adaptive softmax.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        dropatt (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        untie_r (`boolean`, *optional*, defaults to `True`):
            Whether ot not to untie relative position biases.
        init (`str`, *optional*, defaults to `"normal"`):
            Parameter initializer to use.
        init_range (`float`, *optional*, defaults to 0.01):
            Parameters initialized by U(-init_range, init_range).
        proj_init_std (`float`, *optional*, defaults to 0.01):
            Parameters initialized by N(0, init_std)
        init_std (`float`, *optional*, defaults to 0.02):
            Parameters initialized by N(0, init_std)
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers
        eos_token_id (`int`, *optional*, defaults to 0):
            End of stream token id.

    Examples:

    ```python
    >>> from transformers import TransfoXLConfig, TransfoXLModel

    >>> # Initializing a Transformer XL configuration
    >>> configuration = TransfoXLConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = TransfoXLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## TransfoXLTokenizer


    Construct a Transformer-XL tokenizer adapted from Vocab class in [the original
    code](https://github.com/kimiyoung/transformer-xl). The Transformer-XL tokenizer is a word-level tokenizer (no
    sub-word tokenization).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        special (`List[str]`, *optional*):
            A list of special tokens (to be treated by the original implementation of this tokenizer).
        min_freq (`int`, *optional*, defaults to 0):
            The minimum number of times a token has to be present in order to be kept in the vocabulary (otherwise it
            will be mapped to `unk_token`).
        max_size (`int`, *optional*):
            The maximum size of the vocabulary. If left unset, it will default to the size of the vocabulary found
            after excluding the tokens according to the `min_freq` rule.
        lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        delimiter (`str`, *optional*):
            The delimiter used between tokens.
        vocab_file (`str`, *optional*):
            File containing the vocabulary (from the original implementation).
        pretrained_vocab_file (`str`, *optional*):
            File containing the vocabulary as saved with the `save_pretrained()` method.
        never_split (`List[str]`, *optional*):
            List of tokens that should never be split. If no list is specified, will simply use the existing special
            tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str`, *optional*, defaults to `"<eos>"`):
            The end of sequence token.
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<formula>']`):
            A list of additional special tokens (for the HuggingFace functionality).
        language (`str`, *optional*, defaults to `"en"`):
            The language of this tokenizer (used for mose preprocessing).
    

Methods: save_vocabulary

## TransfoXL specific outputs

[[autodoc]] models.deprecated.transfo_xl.modeling_transfo_xl.TransfoXLModelOutput: module 'transformers.models.deprecated' has no attribute 'transfo_xl'

[[autodoc]] models.deprecated.transfo_xl.modeling_transfo_xl.TransfoXLLMHeadModelOutput: module 'transformers.models.deprecated' has no attribute 'transfo_xl'

[[autodoc]] models.deprecated.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLModelOutput: module 'transformers.models.deprecated' has no attribute 'transfo_xl'

[[autodoc]] models.deprecated.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLLMHeadModelOutput: module 'transformers.models.deprecated' has no attribute 'transfo_xl'

<frameworkcontent>
<pt>

## TransfoXLModel

The bare Bert Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TransfoXLConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## TransfoXLLMHeadModel


    The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive
    input embeddings)
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TransfoXLConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## TransfoXLForSequenceClassification


    The Transformer-XL Model transformer with a sequence classification head on top (linear layer).

    [`TransfoXLForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TransfoXLConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

</pt>
<tf>

## TFTransfoXLModel

No docstring available for TFTransfoXLModel

Methods: call

## TFTransfoXLLMHeadModel

No docstring available for TFTransfoXLLMHeadModel

Methods: call

## TFTransfoXLForSequenceClassification

No docstring available for TFTransfoXLForSequenceClassification

Methods: call

</tf>
</frameworkcontent>

## Internal Layers

No docstring available for AdaptiveEmbedding

No docstring available for TFAdaptiveEmbedding
