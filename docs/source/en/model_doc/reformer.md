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

# Reformer

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=reformer">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-reformer-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/reformer-crime-and-punishment">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The Reformer model was proposed in the paper [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451.pdf) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

The abstract from the paper is the following:

*Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can
be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of
Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its
complexity from O(L^2) to O(Llog(L)), where L is the length of the sequence. Furthermore, we use reversible residual
layers instead of the standard residuals, which allows storing activations only once in the training process instead of
N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models
while being much more memory-efficient and much faster on long sequences.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The Authors' code can be
found [here](https://github.com/google/trax/tree/master/trax/models/reformer).

## Usage tips

- Reformer does **not** work with *torch.nn.DataParallel* due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035).
- Use Axial position encoding (see below for more details). It’s a mechanism to avoid having a huge positional encoding matrix (when the sequence length is very big) by factorizing it into smaller matrices.
- Replace traditional attention by LSH (local-sensitive hashing) attention (see below for more details). It’s a technique to avoid computing the full product query-key in the attention layers.
- Avoid storing the intermediate results of each layer by using reversible transformer layers to obtain them during the backward pass (subtracting the residuals from the input of the next layer gives them back) or recomputing them for results inside a given layer (less efficient than storing them but saves memory).
- Compute the feedforward operations by chunks and not on the whole batch.

### Axial Positional Encodings

Axial Positional Encodings were first implemented in Google's [trax library](https://github.com/google/trax/blob/4d99ad4965bab1deba227539758d59f0df0fef48/trax/layers/research/position_encodings.py#L29)
and developed by the authors of this model's paper. In models that are treating very long input sequences, the
conventional position id encodings store an embeddings vector of size \\(d\\) being the `config.hidden_size` for
every position \\(i, \ldots, n_s\\), with \\(n_s\\) being `config.max_embedding_size`. This means that having
a sequence length of \\(n_s = 2^{19} \approx 0.5M\\) and a `config.hidden_size` of \\(d = 2^{10} \approx 1000\\)
would result in a position encoding matrix:

$$X_{i,j}, \text{ with } i \in \left[1,\ldots, d\right] \text{ and } j \in \left[1,\ldots, n_s\right]$$

which alone has over 500M parameters to store. Axial positional encodings factorize \\(X_{i,j}\\) into two matrices:

$$X^{1}_{i,j}, \text{ with } i \in \left[1,\ldots, d^1\right] \text{ and } j \in \left[1,\ldots, n_s^1\right]$$

and

$$X^{2}_{i,j}, \text{ with } i \in \left[1,\ldots, d^2\right] \text{ and } j \in \left[1,\ldots, n_s^2\right]$$

with:

$$d = d^1 + d^2 \text{ and } n_s = n_s^1 \times n_s^2 .$$

Therefore the following holds:

$$X_{i,j} = \begin{cases}
X^{1}_{i, k}, & \text{if }\ i < d^1 \text{ with } k = j \mod n_s^1 \\
X^{2}_{i - d^1, l}, & \text{if } i \ge d^1 \text{ with } l = \lfloor\frac{j}{n_s^1}\rfloor
\end{cases}$$

Intuitively, this means that a position embedding vector \\(x_j \in \mathbb{R}^{d}\\) is now the composition of two
factorized embedding vectors: \\(x^1_{k, l} + x^2_{l, k}\\), where as the `config.max_embedding_size` dimension
\\(j\\) is factorized into \\(k \text{ and } l\\). This design ensures that each position embedding vector
\\(x_j\\) is unique.

Using the above example again, axial position encoding with \\(d^1 = 2^9, d^2 = 2^9, n_s^1 = 2^9, n_s^2 = 2^{10}\\)
can drastically reduced the number of parameters from 500 000 000 to \\(2^{18} + 2^{19} \approx 780 000\\) parameters, this means 85% less memory usage.

In practice, the parameter `config.axial_pos_embds_dim` is set to a tuple \\((d^1, d^2)\\) which sum has to be
equal to `config.hidden_size` and `config.axial_pos_shape` is set to a tuple \\((n_s^1, n_s^2)\\) which
product has to be equal to `config.max_embedding_size`, which during training has to be equal to the *sequence
length* of the `input_ids`.


### LSH Self Attention

In Locality sensitive hashing (LSH) self attention the key and query projection weights are tied. Therefore, the key
query embedding vectors are also tied. LSH self attention uses the locality sensitive hashing mechanism proposed in
[Practical and Optimal LSH for Angular Distance](https://arxiv.org/abs/1509.02897) to assign each of the tied key
query embedding vectors to one of `config.num_buckets` possible buckets. The premise is that the more "similar"
key query embedding vectors (in terms of *cosine similarity*) are to each other, the more likely they are assigned to
the same bucket.

The accuracy of the LSH mechanism can be improved by increasing `config.num_hashes` or directly the argument
`num_hashes` of the forward function so that the output of the LSH self attention better approximates the output
of the "normal" full self attention. The buckets are then sorted and chunked into query key embedding vector chunks
each of length `config.lsh_chunk_length`. For each chunk, the query embedding vectors attend to its key vectors
(which are tied to themselves) and to the key embedding vectors of `config.lsh_num_chunks_before` previous
neighboring chunks and `config.lsh_num_chunks_after` following neighboring chunks.

For more information, see the [original Paper](https://arxiv.org/abs/2001.04451) or this great [blog post](https://www.pragmatic.ml/reformer-deep-dive/).

Note that `config.num_buckets` can also be factorized into a list \\((n_{\text{buckets}}^1,
n_{\text{buckets}}^2)\\). This way instead of assigning the query key embedding vectors to one of \\((1,\ldots,
n_{\text{buckets}})\\) they are assigned to one of \\((1-1,\ldots, n_{\text{buckets}}^1-1, \ldots,
1-n_{\text{buckets}}^2, \ldots, n_{\text{buckets}}^1-n_{\text{buckets}}^2)\\). This is crucial for very long sequences to
save memory.

When training a model from scratch, it is recommended to leave `config.num_buckets=None`, so that depending on the
sequence length a good value for `num_buckets` is calculated on the fly. This value will then automatically be
saved in the config and should be reused for inference.

Using LSH self attention, the memory and time complexity of the query-key matmul operation can be reduced from
\\(\mathcal{O}(n_s \times n_s)\\) to \\(\mathcal{O}(n_s \times \log(n_s))\\), which usually represents the memory
and time bottleneck in a transformer model, with \\(n_s\\) being the sequence length.


### Local Self Attention

Local self attention is essentially a "normal" self attention layer with key, query and value projections, but is
chunked so that in each chunk of length `config.local_chunk_length` the query embedding vectors only attends to
the key embedding vectors in its chunk and to the key embedding vectors of `config.local_num_chunks_before`
previous neighboring chunks and `config.local_num_chunks_after` following neighboring chunks.

Using Local self attention, the memory and time complexity of the query-key matmul operation can be reduced from
\\(\mathcal{O}(n_s \times n_s)\\) to \\(\mathcal{O}(n_s \times \log(n_s))\\), which usually represents the memory
and time bottleneck in a transformer model, with \\(n_s\\) being the sequence length.


### Training

During training, we must ensure that the sequence length is set to a value that can be divided by the least common
multiple of `config.lsh_chunk_length` and `config.local_chunk_length` and that the parameters of the Axial
Positional Encodings are correctly set as described above. Reformer is very memory efficient so that the model can
easily be trained on sequences as long as 64000 tokens.

For training, the [`ReformerModelWithLMHead`] should be used as follows:

```python
input_ids = tokenizer.encode("This is a sentence from the training data", return_tensors="pt")
loss = model(input_ids, labels=input_ids)[0]
```

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)

## ReformerConfig


    This is the configuration class to store the configuration of a [`ReformerModel`]. It is used to instantiate a
    Reformer model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ReFormer
    [google/reformer-crime-and-punishment](https://huggingface.co/google/reformer-crime-and-punishment) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        attention_head_size (`int`, *optional*, defaults to 64):
            Dimensionality of the projected key, query and value vectors
        attn_layers (`List[str]`, *optional*, defaults to `["local", "lsh", "local", "lsh", "local", "lsh"]`):
            List of attention layer types in ascending order. It can be chosen between a LSHSelfAttention layer
            (`"lsh"`) and a LocalSelfAttention layer (`"local"`).

            For more information on LSHSelfAttention layer, see [LSH Self Attention](reformer#lsh-self-attention). For
            more information on LocalSelfAttention layer, see [Local Self Attention](reformer#local-self-attention).
        axial_pos_embds (`bool`, *optional*, defaults to `True`):
            Whether or not to use axial position embeddings. For more information on how axial position embeddings
            work, see [Axial Position Encodings](reformer#axial-positional-encodings).
        axial_norm_std (`float`, *optional*, defaults to 1.0):
            The standard deviation of the normal_initializer for initializing the weight matrices of the axial
            positional encodings.
        axial_pos_shape (`List[int]`, *optional*, defaults to `[64, 64]`):
            The position dims of the axial position encodings. During training, the product of the position dims has to
            be equal to the sequence length.

            For more information on how axial position embeddings work, see [Axial Position
            Encodings](reformer#axial-positional-encodings).
        axial_pos_embds_dim (`List[int]`, *optional*, defaults to `[64, 192]`):
            The embedding dims of the axial position encodings. The sum of the embedding dims has to be equal to the
            hidden size.

            For more information on how axial position embeddings work, see [Axial Position
            Encodings](reformer#axial-positional-encodings).
        chunk_size_lm_head (`int`, *optional*, defaults to 0):
            The chunk size of the final language model feed forward head layer. A chunk size of 0 means that the feed
            forward layer is not chunked. A chunk size of n means that the feed forward layer processes n <
            sequence_length embeddings at a time.

            For more information on feed forward chunking, see [How does Feed Forward Chunking
            work?](../glossary#feed-forward-chunking).
        eos_token_id (`int`, *optional*, defaults to 2):
            The token id for the end-of-sentence token.
        feed_forward_size (`int`, *optional*, defaults to 512):
            Dimensionality of the feed_forward layer in the residual attention block.
        hash_seed (`int`, *optional*):
            Seed that can be used to make local sensitive hashing in `LSHSelfAttention` deterministic. This should only
            be set for testing purposed. For evaluation and training purposes `hash_seed` should be left as `None` to
            ensure fully random rotations in local sensitive hashing scheme.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the feed forward layer in the residual attention
            block. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.05):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the output hidden states of the residual attention blocks.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether or not to use a causal mask in addition to the `attention_mask` passed to [`ReformerModel`]. When
            using the Reformer for causal language modeling, this argument should be set to `True`.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        local_chunk_length (`int`, *optional*, defaults to 64):
            Length of chunk which attends to itself in `LocalSelfAttention`. Chunking reduces memory complexity from
            sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk
            length (chunked self attention).
        local_num_chunks_before (`int`, *optional*, defaults to 1):
            Number of previous neighbouring chunks to attend to in `LocalSelfAttention` layer to itself.
        local_num_chunks_after (`int`, *optional*, defaults to 0):
            Number of following neighbouring chunks to attend to in `LocalSelfAttention` layer in addition to itself.
        local_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities in `LocalSelfAttention`.
        lsh_attn_chunk_length (`int`, *optional*, defaults to 64):
            Length of chunk which attends to itself in `LSHSelfAttention`. Chunking reduces memory complexity from
            sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk
            length (chunked self attention).
        lsh_num_chunks_before (`int`, *optional*, defaults to 1):
            Number of previous neighbouring chunks to attend to in `LSHSelfAttention` layer to itself.
        lsh_num_chunks_after (`int`, *optional*, defaults to 0):
            Number of following neighbouring chunks to attend to in `LSHSelfAttention` layer to itself.
        lsh_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities in `LSHSelfAttention`.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_buckets (`int` or `List[int]`, *optional*):
            Number of buckets, the key query vectors can be "hashed into" using the locality sensitive hashing scheme.
            Each query key vector is hashed into a hash in `1, ..., num_buckets`. The number of buckets can also be
            factorized into a list for improved memory complexity. In this case, each query key vector is hashed into a
            hash in `1-1, 1-2, ..., num_buckets[0]-1, ..., num_buckets[0]-num_buckets[1]` if `num_buckets` is
            factorized into two factors. The number of buckets (or the product the factors) should approximately equal
            sequence length / lsh_chunk_length. If `num_buckets` not set, a good value is calculated on the fly.
        num_hashes (`int`, *optional*, defaults to 1):
            Number of hashing rounds (e.g., number of random rotations) in Local Sensitive Hashing scheme. The higher
            `num_hashes`, the more accurate the `LSHSelfAttention` becomes, but also the more memory and time intensive
            the hashing becomes.
        pad_token_id (`int`, *optional*, defaults to 0):
            The token id for the padding token.
        vocab_size (`int`, *optional*, defaults to 320):\
            Vocabulary size of the Reformer model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`ReformerModel`].
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import ReformerConfig, ReformerModel

    >>> # Initializing a Reformer configuration
    >>> configuration = ReformerConfig()

    >>> # Initializing a Reformer model (with random weights)
    >>> model = ReformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```


## ReformerTokenizer


    Construct a Reformer tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece) .

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        additional_special_tokens (`List[str]`, *optional*, defaults to `[]`):
            Additional special tokens used by the tokenizer.
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
    

Methods: save_vocabulary

## ReformerTokenizerFast


    Construct a "fast" Reformer tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
    

## ReformerModel

The bare Reformer Model transformer outputting raw hidden-stateswithout any specific head on top.
    Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev,
    Łukasz Kaiser, Anselm Levskaya.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ReformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ReformerModelWithLMHead

Reformer Model with a `language modeling` head on top.
    Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev,
    Łukasz Kaiser, Anselm Levskaya.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ReformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ReformerForMaskedLM

Reformer Model with a `language modeling` head on top.
    Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev,
    Łukasz Kaiser, Anselm Levskaya.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ReformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ReformerForSequenceClassification


    Reformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    
    Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev,
    Łukasz Kaiser, Anselm Levskaya.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ReformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ReformerForQuestionAnswering


    Reformer Model with a span classification head on top for extractive question-answering tasks like SQuAD / TriviaQA
    ( a linear layer on top of hidden-states output to compute `span start logits` and `span end logits`.
    
    Reformer was proposed in [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev,
    Łukasz Kaiser, Anselm Levskaya.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ReformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
