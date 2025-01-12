<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Informer

## Overview

The Informer model was proposed in [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) by Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.

This method introduces a Probabilistic Attention mechanism to select the "active" queries rather than the "lazy" queries and provides a sparse Transformer thus mitigating the quadratic compute and memory requirements of vanilla attention.

The abstract from the paper is the following:

*Many real-world applications require the prediction of long sequence time-series, such as electricity consumption planning. Long sequence time-series forecasting (LSTF) demands a high prediction capacity of the model, which is the ability to capture precise long-range dependency coupling between output and input efficiently. Recent studies have shown the potential of Transformer to increase the prediction capacity. However, there are several severe issues with Transformer that prevent it from being directly applicable to LSTF, including quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. To address these issues, we design an efficient transformer-based model for LSTF, named Informer, with three distinctive characteristics: (i) a ProbSparse self-attention mechanism, which achieves O(L logL) in time complexity and memory usage, and has comparable performance on sequences' dependency alignment. (ii) the self-attention distilling highlights dominating attention by halving cascading layer input, and efficiently handles extreme long input sequences. (iii) the generative style decoder, while conceptually simple, predicts the long time-series sequences at one forward operation rather than a step-by-step way, which drastically improves the inference speed of long-sequence predictions. Extensive experiments on four large-scale datasets demonstrate that Informer significantly outperforms existing methods and provides a new solution to the LSTF problem.*

This model was contributed by [elisim](https://huggingface.co/elisim) and [kashif](https://huggingface.co/kashif).
The original code can be found [here](https://github.com/zhouhaoyi/Informer2020).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- Check out the Informer blog-post in HuggingFace blog: [Multivariate Probabilistic Time Series Forecasting with Informer](https://huggingface.co/blog/informer)

## InformerConfig


    This is the configuration class to store the configuration of an [`InformerModel`]. It is used to instantiate an
    Informer model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Informer
    [huggingface/informer-tourism-monthly](https://huggingface.co/huggingface/informer-tourism-monthly) architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        prediction_length (`int`):
            The prediction length for the decoder. In other words, the prediction horizon of the model. This value is
            typically dictated by the dataset and we recommend to set it appropriately.
        context_length (`int`, *optional*, defaults to `prediction_length`):
            The context length for the encoder. If `None`, the context length will be the same as the
            `prediction_length`.
        distribution_output (`string`, *optional*, defaults to `"student_t"`):
            The distribution emission head for the model. Could be either "student_t", "normal" or "negative_binomial".
        loss (`string`, *optional*, defaults to `"nll"`):
            The loss function for the model corresponding to the `distribution_output` head. For parametric
            distributions it is the negative log likelihood (nll) - which currently is the only supported one.
        input_size (`int`, *optional*, defaults to 1):
            The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of
            multivariate targets.
        scaling (`string` or `bool`, *optional* defaults to `"mean"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean".
        lags_sequence (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 5, 6, 7]`):
            The lags of the input time series as covariates often dictated by the frequency of the data. Default is
            `[1, 2, 3, 4, 5, 6, 7]` but we recommend to change it based on the dataset appropriately.
        num_time_features (`int`, *optional*, defaults to 0):
            The number of time features in the input time series.
        num_dynamic_real_features (`int`, *optional*, defaults to 0):
            The number of dynamic real valued features.
        num_static_categorical_features (`int`, *optional*, defaults to 0):
            The number of static categorical features.
        num_static_real_features (`int`, *optional*, defaults to 0):
            The number of static real valued features.
        cardinality (`list[int]`, *optional*):
            The cardinality (number of different values) for each of the static categorical features. Should be a list
            of integers, having the same length as `num_static_categorical_features`. Cannot be `None` if
            `num_static_categorical_features` is > 0.
        embedding_dimension (`list[int]`, *optional*):
            The dimension of the embedding for each of the static categorical features. Should be a list of integers,
            having the same length as `num_static_categorical_features`. Cannot be `None` if
            `num_static_categorical_features` is > 0.
        d_model (`int`, *optional*, defaults to 64):
            Dimensionality of the transformer layers.
        encoder_layers (`int`, *optional*, defaults to 2):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 2):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and decoder. If string, `"gelu"` and
            `"relu"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the encoder, and decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention and fully connected layers for each encoder layer.
        decoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention and fully connected layers for each decoder layer.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability used between the two layers of the feed-forward networks.
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples to generate in parallel for each time step of inference.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.
        attention_type (`str`, *optional*, defaults to "prob"):
            Attention used in encoder. This can be set to "prob" (Informer's ProbAttention) or "full" (vanilla
            transformer's canonical self-attention).
        sampling_factor (`int`, *optional*, defaults to 5):
            ProbSparse sampling factor (only makes affect when `attention_type`="prob"). It is used to control the
            reduced query matrix (Q_reduce) input length.
        distil (`bool`, *optional*, defaults to `True`):
            Whether to use distilling in encoder.

    Example:

    ```python
    >>> from transformers import InformerConfig, InformerModel

    >>> # Initializing an Informer configuration with 12 time steps for prediction
    >>> configuration = InformerConfig(prediction_length=12)

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = InformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## InformerModel

The bare Informer Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TimeSeriesTransformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## InformerForPrediction

The Informer Model with a distribution head on top for time-series forecasting.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TimeSeriesTransformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward