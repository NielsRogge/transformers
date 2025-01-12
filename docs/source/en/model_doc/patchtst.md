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

# PatchTST

## Overview

The PatchTST model was proposed in [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730) by Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong and Jayant Kalagnanam.

At a high level the model vectorizes time series into patches of a given size and encodes the resulting sequence of vectors via a Transformer that then outputs the prediction length forecast via an appropriate head. The model is illustrated in the following figure:

![model](https://github.com/namctin/transformers/assets/8100/150af169-29de-419a-8d98-eb78251c21fa)

The abstract from the paper is the following:

*We propose an efficient design of Transformer-based models for multivariate time series forecasting and self-supervised representation learning. It is based on two key components: (i) segmentation of time series into subseries-level patches which are served as input tokens to Transformer; (ii) channel-independence where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series. Patching design naturally has three-fold benefit: local semantic information is retained in the embedding; computation and memory usage of the attention maps are quadratically reduced given the same look-back window; and the model can attend longer history. Our channel-independent patch time series Transformer (PatchTST) can improve the long-term forecasting accuracy significantly when compared with that of SOTA Transformer-based models. We also apply our model to self-supervised pre-training tasks and attain excellent fine-tuning performance, which outperforms supervised training on large datasets. Transferring of masked pre-trained representation on one dataset to others also produces SOTA forecasting accuracy.*

This model was contributed by [namctin](https://huggingface.co/namctin), [gsinthong](https://huggingface.co/gsinthong), [diepi](https://huggingface.co/diepi), [vijaye12](https://huggingface.co/vijaye12), [wmgifford](https://huggingface.co/wmgifford), and [kashif](https://huggingface.co/kashif). The original code can be found [here](https://github.com/yuqinie98/PatchTST).

## Usage tips

The model can also be used for time series classification and time series regression. See the respective [`PatchTSTForClassification`] and [`PatchTSTForRegression`] classes.

## Resources

- A blog post explaining PatchTST in depth can be found [here](https://huggingface.co/blog/patchtst). The blog can also be opened in Google Colab.

## PatchTSTConfig


    This is the configuration class to store the configuration of an [`PatchTSTModel`]. It is used to instantiate an
    PatchTST model according to the specified arguments, defining the model architecture.
    [ibm/patchtst](https://huggingface.co/ibm/patchtst) architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_input_channels (`int`, *optional*, defaults to 1):
            The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of
            multivariate targets.
        context_length (`int`, *optional*, defaults to 32):
            The context length of the input sequence.
        distribution_output (`str`, *optional*, defaults to `"student_t"`):
            The distribution emission head for the model when loss is "nll". Could be either "student_t", "normal" or
            "negative_binomial".
        loss (`str`, *optional*, defaults to `"mse"`):
            The loss function for the model corresponding to the `distribution_output` head. For parametric
            distributions it is the negative log likelihood ("nll") and for point estimates it is the mean squared
            error "mse".
        patch_length (`int`, *optional*, defaults to 1):
            Define the patch length of the patchification process.
        patch_stride (`int`, *optional*, defaults to 1):
            Define the stride of the patchification process.
        num_hidden_layers (`int`, *optional*, defaults to 3):
            Number of hidden layers.
        d_model (`int`, *optional*, defaults to 128):
            Dimensionality of the transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        share_embedding (`bool`, *optional*, defaults to `True`):
            Sharing the input embedding across all channels.
        channel_attention (`bool`, *optional*, defaults to `False`):
            Activate channel attention block in the Transformer to allow channels to attend each other.
        ffn_dim (`int`, *optional*, defaults to 512):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        norm_type (`str` , *optional*, defaults to `"batchnorm"`):
            Normalization at each Transformer layer. Can be `"batchnorm"` or `"layernorm"`.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            A value added to the denominator for numerical stability of normalization.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention probabilities.
        positional_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability in the positional embedding layer.
        path_dropout (`float`, *optional*, defaults to 0.0):
            The dropout path in the residual block.
        ff_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability used between the two layers of the feed-forward networks.
        bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias in the feed-forward networks.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (string) in the Transformer.`"gelu"` and `"relu"` are supported.
        pre_norm (`bool`, *optional*, defaults to `True`):
            Normalization is applied before self-attention if pre_norm is set to `True`. Otherwise, normalization is
            applied after residual block.
        positional_encoding_type (`str`, *optional*, defaults to `"sincos"`):
            Positional encodings. Options `"random"` and `"sincos"` are supported.
        use_cls_token (`bool`, *optional*, defaults to `False`):
            Whether cls token is used.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        share_projection (`bool`, *optional*, defaults to `True`):
            Sharing the projection layer across different channels in the forecast head.
        scaling (`Union`, *optional*, defaults to `"std"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean".
        do_mask_input (`bool`, *optional*):
            Apply masking during the pretraining.
        mask_type (`str`, *optional*, defaults to `"random"`):
            Masking type. Only `"random"` and `"forecast"` are currently supported.
        random_mask_ratio (`float`, *optional*, defaults to 0.5):
            Masking ratio applied to mask the input data during random pretraining.
        num_forecast_mask_patches (`int` or `list`, *optional*, defaults to `[2]`):
            Number of patches to be masked at the end of each batch sample. If it is an integer,
            all the samples in the batch will have the same number of masked patches. If it is a list,
            samples in the batch will be randomly masked by numbers defined in the list. This argument is only used
            for forecast pretraining.
        channel_consistent_masking (`bool`, *optional*, defaults to `False`):
            If channel consistent masking is True, all the channels will have the same masking pattern.
        unmasked_channel_indices (`list`, *optional*):
            Indices of channels that are not masked during pretraining. Values in the list are number between 1 and
            `num_input_channels`
        mask_value (`int`, *optional*, defaults to 0):
            Values in the masked patches will be filled by `mask_value`.
        pooling_type (`str`, *optional*, defaults to `"mean"`):
            Pooling of the embedding. `"mean"`, `"max"` and `None` are supported.
        head_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for head.
        prediction_length (`int`, *optional*, defaults to 24):
            The prediction horizon that the model will output.
        num_targets (`int`, *optional*, defaults to 1):
            Number of targets for regression and classification tasks. For classification, it is the number of
            classes.
        output_range (`list`, *optional*):
            Output range for regression task. The range of output values can be set to enforce the model to produce
            values within a range.
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples is generated in parallel for probabilistic prediction.


    ```python
    >>> from transformers import PatchTSTConfig, PatchTSTModel

    >>> # Initializing an PatchTST configuration with 12 time steps for prediction
    >>> configuration = PatchTSTConfig(prediction_length=12)

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = PatchTSTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## PatchTSTModel

The bare PatchTST Model outputting raw hidden-states without any specific head.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PatchTSTForPrediction

The PatchTST for prediction model.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PatchTSTForClassification

The PatchTST for classification model.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PatchTSTForPretraining

The PatchTST for pretrain model.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PatchTSTForRegression

The PatchTST for regression model.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
