<!--Copyright 2023 IBM and HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# PatchTSMixer

## Overview

The PatchTSMixer model was proposed in [TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting](https://arxiv.org/pdf/2306.09364.pdf) by Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong and Jayant Kalagnanam.


PatchTSMixer is a lightweight time-series modeling approach based on the MLP-Mixer architecture. In this HuggingFace implementation, we provide PatchTSMixer's capabilities to effortlessly facilitate lightweight mixing across patches, channels, and hidden features for effective multivariate time-series modeling. It also supports various attention mechanisms starting from simple gated attention to more complex self-attention blocks that can be customized accordingly. The model can be pretrained and subsequently used for various downstream tasks such as forecasting, classification and regression.


The abstract from the paper is the following:

*TSMixer is a lightweight neural architecture exclusively composed of multi-layer perceptron (MLP) modules designed for multivariate forecasting and representation learning on patched time series. Our model draws inspiration from the success of MLP-Mixer models in computer vision. We demonstrate the challenges involved in adapting Vision MLP-Mixer for time series and introduce empirically validated components to enhance accuracy. This includes a novel design paradigm of attaching online reconciliation heads to the MLP-Mixer backbone, for explicitly modeling the time-series properties such as hierarchy and channel-correlations. We also propose a Hybrid channel modeling approach to effectively handle noisy channel interactions and generalization across diverse datasets, a common challenge in existing patch channel-mixing methods. Additionally, a simple gated attention mechanism is introduced in the backbone to prioritize important features. By incorporating these lightweight components, we significantly enhance the learning capability of simple MLP structures, outperforming complex Transformer models with minimal computing usage. Moreover, TSMixer's modular design enables compatibility with both supervised and masked self-supervised learning methods, making it a promising building block for time-series Foundation Models. TSMixer outperforms state-of-the-art MLP and Transformer models in forecasting by a considerable margin of 8-60%. It also outperforms the latest strong benchmarks of Patch-Transformer models (by 1-2%) with a significant reduction in memory and runtime (2-3X).*

This model was contributed by [ajati](https://huggingface.co/ajati), [vijaye12](https://huggingface.co/vijaye12), 
[gsinthong](https://huggingface.co/gsinthong), [namctin](https://huggingface.co/namctin),
[wmgifford](https://huggingface.co/wmgifford), [kashif](https://huggingface.co/kashif).

## Usage example

The code snippet below shows how to randomly initialize a PatchTSMixer model. The model is compatible with the [Trainer API](../trainer.md).

```python

from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
from transformers import Trainer, TrainingArguments,


config = PatchTSMixerConfig(context_length = 512, prediction_length = 96)
model = PatchTSMixerForPrediction(config)
trainer = Trainer(model=model, args=training_args, 
            train_dataset=train_dataset,
            eval_dataset=valid_dataset)
trainer.train()
results = trainer.evaluate(test_dataset)
```

## Usage tips

The model can also be used for time series classification and time series regression. See the respective [`PatchTSMixerForTimeSeriesClassification`] and [`PatchTSMixerForRegression`] classes.

## Resources

- A blog post explaining PatchTSMixer in depth can be found [here](https://huggingface.co/blog/patchtsmixer). The blog can also be opened in Google Colab.

## PatchTSMixerConfig


    This is the configuration class to store the configuration of a [`PatchTSMixerModel`]. It is used to instantiate a
    PatchTSMixer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PatchTSMixer
    [ibm/patchtsmixer-etth1-pretrain](https://huggingface.co/ibm/patchtsmixer-etth1-pretrain) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        context_length (`int`, *optional*, defaults to 32):
            The context/history length for the input sequence.
        patch_length (`int`, *optional*, defaults to 8):
            The patch length for the input sequence.
        num_input_channels (`int`, *optional*, defaults to 1):
            Number of input variates. For Univariate, set it to 1.
        patch_stride (`int`, *optional*, defaults to 8):
            Determines the overlap between two consecutive patches. Set it to patch_length (or greater), if we want
            non-overlapping patches.
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples to generate in parallel for probabilistic forecast.
        d_model (`int`, *optional*, defaults to 8):
            Hidden dimension of the model. Recommended to set it as a multiple of patch_length (i.e. 2-5X of
            patch_length). Larger value indicates more complex model.
        expansion_factor (`int`, *optional*, defaults to 2):
            Expansion factor to use inside MLP. Recommended range is 2-5. Larger value indicates more complex model.
        num_layers (`int`, *optional*, defaults to 3):
            Number of layers to use. Recommended range is 3-15. Larger value indicates more complex model.
        dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `PatchTSMixer` backbone. Recommended range is 0.2-0.7
        mode (`str`, *optional*, defaults to `"common_channel"`):
            Mixer Mode. Determines how to process the channels. Allowed values: "common_channel", "mix_channel". In
            "common_channel" mode, we follow Channel-independent modelling with no explicit channel-mixing. Channel
            mixing happens in an implicit manner via shared weights across channels. (preferred first approach) In
            "mix_channel" mode, we follow explicit channel-mixing in addition to patch and feature mixer. (preferred
            approach when channel correlations are very important to model)
        gated_attn (`bool`, *optional*, defaults to `True`):
            Enable Gated Attention.
        norm_mlp (`str`, *optional*, defaults to `"LayerNorm"`):
            Normalization layer (BatchNorm or LayerNorm).
        self_attn (`bool`, *optional*, defaults to `False`):
            Enable Tiny self attention across patches. This can be enabled when the output of Vanilla PatchTSMixer with
            gated attention is not satisfactory. Enabling this leads to explicit pair-wise attention and modelling
            across patches.
        self_attn_heads (`int`, *optional*, defaults to 1):
            Number of self-attention heads. Works only when `self_attn` is set to `True`.
        use_positional_encoding (`bool`, *optional*, defaults to `False`):
            Enable the use of positional embedding for the tiny self-attention layers. Works only when `self_attn` is
            set to `True`.
        positional_encoding_type (`str`, *optional*, defaults to `"sincos"`):
            Positional encodings. Options `"random"` and `"sincos"` are supported. Works only when
            `use_positional_encoding` is set to `True`
        scaling (`string` or `bool`, *optional*, defaults to `"std"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean".
        loss (`string`, *optional*, defaults to `"mse"`):
            The loss function for the model corresponding to the `distribution_output` head. For parametric
            distributions it is the negative log likelihood ("nll") and for point estimates it is the mean squared
            error "mse".
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        post_init (`bool`, *optional*, defaults to `False`):
            Whether to use custom weight initialization from `transformers` library, or the default initialization in
            `PyTorch`. Setting it to `False` performs `PyTorch` weight initialization.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            A value added to the denominator for numerical stability of normalization.
        mask_type (`str`, *optional*, defaults to `"random"`):
            Type of masking to use for Masked Pretraining mode. Allowed values are "random", "forecast". In Random
            masking, points are masked randomly. In Forecast masking, points are masked towards the end.
        random_mask_ratio (`float`, *optional*, defaults to 0.5):
            Masking ratio to use when `mask_type` is `random`. Higher value indicates more masking.
        num_forecast_mask_patches (`int` or `list`, *optional*, defaults to `[2]`):
            Number of patches to be masked at the end of each batch sample. If it is an integer, all the samples in the
            batch will have the same number of masked patches. If it is a list, samples in the batch will be randomly
            masked by numbers defined in the list. This argument is only used for forecast pretraining.
        mask_value (`float`, *optional*, defaults to `0.0`):
            Mask value to use.
        masked_loss (`bool`, *optional*, defaults to `True`):
            Whether to compute pretraining loss only at the masked portions, or on the entire output.
        channel_consistent_masking (`bool`, *optional*, defaults to `True`):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels.
        unmasked_channel_indices (`list`, *optional*):
            Channels that are not masked during pretraining.
        head_dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `PatchTSMixer` head.
        distribution_output (`string`, *optional*, defaults to `"student_t"`):
            The distribution emission head for the model when loss is "nll". Could be either "student_t", "normal" or
            "negative_binomial".
        prediction_length (`int`, *optional*, defaults to 16):
            Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
        prediction_channel_indices (`list`, *optional*):
            List of channel indices to forecast. If None, forecast all channels. Target data is expected to have all
            channels and we explicitly filter the channels in prediction and target before loss computation.
        num_targets (`int`, *optional*, defaults to 3):
            Number of targets (dimensionality of the regressed variable) for a regression task.
        output_range (`list`, *optional*):
            Output range to restrict for the regression task. Defaults to None.
        head_aggregation (`str`, *optional*, defaults to `"max_pool"`):
            Aggregation mode to enable for classification or regression task. Allowed values are `None`, "use_last",
            "max_pool", "avg_pool".

    Example:

    ```python
    >>> from transformers import PatchTSMixerConfig, PatchTSMixerModel

    >>> # Initializing a default PatchTSMixer configuration
    >>> configuration = PatchTSMixerConfig()

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = PatchTSMixerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```


## PatchTSMixerModel

The PatchTSMixer Model for time-series forecasting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSMixerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        mask_input (`bool`, *optional*, defaults to `False`):
            If True, Masking will be enabled. False otherwise.


Methods: forward


## PatchTSMixerForPrediction


    `PatchTSMixer` for forecasting application.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.

    Returns:
        `None`.
    

Methods: forward


## PatchTSMixerForTimeSeriesClassification


    `PatchTSMixer` for classification application.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.

    Returns:
        `None`.
    

Methods: forward


## PatchTSMixerForPretraining


    `PatchTSMixer` for mask pretraining.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.

    Returns:
        `None`.
    

Methods: forward


## PatchTSMixerForRegression


    `PatchTSMixer` for regression application.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.

    Returns:
        `None`.
    

Methods: forward