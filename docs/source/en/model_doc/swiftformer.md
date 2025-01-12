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

# SwiftFormer

## Overview

The SwiftFormer model was proposed in [SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications](https://arxiv.org/abs/2303.15446) by Abdelrahman Shaker, Muhammad Maaz, Hanoona Rasheed, Salman Khan, Ming-Hsuan Yang, Fahad Shahbaz Khan.

The SwiftFormer paper introduces a novel efficient additive attention mechanism that effectively replaces the quadratic matrix multiplication operations in the self-attention computation with linear element-wise multiplications. A series of models called 'SwiftFormer' is built based on this, which achieves state-of-the-art performance in terms of both accuracy and mobile inference speed. Even their small variant achieves 78.5% top-1 ImageNet1K accuracy with only 0.8 ms latency on iPhone 14, which is more accurate and 2× faster compared to MobileViT-v2.

The abstract from the paper is the following:

*Self-attention has become a defacto choice for capturing global context in various vision applications. However, its quadratic computational complexity with respect to image resolution limits its use in real-time applications, especially for deployment on resource-constrained mobile devices. Although hybrid approaches have been proposed to combine the advantages of convolutions and self-attention for a better speed-accuracy trade-off, the expensive matrix multiplication operations in self-attention remain a bottleneck. In this work, we introduce a novel efficient additive attention mechanism that effectively replaces the quadratic matrix multiplication operations with linear element-wise multiplications. Our design shows that the key-value interaction can be replaced with a linear layer without sacrificing any accuracy. Unlike previous state-of-the-art methods, our efficient formulation of self-attention enables its usage at all stages of the network. Using our proposed efficient additive attention, we build a series of models called "SwiftFormer" which achieves state-of-the-art performance in terms of both accuracy and mobile inference speed. Our small variant achieves 78.5% top-1 ImageNet-1K accuracy with only 0.8 ms latency on iPhone 14, which is more accurate and 2x faster compared to MobileViT-v2.*

This model was contributed by [shehan97](https://huggingface.co/shehan97). The TensorFlow version was contributed by [joaocmd](https://huggingface.co/joaocmd).
The original code can be found [here](https://github.com/Amshaker/SwiftFormer).

## SwiftFormerConfig


    This is the configuration class to store the configuration of a [`SwiftFormerModel`]. It is used to instantiate an
    SwiftFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SwiftFormer
    [MBZUAI/swiftformer-xs](https://huggingface.co/MBZUAI/swiftformer-xs) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels
        depths (`List[int]`, *optional*, defaults to `[3, 3, 6, 4]`):
            Depth of each stage
        embed_dims (`List[int]`, *optional*, defaults to `[48, 56, 112, 220]`):
            The embedding dimension at each stage
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of size of the hidden dimensionality of an MLP to the dimensionality of its input.
        downsamples (`List[bool]`, *optional*, defaults to `[True, True, True, True]`):
            Whether or not to downsample inputs between two stages.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (string). `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        down_patch_size (`int`, *optional*, defaults to 3):
            The size of patches in downsampling layers.
        down_stride (`int`, *optional*, defaults to 2):
            The stride of convolution kernels in downsampling layers.
        down_pad (`int`, *optional*, defaults to 1):
            Padding in downsampling layers.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Rate at which to increase dropout probability in DropPath.
        drop_mlp_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for the MLP component of SwiftFormer.
        drop_conv_encoder_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for the ConvEncoder component of SwiftFormer.
        use_layer_scale (`bool`, *optional*, defaults to `True`):
            Whether to scale outputs from token mixers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-05):
            Factor by which outputs from token mixers are scaled.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.


    Example:

    ```python
    >>> from transformers import SwiftFormerConfig, SwiftFormerModel

    >>> # Initializing a SwiftFormer swiftformer-base-patch16-224 style configuration
    >>> configuration = SwiftFormerConfig()

    >>> # Initializing a model (with random weights) from the swiftformer-base-patch16-224 style configuration
    >>> model = SwiftFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## SwiftFormerModel

The bare SwiftFormer Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SwiftFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## SwiftFormerForImageClassification


    SwiftFormer Model transformer with an image classification head on top (e.g. for ImageNet).
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SwiftFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## TFSwiftFormerModel

No docstring available for TFSwiftFormerModel

Methods: call

## TFSwiftFormerForImageClassification

No docstring available for TFSwiftFormerForImageClassification

Methods: call
