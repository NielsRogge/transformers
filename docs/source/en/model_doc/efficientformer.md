<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# EfficientFormer

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The EfficientFormer model was proposed in [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)
by Yanyu Li, Geng Yuan, Yang Wen, Eric Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren.  EfficientFormer proposes a
dimension-consistent pure transformer that can be run on mobile devices for dense prediction tasks like image classification, object
detection and semantic segmentation.

The abstract from the paper is the following:

*Vision Transformers (ViT) have shown rapid progress in computer vision tasks, achieving promising results on various benchmarks.
However, due to the massive number of parameters and model design, e.g., attention mechanism, ViT-based models are generally
times slower than lightweight convolutional networks. Therefore, the deployment of ViT for real-time applications is particularly
challenging, especially on resource-constrained hardware such as mobile devices. Recent efforts try to reduce the computation
complexity of ViT through network architecture search or hybrid design with MobileNet block, yet the inference speed is still
unsatisfactory. This leads to an important question: can transformers run as fast as MobileNet while obtaining high performance?
To answer this, we first revisit the network architecture and operators used in ViT-based models and identify inefficient designs.
Then we introduce a dimension-consistent pure transformer (without MobileNet blocks) as a design paradigm.
Finally, we perform latency-driven slimming to get a series of final models dubbed EfficientFormer.
Extensive experiments show the superiority of EfficientFormer in performance and speed on mobile devices.
Our fastest model, EfficientFormer-L1, achieves 79.2% top-1 accuracy on ImageNet-1K with only 1.6 ms inference latency on
iPhone 12 (compiled with CoreML), which { runs as fast as MobileNetV2×1.4 (1.6 ms, 74.7% top-1),} and our largest model,
EfficientFormer-L7, obtains 83.3% accuracy with only 7.0 ms latency. Our work proves that properly designed transformers can
reach extremely low latency on mobile devices while maintaining high performance.*

This model was contributed by [novice03](https://huggingface.co/novice03) and [Bearnardd](https://huggingface.co/Bearnardd).
The original code can be found [here](https://github.com/snap-research/EfficientFormer). The TensorFlow version of this model was added by [D-Roberts](https://huggingface.co/D-Roberts).

## Documentation resources

- [Image classification task guide](../tasks/image_classification)

## EfficientFormerConfig


    This is the configuration class to store the configuration of an [`EfficientFormerModel`]. It is used to
    instantiate an EfficientFormer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the EfficientFormer
    [snap-research/efficientformer-l1](https://huggingface.co/snap-research/efficientformer-l1) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        depths (`List(int)`, *optional*, defaults to `[3, 2, 6, 4]`)
            Depth of each stage.
        hidden_sizes (`List(int)`, *optional*, defaults to `[48, 96, 224, 448]`)
            Dimensionality of each stage.
        downsamples (`List(bool)`, *optional*, defaults to `[True, True, True, True]`)
            Whether or not to downsample inputs between two stages.
        dim (`int`, *optional*, defaults to 448):
            Number of channels in Meta3D layers
        key_dim (`int`, *optional*, defaults to 32):
            The size of the key in meta3D block.
        attention_ratio (`int`, *optional*, defaults to 4):
            Ratio of the dimension of the query and value to the dimension of the key in MSHA block
        resolution (`int`, *optional*, defaults to 7)
            Size of each patch
        num_hidden_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the 3D MetaBlock.
        mlp_expansion_ratio (`int`, *optional*, defaults to 4):
            Ratio of size of the hidden dimensionality of an MLP to the dimensionality of its input.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        pool_size (`int`, *optional*, defaults to 3):
            Kernel size of pooling layers.
        downsample_patch_size (`int`, *optional*, defaults to 3):
            The size of patches in downsampling layers.
        downsample_stride (`int`, *optional*, defaults to 2):
            The stride of convolution kernels in downsampling layers.
        downsample_pad (`int`, *optional*, defaults to 1):
            Padding in downsampling layers.
        drop_path_rate (`int`, *optional*, defaults to 0):
            Rate at which to increase dropout probability in DropPath.
        num_meta3d_blocks (`int`, *optional*, defaults to 1):
            The number of 3D MetaBlocks in the last stage.
        distillation (`bool`, *optional*, defaults to `True`):
            Whether to add a distillation head.
        use_layer_scale (`bool`, *optional*, defaults to `True`):
            Whether to scale outputs from token mixers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-5):
            Factor by which outputs from token mixers are scaled.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to `224`):
            The size (resolution) of each image.

    Example:

    ```python
    >>> from transformers import EfficientFormerConfig, EfficientFormerModel

    >>> # Initializing a EfficientFormer efficientformer-l1 style configuration
    >>> configuration = EfficientFormerConfig()

    >>> # Initializing a EfficientFormerModel (with random weights) from the efficientformer-l3 style configuration
    >>> model = EfficientFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## EfficientFormerImageProcessor


    Constructs a EfficientFormer image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize:
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    

Methods: preprocess

<frameworkcontent>
<pt>

## EfficientFormerModel

The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`EfficientFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## EfficientFormerForImageClassification


    EfficientFormer Model transformer with an image classification head on top (a linear layer on top of the final
    hidden state of the [CLS] token) e.g. for ImageNet.
    
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`EfficientFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## EfficientFormerForImageClassificationWithTeacher


    EfficientFormer Model transformer with image classification heads on top (a linear layer on top of the final hidden
    state of the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for
    ImageNet.

    <Tip warning={true}>

           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.

    </Tip>
    
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`EfficientFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

</pt>
<tf>

## TFEfficientFormerModel

No docstring available for TFEfficientFormerModel

Methods: call

## TFEfficientFormerForImageClassification

No docstring available for TFEfficientFormerForImageClassification

Methods: call

## TFEfficientFormerForImageClassificationWithTeacher

No docstring available for TFEfficientFormerForImageClassificationWithTeacher

Methods: call

</tf>
</frameworkcontent>
