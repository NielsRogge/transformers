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

# PoolFormer

## Overview

The PoolFormer model was proposed in [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)  by Sea AI Labs. Instead of designing complicated token mixer to achieve SOTA performance, the target of this work is to demonstrate the competence of transformer models largely stem from the general architecture MetaFormer.

The abstract from the paper is the following:

*Transformers have shown great potential in computer vision tasks. A common belief is their attention-based token mixer module contributes most to their competence. However, recent works show the attention-based module in transformers can be replaced by spatial MLPs and the resulted models still perform quite well. Based on this observation, we hypothesize that the general architecture of the transformers, instead of the specific token mixer module, is more essential to the model's performance. To verify this, we deliberately replace the attention module in transformers with an embarrassingly simple spatial pooling operator to conduct only the most basic token mixing. Surprisingly, we observe that the derived model, termed as PoolFormer, achieves competitive performance on multiple computer vision tasks. For example, on ImageNet-1K, PoolFormer achieves 82.1% top-1 accuracy, surpassing well-tuned vision transformer/MLP-like baselines DeiT-B/ResMLP-B24 by 0.3%/1.1% accuracy with 35%/52% fewer parameters and 48%/60% fewer MACs. The effectiveness of PoolFormer verifies our hypothesis and urges us to initiate the concept of "MetaFormer", a general architecture abstracted from transformers without specifying the token mixer. Based on the extensive experiments, we argue that MetaFormer is the key player in achieving superior results for recent transformer and MLP-like models on vision tasks. This work calls for more future research dedicated to improving MetaFormer instead of focusing on the token mixer modules. Additionally, our proposed PoolFormer could serve as a starting baseline for future MetaFormer architecture design.*

The figure below illustrates the architecture of PoolFormer. Taken from the [original paper](https://arxiv.org/abs/2111.11418).

<img width="600" src="https://user-images.githubusercontent.com/15921929/142746124-1ab7635d-2536-4a0e-ad43-b4fe2c5a525d.png"/>

This model was contributed by [heytanay](https://huggingface.co/heytanay). The original code can be found [here](https://github.com/sail-sg/poolformer).

## Usage tips

- PoolFormer has a hierarchical architecture, where instead of Attention, a simple Average Pooling layer is present. All checkpoints of the model can be found on the [hub](https://huggingface.co/models?other=poolformer).
- One can use [`PoolFormerImageProcessor`] to prepare images for the model.
- As most models, PoolFormer comes in different sizes, the details of which can be found in the table below.

| **Model variant** | **Depths**    | **Hidden sizes**    | **Params (M)** | **ImageNet-1k Top 1** |
| :---------------: | ------------- | ------------------- | :------------: | :-------------------: |
| s12               | [2, 2, 6, 2]  | [64, 128, 320, 512] | 12             | 77.2                  |
| s24               | [4, 4, 12, 4] | [64, 128, 320, 512] | 21             | 80.3                  |
| s36               | [6, 6, 18, 6] | [64, 128, 320, 512] | 31             | 81.4                  |
| m36               | [6, 6, 18, 6] | [96, 192, 384, 768] | 56             | 82.1                  |
| m48               | [8, 8, 24, 8] | [96, 192, 384, 768] | 73             | 82.5                  |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with PoolFormer.

<PipelineTag pipeline="image-classification"/>

- [`PoolFormerForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## PoolFormerConfig


    This is the configuration class to store the configuration of [`PoolFormerModel`]. It is used to instantiate a
    PoolFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PoolFormer
    [sail/poolformer_s12](https://huggingface.co/sail/poolformer_s12) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels in the input image.
        patch_size (`int`, *optional*, defaults to 16):
            The size of the input patch.
        stride (`int`, *optional*, defaults to 16):
            The stride of the input patch.
        pool_size (`int`, *optional*, defaults to 3):
            The size of the pooling window.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the number of channels in the output of the MLP to the number of channels in the input.
        depths (`list`, *optional*, defaults to `[2, 2, 6, 2]`):
            The depth of each encoder block.
        hidden_sizes (`list`, *optional*, defaults to `[64, 128, 320, 512]`):
            The hidden sizes of each encoder block.
        patch_sizes (`list`, *optional*, defaults to `[7, 3, 3, 3]`):
            The size of the input patch for each encoder block.
        strides (`list`, *optional*, defaults to `[4, 2, 2, 2]`):
            The stride of the input patch for each encoder block.
        padding (`list`, *optional*, defaults to `[2, 1, 1, 1]`):
            The padding of the input patch for each encoder block.
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            The number of encoder blocks.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The dropout rate for the dropout layers.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function for the hidden layers.
        use_layer_scale (`bool`, *optional*, defaults to `True`):
            Whether to use layer scale.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-05):
            The initial value for the layer scale.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The initializer range for the weights.

    Example:

    ```python
    >>> from transformers import PoolFormerConfig, PoolFormerModel

    >>> # Initializing a PoolFormer sail/poolformer_s12 style configuration
    >>> configuration = PoolFormerConfig()

    >>> # Initializing a model (with random weights) from the sail/poolformer_s12 style configuration
    >>> model = PoolFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    

## PoolFormerFeatureExtractor

No docstring available for PoolFormerFeatureExtractor

Methods: __call__

## PoolFormerImageProcessor


    Constructs a PoolFormer image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method. If crop_pct is
            unset:
            - size is `{"height": h, "width": w}`: the image is resized to `(h, w)`.
            - size is `{"shortest_edge": s}`: the shortest edge of the image is resized to s whilst maintaining the
              aspect ratio.

            If crop_pct is set:
            - size is `{"height": h, "width": w}`: the image is resized to `(int(floor(h/crop_pct)),
              int(floor(w/crop_pct)))`
            - size is `{"height": c, "width": c}`: the shortest edge of the image is resized to `int(floor(c/crop_pct)`
              whilst maintaining the aspect ratio.
            - size is `{"shortest_edge": c}`: the shortest edge of the image is resized to `int(floor(c/crop_pct)`
              whilst maintaining the aspect ratio.
        crop_pct (`float`, *optional*, defaults to 0.9):
            Percentage of the image to crop from the center. Can be overridden by `crop_pct` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image
            is padded with 0's and then center cropped. Can be overridden by `do_center_crop` in the `preprocess`
            method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after applying center crop. Only has an effect if `do_center_crop` is set to `True`. Can
            be overridden by the `crop_size` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    

Methods: preprocess

## PoolFormerModel

The bare PoolFormer Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PoolFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PoolFormerForImageClassification


    PoolFormer Model transformer with an image classification head on top
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PoolFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
