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

# LeViT

## Overview

The LeViT model was proposed in [LeViT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2104.01136) by Ben Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, Matthijs Douze. LeViT improves the [Vision Transformer (ViT)](vit) in performance and efficiency by a few architectural differences such as activation maps with decreasing resolutions in Transformers and the introduction of an attention bias to integrate positional information.

The abstract from the paper is the following:

*We design a family of image classification architectures that optimize the trade-off between accuracy
and efficiency in a high-speed regime. Our work exploits recent findings in attention-based architectures,
which are competitive on highly parallel processing hardware. We revisit principles from the extensive
literature on convolutional neural networks to apply them to transformers, in particular activation maps
with decreasing resolutions. We also introduce the attention bias, a new way to integrate positional information
in vision transformers. As a result, we propose LeVIT: a hybrid neural network for fast inference image classification.
We consider different measures of efficiency on different hardware platforms, so as to best reflect a wide range of
application scenarios. Our extensive experiments empirically validate our technical choices and show they are suitable
to most architectures. Overall, LeViT significantly outperforms existing convnets and vision transformers with respect
to the speed/accuracy tradeoff. For example, at 80% ImageNet top-1 accuracy, LeViT is 5 times faster than EfficientNet on CPU. *

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/levit_architecture.png"
alt="drawing" width="600"/>

<small> LeViT Architecture. Taken from the <a href="https://arxiv.org/abs/2104.01136">original paper</a>.</small>

This model was contributed by [anugunj](https://huggingface.co/anugunj). The original code can be found [here](https://github.com/facebookresearch/LeViT).

## Usage tips

- Compared to ViT, LeViT models use an additional distillation head to effectively learn from a teacher (which, in the LeViT paper, is a ResNet like-model). The distillation head is learned through backpropagation under supervision of a ResNet like-model. They also draw inspiration from convolution neural networks to use activation maps with decreasing resolutions to increase the efficiency.
- There are 2 ways to fine-tune distilled models, either (1) in a classic way, by only placing a prediction head on top
  of the final hidden state and not using the distillation head, or (2) by placing both a prediction head and distillation
  head on top of the final hidden state. In that case, the prediction head is trained using regular cross-entropy between
  the prediction of the head and the ground-truth label, while the distillation prediction head is trained using hard distillation
  (cross-entropy between the prediction of the distillation head and the label predicted by the teacher). At inference time,
  one takes the average prediction between both heads as final prediction. (2) is also called "fine-tuning with distillation",
  because one relies on a teacher that has already been fine-tuned on the downstream dataset. In terms of models, (1) corresponds
  to [`LevitForImageClassification`] and (2) corresponds to [`LevitForImageClassificationWithTeacher`].
- All released checkpoints were pre-trained and fine-tuned on  [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)
  (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes). only. No external data was used. This is in
  contrast with the original ViT model, which used external data like the JFT-300M dataset/Imagenet-21k for
  pre-training.
- The authors of LeViT released 5 trained LeViT models, which you can directly plug into [`LevitModel`] or [`LevitForImageClassification`].
  Techniques like data augmentation, optimization, and regularization were used in order to simulate training on a much larger dataset
  (while only using ImageNet-1k for pre-training). The 5 variants available are (all trained on images of size 224x224):
  *facebook/levit-128S*, *facebook/levit-128*, *facebook/levit-192*, *facebook/levit-256* and
  *facebook/levit-384*. Note that one should use [`LevitImageProcessor`] in order to
  prepare images for the model.
- [`LevitForImageClassificationWithTeacher`] currently supports only inference and not training or fine-tuning.
- You can check out demo notebooks regarding inference as well as fine-tuning on custom data [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer)
  (you can just replace [`ViTFeatureExtractor`] by [`LevitImageProcessor`] and [`ViTForImageClassification`] by [`LevitForImageClassification`] or [`LevitForImageClassificationWithTeacher`]).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LeViT.

<PipelineTag pipeline="image-classification"/>

- [`LevitForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## LevitConfig


    This is the configuration class to store the configuration of a [`LevitModel`]. It is used to instantiate a LeViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LeViT
    [facebook/levit-128S](https://huggingface.co/facebook/levit-128S) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size of the input image.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the initial convolution layers of patch embedding.
        stride (`int`, *optional*, defaults to 2):
            The stride size for the initial convolution layers of patch embedding.
        padding (`int`, *optional*, defaults to 1):
            The padding size for the initial convolution layers of patch embedding.
        patch_size (`int`, *optional*, defaults to 16):
            The patch size for embeddings.
        hidden_sizes (`List[int]`, *optional*, defaults to `[128, 256, 384]`):
            Dimension of each of the encoder blocks.
        num_attention_heads (`List[int]`, *optional*, defaults to `[4, 8, 12]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        depths (`List[int]`, *optional*, defaults to `[4, 4, 4]`):
            The number of layers in each encoder block.
        key_dim (`List[int]`, *optional*, defaults to `[16, 16, 16]`):
            The size of key in each of the encoder blocks.
        drop_path_rate (`int`, *optional*, defaults to 0):
            The dropout probability for stochastic depths, used in the blocks of the Transformer encoder.
        mlp_ratios (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        attention_ratios (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Ratio of the size of the output dimension compared to input dimension of attention layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import LevitConfig, LevitModel

    >>> # Initializing a LeViT levit-128S style configuration
    >>> configuration = LevitConfig()

    >>> # Initializing a model (with random weights) from the levit-128S style configuration
    >>> model = LevitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## LevitFeatureExtractor

No docstring available for LevitFeatureExtractor

Methods: __call__

## LevitImageProcessor

  
    Constructs a LeViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Wwhether to resize the shortest edge of the input to int(256/224 *`size`). Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]`, *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the output image after resizing. If size is a dict with keys "width" and "height", the image will
            be resized to `(size["height"], size["width"])`. If size is a dict with key "shortest_edge", the shortest
            edge value `c` is rescaled to `int(c * (256/224))`. The smaller edge of the image will be matched to this
            value i.e, if height > width, then image will be rescaled to `(size["shortest_egde"] * height / width,
            size["shortest_egde"])`. Can be overridden by the `size` parameter in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether or not to center crop the input to `(crop_size["height"], crop_size["width"])`. Can be overridden
            by the `do_center_crop` parameter in the `preprocess` method.
        crop_size (`Dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Desired image size after `center_crop`. Can be overridden by the `crop_size` parameter in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`List[int]`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`List[int]`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    

Methods: preprocess

## LevitModel

The bare Levit model outputting raw features without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LevitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## LevitForImageClassification


    Levit Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LevitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## LevitForImageClassificationWithTeacher


    LeViT Model transformer with image classification heads on top (a linear layer on top of the final hidden state and
    a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet. .. warning::
           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LevitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
