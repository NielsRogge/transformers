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

# Convolutional Vision Transformer (CvT)

## Overview

The CvT model was proposed in [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808) by Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan and Lei Zhang. The Convolutional vision Transformer (CvT) improves the [Vision Transformer (ViT)](vit) in performance and efficiency by introducing convolutions into ViT to yield the best of both designs.

The abstract from the paper is the following:

*We present in this paper a new architecture, named Convolutional vision Transformer (CvT), that improves Vision Transformer (ViT) 
in performance and efficiency by introducing convolutions into ViT to yield the best of both designs. This is accomplished through 
two primary modifications: a hierarchy of Transformers containing a new convolutional token embedding, and a convolutional Transformer 
block leveraging a convolutional projection. These changes introduce desirable properties of convolutional neural networks (CNNs) 
to the ViT architecture (\ie shift, scale, and distortion invariance) while maintaining the merits of Transformers (\ie dynamic attention, 
global context, and better generalization). We validate CvT by conducting extensive experiments, showing that this approach achieves 
state-of-the-art performance over other Vision Transformers and ResNets on ImageNet-1k, with fewer parameters and lower FLOPs. In addition, 
performance gains are maintained when pretrained on larger datasets (\eg ImageNet-22k) and fine-tuned to downstream tasks. Pre-trained on 
ImageNet-22k, our CvT-W24 obtains a top-1 accuracy of 87.7\% on the ImageNet-1k val set. Finally, our results show that the positional encoding, 
a crucial component in existing Vision Transformers, can be safely removed in our model, simplifying the design for higher resolution vision tasks.*

This model was contributed by [anugunj](https://huggingface.co/anugunj). The original code can be found [here](https://github.com/microsoft/CvT).

## Usage tips

- CvT models are regular Vision Transformers, but trained with convolutions. They outperform the [original model (ViT)](vit) when fine-tuned on ImageNet-1K and CIFAR-100.
- You can check out demo notebooks regarding inference as well as fine-tuning on custom data [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) (you can just replace [`ViTFeatureExtractor`] by [`AutoImageProcessor`] and [`ViTForImageClassification`] by [`CvtForImageClassification`]).
- The available checkpoints are either (1) pre-trained on [ImageNet-22k](http://www.image-net.org/) (a collection of 14 million images and 22k classes) only, (2) also fine-tuned on ImageNet-22k or (3) also fine-tuned on [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/) (also referred to as ILSVRC 2012, a collection of 1.3 million
  images and 1,000 classes).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with CvT.

<PipelineTag pipeline="image-classification"/>

- [`CvtForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## CvtConfig


    This is the configuration class to store the configuration of a [`CvtModel`]. It is used to instantiate a CvT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CvT
    [microsoft/cvt-13](https://huggingface.co/microsoft/cvt-13) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3]`):
            The kernel size of each encoder's patch embedding.
        patch_stride (`List[int]`, *optional*, defaults to `[4, 2, 2]`):
            The stride size of each encoder's patch embedding.
        patch_padding (`List[int]`, *optional*, defaults to `[2, 1, 1]`):
            The padding size of each encoder's patch embedding.
        embed_dim (`List[int]`, *optional*, defaults to `[64, 192, 384]`):
            Dimension of each of the encoder blocks.
        num_heads (`List[int]`, *optional*, defaults to `[1, 3, 6]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        depth (`List[int]`, *optional*, defaults to `[1, 2, 10]`):
            The number of layers in each encoder block.
        mlp_ratios (`List[float]`, *optional*, defaults to `[4.0, 4.0, 4.0, 4.0]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        attention_drop_rate (`List[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`):
            The dropout ratio for the attention probabilities.
        drop_rate (`List[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`):
            The dropout ratio for the patch embeddings probabilities.
        drop_path_rate (`List[float]`, *optional*, defaults to `[0.0, 0.0, 0.1]`):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        qkv_bias (`List[bool]`, *optional*, defaults to `[True, True, True]`):
            The bias bool for query, key and value in attentions
        cls_token (`List[bool]`, *optional*, defaults to `[False, False, True]`):
            Whether or not to add a classification token to the output of each of the last 3 stages.
        qkv_projection_method (`List[string]`, *optional*, defaults to ["dw_bn", "dw_bn", "dw_bn"]`):
            The projection method for query, key and value Default is depth-wise convolutions with batch norm. For
            Linear projection use "avg".
        kernel_qkv (`List[int]`, *optional*, defaults to `[3, 3, 3]`):
            The kernel size for query, key and value in attention layer
        padding_kv (`List[int]`, *optional*, defaults to `[1, 1, 1]`):
            The padding size for key and value in attention layer
        stride_kv (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            The stride size for key and value in attention layer
        padding_q (`List[int]`, *optional*, defaults to `[1, 1, 1]`):
            The padding size for query in attention layer
        stride_q (`List[int]`, *optional*, defaults to `[1, 1, 1]`):
            The stride size for query in attention layer
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import CvtConfig, CvtModel

    >>> # Initializing a Cvt msft/cvt style configuration
    >>> configuration = CvtConfig()

    >>> # Initializing a model (with random weights) from the msft/cvt style configuration
    >>> model = CvtModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

<frameworkcontent>
<pt>

## CvtModel

The bare Cvt Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## CvtForImageClassification


    Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

</pt>
<tf>

## TFCvtModel

No docstring available for TFCvtModel

Methods: call

## TFCvtForImageClassification

No docstring available for TFCvtForImageClassification

Methods: call

</tf>
</frameworkcontent>
