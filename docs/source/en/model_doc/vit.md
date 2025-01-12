<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Vision Transformer (ViT)

## Overview

The Vision Transformer (ViT) model was proposed in [An Image is Worth 16x16 Words: Transformers for Image Recognition
at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk
Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob
Uszkoreit, Neil Houlsby. It's the first paper that successfully trains a Transformer encoder on ImageNet, attaining
very good results compared to familiar convolutional architectures.

The abstract from the paper is the following:

*While the Transformer architecture has become the de-facto standard for natural language processing tasks, its
applications to computer vision remain limited. In vision, attention is either applied in conjunction with
convolutional networks, or used to replace certain components of convolutional networks while keeping their overall
structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to
sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of
data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.),
Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring
substantially fewer computational resources to train.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg"
alt="drawing" width="600"/>

<small> ViT architecture. Taken from the <a href="https://arxiv.org/abs/2010.11929">original paper.</a> </small>

Following the original Vision Transformer, some follow-up works have been made:

- [DeiT](deit) (Data-efficient Image Transformers) by Facebook AI. DeiT models are distilled vision transformers.
  The authors of DeiT also released more efficiently trained ViT models, which you can directly plug into [`ViTModel`] or
  [`ViTForImageClassification`]. There are 4 variants available (in 3 different sizes): *facebook/deit-tiny-patch16-224*,
  *facebook/deit-small-patch16-224*, *facebook/deit-base-patch16-224* and *facebook/deit-base-patch16-384*. Note that one should
  use [`DeiTImageProcessor`] in order to prepare images for the model.

- [BEiT](beit) (BERT pre-training of Image Transformers) by Microsoft Research. BEiT models outperform supervised pre-trained
  vision transformers using a self-supervised method inspired by BERT (masked image modeling) and based on a VQ-VAE.

- DINO (a method for self-supervised training of Vision Transformers) by Facebook AI. Vision Transformers trained using
  the DINO method show very interesting properties not seen with convolutional models. They are capable of segmenting
  objects, without having ever been trained to do so. DINO checkpoints can be found on the [hub](https://huggingface.co/models?other=dino).

- [MAE](vit_mae) (Masked Autoencoders) by Facebook AI. By pre-training Vision Transformers to reconstruct pixel values for a high portion
  (75%) of masked patches (using an asymmetric encoder-decoder architecture), the authors show that this simple method outperforms
  supervised pre-training after fine-tuning.

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code (written in JAX) can be
found [here](https://github.com/google-research/vision_transformer).

Note that we converted the weights from Ross Wightman's [timm library](https://github.com/rwightman/pytorch-image-models),
who already converted the weights from JAX to PyTorch. Credits go to him!

## Usage tips

- To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches,
  which are then linearly embedded. A [CLS] token is added to serve as representation of an entire image, which can be
  used for classification. The authors also add absolute position embeddings, and feed the resulting sequence of
  vectors to a standard Transformer encoder.
- As the Vision Transformer expects each image to be of the same size (resolution), one can use
  [`ViTImageProcessor`] to resize (or rescale) and normalize images for the model.
- Both the patch resolution and image resolution used during pre-training or fine-tuning are reflected in the name of
  each checkpoint. For example, `google/vit-base-patch16-224` refers to a base-sized architecture with patch
  resolution of 16x16 and fine-tuning resolution of 224x224. All checkpoints can be found on the [hub](https://huggingface.co/models?search=vit).
- The available checkpoints are either (1) pre-trained on [ImageNet-21k](http://www.image-net.org/) (a collection of
  14 million images and 21k classes) only, or (2) also fine-tuned on [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/) (also referred to as ILSVRC 2012, a collection of 1.3 million
  images and 1,000 classes).
- The Vision Transformer was pre-trained using a resolution of 224x224. During fine-tuning, it is often beneficial to
  use a higher resolution than pre-training [(Touvron et al., 2019)](https://arxiv.org/abs/1906.06423), [(Kolesnikov
  et al., 2020)](https://arxiv.org/abs/1912.11370). In order to fine-tune at higher resolution, the authors perform
  2D interpolation of the pre-trained position embeddings, according to their location in the original image.
- The best results are obtained with supervised pre-training, which is not the case in NLP. The authors also performed
  an experiment with a self-supervised pre-training objective, namely masked patched prediction (inspired by masked
  language modeling). With this approach, the smaller ViT-B/16 model achieves 79.9% accuracy on ImageNet, a significant
  improvement of 2% to training from scratch, but still 4% behind supervised pre-training.

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="sdpa", torch_dtype=torch.float16)
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `google/vit-base-patch16-224` model, we saw the following speedups during inference.

|   Batch size |   Average inference time (ms), eager mode |   Average inference time (ms), sdpa model |   Speed up, Sdpa / Eager (x) |
|--------------|-------------------------------------------|-------------------------------------------|------------------------------|
|            1 |                                         7 |                                         6 |                      1.17 |
|            2 |                                         8 |                                         6 |                      1.33 |
|            4 |                                         8 |                                         6 |                      1.33 |
|            8 |                                         8 |                                         6 |                      1.33 |

## Resources

Demo notebooks regarding inference as well as fine-tuning ViT on custom data can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer).
A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViT. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

`ViTForImageClassification` is supported by:
<PipelineTag pipeline="image-classification"/>

- A blog post on how to [Fine-Tune ViT for Image Classification with Hugging Face Transformers](https://huggingface.co/blog/fine-tune-vit)
- A blog post on [Image Classification with Hugging Face Transformers and `Keras`](https://www.philschmid.de/image-classification-huggingface-transformers-keras)
- A notebook on [Fine-tuning for Image Classification with Hugging Face Transformers](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb)
- A notebook on how to [Fine-tune the Vision Transformer on CIFAR-10 with the Hugging Face Trainer](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb)
- A notebook on how to [Fine-tune the Vision Transformer on CIFAR-10 with PyTorch Lightning](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_PyTorch_Lightning.ipynb)

⚗️ Optimization

- A blog post on how to [Accelerate Vision Transformer (ViT) with Quantization using Optimum](https://www.philschmid.de/optimizing-vision-transformer)

⚡️ Inference

- A notebook on [Quick demo: Vision Transformer (ViT) by Google Brain](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Quick_demo_of_HuggingFace_version_of_Vision_Transformer_inference.ipynb)

🚀 Deploy

- A blog post on [Deploying Tensorflow Vision Models in Hugging Face with TF Serving](https://huggingface.co/blog/tf-serving-vision)
- A blog post on [Deploying Hugging Face ViT on Vertex AI](https://huggingface.co/blog/deploy-vertex-ai)
- A blog post on [Deploying Hugging Face ViT on Kubernetes with TF Serving](https://huggingface.co/blog/deploy-tfserving-kubernetes)

## ViTConfig


    This is the configuration class to store the configuration of a [`ViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        encoder_stride (`int`, *optional*, defaults to 16):
           Factor to increase the spatial resolution by in the decoder head for masked image modeling.

    Example:

    ```python
    >>> from transformers import ViTConfig, ViTModel

    >>> # Initializing a ViT vit-base-patch16-224 style configuration
    >>> configuration = ViTConfig()

    >>> # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
    >>> model = ViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## ViTFeatureExtractor

No docstring available for ViTFeatureExtractor

Methods: __call__

## ViTImageProcessor


    Constructs a ViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*):
            Whether to convert the image to RGB.
    

Methods: preprocess

## ViTImageProcessorFast


    Constructs a ViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*):
            Whether to convert the image to RGB.
    

Methods: preprocess

<frameworkcontent>
<pt>

## ViTModel

The bare ViT Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ViTForMaskedImageModeling

ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ViTForImageClassification


    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

</pt>
<tf>

## TFViTModel

No docstring available for TFViTModel

Methods: call

## TFViTForImageClassification

No docstring available for TFViTForImageClassification

Methods: call

</tf>
<jax>

## FlaxVitModel

No docstring available for FlaxViTModel

Methods: __call__

## FlaxViTForImageClassification

No docstring available for FlaxViTForImageClassification

Methods: __call__

</jax>
</frameworkcontent>
