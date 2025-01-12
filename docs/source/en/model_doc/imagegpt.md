<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->

# ImageGPT

## Overview

The ImageGPT model was proposed in [Generative Pretraining from Pixels](https://openai.com/blog/image-gpt) by Mark
Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever. ImageGPT (iGPT) is a GPT-2-like
model trained to predict the next pixel value, allowing for both unconditional and conditional image generation.

The abstract from the paper is the following:

*Inspired by progress in unsupervised representation learning for natural language, we examine whether similar models
can learn useful representations for images. We train a sequence Transformer to auto-regressively predict pixels,
without incorporating knowledge of the 2D input structure. Despite training on low-resolution ImageNet without labels,
we find that a GPT-2 scale model learns strong image representations as measured by linear probing, fine-tuning, and
low-data classification. On CIFAR-10, we achieve 96.3% accuracy with a linear probe, outperforming a supervised Wide
ResNet, and 99.0% accuracy with full fine-tuning, matching the top supervised pre-trained models. We are also
competitive with self-supervised benchmarks on ImageNet when substituting pixels for a VQVAE encoding, achieving 69.0%
top-1 accuracy on a linear probe of our features.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/imagegpt_architecture.png"
alt="drawing" width="600"/>

<small> Summary of the approach. Taken from the [original paper](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf). </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr), based on [this issue](https://github.com/openai/image-gpt/issues/7). The original code can be found
[here](https://github.com/openai/image-gpt).

## Usage tips

- ImageGPT is almost exactly the same as [GPT-2](gpt2), with the exception that a different activation
  function is used (namely "quick gelu"), and the layer normalization layers don't mean center the inputs. ImageGPT
  also doesn't have tied input- and output embeddings.
- As the time- and memory requirements of the attention mechanism of Transformers scales quadratically in the sequence
  length, the authors pre-trained ImageGPT on smaller input resolutions, such as 32x32 and 64x64. However, feeding a
  sequence of 32x32x3=3072 tokens from 0..255 into a Transformer is still prohibitively large. Therefore, the authors
  applied k-means clustering to the (R,G,B) pixel values with k=512. This way, we only have a 32*32 = 1024-long
  sequence, but now of integers in the range 0..511. So we are shrinking the sequence length at the cost of a bigger
  embedding matrix. In other words, the vocabulary size of ImageGPT is 512, + 1 for a special "start of sentence" (SOS)
  token, used at the beginning of every sequence. One can use [`ImageGPTImageProcessor`] to prepare
  images for the model.
- Despite being pre-trained entirely unsupervised (i.e. without the use of any labels), ImageGPT produces fairly
  performant image features useful for downstream tasks, such as image classification. The authors showed that the
  features in the middle of the network are the most performant, and can be used as-is to train a linear model (such as
  a sklearn logistic regression model for example). This is also referred to as "linear probing". Features can be
  easily obtained by first forwarding the image through the model, then specifying `output_hidden_states=True`, and
  then average-pool the hidden states at whatever layer you like.
- Alternatively, one can further fine-tune the entire model on a downstream dataset, similar to BERT. For this, you can
  use [`ImageGPTForImageClassification`].
- ImageGPT comes in different sizes: there's ImageGPT-small, ImageGPT-medium and ImageGPT-large. The authors did also
  train an XL variant, which they didn't release. The differences in size are summarized in the following table:

| **Model variant** | **Depths** | **Hidden sizes** | **Decoder hidden size** | **Params (M)** | **ImageNet-1k Top 1** |
|---|---|---|---|---|---|
| MiT-b0 | [2, 2, 2, 2] | [32, 64, 160, 256] | 256 | 3.7 | 70.5 |
| MiT-b1 | [2, 2, 2, 2] | [64, 128, 320, 512] | 256 | 14.0 | 78.7 |
| MiT-b2 | [3, 4, 6, 3] | [64, 128, 320, 512] | 768 | 25.4 | 81.6 |
| MiT-b3 | [3, 4, 18, 3] | [64, 128, 320, 512] | 768 | 45.2 | 83.1 |
| MiT-b4 | [3, 8, 27, 3] | [64, 128, 320, 512] | 768 | 62.6 | 83.6 |
| MiT-b5 | [3, 6, 40, 3] | [64, 128, 320, 512] | 768 | 82.0 | 83.8 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ImageGPT.

<PipelineTag pipeline="image-classification"/>

- Demo notebooks for ImageGPT can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ImageGPT).
- [`ImageGPTForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## ImageGPTConfig


    This is the configuration class to store the configuration of a [`ImageGPTModel`] or a [`TFImageGPTModel`]. It is
    used to instantiate a GPT-2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ImageGPT
    [openai/imagegpt-small](https://huggingface.co/openai/imagegpt-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ImageGPTModel`] or [`TFImageGPTModel`].
        n_positions (`int`, *optional*, defaults to 32*32):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 512):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"quick_gelu"`):
            Activation function (can be one of the activation functions defined in src/transformers/activations.py).
            Defaults to "quick_gelu".
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import ImageGPTConfig, ImageGPTModel

    >>> # Initializing a ImageGPT configuration
    >>> configuration = ImageGPTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ImageGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## ImageGPTFeatureExtractor

No docstring available for ImageGPTFeatureExtractor

Methods: __call__

## ImageGPTImageProcessor


    Constructs a ImageGPT image processor. This image processor can be used to resize images to a smaller resolution
    (such as 32x32 or 64x64), normalize them and finally color quantize them to obtain sequences of "pixel values"
    (color clusters).

    Args:
        clusters (`np.ndarray` or `List[List[int]]`, *optional*):
            The color clusters to use, of shape `(n_clusters, 3)` when color quantizing. Can be overriden by `clusters`
            in `preprocess`.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's dimensions to `(size["height"], size["width"])`. Can be overridden by
            `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 256, "width": 256}`):
            Size of the image after resizing. Can be overridden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image pixel value to between [-1, 1]. Can be overridden by `do_normalize` in
            `preprocess`.
        do_color_quantize (`bool`, *optional*, defaults to `True`):
            Whether to color quantize the image. Can be overridden by `do_color_quantize` in `preprocess`.
    

Methods: preprocess

## ImageGPTModel

The bare ImageGPT Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ImageGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ImageGPTForCausalImageModeling


    The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ImageGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ImageGPTForImageClassification


    The ImageGPT Model transformer with an image classification head on top (linear layer).
    [`ImageGPTForImageClassification`] average-pools the hidden states in order to do the classification.
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ImageGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
