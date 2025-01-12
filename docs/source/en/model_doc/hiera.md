<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Hiera

## Overview

Hiera was proposed in [Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://arxiv.org/abs/2306.00989) by Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer

The paper introduces "Hiera," a hierarchical Vision Transformer that simplifies the architecture of modern hierarchical vision transformers by removing unnecessary components without compromising on accuracy or efficiency. Unlike traditional transformers that add complex vision-specific components to improve supervised classification performance, Hiera demonstrates that such additions, often termed "bells-and-whistles," are not essential for high accuracy. By leveraging a strong visual pretext task (MAE) for pretraining, Hiera retains simplicity and achieves superior accuracy and speed both in inference and training across various image and video recognition tasks. The approach suggests that spatial biases required for vision tasks can be effectively learned through proper pretraining, eliminating the need for added architectural complexity. 

The abstract from the paper is the following:

*Modern hierarchical vision transformers have added several vision-specific components in the pursuit of supervised classification performance. While these components lead to effective accuracies and attractive FLOP counts, the added complexity actually makes these transformers slower than their vanilla ViT counterparts. In this paper, we argue that this additional bulk is unnecessary. By pretraining with a strong visual pretext task (MAE), we can strip out all the bells-and-whistles from a state-of-the-art multi-stage vision transformer without losing accuracy. In the process, we create Hiera, an extremely simple hierarchical vision transformer that is more accurate than previous models while being significantly faster both at inference and during training. We evaluate Hiera on a variety of tasks for image and video recognition. Our code and models are available at https://github.com/facebookresearch/hiera.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/hiera_overview.png"
alt="drawing" width="600"/>

<small> Hiera architecture. Taken from the <a href="https://arxiv.org/abs/2306.00989">original paper.</a> </small>

This model was a joint contribution by [EduardoPacheco](https://huggingface.co/EduardoPacheco) and [namangarg110](https://huggingface.co/namangarg110). The original code can be found [here] (https://github.com/facebookresearch/hiera).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Hiera. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="image-classification"/>

- [`HieraForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

## HieraConfig


    This is the configuration class to store the configuration of a [`HieraModel`]. It is used to instantiate a Hiera
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Hiera
    [facebook/hiera-base-224](https://huggingface.co/facebook/hiera-base-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        embed_dim (`int`, *optional*, defaults to 96):
            Dimensionality of patch embedding.
        image_size (`list(int)`, *optional*, defaults to `[224, 224]`):
            The size (resolution) of input in the format (height, width) for images
            and (frames, height, width) for videos.
        patch_size (`list(int)`, *optional*, defaults to `[7, 7]`):
            The size (resolution) of each patch.
        patch_stride (`list(int)`, *optional*, defaults to `[4, 4]`):
            The stride of the patch.
        patch_padding (`list(int)`, *optional*, defaults to `[3, 3]`):
            The padding of the patch.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of mlp hidden dim to embedding dim.
        depths (`list(int)`, *optional*, defaults to `[2, 3, 16, 3]`):
            Depth of each layer in the Transformer encoder.
        num_heads (`list(int)`, *optional*, defaults to `[1, 2, 4, 8]`):
            Number of attention heads in each layer of the Transformer encoder.
        embed_dim_multiplier (`float`, *optional*, defaults to 2.0):
            The multiplier to the dimensionality of patch embedding in each layer of the Transformer encoder.
        num_query_pool (`int`, *optional*, defaults to 3):
            The number of query pool stages.
        query_stride (`list(int)`, *optional*, defaults to `[2, 2]`):
            The stride of the query pool.
        masked_unit_size (`list(int)`, *optional*, defaults to `[8, 8]`):
            The size of the masked unit.
        masked_unit_attention (`list(bool)`, *optional*, defaults to `[True, True, False, False]`):
            Whether to use masked unit attention in each layer of the Transformer encoder.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The drop path rate.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices and
            the zero_initializer for initializing all bias vectors.
        layer_norm_init (`float`, *optional*, defaults to 1.0):
            The initial weight value for layer normalization layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        decoder_hidden_size (`int`, *optional*):
            Dimensionality of decoder embeddings for MAE pretraining.
        decoder_depth (`int`, *optional*):
            Depth of the decoder for MAE pretraining.
        decoder_num_heads (`int`, *optional*):
            Number of attention heads in each layer of the decoder for MAE pretraining.
        normalize_pixel_loss (`bool`, *optional*, defaults to `True`):
            Whether to normalize the pixel loss by the number of pixels.
        mask_ratio (`float`, *optional*, defaults to 0.6):
            The ratio of masked tokens in the input.
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.


    Example:

    ```python
    >>> from transformers import HieraConfig, HieraModel

    >>> # Initializing a Hiera hiera-base-patch16-224 style configuration
    >>> configuration = HieraConfig()

    >>> # Initializing a model (with random weights) from the hiera-base-patch16-224 style configuration
    >>> model = HieraModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## HieraModel

The bare Hiera Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`HieraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

        add_pooling_layer (`bool`, *optional*, defaults to `True`):
                Whether or not to apply pooling layer.
        is_mae (`bool`, *optional*, defaults to `False`):
                Whether or not to run the model on MAE mode.
    

Methods: forward

## HieraForPreTraining

The Hiera Model transformer with the decoder on top for self-supervised pre-training.

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`HieraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
  
## HieraForImageClassification


    Hiera Model transformer with an image classification head on top (a linear layer on top of the final hidden state with
    average pooling) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune Hiera on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`HieraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
