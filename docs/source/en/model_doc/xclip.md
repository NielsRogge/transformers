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

# X-CLIP

## Overview

The X-CLIP model was proposed in [Expanding Language-Image Pretrained Models for General Video Recognition](https://arxiv.org/abs/2208.02816) by Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, Haibin Ling.
X-CLIP is a minimal extension of [CLIP](clip) for video. The model consists of a text encoder, a cross-frame vision encoder, a multi-frame integration Transformer, and a video-specific prompt generator.

The abstract from the paper is the following:

*Contrastive language-image pretraining has shown great success in learning visual-textual joint representation from web-scale data, demonstrating remarkable "zero-shot" generalization ability for various image tasks. However, how to effectively expand such new language-image pretraining methods to video domains is still an open problem. In this work, we present a simple yet effective approach that adapts the pretrained language-image models to video recognition directly, instead of pretraining a new model from scratch. More concretely, to capture the long-range dependencies of frames along the temporal dimension, we propose a cross-frame attention mechanism that explicitly exchanges information across frames. Such module is lightweight and can be plugged into pretrained language-image models seamlessly. Moreover, we propose a video-specific prompting scheme, which leverages video content information for generating discriminative textual prompts. Extensive experiments demonstrate that our approach is effective and can be generalized to different video recognition scenarios. In particular, under fully-supervised settings, our approach achieves a top-1 accuracy of 87.1% on Kinectics-400, while using 12 times fewer FLOPs compared with Swin-L and ViViT-H. In zero-shot experiments, our approach surpasses the current state-of-the-art methods by +7.6% and +14.9% in terms of top-1 accuracy under two popular protocols. In few-shot scenarios, our approach outperforms previous best methods by +32.1% and +23.1% when the labeled data is extremely limited.*

Tips:

- Usage of X-CLIP is identical to [CLIP](clip).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/xclip_architecture.png"
alt="drawing" width="600"/>

<small> X-CLIP architecture. Taken from the <a href="https://arxiv.org/abs/2208.02816">original paper.</a> </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/microsoft/VideoX/tree/master/X-CLIP).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with X-CLIP.

- Demo notebooks for X-CLIP can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/X-CLIP).

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## XCLIPProcessor


    Constructs an X-CLIP processor which wraps a VideoMAE image processor and a CLIP tokenizer into a single processor.

    [`XCLIPProcessor`] offers all the functionalities of [`VideoMAEImageProcessor`] and [`CLIPTokenizerFast`]. See the
    [`~XCLIPProcessor.__call__`] and [`~XCLIPProcessor.decode`] for more information.

    Args:
        image_processor ([`VideoMAEImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizerFast`], *optional*):
            The tokenizer is a required input.
    

## XCLIPConfig


    [`XCLIPConfig`] is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to
    instantiate X-CLIP model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`XCLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`XCLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        prompt_layers (`int`, *optional*, defaults to 2):
            Number of layers in the video specific prompt generator.
        prompt_alpha (`float`, *optional*, defaults to 0.1):
            Alpha value to use in the video specific prompt generator.
        prompt_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the video specific prompt generator. If string,
            `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        prompt_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the cross-attention of the video specific prompt generator.
        prompt_attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers in the video specific prompt generator.
        prompt_projection_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the projection layers in the video specific prompt generator.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* parameter. Default is used as per the original XCLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    

Methods: from_text_vision_configs

## XCLIPTextConfig


    This is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to instantiate an X-CLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the X-CLIP text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`XCLIPModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import XCLIPTextModel, XCLIPTextConfig

    >>> # Initializing a XCLIPTextModel with microsoft/xclip-base-patch32 style configuration
    >>> configuration = XCLIPTextConfig()

    >>> # Initializing a XCLIPTextConfig from the microsoft/xclip-base-patch32 style configuration
    >>> model = XCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## XCLIPVisionConfig


    This is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to instantiate an X-CLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        mit_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers of the Multiframe Integration Transformer (MIT).
        mit_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Multiframe Integration Transformer
            (MIT).
        mit_num_hidden_layers (`int`, *optional*, defaults to 1):
            Number of hidden layers in the Multiframe Integration Transformer (MIT).
        mit_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Multiframe Integration Transformer (MIT).
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"`, `"gelu_new"` and `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate.

    Example:

    ```python
    >>> from transformers import XCLIPVisionModel, XCLIPVisionConfig

    >>> # Initializing a XCLIPVisionModel with microsoft/xclip-base-patch32 style configuration
    >>> configuration = XCLIPVisionConfig()

    >>> # Initializing a XCLIPVisionModel model from the microsoft/xclip-base-patch32 style configuration
    >>> model = XCLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## XCLIPModel


    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`XCLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - get_text_features
    - get_video_features

## XCLIPTextModel

No docstring available for XCLIPTextModel

Methods: forward

## XCLIPVisionModel

No docstring available for XCLIPVisionModel

Methods: forward
