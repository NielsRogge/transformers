<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# InstructBlipVideo

## Overview

## Overview

The InstructBLIPVideo is an extension of the models proposed in [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500) by Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi.
InstructBLIPVideo uses the same architecture as [InstructBLIP](instructblip) and works with the same checkpoints as [InstructBLIP](instructblip). The only difference is the ability to process videos.

The abstract from the paper is the following:

*General-purpose language models that can solve various language-domain tasks have emerged driven by the pre-training and instruction-tuning pipeline. However, building general-purpose vision-language models is challenging due to the increased task discrepancy introduced by the additional visual input. Although vision-language pre-training has been widely studied, vision-language instruction tuning remains relatively less explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pre-trained BLIP-2 models. We gather a wide variety of 26 publicly available datasets, transform them into instruction tuning format and categorize them into two clusters for held-in instruction tuning and held-out zero-shot evaluation. Additionally, we introduce instruction-aware visual feature extraction, a crucial method that enables the model to extract informative features tailored to the given instruction. The resulting InstructBLIP models achieve state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and the larger Flamingo. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA IMG). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/instructblip_architecture.jpg"
alt="drawing" width="600"/>

<small> InstructBLIPVideo architecture. Taken from the <a href="https://arxiv.org/abs/2305.06500">original paper.</a> </small>

This model was contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip).

## Usage tips

- The model was trained by sampling 4 frames per video, so it's recommended to sample 4 frames

> [!NOTE]
> BLIP models after release v4.46 will raise warnings about adding `processor.num_query_tokens = {{num_query_tokens}}` and expand model embeddings layer to add special `<image>` token. It is strongly recommended to add the attributes to the processor if you own the model checkpoint, or open a PR if it is not owned by you. Adding these attributes means that BLIP will add the number of query tokens required per image and expand the text with as many `<image>` placeholders as there will be query tokens. Usually it is around 500 tokens per image, so make sure that the text is not truncated as otherwise there wil be failure when merging the embeddings.
The attributes can be obtained from model config, as `model.config.num_query_tokens` and model embeddings expansion can be done by following [this link](https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042).

## InstructBlipVideoConfig


    [`InstructBlipVideoConfig`] is the configuration class to store the configuration of a
    [`InstructBlipVideoForConditionalGeneration`]. It is used to instantiate a Instructblipvideo model according to the specified
    arguments, defining the vision model, Q-Former model and language model configs. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the Instructblipvideo
    [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`InstructBlipVideoVisionConfig`].
        qformer_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`InstructBlipVideoQFormerConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        num_query_tokens (`int`, *optional*, defaults to 32):
            The number of query tokens passed through the Transformer.

        video_token_index (`int`, *optional*):
            Token index of special video token.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     InstructBlipVideoVisionConfig,
    ...     InstructBlipVideoQFormerConfig,
    ...     OPTConfig,
    ...     InstructBlipVideoConfig,
    ...     InstructBlipVideoForConditionalGeneration,
    ... )

    >>> # Initializing a InstructBlipVideoConfig with Salesforce/instruct-blip-flan-t5 style configuration
    >>> configuration = InstructBlipVideoConfig()

    >>> # Initializing a InstructBlipVideoForConditionalGeneration (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
    >>> model = InstructBlipVideoForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a InstructBlipVideoConfig from a InstructBlipVideoVisionConfig, InstructBlipVideoQFormerConfig and any PretrainedConfig

    >>> # Initializing Instructblipvideo vision, Instructblipvideo Q-Former and language model configurations
    >>> vision_config = InstructBlipVideoVisionConfig()
    >>> qformer_config = InstructBlipVideoQFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = InstructBlipVideoConfig.from_text_vision_configs(vision_config, qformer_config, text_config)
    ```

Methods: from_vision_qformer_text_configs

## InstructBlipVideoVisionConfig


    This is the configuration class to store the configuration of a [`InstructBlipVideoVisionModel`]. It is used to
    instantiate a InstructBlipVideo vision encoder according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of the InstructBlipVideo
    [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1408):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 39):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported. to 1e-5): The epsilon used by the layer
            normalization layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries and values in the self-attention layers.

    Example:

    ```python
    >>> from transformers import InstructBlipVideoVisionConfig, InstructBlipVideoVisionModel

    >>> # Initializing a InstructBlipVideoVisionConfig with Salesforce/instruct-blip-flan-t5 style configuration
    >>> configuration = InstructBlipVideoVisionConfig()

    >>> # Initializing a InstructBlipVideoVisionModel (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
    >>> model = InstructBlipVideoVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## InstructBlipVideoQFormerConfig


    This is the configuration class to store the configuration of a [`InstructBlipVideoQFormerModel`]. It is used to
    instantiate a InstructBlipVideo Querying Transformer (Q-Former) model according to the specified arguments, defining the
    model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the InstructBlipVideo [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5)
    architecture. Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Note that [`InstructBlipVideoQFormerModel`] is very similar to [`BertLMHeadModel`] with interleaved cross-attention.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Q-Former model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling the model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Token id used for padding sequences.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        cross_attention_frequency (`int`, *optional*, defaults to 2):
            The frequency of adding cross-attention to the Transformer layers.
        encoder_hidden_size (`int`, *optional*, defaults to 1408):
            The hidden size of the hidden states for cross-attention.

    Examples:

    ```python
    >>> from transformers import InstructBlipVideoQFormerConfig, InstructBlipVideoQFormerModel

    >>> # Initializing a InstructBlipVideo Salesforce/instruct-blip-flan-t5 style configuration
    >>> configuration = InstructBlipVideoQFormerConfig()

    >>> # Initializing a model (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
    >>> model = InstructBlipVideoQFormerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## InstructBlipVideoProcessor


    Constructs an InstructBLIPVideo processor which wraps a InstructBLIP image processor and a LLaMa/T5 tokenizer into a single
    processor.

    [`InstructBlipVideoProcessor`] offers all the functionalities of [`InstructBlipVideoImageProcessor`] and [`AutoTokenizer`]. See the
    docstring of [`~InstructBlipVideoProcessor.__call__`] and [`~InstructBlipVideoProcessor.decode`] for more information.

    Args:
        image_processor (`InstructBlipVideoImageProcessor`):
            An instance of [`InstructBlipVideoImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
        qformer_tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.
        num_query_tokens (`int`, *optional*):
            Number of tokens used by the Qformer as queries, should be same as in model's config.
    

## InstructBlipVideoImageProcessor


    Constructs a InstructBLIPVideo image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    

Methods: preprocess

## InstructBlipVideoVisionModel

No docstring available for InstructBlipVideoVisionModel

Methods: forward

## InstructBlipVideoQFormerModel


    Querying Transformer (Q-Former), used in InstructBlipVideo. Slightly modified from BLIP-2 as it also takes the
    instruction as input.
    

Methods: forward

## InstructBlipVideoForConditionalGeneration


    InstructBlipVideo Model for generating text given an image and an optional text prompt. The model consists of a vision
    encoder, Querying Transformer (Q-Former) and a language model.

    One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
    the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`InstructBlipVideoConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - generate