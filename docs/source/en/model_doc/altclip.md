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

# AltCLIP

## Overview

The AltCLIP model was proposed in [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679v2) by Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Fulong Ye, Qinghong Yang, Ledell Wu. AltCLIP
(Altering the Language Encoder in CLIP) is a neural network trained on a variety of image-text and text-text pairs. By switching CLIP's
text encoder with a pretrained multilingual text encoder XLM-R, we could obtain very close performances with CLIP on almost all tasks, and extended original CLIP's capabilities such as multilingual understanding.

The abstract from the paper is the following:

*In this work, we present a conceptually simple and effective method to train a strong bilingual multimodal representation model. 
Starting from the pretrained multimodal representation model CLIP released by OpenAI, we switched its text encoder with a pretrained 
multilingual text encoder XLM-R, and aligned both languages and image representations by a two-stage training schema consisting of 
teacher learning and contrastive learning. We validate our method through evaluations of a wide range of tasks. We set new state-of-the-art 
performances on a bunch of tasks including ImageNet-CN, Flicker30k- CN, and COCO-CN. Further, we obtain very close performances with 
CLIP on almost all tasks, suggesting that one can simply alter the text encoder in CLIP for extended capabilities such as multilingual understanding.*

This model was contributed by [jongjyh](https://huggingface.co/jongjyh).

## Usage tips and example

The usage of AltCLIP is very similar to the CLIP. the difference between CLIP is the text encoder. Note that we use bidirectional attention instead of casual attention
and we take the [CLS] token in XLM-R to represent text embedding.

AltCLIP is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image
classification. AltCLIP uses a ViT like transformer to get visual features and a bidirectional language model to get the text
features. Both the text and visual features are then projected to a latent space with identical dimension. The dot
product between the projected image and text features is then used as a similar score.

To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches,
which are then linearly embedded. A [CLS] token is added to serve as representation of an entire image. The authors
also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder.
The [`CLIPImageProcessor`] can be used to resize (or rescale) and normalize images for the model.

The [`AltCLIPProcessor`] wraps a [`CLIPImageProcessor`] and a [`XLMRobertaTokenizer`] into a single instance to both
encode the text and prepare the images. The following example shows how to get the image-text similarity scores using
[`AltCLIPProcessor`] and [`AltCLIPModel`].

```python
>>> from PIL import Image
>>> import requests

>>> from transformers import AltCLIPModel, AltCLIPProcessor

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

<Tip>

This model is based on `CLIPModel`, use it like you would use the original [CLIP](clip).

</Tip>

## AltCLIPConfig


    This is the configuration class to store the configuration of a [`AltCLIPModel`]. It is used to instantiate an
    AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AltCLIP
    [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AltCLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AltCLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import AltCLIPConfig, AltCLIPModel

    >>> # Initializing a AltCLIPConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPConfig()

    >>> # Initializing a AltCLIPModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AltCLIPConfig from a AltCLIPTextConfig and a AltCLIPVisionConfig

    >>> # Initializing a AltCLIPText and AltCLIPVision configuration
    >>> config_text = AltCLIPTextConfig()
    >>> config_vision = AltCLIPVisionConfig()

    >>> config = AltCLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```

Methods: from_text_vision_configs

## AltCLIPTextConfig


    This is the configuration class to store the configuration of a [`AltCLIPTextModel`]. It is used to instantiate a
    AltCLIP text model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the AltCLIP
    [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 250002):
            Vocabulary size of the AltCLIP model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`AltCLIPTextModel`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 514):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 1):
            The vocabulary size of the `token_type_ids` passed when calling [`AltCLIPTextModel`]
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 0.02):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 1): The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to 0): The id of the *beginning-of-sequence* token.
        eos_token_id (`Union[int, List[int]]`, *optional*, defaults to 2):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        project_dim (`int`, *optional*, defaults to 768):
            The dimensions of the teacher model before the mapping layer.

    Examples:

    ```python
    >>> from transformers import AltCLIPTextModel, AltCLIPTextConfig

    >>> # Initializing a AltCLIPTextConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPTextConfig()

    >>> # Initializing a AltCLIPTextModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## AltCLIPVisionConfig


    This is the configuration class to store the configuration of a [`AltCLIPModel`]. It is used to instantiate an
    AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AltCLIP
    [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import AltCLIPVisionConfig, AltCLIPVisionModel

    >>> # Initializing a AltCLIPVisionConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPVisionConfig()

    >>> # Initializing a AltCLIPVisionModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## AltCLIPProcessor


    Constructs a AltCLIP processor which wraps a CLIP image processor and a XLM-Roberta tokenizer into a single
    processor.

    [`AltCLIPProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`XLMRobertaTokenizerFast`]. See
    the [`~AltCLIPProcessor.__call__`] and [`~AltCLIPProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`XLMRobertaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    

## AltCLIPModel

No docstring available for AltCLIPModel

Methods: forward
    - get_text_features
    - get_image_features

## AltCLIPTextModel

No docstring available for AltCLIPTextModel

Methods: forward

## AltCLIPVisionModel

No docstring available for AltCLIPVisionModel

Methods: forward