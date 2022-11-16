# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

GIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/git-base": "https://huggingface.co/microsoft/git-base/resolve/main/config.json",
}


# Copied from transformers.models.clip.configuration_clip.CLIPVisionConfig with CLIP->GIT, clip->git
class GITVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GITModel`]. It is used to instantiate an GIT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GIT
    [openai/git-vit-base-patch32](https://huggingface.co/openai/git-vit-base-patch32) architecture.

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
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported. layer_norm_eps (`float`, *optional*,
            defaults to 1e-5): The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import GITVisionConfig, GITVisionModel

    >>> # Initializing a GITVisionConfig with openai/git-vit-base-patch32 style configuration
    >>> configuration = GITVisionConfig()

    >>> # Initializing a GITVisionModel (with random weights) from the openai/git-vit-base-patch32 style configuration
    >>> model = GITVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "git_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from GITConfig
        if config_dict.get("model_type") == "git":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class GITConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GITModel`]. It is used to instantiate a GIT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GIT
    [microsoft/git-base](https://huggingface.co/microsoft/git-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the GIT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GITModel`].
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
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`GITModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import GITConfig, GITModel

    >>> # Initializing a GIT microsoft/git-base style configuration
    >>> configuration = GITConfig()

    >>> # Initializing a model (with random weights) from the microsoft/git-base style configuration
    >>> model = GITModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "git"

    def __init__(
        self,
        vision_config=None,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        tie_word_embeddings=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vision_config = vision_config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.tie_word_embeddings = tie_word_embeddings