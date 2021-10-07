# coding=utf-8
# Copyright Deepmind and The HuggingFace Inc. team. All rights reserved.
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
""" Perceiver model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "deepmind/language-perceiver": "https://huggingface.co/deepmind/language-perceiver/resolve/main/config.json",
    # See all Perceiver models at https://huggingface.co/models?filter=perceiver
}


class PerceiverConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.PerceiverModel`. It is used
    to instantiate an Perceiver model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Perceiver
    `deepmind/language-perceiver <https://huggingface.co/deepmind/language-perceiver>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        num_latents (:obj:`int`, `optional`, defaults to 256):
            The number of latents.
        d_latents (:obj:`int`, `optional`, defaults to 1280):
            Dimension of the latent embeddings.
        d_model (:obj:`int`, `optional`, defaults to 768):
            Dimension of the inputs.
        num_blocks (:obj:`int`, `optional`, defaults to 1):
            Number of blocks in the Transformer encoder.
        num_self_attends_per_block (:obj:`int`, `optional`, defaults to 26):
            The number of self-attention layers per block.
        num_self_attention_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each self-attention layer in the Transformer encoder.
        num_cross_attention_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each cross-attention layer in the Transformer encoder.
        qk_channels (:obj:`int`, `optional`):
            Dimension to project the queries + keys before applying attention in the cross-attention and self-attention
            layers of the encoder. Will default to preserving the dimension of the queries if not specified.
        v_channels (:obj:`int`, `optional`):
            Dimension to project the values before applying attention in the cross-attention and self-attention layers
            of the encoder. Will default to preserving the dimension of the queries if not specified.
        cross_attention_shape_for_attention (:obj:`str`, `optional`, defaults to :obj:`'kv'`):
            Dimension to use when downsampling the queries and keys in the cross-attention layer of the encoder.
        self_attention_widening_factor (:obj:`int`, `optional`, defaults to 1):
            Dimension of the feed-forward layer in the cross-attention layer of the Transformer encoder.
        cross_attention_widening_factor (:obj:`int`, `optional`, defaults to 1):
            Dimension of the feed-forward layer in the self-attention layers of the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_query_residual (:obj:`float`, `optional`, defaults to :obj:`True`):
            Whether to add a query residual in the cross-attention layer of the encoder.
        vocab_size (:obj:`int`, `optional`, defaults to 262):
            Vocabulary size of the Perceiver model.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        train_size (:obj:`List[int]`, `optional`, defaults to [368, 496]):
            Training size of the images for the flow model.

    Example::

        >>> from transformers import PerceiverModel, PerceiverConfig

        >>> # Initializing a Perceiver deepmind/language-perceiver style configuration
        >>> configuration = PerceiverConfig()

        >>> # Initializing a model from the deepmind/language-perceiver style configuration
        >>> model = PerceiverModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "perceiver"

    def __init__(
        self,
        num_latents=256,
        d_latents=1280,
        d_model=768,
        num_blocks=1,
        num_self_attends_per_block=26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        position_embedding_init_scale=0.02,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        use_query_residual=True,
        vocab_size=262,
        max_position_embeddings=2048,
        train_size=[368, 496],
        num_frames=16,
        audio_samples_per_frame=1920,  # 48000 // 25
        samples_per_patch=16,
        image_size=56,
        output_shape=[1, 16, 224, 224],
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_latents = num_latents
        self.d_latents = d_latents
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.cross_attention_shape_for_attention = cross_attention_shape_for_attention
        self.self_attention_widening_factor = self_attention_widening_factor
        self.cross_attention_widening_factor = cross_attention_widening_factor
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_query_residual = use_query_residual
        # masked language modeling attributes
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        # image classification attributes
        self.image_size = image_size
        # flow attributes
        self.train_size = train_size
        # multimodal autoencoding attributes
        self.num_frames = num_frames
        self.audio_samples_per_frame = audio_samples_per_frame
        self.samples_per_patch = samples_per_patch
        self.output_shape = output_shape
