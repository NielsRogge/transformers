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

# Trajectory Transformer

<Tip warning={true}>

This model is in maintenance mode only, so we won't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

</Tip>

## Overview

The Trajectory Transformer model was proposed in [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039)  by Michael Janner, Qiyang Li, Sergey Levine.

The abstract from the paper is the following:

*Reinforcement learning (RL) is typically concerned with estimating stationary policies or single-step models,
leveraging the Markov property to factorize problems in time. However, we can also view RL as a generic sequence
modeling problem, with the goal being to produce a sequence of actions that leads to a sequence of high rewards.
Viewed in this way, it is tempting to consider whether high-capacity sequence prediction models that work well
in other domains, such as natural-language processing, can also provide effective solutions to the RL problem.
To this end, we explore how RL can be tackled with the tools of sequence modeling, using a Transformer architecture
to model distributions over trajectories and repurposing beam search as a planning algorithm. Framing RL as sequence
modeling problem simplifies a range of design decisions, allowing us to dispense with many of the components common
in offline RL algorithms. We demonstrate the flexibility of this approach across long-horizon dynamics prediction,
imitation learning, goal-conditioned RL, and offline RL. Further, we show that this approach can be combined with
existing model-free algorithms to yield a state-of-the-art planner in sparse-reward, long-horizon tasks.*

This model was contributed by [CarlCochet](https://huggingface.co/CarlCochet). The original code can be found [here](https://github.com/jannerm/trajectory-transformer).

## Usage tips

This Transformer is used for deep reinforcement learning. To use it, you need to create sequences from
actions, states and rewards from all previous timesteps. This model will treat all these elements together
as one big sequence (a trajectory).

## TrajectoryTransformerConfig


    This is the configuration class to store the configuration of a [`TrajectoryTransformerModel`]. It is used to
    instantiate an TrajectoryTransformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    TrajectoryTransformer
    [CarlCochet/trajectory-transformer-halfcheetah-medium-v2](https://huggingface.co/CarlCochet/trajectory-transformer-halfcheetah-medium-v2)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 100):
            Vocabulary size of the TrajectoryTransformer model. Defines the number of different tokens that can be
            represented by the `trajectories` passed when calling [`TrajectoryTransformerModel`]
        action_weight (`int`, *optional*, defaults to 5):
            Weight of the action in the loss function
        reward_weight (`int`, *optional*, defaults to 1):
            Weight of the reward in the loss function
        value_weight (`int`, *optional*, defaults to 1):
            Weight of the value in the loss function
        block_size (`int`, *optional*, defaults to 249):
            Size of the blocks in the trajectory transformer.
        action_dim (`int`, *optional*, defaults to 6):
            Dimension of the action space.
        observation_dim (`int`, *optional*, defaults to 17):
            Dimension of the observation space.
        transition_dim (`int`, *optional*, defaults to 25):
            Dimension of the transition space.
        n_layer (`int`, *optional*, defaults to 4):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_embd (`int`, *optional*, defaults to 128):
            Dimensionality of the embeddings and hidden states.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        kaiming_initializer_range (`float, *optional*, defaults to 1):
            A coefficient scaling the negative slope of the kaiming initializer rectifier for EinLinear layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import TrajectoryTransformerConfig, TrajectoryTransformerModel

    >>> # Initializing a TrajectoryTransformer CarlCochet/trajectory-transformer-halfcheetah-medium-v2 style configuration
    >>> configuration = TrajectoryTransformerConfig()

    >>> # Initializing a model (with random weights) from the CarlCochet/trajectory-transformer-halfcheetah-medium-v2 style configuration
    >>> model = TrajectoryTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## TrajectoryTransformerModel

The bare TrajectoryTransformer Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TrajectoryTransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
the full GPT language model, with a context size of block_size

Methods: forward
