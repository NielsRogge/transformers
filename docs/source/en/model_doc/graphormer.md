<!--Copyright 2022 The HuggingFace Team and Microsoft. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Graphormer

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The Graphormer model was proposed in [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234)  by
Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen and Tie-Yan Liu. It is a Graph Transformer model, modified to allow computations on graphs instead of text sequences by generating embeddings and features of interest during preprocessing and collation, then using a modified attention.

The abstract from the paper is the following:

*The Transformer architecture has become a dominant choice in many domains, such as natural language processing and computer vision. Yet, it has not achieved competitive performance on popular leaderboards of graph-level prediction compared to mainstream GNN variants. Therefore, it remains a mystery how Transformers could perform well for graph representation learning. In this paper, we solve this mystery by presenting Graphormer, which is built upon the standard Transformer architecture, and could attain excellent results on a broad range of graph representation learning tasks, especially on the recent OGB Large-Scale Challenge. Our key insight to utilizing Transformer in the graph is the necessity of effectively encoding the structural information of a graph into the model. To this end, we propose several simple yet effective structural encoding methods to help Graphormer better model graph-structured data. Besides, we mathematically characterize the expressive power of Graphormer and exhibit that with our ways of encoding the structural information of graphs, many popular GNN variants could be covered as the special cases of Graphormer.*

This model was contributed by [clefourrier](https://huggingface.co/clefourrier). The original code can be found [here](https://github.com/microsoft/Graphormer).

## Usage tips

This model will not work well on large graphs (more than 100 nodes/edges), as it will make the memory explode.
You can reduce the batch size, increase your RAM, or decrease the `UNREACHABLE_NODE_DISTANCE` parameter in algos_graphormer.pyx, but it will be hard to go above 700 nodes/edges.

This model does not use a tokenizer, but instead a special collator during training.

## GraphormerConfig


    This is the configuration class to store the configuration of a [`~GraphormerModel`]. It is used to instantiate an
    Graphormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Graphormer
    [graphormer-base-pcqm4mv1](https://huggingface.co/graphormer-base-pcqm4mv1) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_classes (`int`, *optional*, defaults to 1):
            Number of target classes or labels, set to n for binary classification of n tasks.
        num_atoms (`int`, *optional*, defaults to 512*9):
            Number of node types in the graphs.
        num_edges (`int`, *optional*, defaults to 512*3):
            Number of edges types in the graph.
        num_in_degree (`int`, *optional*, defaults to 512):
            Number of in degrees types in the input graphs.
        num_out_degree (`int`, *optional*, defaults to 512):
            Number of out degrees types in the input graphs.
        num_edge_dis (`int`, *optional*, defaults to 128):
            Number of edge dis in the input graphs.
        multi_hop_max_dist (`int`, *optional*, defaults to 20):
            Maximum distance of multi hop edges between two nodes.
        spatial_pos_max (`int`, *optional*, defaults to 1024):
            Maximum distance between nodes in the graph attention bias matrices, used during preprocessing and
            collation.
        edge_type (`str`, *optional*, defaults to multihop):
            Type of edge relation chosen.
        max_nodes (`int`, *optional*, defaults to 512):
            Maximum number of nodes which can be parsed for the input graphs.
        share_input_output_embed (`bool`, *optional*, defaults to `False`):
            Shares the embedding layer between encoder and decoder - careful, True is not implemented.
        num_layers (`int`, *optional*, defaults to 12):
            Number of layers.
        embedding_dim (`int`, *optional*, defaults to 768):
            Dimension of the embedding layer in encoder.
        ffn_embedding_dim (`int`, *optional*, defaults to 768):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads in the encoder.
        self_attention (`bool`, *optional*, defaults to `True`):
            Model is self attentive (False not implemented).
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention weights.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the activation of the linear transformer layer.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        bias (`bool`, *optional*, defaults to `True`):
            Uses bias in the attention module - unsupported at the moment.
        embed_scale(`float`, *optional*, defaults to None):
            Scaling factor for the node embeddings.
        num_trans_layers_to_freeze (`int`, *optional*, defaults to 0):
            Number of transformer layers to freeze.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Normalize features before encoding the graph.
        pre_layernorm (`bool`, *optional*, defaults to `False`):
            Apply layernorm before self attention and the feed forward network. Without this, post layernorm will be
            used.
        apply_graphormer_init (`bool`, *optional*, defaults to `False`):
            Apply a custom graphormer initialisation to the model before training.
        freeze_embeddings (`bool`, *optional*, defaults to `False`):
            Freeze the embedding layer, or train it along the model.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Apply the layer norm before each encoder block.
        q_noise (`float`, *optional*, defaults to 0.0):
            Amount of quantization noise (see "Training with Quantization Noise for Extreme Model Compression"). (For
            more detail, see fairseq's documentation on quant_noise).
        qn_block_size (`int`, *optional*, defaults to 8):
            Size of the blocks for subsequent quantization with iPQ (see q_noise).
        kdim (`int`, *optional*, defaults to None):
            Dimension of the key in the attention, if different from the other values.
        vdim (`int`, *optional*, defaults to None):
            Dimension of the value in the attention, if different from the other values.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        traceable (`bool`, *optional*, defaults to `False`):
            Changes return value of the encoder's inner_state to stacked tensors.

        Example:
            ```python
            >>> from transformers import GraphormerForGraphClassification, GraphormerConfig

            >>> # Initializing a Graphormer graphormer-base-pcqm4mv2 style configuration
            >>> configuration = GraphormerConfig()

            >>> # Initializing a model from the graphormer-base-pcqm4mv1 style configuration
            >>> model = GraphormerForGraphClassification(configuration)

            >>> # Accessing the model configuration
            >>> configuration = model.config
            ```
    

## GraphormerModel

The Graphormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    GraphormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.
    

Methods: forward

## GraphormerForGraphClassification


    This model can be used for graph-level classification or regression tasks.

    It can be trained on
    - regression (by setting config.num_classes to 1); there should be one float-type label per graph
    - one task classification (by setting config.num_classes to the number of classes); there should be one integer
      label per graph
    - binary multi-task classification (by setting config.num_classes to the number of labels); there should be a list
      of integer labels for each graph.
    

Methods: forward
