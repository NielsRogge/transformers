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

# Dilated Neighborhood Attention Transformer

## Overview

DiNAT was proposed in [Dilated Neighborhood Attention Transformer](https://arxiv.org/abs/2209.15001)
by Ali Hassani and Humphrey Shi.

It extends [NAT](nat) by adding a Dilated Neighborhood Attention pattern to capture global context,
and shows significant performance improvements over it.

The abstract from the paper is the following:

*Transformers are quickly becoming one of the most heavily applied deep learning architectures across modalities,
domains, and tasks. In vision, on top of ongoing efforts into plain transformers, hierarchical transformers have
also gained significant attention, thanks to their performance and easy integration into existing frameworks.
These models typically employ localized attention mechanisms, such as the sliding-window Neighborhood Attention (NA)
or Swin Transformer's Shifted Window Self Attention. While effective at reducing self attention's quadratic complexity,
local attention weakens two of the most desirable properties of self attention: long range inter-dependency modeling,
and global receptive field. In this paper, we introduce Dilated Neighborhood Attention (DiNA), a natural, flexible and
efficient extension to NA that can capture more global context and expand receptive fields exponentially at no
additional cost. NA's local attention and DiNA's sparse global attention complement each other, and therefore we
introduce Dilated Neighborhood Attention Transformer (DiNAT), a new hierarchical vision transformer built upon both.
DiNAT variants enjoy significant improvements over strong baselines such as NAT, Swin, and ConvNeXt.
Our large model is faster and ahead of its Swin counterpart by 1.5% box AP in COCO object detection,
1.3% mask AP in COCO instance segmentation, and 1.1% mIoU in ADE20K semantic segmentation.
Paired with new frameworks, our large variant is the new state of the art panoptic segmentation model on COCO (58.2 PQ)
and ADE20K (48.5 PQ), and instance segmentation model on Cityscapes (44.5 AP) and ADE20K (35.4 AP) (no extra data).
It also matches the state of the art specialized semantic segmentation models on ADE20K (58.2 mIoU),
and ranks second on Cityscapes (84.5 mIoU) (no extra data). *

<img
src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dilated-neighborhood-attention-pattern.jpg"
alt="drawing" width="600"/>

<small> Neighborhood Attention with different dilation values.
Taken from the <a href="https://arxiv.org/abs/2209.15001">original paper</a>.</small>

This model was contributed by [Ali Hassani](https://huggingface.co/alihassanijr).
The original code can be found [here](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer).

## Usage tips

DiNAT can be used as a *backbone*. When `output_hidden_states = True`,
it will output both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, height, width, num_channels)`.

Notes:
- DiNAT depends on [NATTEN](https://github.com/SHI-Labs/NATTEN/)'s implementation of Neighborhood Attention and Dilated Neighborhood Attention.
You can install it with pre-built wheels for Linux by referring to [shi-labs.com/natten](https://shi-labs.com/natten), or build on your system by running `pip install natten`.
Note that the latter will likely take time to compile. NATTEN does not support Windows devices yet.
- Patch size of 4 is only supported at the moment.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DiNAT.

<PipelineTag pipeline="image-classification"/>

- [`DinatForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DinatConfig


    This is the configuration class to store the configuration of a [`DinatModel`]. It is used to instantiate a Dinat
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Dinat
    [shi-labs/dinat-mini-in1k-224](https://huggingface.co/shi-labs/dinat-mini-in1k-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each patch. NOTE: Only patch size of 4 is supported at the moment.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embed_dim (`int`, *optional*, defaults to 64):
            Dimensionality of patch embedding.
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 5]`):
            Number of layers in each level of the encoder.
        num_heads (`List[int]`, *optional*, defaults to `[2, 4, 8, 16]`):
            Number of attention heads in each layer of the Transformer encoder.
        kernel_size (`int`, *optional*, defaults to 7):
            Neighborhood Attention kernel size.
        dilations (`List[List[int]]`, *optional*, defaults to `[[1, 8, 1], [1, 4, 1, 4], [1, 2, 1, 2, 1, 2], [1, 1, 1, 1, 1]]`):
            Dilation value of each NA layer in the Transformer encoder.
        mlp_ratio (`float`, *optional*, defaults to 3.0):
            Ratio of MLP hidden dimensionality to embedding dimensionality.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not a learnable bias should be added to the queries, keys and values.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        layer_scale_init_value (`float`, *optional*, defaults to 0.0):
            The initial value for the layer scale. Disabled if <=0.
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
    >>> from transformers import DinatConfig, DinatModel

    >>> # Initializing a Dinat shi-labs/dinat-mini-in1k-224 style configuration
    >>> configuration = DinatConfig()

    >>> # Initializing a model (with random weights) from the shi-labs/dinat-mini-in1k-224 style configuration
    >>> model = DinatModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## DinatModel

The bare Dinat Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DinatConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## DinatForImageClassification


    Dinat Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DinatConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
