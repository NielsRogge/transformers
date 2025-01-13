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

# OneFormer

## Overview

The OneFormer model was proposed in [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220) by Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi. OneFormer is a universal image segmentation framework that can be trained on a single panoptic dataset to perform semantic, instance, and panoptic segmentation tasks. OneFormer uses a task token to condition the model on the task in focus, making the architecture task-guided for training, and task-dynamic for inference.

<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_teaser.png"/>

The abstract from the paper is the following:

*Universal Image Segmentation is not a new concept. Past attempts to unify image segmentation in the last decades include scene parsing, panoptic segmentation, and, more recently, new panoptic architectures. However, such panoptic architectures do not truly unify image segmentation because they need to be trained individually on the semantic, instance, or panoptic segmentation to achieve the best performance. Ideally, a truly universal framework should be trained only once and achieve SOTA performance across all three image segmentation tasks. To that end, we propose OneFormer, a universal image segmentation framework that unifies segmentation with a multi-task train-once design. We first propose a task-conditioned joint training strategy that enables training on ground truths of each domain (semantic, instance, and panoptic segmentation) within a single multi-task training process. Secondly, we introduce a task token to condition our model on the task at hand, making our model task-dynamic to support multi-task training and inference. Thirdly, we propose using a query-text contrastive loss during training to establish better inter-task and inter-class distinctions. Notably, our single OneFormer model outperforms specialized Mask2Former models across all three segmentation tasks on ADE20k, CityScapes, and COCO, despite the latter being trained on each of the three tasks individually with three times the resources. With new ConvNeXt and DiNAT backbones, we observe even more performance improvement. We believe OneFormer is a significant step towards making image segmentation more universal and accessible.*

The figure below illustrates the architecture of OneFormer. Taken from the [original paper](https://arxiv.org/abs/2211.06220).

<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_architecture.png"/>

This model was contributed by [Jitesh Jain](https://huggingface.co/praeclarumjj3). The original code can be found [here](https://github.com/SHI-Labs/OneFormer).

## Usage tips

-  OneFormer requires two inputs during inference: *image* and *task token*. 
- During training, OneFormer only uses panoptic annotations.
- If you want to train the model in a distributed environment across multiple nodes, then one should update the
  `get_num_masks` function inside in the `OneFormerLoss` class of `modeling_oneformer.py`. When training on multiple nodes, this should be
  set to the average number of target masks across all nodes, as can be seen in the original implementation [here](https://github.com/SHI-Labs/OneFormer/blob/33ebb56ed34f970a30ae103e786c0cb64c653d9a/oneformer/modeling/criterion.py#L287).
- One can use [`OneFormerProcessor`] to prepare input images and task inputs for the model and optional targets for the model. [`OneFormerProcessor`] wraps [`OneFormerImageProcessor`] and [`CLIPTokenizer`] into a single instance to both prepare the images and encode the task inputs.
- To get the final segmentation, depending on the task, you can call [`~OneFormerProcessor.post_process_semantic_segmentation`] or [`~OneFormerImageProcessor.post_process_instance_segmentation`] or [`~OneFormerImageProcessor.post_process_panoptic_segmentation`]. All three tasks can be solved using [`OneFormerForUniversalSegmentation`] output, panoptic segmentation accepts an optional `label_ids_to_fuse` argument to fuse instances of the target object/s (e.g. sky) together.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with OneFormer.

- Demo notebooks regarding inference + fine-tuning on custom data can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/OneFormer).

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we will review it.
The resource should ideally demonstrate something new instead of duplicating an existing resource.

## OneFormer specific outputs

[[autodoc]] models.oneformer.modeling_oneformer.OneFormerModelOutput

[[autodoc]] models.oneformer.modeling_oneformer.OneFormerForUniversalSegmentationOutput

## OneFormerConfig


    This is the configuration class to store the configuration of a [`OneFormerModel`]. It is used to instantiate a
    OneFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OneFormer
    [shi-labs/oneformer_ade20k_swin_tiny](https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny) architecture
    trained on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig`, *optional*, defaults to `SwinConfig`):
            The configuration of the backbone model.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        ignore_value (`int`, *optional*, defaults to 255):
            Values to be ignored in GT label while calculating loss.
        num_queries (`int`, *optional*, defaults to 150):
            Number of object queries.
        no_object_weight (`float`, *optional*, defaults to 0.1):
            Weight for no-object class predictions.
        class_weight (`float`, *optional*, defaults to 2.0):
            Weight for Classification CE loss.
        mask_weight (`float`, *optional*, defaults to 5.0):
            Weight for binary CE loss.
        dice_weight (`float`, *optional*, defaults to 5.0):
            Weight for dice loss.
        contrastive_weight (`float`, *optional*, defaults to 0.5):
            Weight for contrastive loss.
        contrastive_temperature (`float`, *optional*, defaults to 0.07):
            Initial value for scaling the contrastive logits.
        train_num_points (`int`, *optional*, defaults to 12544):
            Number of points to sample while calculating losses on mask predictions.
        oversample_ratio (`float`, *optional*, defaults to 3.0):
            Ratio to decide how many points to oversample.
        importance_sample_ratio (`float`, *optional*, defaults to 0.75):
            Ratio of points that are sampled via importance sampling.
        init_std (`float`, *optional*, defaults to 0.02):
            Standard deviation for normal intialization.
        init_xavier_std (`float`, *optional*, defaults to 1.0):
            Standard deviation for xavier uniform initialization.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon for layer normalization.
        is_training (`bool`, *optional*, defaults to `False`):
            Whether to run in training or inference mode.
        use_auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether to calculate loss using intermediate predictions from transformer decoder.
        output_auxiliary_logits (`bool`, *optional*, defaults to `True`):
            Whether to return intermediate predictions from transformer decoder.
        strides (`list`, *optional*, defaults to `[4, 8, 16, 32]`):
            List containing the strides for feature maps in the encoder.
        task_seq_len (`int`, *optional*, defaults to 77):
            Sequence length for tokenizing text list input.
        text_encoder_width (`int`, *optional*, defaults to 256):
            Hidden size for text encoder.
        text_encoder_context_length (`int`, *optional*, defaults to 77):
            Input sequence length for text encoder.
        text_encoder_num_layers (`int`, *optional*, defaults to 6):
            Number of layers for transformer in text encoder.
        text_encoder_vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size for tokenizer.
        text_encoder_proj_layers (`int`, *optional*, defaults to 2):
            Number of layers in MLP for project text queries.
        text_encoder_n_ctx (`int`, *optional*, defaults to 16):
            Number of learnable text context queries.
        conv_dim (`int`, *optional*, defaults to 256):
            Feature map dimension to map outputs from the backbone.
        mask_dim (`int`, *optional*, defaults to 256):
            Dimension for feature maps in pixel decoder.
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimension for hidden states in transformer decoder.
        encoder_feedforward_dim (`int`, *optional*, defaults to 1024):
            Dimension for FFN layer in pixel decoder.
        norm (`str`, *optional*, defaults to `"GN"`):
            Type of normalization.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of layers in pixel decoder.
        decoder_layers (`int`, *optional*, defaults to 10):
            Number of layers in transformer decoder.
        use_task_norm (`bool`, *optional*, defaults to `True`):
            Whether to normalize the task token.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in transformer layers in the pixel and transformer decoders.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability for pixel and transformer decoders.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Dimension for FFN layer in transformer decoder.
        pre_norm (`bool`, *optional*, defaults to `False`):
            Whether to normalize hidden states before attention layers in transformer decoder.
        enforce_input_proj (`bool`, *optional*, defaults to `False`):
            Whether to project hidden states in transformer decoder.
        query_dec_layers (`int`, *optional*, defaults to 2):
            Number of layers in query transformer.
        common_stride (`int`, *optional*, defaults to 4):
            Common stride used for features in pixel decoder.

    Examples:
    ```python
    >>> from transformers import OneFormerConfig, OneFormerModel

    >>> # Initializing a OneFormer shi-labs/oneformer_ade20k_swin_tiny configuration
    >>> configuration = OneFormerConfig()
    >>> # Initializing a model (with random weights) from the shi-labs/oneformer_ade20k_swin_tiny style configuration
    >>> model = OneFormerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    

## OneFormerImageProcessor


    Constructs a OneFormer image processor. The image processor can be used to prepare image(s), task input(s) and
    optional text inputs and targets for the model.

    This image processor inherits from [`BaseImageProcessor`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 800):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a
            sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of
            the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size *
            height / width, size)`.
        resample (`int`, *optional*, defaults to `Resampling.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
            `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
            `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
            to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input to a certain `scale`.
        rescale_factor (`float`, *optional*, defaults to `1/ 255`):
            Rescale the input by the given factor. Only has an effect if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
        ignore_index (`int`, *optional*):
            Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
            denoted with 0 (background) will be replaced with `ignore_index`.
        do_reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0
            is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k).
            The background label will be replaced by `ignore_index`.
        repo_path (`str`, *optional*, defaults to `"shi-labs/oneformer_demo"`):
            Path to hub repo or local directory containing the JSON file with class information for the dataset.
            If unset, will look for `class_info_file` in the current working directory.
        class_info_file (`str`, *optional*):
            JSON file containing class information for the dataset. See `shi-labs/oneformer_demo/cityscapes_panoptic.json` for an example.
        num_text (`int`, *optional*):
            Number of text entries in the text input list.
        num_labels (`int`, *optional*):
            The number of labels in the segmentation map.
    

Methods: preprocess
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## OneFormerProcessor


    Constructs an OneFormer processor which wraps [`OneFormerImageProcessor`] and
    [`CLIPTokenizer`]/[`CLIPTokenizerFast`] into a single processor that inherits both the image processor and
    tokenizer functionalities.

    Args:
        image_processor ([`OneFormerImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`]):
            The tokenizer is a required input.
        max_seq_len (`int`, *optional*, defaults to 77)):
            Sequence length for input text list.
        task_seq_len (`int`, *optional*, defaults to 77):
            Sequence length for input task token.
    

## OneFormerModel

The bare OneFormer Model outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`OneFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## OneFormerForUniversalSegmentation

OneFormer Model for instance, semantic and panoptic image segmentation.
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`OneFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    