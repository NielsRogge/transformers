<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# VisionTextDualEncoder

## Overview

The [`VisionTextDualEncoderModel`] can be used to initialize a vision-text dual encoder model with
any pretrained vision autoencoding model as the vision encoder (*e.g.* [ViT](vit), [BEiT](beit), [DeiT](deit)) and any pretrained text autoencoding model as the text encoder (*e.g.* [RoBERTa](roberta), [BERT](bert)). Two projection layers are added on top of both the vision and text encoder to project the output embeddings
to a shared latent space. The projection layers are randomly initialized so the model should be fine-tuned on a
downstream task. This model can be used to align the vision-text embeddings using CLIP like contrastive image-text
training and then can be used for zero-shot vision tasks such image-classification or retrieval.

In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how
leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvement on
new zero-shot vision tasks such as image classification or retrieval.

## VisionTextDualEncoderConfig


    [`VisionTextDualEncoderConfig`] is the configuration class to store the configuration of a
    [`VisionTextDualEncoderModel`]. It is used to instantiate [`VisionTextDualEncoderModel`] model according to the
    specified arguments, defining the text model and vision model configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Examples:

    ```python
    >>> from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig, VisionTextDualEncoderModel

    >>> # Initializing a BERT and ViT configuration
    >>> config_vision = ViTConfig()
    >>> config_text = BertConfig()

    >>> config = VisionTextDualEncoderConfig.from_vision_text_configs(config_vision, config_text, projection_dim=512)

    >>> # Initializing a BERT and ViT model (with random weights)
    >>> model = VisionTextDualEncoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_vision = model.config.vision_config
    >>> config_text = model.config.text_config

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("vit-bert")

    >>> # loading model and config from pretrained folder
    >>> vision_text_config = VisionTextDualEncoderConfig.from_pretrained("vit-bert")
    >>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert", config=vision_text_config)
    ```

## VisionTextDualEncoderProcessor


    Constructs a VisionTextDualEncoder processor which wraps an image processor and a tokenizer into a single
    processor.

    [`VisionTextDualEncoderProcessor`] offers all the functionalities of [`AutoImageProcessor`] and [`AutoTokenizer`].
    See the [`~VisionTextDualEncoderProcessor.__call__`] and [`~VisionTextDualEncoderProcessor.decode`] for more
    information.

    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The tokenizer is a required input.
    

<frameworkcontent>
<pt>

## VisionTextDualEncoderModel


    This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model
    as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded
    via the [`~AutoModel.from_pretrained`] method. The projection layers are automatically added to the model and
    should be fine-tuned on a downstream task, like contrastive image-text modeling.

    In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how
    leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment
    on new zero-shot vision tasks such as image classification or retrieval.

    After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other
    models (see the examples for more information).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

</pt>
<tf>

## FlaxVisionTextDualEncoderModel

No docstring available for FlaxVisionTextDualEncoderModel

Methods: __call__

</tf>
<jax>

## TFVisionTextDualEncoderModel

No docstring available for TFVisionTextDualEncoderModel

Methods: call

</jax>
</frameworkcontent>
