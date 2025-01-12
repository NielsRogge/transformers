<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TimmWrapper

## Overview

Helper class to enable loading timm models to be used with the transformers library and its autoclasses.

```python
>>> import torch
>>> from PIL import Image
>>> from urllib.request import urlopen
>>> from transformers import AutoModelForImageClassification, AutoImageProcessor

>>> # Load image
>>> image = Image.open(urlopen(
...     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
... ))

>>> # Load model and image processor
>>> checkpoint = "timm/resnet50.a1_in1k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModelForImageClassification.from_pretrained(checkpoint).eval()

>>> # Preprocess image
>>> inputs = image_processor(image)

>>> # Forward pass
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # Get top 5 predictions
>>> top5_probabilities, top5_class_indices = torch.topk(logits.softmax(dim=1) * 100, k=5)
```

## TimmWrapperConfig


    This is the configuration class to store the configuration for a timm backbone [`TimmWrapper`].

    It is used to instantiate a timm model according to the specified arguments, defining the model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Config loads imagenet label descriptions and stores them in `id2label` attribute, `label2id` attribute for default
    imagenet models is set to `None` due to occlusions in the label descriptions.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        do_pooling (`bool`, *optional*, defaults to `True`):
            Whether to do pooling for the last_hidden_state in `TimmWrapperModel` or not.

    Example:
    ```python
    >>> from transformers import TimmWrapperModel

    >>> # Initializing a timm model
    >>> model = TimmWrapperModel.from_pretrained("timm/resnet18.a1_in1k")

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    

## TimmWrapperImageProcessor


    Wrapper class for timm models to be used within transformers.

    Args:
        pretrained_cfg (`Dict[str, Any]`):
            The configuration of the pretrained model used to resolve evaluation and
            training transforms.
        architecture (`Optional[str]`, *optional*):
            Name of the architecture of the model.
    

Methods: preprocess

## TimmWrapperModel


    Wrapper class for timm models to be used in transformers.
    

Methods: forward

## TimmWrapperForImageClassification


    Wrapper class for timm models to be used in transformers for image classification.
    

Methods: forward
