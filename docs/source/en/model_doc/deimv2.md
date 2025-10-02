<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->


# DEIMv2

## Overview

The DEIMv2 model was proposed by the Intellindust AI Lab team. It extends the RT-DETR one-stage detector by swapping
the ResNet backbone for a DINOv3 feature extractor enhanced with a Spatial Tuning Adapter (STA) and by supporting the
HGNetV2 backbone for lightweight variants. Multi-scale feature maps produced by the backbone are refined by a hybrid
encoder before a decoder iteratively improves the top-K object proposals.

Tips:

* The implementation currently focuses on providing a modular PyTorch draft that mirrors the RT-DETR API and is ready
  to integrate AutoBackbone-powered backbones.

This model was contributed by [Intellindust AI Lab](https://huggingface.co/Intellindust-AI-Lab).
The original code can be found [here](https://github.com/Intellindust-AI-Lab/DEIMv2).

## Usage examples

```python
>>> import torch
>>> from transformers import Deimv2Config, Deimv2Model

>>> config = Deimv2Config()
>>> model = Deimv2Model(config)
>>> pixel_values = torch.randn(1, 3, 640, 640)
>>> outputs = model(pixel_values=pixel_values)
>>> last_hidden_state = outputs.last_hidden_state
```

## Deimv2Config

[[autodoc]] Deimv2Config

## Deimv2ForObjectDetection

[[autodoc]] Deimv2ForObjectDetection

## Deimv2Model

[[autodoc]] Deimv2Model
    - forward

## Deimv2PreTrainedModel

[[autodoc]] Deimv2PreTrainedModel
    - forward
