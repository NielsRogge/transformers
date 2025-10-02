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

DEIMv2 is an object detection architecture released by the Intellindust AI Lab. It builds on top of a transformer-based detection
head reminiscent of DETR while leveraging a DINOv3 Vision Transformer backbone for high-quality multi-scale image features. The
official repository with training and evaluation utilities is available on [GitHub](https://github.com/Intellindust-AI-Lab/DEIMv2).

At the time of writing no formal paper has been published. The maintainers describe the approach in the repository documentation,
highlighting the focus on real-time inference and strong zero-shot transfer.

Tips:

- The reference implementation pre-trains the DINOv3 backbone separately. In Transformers you can configure
  [`Deimv2Config.backbone`] to point to an available DINOv3 checkpoint once weights are converted.
- The config exposes `backbone_out_features` to control which DINOv3 hidden states are transformed into spatial feature maps for
  the detection head.

This model was contributed by [gpt-5-codex](https://huggingface.co/gpt-5-codex). The original code can be found in the
[Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) repository.

## Usage examples

```python
>>> from transformers import AutoImageProcessor, Deimv2ForObjectDetection
>>> import torch

>>> model = Deimv2ForObjectDetection(Deimv2Config())
>>> processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

>>> dummy_image = torch.rand(3, 64, 64)
>>> inputs = processor(images=dummy_image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits  # object class predictions
>>> boxes = outputs.pred_boxes  # normalized bounding boxes
```

## Deimv2Config

[[autodoc]] Deimv2Config

## Deimv2OnnxConfig

[[autodoc]] Deimv2OnnxConfig

## Deimv2ForObjectDetection

[[autodoc]] Deimv2ForObjectDetection

## Deimv2ForSegmentation

[[autodoc]] Deimv2ForSegmentation

## Deimv2Model

[[autodoc]] Deimv2Model
    - forward

## Deimv2PreTrainedModel

[[autodoc]] Deimv2PreTrainedModel
    - forward