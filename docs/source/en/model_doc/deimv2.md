<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-09-26 and added to Hugging Face Transformers in 2026.*

# DEIMv2

## Overview

The DEIMv2 model was introduced in [Real-Time Object Detection Meets DINOv3](https://arxiv.org/abs/2509.20787) by
Shihua Huang, Yongjie Hou, Longfei Liu, Xuanlong Yu, and Xi Shen.

DEIMv2 is a real-time object detector built on top of a DETR-style architecture. It combines DEIM/RT-DETR-style
decoder design with stronger visual backbones (including DINOv3-based variants) to improve the speed/accuracy tradeoff
for object detection.

The abstract from the paper is the following:

*DEIMv2 is an evolution of the DEIM framework while leveraging the rich features from DINOv3. Our method is designed
with various model sizes, from an ultra-light version up to S, M, L, and X, to be adaptable for a wide range of
scenarios. Across these variants, DEIMv2 achieves state-of-the-art performance, with the S-sized model notably
surpassing 50 AP on the challenging COCO benchmark.*

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/Intellindust-AI-Lab/DEIMv2).

## Usage tips

```python
>>> import torch
>>> from transformers import AutoImageProcessor, Deimv2ForObjectDetection
>>> from transformers.image_utils import load_image

>>> image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")

>>> image_processor = AutoImageProcessor.from_pretrained("Intellindust/DEIMv2_HGNetv2_ATTO_COCO")
>>> model = Deimv2ForObjectDetection.from_pretrained("Intellindust/DEIMv2_HGNetv2_ATTO_COCO")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = image_processor.post_process_object_detection(
...     outputs, threshold=0.5, target_sizes=[(image.height, image.width)]
... )[0]

>>> for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
...     score = score.item()
...     label = model.config.id2label[label_id.item()]
...     box = [round(x, 2) for x in box.tolist()]
...     print(label, score, box)
```

## Deimv2Config

[[autodoc]] Deimv2Config

## Deimv2ForObjectDetection

[[autodoc]] Deimv2ForObjectDetection
    - forward

## Deimv2Model

[[autodoc]] Deimv2Model
    - forward

## Deimv2PreTrainedModel

[[autodoc]] Deimv2PreTrainedModel
    - forward

## Deimv2ImageProcessor

[[autodoc]] Deimv2ImageProcessor

## Deimv2ImageProcessorFast

[[autodoc]] Deimv2ImageProcessorFast
