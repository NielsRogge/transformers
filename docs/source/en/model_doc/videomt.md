<!--Copyright 2026 the HuggingFace Team. All rights reserved.

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


# VidEoMT

## Overview

The VidEoMT model was proposed in [<INSERT PAPER NAME HERE>](VidEoMT: Your ViT is Secretly Also a Video Segmentation Model) by Narges Norouzi, Idil Esen Zulfikar, Niccol`o Cavagnero, Tommie Kerssies, Bastian Leibe, Gijs Dubbelman, Daan de Geus. Video Encoder-only Mask Transformer (VidEoMT) is a lightweight encoder-only model for online video segmentation built on a plain [Vision Transformer (ViT)](vit). It is a minimal extension of [EoMT](eomt) to video which performs both spatial and temporal reasoning within the ViT encoder, without relying on dedicated tracking modules or heavy task-specific heads.

The abstract from the paper is the following:

**Existing online video segmentation models typically combine a per-frame segmenter with complex specialized tracking modules. While effective, these modules introduce significant architectural complexity and computational overhead. Recent studies suggest that plain Vision Transformer (ViT) encoders, when scaled with sufficient capacity and large-scale pre-training, can conduct accurate image segmentation without requiring specialized modules. Motivated by this observation, we propose the Video Encoder-only Mask Transformer (VidEoMT), a simple encoder-only video segmentation model that eliminates the need for dedicated tracking modules. To enable temporal modeling in an encoder-only ViT, VidEoMT introduces a lightweight query propagation mechanism that carries information across frames by reusing queries from the previous frame. To balance this with adaptability to new content, it employs a query fusion strategy that combines the propagated queries with a set of temporally-agnostic learned queries. As a result, VidEoMT attains the benefits of a tracker without added complexity, achieving competitive accuracy while being 5x--10x faster, running at up to 160 FPS with a ViT-L backbone.**

Tips:

- To do

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/tue-mps/videomt).

## Usage examples

<INSERT SOME NICE EXAMPLES HERE>

## VideomtConfig

[[autodoc]] VideomtConfig

## VideomtPreTrainedModel

[[autodoc]] VideomtPreTrainedModel
    - forward

## VideomtForUniversalSegmentation

[[autodoc]] VideomtForUniversalSegmentation