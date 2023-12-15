<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Vip-LLaVa

## Overview

The Vip-LLaVa model was proposed in [Making Large Multimodal Models Understand Arbitrary Visual Prompts](https://arxiv.org/abs/2312.00784) by Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, Yong Jae Lee.

Vip-LLaVa enhances the training protocol of [LLaVa](llava), a multimodal LLM, by marking images and interact with the model using natural cues like a "red bounding box" or "pointed arrow" during training.

The abstract from the paper is the following:

*While existing large vision-language multimodal models focus on whole image understanding, there is a prominent gap in achieving region-specific comprehension. Current approaches that use textual coordinates or spatial encodings often fail to provide a user-friendly interface for visual prompting. To address this challenge, we introduce a novel multimodal model capable of decoding arbitrary visual prompts. This allows users to intuitively mark images and interact with the model using natural cues like a "red bounding box" or "pointed arrow". Our simple design directly overlays visual markers onto the RGB image, eliminating the need for complex region encodings, yet achieves state-of-the-art performance on region-understanding tasks like Visual7W, PointQA, and Visual Commonsense Reasoning benchmark. Furthermore, we present ViP-Bench, a comprehensive benchmark to assess the capability of models in understanding visual prompts across multiple dimensions, enabling future research in this domain. Code, data, and model are publicly available.*

## Usage tips

- The architecture is similar to the LLaVa architecture except that the multi-modal projector takes a set of concatenated vision hidden states and has an additional layernorm layer on that module.

- We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to call `processor.tokenizer.padding_side = "left"` before generating.

- Note the model has not been explicitly trained to process multiple images in the same prompt, although this is technically possible, you may experience inaccurate results.

- For better results, we recommend users to prompt the model with the correct prompt format: 

```bash
"USER: <image>\n<prompt>ASSISTANT:"
```

For multiple turns conversation:

```bash
"USER: <image>\n<prompt1>ASSISTANT: <answer1>USER: <prompt2>ASSISTANT: <answer2>USER: <prompt3>ASSISTANT:"
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vipllava_architecture.png"
alt="drawing" width="600"/>

<small> Vip-LLaVa overview. Taken from the <a href="https://arxiv.org/abs/2304.08485">original paper.</a> </small>

The original code can be found [here](https://github.com/mu-cai/ViP-LLaVA).

This model was contributed by [Younes Belkada](https://huggingface.co/ybelkada)

### Using Flash Attention 2

Flash Attention 2 is an even faster, optimized version of the previous optimization, please refer to the [Flash Attention 2 section of performance docs](https://huggingface.co/docs/transformers/perf_infer_gpu_one).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BEiT.

<PipelineTag pipeline="image-to-text"/>

- A [Google Colab demo](https://colab.research.google.com/drive/1qsl6cd2c8gGtEW1xV5io7S8NHh-Cp1TV?usp=sharing) on how to run Llava on a free-tier Google colab instance leveraging 4-bit inference. One can simply replace `LlavaForConditionalGeneration` by `VipLlavaForConditionalGeneration`.
- A [similar notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Vip-LLaVa/Inference_with_LLaVa_for_multimodal_generation.ipynb) showcasing batched inference. 🌎

## VipLlavaConfig

[[autodoc]] VipLlavaConfig

## VipLlavaForConditionalGeneration

[[autodoc]] VipLlavaForConditionalGeneration
    - forward
