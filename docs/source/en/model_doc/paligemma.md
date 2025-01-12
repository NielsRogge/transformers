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

# PaliGemma

## Overview

The PaliGemma model was proposed in [PaliGemma – Google's Cutting-Edge Open Vision Language Model](https://huggingface.co/blog/paligemma) by Google. It is a 3B vision-language model composed by a [SigLIP](siglip) vision encoder and a [Gemma](gemma) language decoder linked by a multimodal linear projection. It cuts an image into a fixed number of VIT tokens and prepends it to an optional prompt. One particularity is that the model uses full block attention on all the image tokens plus the input text tokens. It comes in 3 resolutions, 224x224, 448x448 and 896x896 with 3 base models, with 55 fine-tuned versions for different tasks, and 2 mix models.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/paligemma/paligemma_arch.png"
alt="drawing" width="600"/>

<small> PaliGemma architecture. Taken from the <a href="https://huggingface.co/blog/paligemma">blog post.</a> </small>

This model was contributed by [Molbap](https://huggingface.co/Molbap).

## Usage tips

- PaliGemma is not meant for conversational use, and it works best when fine-tuning to a specific use case. Some downstream tasks on which PaliGemma can be fine-tuned include image captioning, visual question answering (VQA), object detection, referring expression segmentation and document understanding.
- One can use `PaliGemmaProcessor` to prepare images, text and optional labels for the model. When fine-tuning a PaliGemma model, the `suffix` argument can be passed to the processor which creates the `labels` for the model:

```python
prompt = "What is on the flower?"
answer = "a bee"
inputs = processor(images=raw_image, text=prompt, suffix=answer, return_tensors="pt")
```

## Usage Example

The model can accept a single or multiple images. According to the [paper](https://arxiv.org/abs/2407.07726v1), the checkpoint PaliGemma can transfer to tasks which take multiple images as input. NLVR2 is one such task, which asks one question about two images, and requires looking at both to give the correct answer. Here's an example code for single and multi image inference.

### Single-image Inference

```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "What is on the flower?"
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(raw_image, prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20)

print(processor.decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ])
```

### Multi-image Inference

```python
model_id = "google/paligemma-3b-ft-nlvr2-448"  # checkpoint tuned for multiple images
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = PaliGemmaProcessor.from_pretrained(model_id)

prompt = "answer en Which of the two pictures shows a snowman, first or second?"
stop_sign_image = Image.open(
    requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw
)
snow_image = Image.open(
    requests.get(
        "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg", stream=True
    ).raw
)

inputs = processor(images=[[snow_image, stop_sign_image]], text=prompt, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=20)
print(processor.decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ])

```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with PaliGemma. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- A blog post introducing all the features of PaliGemma can be found [here](https://huggingface.co/blog/paligemma).
- Demo notebooks on how to fine-tune PaliGemma for VQA with the Trainer API along with inference can be found [here](https://github.com/huggingface/notebooks/tree/main/examples/paligemma).
- Demo notebooks on how to fine-tune PaliGemma on a custom dataset (receipt image -> JSON) along with inference can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma). 🌎

## PaliGemmaConfig


    This is the configuration class to store the configuration of a [`PaliGemmaForConditionalGeneration`]. It is used to instantiate an
    PaliGemmamodel according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PaliGemma-2B.

    e.g. [paligemma-hf/paligemma-2b](https://huggingface.co/paligemma-hf/paligemma-2b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`PaliGemmaVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 256000):
            The image token index to encode the image prompt.
        vocab_size (`int`, *optional*, defaults to 257152):
            Vocabulary size of the PaliGemmamodel. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~PaliGemmaForConditionalGeneration`]
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimension of the multimodal projection space.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden layer of the Language model.

    Example:

    ```python
    >>> from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig, SiglipVisionConfig, GemmaConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a PaliGemma config
    >>> text_config = GemmaConfig()

    >>> # Initializing a PaliGemma paligemma-3b-224 style configuration
    >>> configuration = PaliGemmaConfig(vision_config, text_config)

    >>> # Initializing a model from the paligemma-3b-224 style configuration
    >>> model = PaliGemmaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## PaliGemmaProcessor


    Constructs a PaliGemma processor which wraps a PaliGemma image processor and a PaliGemma tokenizer into a single processor.

    [`PaliGemmaProcessor`] offers all the functionalities of [`SiglipImageProcessor`] and [`GemmaTokenizerFast`]. See the
    [`~PaliGemmaProcessor.__call__`] and [`~PaliGemmaProcessor.decode`] for more information.

    Args:
        image_processor ([`SiglipImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`GemmaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    

## PaliGemmaForConditionalGeneration

The PALIGEMMA model which consists of a vision backbone and a language model.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PaliGemmaConfig`] or [`PaliGemmaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
