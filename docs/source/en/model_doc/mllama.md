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

# Mllama

## Overview

The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text \+ images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image.

**Model Architecture:** Llama 3.2-Vision is built on top of Llama 3.1 text-only model, which is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety. To support image recognition tasks, the Llama 3.2-Vision model uses a separately trained vision adapter that integrates with the pre-trained Llama 3.1 language model. The adapter consists of a series of cross-attention layers that feed image encoder representations into the core LLM.

## Usage Tips

- For image+text and text inputs use `MllamaForConditionalGeneration`.
- For text-only inputs use `MllamaForCausalLM` for generation to avoid loading vision tower.
- Each sample can contain multiple images, and the number of images can vary between samples. The processor will pad the inputs to the maximum number of images across samples and to a maximum number of tiles within each image.
- The text passed to the processor should have the `"<|image|>"` tokens where the images should be inserted.
- The processor has its own `apply_chat_template` method to convert chat messages to text that can then be passed as text to the processor.


<Tip warning={true}>

Mllama has an extra token used as a placeholder for image positions in the text. It means that input ids and an input embedding layer will have an extra token. But since the weights for input and output embeddings are not tied, the `lm_head` layer has one less token and will fail if you want to calculate loss on image tokens or apply some logit processors. In case you are training, make sure to mask out special `"<|image|>"` tokens in the `labels` as the model should not be trained on predicting them.

Otherwise if you see CUDA-side index erros when generating, use the below code to expand the `lm_head` by one more token. 


```python
old_embeddings = model.get_output_embeddings()

num_tokens = model.vocab_size + 1
resized_embeddings = model._get_resized_lm_head(old_embeddings, new_num_tokens=num_tokens, mean_resizing=True)
resized_embeddings.requires_grad_(old_embeddings.weight.requires_grad)
model.set_output_embeddings(resized_embeddings)
```
</Tip>


## Usage Example

#### Instruct model
```python
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What does the image show?"}
            ]
        }
    ],
]
text = processor.apply_chat_template(messages, add_generation_prompt=True)

url = "https://llava-vl.github.io/static/images/view.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=25)
print(processor.decode(output[0]))
```

#### Base model
```python
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision"
model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "<|image|>If I had to write a haiku for this one"
url = "https://llava-vl.github.io/static/images/view.jpg"
raw_image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, do_sample=False, max_new_tokens=25)
print(processor.decode(output[0], skip_special_tokens=True))
```


## MllamaConfig


    This is the configuration class to store the configuration of a [`MllamaForConditionalGeneration`]. It is used to instantiate an
    Mllama model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-9B.

    e.g. [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MllamaVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MllamaTextConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 128256):
            The image token index to encode the image prompt.

    Example:

    ```python
    >>> from transformers import MllamaForConditionalGeneration, MllamaConfig, MllamaVisionConfig, MllamaTextConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = MllamaVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = MllamaTextConfig()

    >>> # Initializing a mllama-11b style configuration
    >>> configuration = MllamaConfig(vision_config, text_config)

    >>> # Initializing a model from the mllama-11b style configuration
    >>> model = MllamaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## MllamaProcessor


    Constructs a Mllama processor which wraps [`MllamaImageProcessor`] and
    [`PretrainedTokenizerFast`] into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~MllamaProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more
    information.
    The preferred way of passing kwargs is as a dictionary per modality, see usage example below.
        ```python
        from transformers import MllamaProcessor
        from PIL import Image

        processor = MllamaProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

        processor(
            images=your_pil_image,
            text=["<|image|>If I had to write a haiku for this one"],
            images_kwargs = {"size": {"height": 448, "width": 448}},
            text_kwargs = {"padding": "right"},
            common_kwargs = {"return_tensors": "pt"},
        )
        ```

    Args:
        image_processor ([`MllamaImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`]):
            The tokenizer is a required input.

    


## MllamaImageProcessor


    Constructs a Mllama image processor.

    Args:
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB. This is useful if the input image is of a different format e.g. RGBA.
            Only has an effect if the input image is in the PIL format.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Size of the image tile. Should be a dictionary containing 'height' and 'width' keys, both with integer values.
            The height and width values should be equal.
        resample (`int`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
            has an effect if `do_resize` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to 0.0):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
            `True`.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether or not to pad the images to the largest height and width in the batch.
        max_image_tiles (`int`, *optional*, defaults to 4):
            The maximum number of tiles to split the image into.
    

## MllamaForConditionalGeneration

The Mllama model which consists of a vision encoder and a language model.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MllamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MllamaForCausalLM

The Mllama Text Model with a language modeling head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MllamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MllamaTextModel

The Mllama Text Model which consists of transformer with self and cross attention layers.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MllamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MllamaForCausalLM

The Mllama Text Model with a language modeling head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MllamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## MllamaVisionModel

The Mllama Vision Model which consists of two vision encoders.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MllamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
