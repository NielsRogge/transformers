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

# Idefics2

## Overview

The Idefics2 model was proposed in [What matters when building vision-language models?](https://arxiv.org/abs/2405.02246) by Léo Tronchon, Hugo Laurencon, Victor Sanh. The accompanying blog post can be found [here](https://huggingface.co/blog/idefics2).

Idefics2 is an open multimodal model that accepts arbitrary sequences of image and text inputs and produces text
outputs. The model can answer questions about images, describe visual content, create stories grounded on multiple
images, or simply behave as a pure language model without visual inputs. It improves upon IDEFICS-1, notably on
document understanding, OCR, or visual reasoning. Idefics2 is lightweight (8 billion parameters) and treats
images in their native aspect ratio and resolution, which allows for varying inference efficiency.

The abstract from the paper is the following:

*The growing interest in vision-language models (VLMs) has been driven by improvements in large language models and vision transformers. Despite the abundance of literature on this subject, we observe that critical decisions regarding the design of VLMs are often not justified. We argue that these unsupported decisions impede progress in the field by making it difficult to identify which choices improve model performance. To address this issue, we conduct extensive experiments around pre-trained models, architecture choice, data, and training methods. Our consolidation of findings includes the development of Idefics2, an efficient foundational VLM of 8 billion parameters. Idefics2 achieves state-of-the-art performance within its size category across various multimodal benchmarks, and is often on par with models four times its size. We release the model (base, instructed, and chat) along with the datasets created for its training.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/idefics2_architecture.png"
alt="drawing" width="600"/>

<small> Idefics2 architecture. Taken from the <a href="https://arxiv.org/abs/2405.02246">original paper.</a> </small>

This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts).
The original code can be found [here](https://huggingface.co/HuggingFaceM4/idefics2).

## Usage tips

- Each sample can contain multiple images, and the number of images can vary between samples. The processor will pad the inputs to the maximum number of images in a batch for input to the model.
- The processor has a `do_image_splitting` option. If `True`, each input image will be split into 4 sub-images, and concatenated with the original to form 5 images. This is useful for increasing model performance. Make sure `processor.image_processor.do_image_splitting` is set to `False` if the model was not trained with this option.
- `text` passed to the processor should have the `<image>` tokens where the images should be inserted. And `<end_of_utterance>` at the end of each utterance if the text is a chat message.
- The processor has its own `apply_chat_template` method to convert chat messages to text that can then be passed as `text` to the processor.

Example of how to use the processor on chat messages:

```python
import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, image_2]

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What’s the difference between these two images?"},
        {"type": "image"},
        {"type": "image"},
    ],
}]

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")
model.to(device)

# at inference time, one needs to pass `add_generation_prompt=True` in order to make sure the model completes the prompt
text = processor.apply_chat_template(messages, add_generation_prompt=True)
print(text)
# 'User: What’s the difference between these two images?<image><image><end_of_utterance>\nAssistant:'

inputs = processor(images=images, text=text, return_tensors="pt").to(device)

generated_text = model.generate(**inputs, max_new_tokens=500)
generated_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
print("Generated text:", generated_text)
```

- During training, it's important to determine which tokens the model should not learn. For Idefics2, this typically comes down to the image and padding tokens. This means that one can create the labels as follows:

```python
import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, image_2]

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What’s the difference between these two images?"},
        {"type": "image"},
        {"type": "image"},
    ],
},
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "The difference is that one image is about dogs and the other one about cats."},
    ],
}]

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")
model.to(device)

text = processor.apply_chat_template(messages, add_generation_prompt=False)
inputs = processor(images=images, text=text, return_tensors="pt").to(device)

labels = inputs.input_ids.clone()
labels[labels == processor.tokenizer.pad_token_id] = -100
labels[labels == model.config.image_token_id] = -100

inputs["labels"] = labels

outputs = model(**inputs)
loss = outputs.loss
loss.backward()
```

Do note that when training Idefics2 on multi-turn conversations between a user and an assistant, one typically also sets all the tokens corresponding to the user messages to -100.

## Model optimizations: Flash Attention

The code snippets above showcase inference without any optimization tricks. However, one can drastically speed up the model by leveraging [Flash Attention](../perf_train_gpu_one#flash-attention-2), which is a faster implementation of the attention mechanism used inside the model.

First, make sure to install the latest version of Flash Attention 2 to include the sliding window attention feature.

```bash
pip install -U flash-attn --no-build-isolation
```

Make also sure that you have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). Make also sure to load your model in half-precision (e.g. `torch.float16`)

To load and run a model using Flash Attention-2, simply change the code snippet above with the following change:

```diff
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
+    torch_dtype=torch.float16,    
+    attn_implementation="flash_attention_2",
).to(device)
```

## Shrinking down Idefics2 using quantization

As the Idefics2 model has 8 billion parameters, that would require about 16GB of GPU RAM in half precision (float16), since each parameter is stored in 2 bytes. However, one can shrink down the size of the model using [quantization](../quantization.md). If the model is quantized to 4 bits (or half a byte per parameter), that requires only about 3.5GB of RAM.

Quantizing a model is as simple as passing a `quantization_config` to the model. One can change the code snippet above with the changes below. We'll leverage the BitsAndyBytes quantization (but refer to [this page](../quantization.md) for other quantization methods):

```diff
+ from transformers import BitsAndBytesConfig

+ quantization_config = BitsAndBytesConfig(
+    load_in_4bit=True,
+    bnb_4bit_quant_type="nf4",
+    bnb_4bit_use_double_quant=True,
+    bnb_4bit_compute_dtype=torch.float16
+ )
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
+    torch_dtype=torch.float16,    
+    quantization_config=quantization_config,
).to(device)
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Idefics2. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- A notebook on how to fine-tune Idefics2 on a custom dataset using the [Trainer](../main_classes/trainer.md) can be found [here](https://colab.research.google.com/drive/1NtcTgRbSBKN7pYD3Vdx1j9m8pt3fhFDB?usp=sharing). It supports both full fine-tuning as well as (quantized) LoRa.
- A script regarding how to fine-tune Idefics2 using the TRL library can be found [here](https://gist.github.com/edbeeching/228652fc6c2b29a1641be5a5778223cb).
- Demo notebook regarding fine-tuning Idefics2 for JSON extraction use cases can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Idefics2). 🌎

## Idefics2Config


    This is the configuration class to store the configuration of a [`Idefics2Model`]. It is used to instantiate a
    Idefics2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the model of the Idefics2
    [HuggingFaceM4/idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should cache the key/value pairs of the attention mechanism.
        image_token_id (`int`, *optional*, defaults to 32001):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`IdeficsVisionConfig` or `dict`, *optional*):
            Custom vision config or dict
        perceiver_config (`IdeficsPerceiverConfig` or `dict`, *optional*):
            Custom perceiver config or dict
        text_config (`MistralConfig` or `dict`, *optional*):
            Custom text config or dict for the text model

    Example:
    ```python
    >>> from transformers import Idefics2Model, Idefics2Config
    >>> # Initializing configuration
    >>> configuration = Idefics2Config()
    >>> # Initializing a model from the configuration
    >>> model = Idefics2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```


## Idefics2Model

Idefics2 model consisting of a SIGLIP vision encoder and Mistral language decoder
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Idefics2Config`] or [`Idefics2VisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward


## Idefics2ForConditionalGeneration

The Idefics2 Model with a language modeling head. It is made up a SigLIP vision encoder, with a language modeling head on top. 
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Idefics2Config`] or [`Idefics2VisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward



    Constructs a Idefics image processor.

    Args:
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB. This is useful if the input image is of a different format e.g. RGBA.
            Only has an effect if the input image is in the PIL format.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image. The longest edge of the image is resized to  be <= `size["longest_edge"]`, with the
            shortest edge resized to keep the input aspect ratio, with a minimum size of `size["shortest_edge"]`.
        size (`Dict`, *optional*):
            Controls the size of the output image. This is a dictionary containing the keys "shortest_edge" and "longest_edge".
        resample (`Resampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image. If set to `True`, the image is rescaled to have pixel values between 0 and 1.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. If set to `True`, the image is normalized to have a mean of `image_mean` and
            a standard deviation of `image_std`.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether or not to pad the images to the largest height and width in the batch and number of images per
            sample in the batch, such that the returned tensor is of shape (batch_size, max_num_images, num_channels, max_height, max_width).
        do_image_splitting (`bool`, *optional*, defaults to `False`):
            Whether to split the image into a sequence 4 equal sub-images concatenated with the original image. That
            strategy was first introduced in https://arxiv.org/abs/2311.06607.
    

Methods: preprocess



    Constructs a IDEFICS2 processor which wraps a LLama tokenizer and IDEFICS2 image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`Idefics2ImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`Idefics2ImageProcessor`):
            An instance of [`Idefics2ImageProcessor`]. The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
        image_seq_len (`int`, *optional*, defaults to 64):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            config.perceiver_config.resampler_n_latents value for the model used.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    

Methods: __call__