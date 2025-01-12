<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->

# Donut

## Overview

The Donut model was proposed in [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) by
Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park.
Donut consists of an image Transformer encoder and an autoregressive text Transformer decoder to perform document understanding
tasks such as document image classification, form understanding and visual question answering.

The abstract from the paper is the following:

*Understanding document images (e.g., invoices) is a core but challenging task since it requires complex functions such as reading text and a holistic understanding of the document. Current Visual Document Understanding (VDU) methods outsource the task of reading text to off-the-shelf Optical Character Recognition (OCR) engines and focus on the understanding task with the OCR outputs. Although such OCR-based approaches have shown promising performance, they suffer from 1) high computational costs for using OCR; 2) inflexibility of OCR models on languages or types of document; 3) OCR error propagation to the subsequent process. To address these issues, in this paper, we introduce a novel OCR-free VDU model named Donut, which stands for Document understanding transformer. As the first step in OCR-free VDU research, we propose a simple architecture (i.e., Transformer) with a pre-training objective (i.e., cross-entropy loss). Donut is conceptually simple yet effective. Through extensive experiments and analyses, we show a simple OCR-free VDU model, Donut, achieves state-of-the-art performances on various VDU tasks in terms of both speed and accuracy. In addition, we offer a synthetic data generator that helps the model pre-training to be flexible in various languages and domains.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/donut_architecture.jpg"
alt="drawing" width="600"/>

<small> Donut high-level overview. Taken from the <a href="https://arxiv.org/abs/2111.15664">original paper</a>. </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found
[here](https://github.com/clovaai/donut).

## Usage tips

- The quickest way to get started with Donut is by checking the [tutorial
  notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut), which show how to use the model
  at inference time as well as fine-tuning on custom data.
- Donut is always used within the [VisionEncoderDecoder](vision-encoder-decoder) framework.

## Inference examples

Donut's [`VisionEncoderDecoder`] model accepts images as input and makes use of
[`~generation.GenerationMixin.generate`] to autoregressively generate text given the input image.

The [`DonutImageProcessor`] class is responsible for preprocessing the input image and
[`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`] decodes the generated target tokens to the target string. The
[`DonutProcessor`] wraps [`DonutImageProcessor`] and [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]
into a single instance to both extract the input features and decode the predicted token ids.

- Step-by-step Document Image Classification

```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # load document image
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[1]["image"]

>>> # prepare decoder inputs
>>> task_prompt = "<s_rvlcdip>"
>>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
>>> print(processor.token2json(sequence))
{'class': 'advertisement'}
```

- Step-by-step Document Parsing

```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # load document image
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[2]["image"]

>>> # prepare decoder inputs
>>> task_prompt = "<s_cord-v2>"
>>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
>>> print(processor.token2json(sequence))
{'menu': {'nm': 'CINNAMON SUGAR', 'unitprice': '17,000', 'cnt': '1 x', 'price': '17,000'}, 'sub_total': {'subtotal_price': '17,000'}, 'total': {'total_price': '17,000', 'cashprice': '20,000', 'changeprice': '3,000'}}
```

- Step-by-step Document Visual Question Answering (DocVQA)

```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # load document image from the DocVQA dataset
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[0]["image"]

>>> # prepare decoder inputs
>>> task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
>>> question = "When is the coffee break?"
>>> prompt = task_prompt.replace("{user_input}", question)
>>> decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
>>> print(processor.token2json(sequence))
{'question': 'When is the coffee break?', 'answer': '11-14 to 11:39 a.m.'}
```

See the [model hub](https://huggingface.co/models?filter=donut) to look for Donut checkpoints.

## Training

We refer to the [tutorial notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut).

## DonutSwinConfig


    This is the configuration class to store the configuration of a [`DonutSwinModel`]. It is used to instantiate a
    Donut model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Donut
    [naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embed_dim (`int`, *optional*, defaults to 96):
            Dimensionality of patch embedding.
        depths (`list(int)`, *optional*, defaults to `[2, 2, 6, 2]`):
            Depth of each layer in the Transformer encoder.
        num_heads (`list(int)`, *optional*, defaults to `[3, 6, 12, 24]`):
            Number of attention heads in each layer of the Transformer encoder.
        window_size (`int`, *optional*, defaults to 7):
            Size of windows.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of MLP hidden dimensionality to embedding dimensionality.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not a learnable bias should be added to the queries, keys and values.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        use_absolute_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to add absolute position embeddings to the patch embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import DonutSwinConfig, DonutSwinModel

    >>> # Initializing a Donut naver-clova-ix/donut-base style configuration
    >>> configuration = DonutSwinConfig()

    >>> # Randomly initializing a model from the naver-clova-ix/donut-base style configuration
    >>> model = DonutSwinModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## DonutImageProcessor


    Constructs a Donut image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to resize the image using thumbnail method.
        do_align_long_axis (`bool`, *optional*, defaults to `False`):
            Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image. If `random_padding` is set to `True` in `preprocess`, each image is padded with a
            random amont of padding on each size, up to the largest image size in the batch. Otherwise, all images are
            padded to the largest image size in the batch.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Image standard deviation.
    

Methods: preprocess

## DonutFeatureExtractor

No docstring available for DonutFeatureExtractor

Methods: __call__

## DonutProcessor


    Constructs a Donut processor which wraps a Donut image processor and an XLMRoBERTa tokenizer into a single
    processor.

    [`DonutProcessor`] offers all the functionalities of [`DonutImageProcessor`] and
    [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. See the [`~DonutProcessor.__call__`] and
    [`~DonutProcessor.decode`] for more information.

    Args:
        image_processor ([`DonutImageProcessor`], *optional*):
            An instance of [`DonutImageProcessor`]. The image processor is a required input.
        tokenizer ([`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`], *optional*):
            An instance of [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. The tokenizer is a required input.
    

Methods: __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## DonutSwinModel

The bare Donut Swin Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DonutSwinConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
