<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->

# Nougat

## Overview

The Nougat model was proposed in [Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/abs/2308.13418) by
Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic. Nougat uses the same architecture as [Donut](donut), meaning an image Transformer
encoder and an autoregressive text Transformer decoder to translate scientific PDFs to markdown, enabling easier access to them.

The abstract from the paper is the following:

*Scientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (Neural Optical Understanding for Academic Documents), a Visual Transformer model that performs an Optical Character Recognition (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the accessibility of scientific knowledge in the digital age, by bridging the gap between human-readable documents and machine-readable text. We release the models and code to accelerate future work on scientific text recognition.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/nougat_architecture.jpg"
alt="drawing" width="600"/>

<small> Nougat high-level overview. Taken from the <a href="https://arxiv.org/abs/2308.13418">original paper</a>. </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found
[here](https://github.com/facebookresearch/nougat).

## Usage tips

- The quickest way to get started with Nougat is by checking the [tutorial
  notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Nougat), which show how to use the model
  at inference time as well as fine-tuning on custom data.
- Nougat is always used within the [VisionEncoderDecoder](vision-encoder-decoder) framework. The model is identical to [Donut](donut) in terms of architecture.

## Inference

Nougat's [`VisionEncoderDecoder`] model accepts images as input and makes use of
[`~generation.GenerationMixin.generate`] to autoregressively generate text given the input image.

The [`NougatImageProcessor`] class is responsible for preprocessing the input image and
[`NougatTokenizerFast`] decodes the generated target tokens to the target string. The
[`NougatProcessor`] wraps [`NougatImageProcessor`] and [`NougatTokenizerFast`] classes
into a single instance to both extract the input features and decode the predicted token ids.

- Step-by-step PDF transcription

```py
>>> from huggingface_hub import hf_hub_download
>>> import re
>>> from PIL import Image

>>> from transformers import NougatProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = NougatProcessor.from_pretrained("facebook/nougat-base")
>>> model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # prepare PDF image for the model
>>> filepath = hf_hub_download(repo_id="hf-internal-testing/fixtures_docvqa", filename="nougat_paper.png", repo_type="dataset")
>>> image = Image.open(filepath)
>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> # generate transcription (here we only generate 30 tokens)
>>> outputs = model.generate(
...     pixel_values.to(device),
...     min_length=1,
...     max_new_tokens=30,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
... )

>>> sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
>>> sequence = processor.post_process_generation(sequence, fix_markdown=False)
>>> # note: we're using repr here such for the sake of printing the \n characters, feel free to just print the sequence
>>> print(repr(sequence))
'\n\n# Nougat: Neural Optical Understanding for Academic Documents\n\n Lukas Blecher\n\nCorrespondence to: lblecher@'
```

See the [model hub](https://huggingface.co/models?filter=nougat) to look for Nougat checkpoints.

<Tip>

The model is identical to [Donut](donut) in terms of architecture.

</Tip>

## NougatImageProcessor


    Constructs a Nougat image processor.

    Args:
        do_crop_margin (`bool`, *optional*, defaults to `True`):
            Whether to crop the image margins.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 896, "width": 672}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to resize the image using thumbnail method.
        do_align_long_axis (`bool`, *optional*, defaults to `False`):
            Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the images to the largest image size in the batch.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Image standard deviation.
    

Methods: preprocess

## NougatTokenizerFast


    Fast tokenizer for Nougat (backed by HuggingFace tokenizers library).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods. This class mainly adds Nougat-specific
    methods for postprocessing the generated text.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.

        clean_up_tokenization_spaces (`str`, *optional*, defaults to `False`):
            Wether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra
            spaces.

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.

        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
    
    Class attributes (overridden by derived classes)

        - **vocab_files_names** (`Dict[str, str]`) -- A dictionary with, as keys, the `__init__` keyword name of each
          vocabulary file required by the model, and as associated values, the filename for saving the associated file
          (string).
        - **pretrained_vocab_files_map** (`Dict[str, Dict[str, str]]`) -- A dictionary of dictionaries, with the
          high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
          low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
          associated pretrained vocabulary file.
        - **model_input_names** (`List[str]`) -- A list of inputs expected in the forward pass of the model.
        - **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
          Should be `'right'` or `'left'`.
        - **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
          applied. Should be `'right'` or `'left'`.

    Args:
        model_max_length (`int`, *optional*):
            The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
            loaded with [`~tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained`], this will be set to the
            value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will
            default to VERY_LARGE_INTEGER (`int(1e30)`).
        padding_side (`str`, *optional*):
            The side on which the model should have padding applied. Should be selected between ['right', 'left'].
            Default value is picked from the class attribute of the same name.
        truncation_side (`str`, *optional*):
            The side on which the model should have truncation applied. Should be selected between ['right', 'left'].
            Default value is picked from the class attribute of the same name.
        chat_template (`str`, *optional*):
            A Jinja template string that will be used to format lists of chat messages. See
            https://huggingface.co/docs/transformers/chat_templating for a full description.
        model_input_names (`List[string]`, *optional*):
            The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
            `"attention_mask"`). Default value is picked from the class attribute of the same name.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the beginning of a sentence. Will be associated to `self.bos_token` and
            `self.bos_token_id`.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the end of a sentence. Will be associated to `self.eos_token` and
            `self.eos_token_id`.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing an out-of-vocabulary token. Will be associated to `self.unk_token` and
            `self.unk_token_id`.
        sep_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token separating two different sentences in the same input (used by BERT for instance). Will be
            associated to `self.sep_token` and `self.sep_token_id`.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation. Will be associated to `self.pad_token` and `self.pad_token_id`.
        cls_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the class of the input (used by BERT for instance). Will be associated to
            `self.cls_token` and `self.cls_token_id`.
        mask_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional special tokens. Add them here to ensure they are skipped when decoding with
            `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end
            of the vocabulary.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
            Whether or not the model should cleanup the spaces that were added when splitting the input text during the
            tokenization process.
        split_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the special tokens should be split during the tokenization process. Passing will affect the
            internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
            `<s>` is the `bos_token`, then `tokenizer.tokenize("<s>") = ['<s>`]. Otherwise, if
            `split_special_tokens=True`, then `tokenizer.tokenize("<s>")` will be give `['<','s', '>']`.

        tokenizer_object ([`tokenizers.Tokenizer`]):
            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
            tokenizers](../fast_tokenizers) for more information.
        tokenizer_file ([`str`]):
            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗
            tokenizers.


## NougatProcessor


    Constructs a Nougat processor which wraps a Nougat image processor and a Nougat tokenizer into a single processor.

    [`NougatProcessor`] offers all the functionalities of [`NougatImageProcessor`] and [`NougatTokenizerFast`]. See the
    [`~NougatProcessor.__call__`] and [`~NougatProcessor.decode`] for more information.

    Args:
        image_processor ([`NougatImageProcessor`]):
            An instance of [`NougatImageProcessor`]. The image processor is a required input.
        tokenizer ([`NougatTokenizerFast`]):
            An instance of [`NougatTokenizerFast`]. The tokenizer is a required input.
    

Methods: __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode
    - post_process_generation