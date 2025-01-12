<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# LayoutLMV2

## Overview

The LayoutLMV2 model was proposed in [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740) by Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu,
Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou. LayoutLMV2 improves [LayoutLM](layoutlm) to obtain
state-of-the-art results across several document image understanding benchmarks:

- information extraction from scanned documents: the [FUNSD](https://guillaumejaume.github.io/FUNSD/) dataset (a
  collection of 199 annotated forms comprising more than 30,000 words), the [CORD](https://github.com/clovaai/cord)
  dataset (a collection of 800 receipts for training, 100 for validation and 100 for testing), the [SROIE](https://rrc.cvc.uab.es/?ch=13) dataset (a collection of 626 receipts for training and 347 receipts for testing)
  and the [Kleister-NDA](https://github.com/applicaai/kleister-nda) dataset (a collection of non-disclosure
  agreements from the EDGAR database, including 254 documents for training, 83 documents for validation, and 203
  documents for testing).
- document image classification: the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset (a collection of
  400,000 images belonging to one of 16 classes).
- document visual question answering: the [DocVQA](https://arxiv.org/abs/2007.00398) dataset (a collection of 50,000
  questions defined on 12,000+ document images).

The abstract from the paper is the following:

*Pre-training of text and layout has proved effective in a variety of visually-rich document understanding tasks due to
its effective model architecture and the advantage of large-scale unlabeled scanned/digital-born documents. In this
paper, we present LayoutLMv2 by pre-training text, layout and image in a multi-modal framework, where new model
architectures and pre-training tasks are leveraged. Specifically, LayoutLMv2 not only uses the existing masked
visual-language modeling task but also the new text-image alignment and text-image matching tasks in the pre-training
stage, where cross-modality interaction is better learned. Meanwhile, it also integrates a spatial-aware self-attention
mechanism into the Transformer architecture, so that the model can fully understand the relative positional
relationship among different text blocks. Experiment results show that LayoutLMv2 outperforms strong baselines and
achieves new state-of-the-art results on a wide variety of downstream visually-rich document understanding tasks,
including FUNSD (0.7895 -> 0.8420), CORD (0.9493 -> 0.9601), SROIE (0.9524 -> 0.9781), Kleister-NDA (0.834 -> 0.852),
RVL-CDIP (0.9443 -> 0.9564), and DocVQA (0.7295 -> 0.8672). The pre-trained LayoutLMv2 model is publicly available at
this https URL.*

LayoutLMv2 depends on `detectron2`, `torchvision` and `tesseract`. Run the
following to install them:
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install torchvision tesseract
```
(If you are developing for LayoutLMv2, note that passing the doctests also requires the installation of these packages.)

## Usage tips

- The main difference between LayoutLMv1 and LayoutLMv2 is that the latter incorporates visual embeddings during
  pre-training (while LayoutLMv1 only adds visual embeddings during fine-tuning).
- LayoutLMv2 adds both a relative 1D attention bias as well as a spatial 2D attention bias to the attention scores in
  the self-attention layers. Details can be found on page 5 of the [paper](https://arxiv.org/abs/2012.14740).
- Demo notebooks on how to use the LayoutLMv2 model on RVL-CDIP, FUNSD, DocVQA, CORD can be found [here](https://github.com/NielsRogge/Transformers-Tutorials).
- LayoutLMv2 uses Facebook AI's [Detectron2](https://github.com/facebookresearch/detectron2/) package for its visual
  backbone. See [this link](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for installation
  instructions.
- In addition to `input_ids`, [`~LayoutLMv2Model.forward`] expects 2 additional inputs, namely
  `image` and `bbox`. The `image` input corresponds to the original document image in which the text
  tokens occur. The model expects each document image to be of size 224x224. This means that if you have a batch of
  document images, `image` should be a tensor of shape (batch_size, 3, 224, 224). This can be either a
  `torch.Tensor` or a `Detectron2.structures.ImageList`. You don't need to normalize the channels, as this is
  done by the model. Important to note is that the visual backbone expects BGR channels instead of RGB, as all models
  in Detectron2 are pre-trained using the BGR format. The `bbox` input are the bounding boxes (i.e. 2D-positions)
  of the input text tokens. This is identical to [`LayoutLMModel`]. These can be obtained using an
  external OCR engine such as Google's [Tesseract](https://github.com/tesseract-ocr/tesseract) (there's a [Python
  wrapper](https://pypi.org/project/pytesseract/) available). Each bounding box should be in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1)
  represents the position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on
  a 0-1000 scale. To normalize, you can use the following function:

```python
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

Here, `width` and `height` correspond to the width and height of the original document in which the token
occurs (before resizing the image). Those can be obtained using the Python Image Library (PIL) library for example, as
follows:

```python
from PIL import Image

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
)

width, height = image.size
```

However, this model includes a brand new [`~transformers.LayoutLMv2Processor`] which can be used to directly
prepare data for the model (including applying OCR under the hood). More information can be found in the "Usage"
section below.

- Internally, [`~transformers.LayoutLMv2Model`] will send the `image` input through its visual backbone to
  obtain a lower-resolution feature map, whose shape is equal to the `image_feature_pool_shape` attribute of
  [`~transformers.LayoutLMv2Config`]. This feature map is then flattened to obtain a sequence of image tokens. As
  the size of the feature map is 7x7 by default, one obtains 49 image tokens. These are then concatenated with the text
  tokens, and send through the Transformer encoder. This means that the last hidden states of the model will have a
  length of 512 + 49 = 561, if you pad the text tokens up to the max length. More generally, the last hidden states
  will have a shape of `seq_length` + `image_feature_pool_shape[0]` *
  `config.image_feature_pool_shape[1]`.
- When calling [`~transformers.LayoutLMv2Model.from_pretrained`], a warning will be printed with a long list of
  parameter names that are not initialized. This is not a problem, as these parameters are batch normalization
  statistics, which are going to have values when fine-tuning on a custom dataset.
- If you want to train the model in a distributed environment, make sure to call [`synchronize_batch_norm`] on the
  model in order to properly synchronize the batch normalization layers of the visual backbone.

In addition, there's LayoutXLM, which is a multilingual version of LayoutLMv2. More information can be found on
[LayoutXLM's documentation page](layoutxlm).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LayoutLMv2. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-classification"/>

- A notebook on how to [finetune LayoutLMv2 for text-classification on RVL-CDIP dataset](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb).
- See also: [Text classification task guide](../tasks/sequence_classification)

<PipelineTag pipeline="question-answering"/>

- A notebook on how to [finetune LayoutLMv2 for question-answering on DocVQA dataset](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb).
- See also: [Question answering task guide](../tasks/question_answering)
- See also: [Document question answering task guide](../tasks/document_question_answering)


<PipelineTag pipeline="token-classification"/>

- A notebook on how to [finetune LayoutLMv2 for token-classification on CORD dataset](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/CORD/Fine_tuning_LayoutLMv2ForTokenClassification_on_CORD.ipynb).
- A notebook on how to [finetune LayoutLMv2 for token-classification on FUNSD dataset](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb).
- See also: [Token classification task guide](../tasks/token_classification)

## Usage: LayoutLMv2Processor

The easiest way to prepare data for the model is to use [`LayoutLMv2Processor`], which internally
combines a image processor ([`LayoutLMv2ImageProcessor`]) and a tokenizer
([`LayoutLMv2Tokenizer`] or [`LayoutLMv2TokenizerFast`]). The image processor
handles the image modality, while the tokenizer handles the text modality. A processor combines both, which is ideal
for a multi-modal model like LayoutLMv2. Note that you can still use both separately, if you only want to handle one
modality.

```python
from transformers import LayoutLMv2ImageProcessor, LayoutLMv2TokenizerFast, LayoutLMv2Processor

image_processor = LayoutLMv2ImageProcessor()  # apply_ocr is set to True by default
tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
processor = LayoutLMv2Processor(image_processor, tokenizer)
```

In short, one can provide a document image (and possibly additional data) to [`LayoutLMv2Processor`],
and it will create the inputs expected by the model. Internally, the processor first uses
[`LayoutLMv2ImageProcessor`] to apply OCR on the image to get a list of words and normalized
bounding boxes, as well to resize the image to a given size in order to get the `image` input. The words and
normalized bounding boxes are then provided to [`LayoutLMv2Tokenizer`] or
[`LayoutLMv2TokenizerFast`], which converts them to token-level `input_ids`,
`attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide word labels to the processor,
which are turned into token-level `labels`.

[`LayoutLMv2Processor`] uses [PyTesseract](https://pypi.org/project/pytesseract/), a Python
wrapper around Google's Tesseract OCR engine, under the hood. Note that you can still use your own OCR engine of
choice, and provide the words and normalized boxes yourself. This requires initializing
[`LayoutLMv2ImageProcessor`] with `apply_ocr` set to `False`.

In total, there are 5 use cases that are supported by the processor. Below, we list them all. Note that each of these
use cases work for both batched and non-batched inputs (we illustrate them for non-batched inputs).

**Use case 1: document image classification (training, inference) + token classification (inference), apply_ocr =
True**

This is the simplest case, in which the processor (actually the image processor) will perform OCR on the image to get
the words and normalized bounding boxes.

```python
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
encoding = processor(
    image, return_tensors="pt"
)  # you can also add all tokenizer parameters here such as padding, truncation
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

**Use case 2: document image classification (training, inference) + token classification (inference), apply_ocr=False**

In case one wants to do OCR themselves, one can initialize the image processor with `apply_ocr` set to
`False`. In that case, one should provide the words and corresponding (normalized) bounding boxes themselves to
the processor.

```python
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
words = ["hello", "world"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
encoding = processor(image, words, boxes=boxes, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

**Use case 3: token classification (training), apply_ocr=False**

For token classification tasks (such as FUNSD, CORD, SROIE, Kleister-NDA), one can also provide the corresponding word
labels in order to train a model. The processor will then convert these into token-level `labels`. By default, it
will only label the first wordpiece of a word, and label the remaining wordpieces with -100, which is the
`ignore_index` of PyTorch's CrossEntropyLoss. In case you want all wordpieces of a word to be labeled, you can
initialize the tokenizer with `only_label_first_subword` set to `False`.

```python
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
words = ["hello", "world"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
word_labels = [1, 2]
encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'labels', 'image'])
```

**Use case 4: visual question answering (inference), apply_ocr=True**

For visual question answering tasks (such as DocVQA), you can provide a question to the processor. By default, the
processor will apply OCR on the image, and create [CLS] question tokens [SEP] word tokens [SEP].

```python
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
question = "What's his name?"
encoding = processor(image, question, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

**Use case 5: visual question answering (inference), apply_ocr=False**

For visual question answering tasks (such as DocVQA), you can provide a question to the processor. If you want to
perform OCR yourself, you can provide your own words and (normalized) bounding boxes to the processor.

```python
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
question = "What's his name?"
words = ["hello", "world"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

## LayoutLMv2Config


    This is the configuration class to store the configuration of a [`LayoutLMv2Model`]. It is used to instantiate an
    LayoutLMv2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LayoutLMv2
    [microsoft/layoutlmv2-base-uncased](https://huggingface.co/microsoft/layoutlmv2-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the LayoutLMv2 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`LayoutLMv2Model`] or [`TFLayoutLMv2Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`LayoutLMv2Model`] or
            [`TFLayoutLMv2Model`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum value that the 2D position embedding might ever be used with. Typically set this to something
            large just in case (e.g., 1024).
        max_rel_pos (`int`, *optional*, defaults to 128):
            The maximum number of relative positions to be used in the self-attention mechanism.
        rel_pos_bins (`int`, *optional*, defaults to 32):
            The number of relative position bins to be used in the self-attention mechanism.
        fast_qkv (`bool`, *optional*, defaults to `True`):
            Whether or not to use a single matrix for the queries, keys, values in the self-attention layers.
        max_rel_2d_pos (`int`, *optional*, defaults to 256):
            The maximum number of relative 2D positions in the self-attention mechanism.
        rel_2d_pos_bins (`int`, *optional*, defaults to 64):
            The number of 2D relative position bins in the self-attention mechanism.
        image_feature_pool_shape (`List[int]`, *optional*, defaults to [7, 7, 256]):
            The shape of the average-pooled feature map.
        coordinate_size (`int`, *optional*, defaults to 128):
            Dimension of the coordinate embeddings.
        shape_size (`int`, *optional*, defaults to 128):
            Dimension of the width and height embeddings.
        has_relative_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use a relative attention bias in the self-attention mechanism.
        has_spatial_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use a spatial attention bias in the self-attention mechanism.
        has_visual_segment_embedding (`bool`, *optional*, defaults to `False`):
            Whether or not to add visual segment embeddings.
        detectron2_config_args (`dict`, *optional*):
            Dictionary containing the configuration arguments of the Detectron2 visual backbone. Refer to [this
            file](https://github.com/microsoft/unilm/blob/master/layoutlmft/layoutlmft/models/layoutlmv2/detectron2_config.py)
            for details regarding default values.

    Example:

    ```python
    >>> from transformers import LayoutLMv2Config, LayoutLMv2Model

    >>> # Initializing a LayoutLMv2 microsoft/layoutlmv2-base-uncased style configuration
    >>> configuration = LayoutLMv2Config()

    >>> # Initializing a model (with random weights) from the microsoft/layoutlmv2-base-uncased style configuration
    >>> model = LayoutLMv2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## LayoutLMv2FeatureExtractor

No docstring available for LayoutLMv2FeatureExtractor

Methods: __call__

## LayoutLMv2ImageProcessor


    Constructs a LayoutLMv2 image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to `(size["height"], size["width"])`. Can be
            overridden by `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        apply_ocr (`bool`, *optional*, defaults to `True`):
            Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by
            `apply_ocr` in `preprocess`.
        ocr_lang (`str`, *optional*):
            The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
            used. Can be overridden by `ocr_lang` in `preprocess`.
        tesseract_config (`str`, *optional*, defaults to `""`):
            Any additional custom configuration flags that are forwarded to the `config` parameter when calling
            Tesseract. For example: '--psm 6'. Can be overridden by `tesseract_config` in `preprocess`.
    

Methods: preprocess

## LayoutLMv2Tokenizer


    Construct a LayoutLMv2 tokenizer. Based on WordPiece. [`LayoutLMv2Tokenizer`] can be used to turn words, word-level
    bounding boxes and optional word labels to token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`, and
    optional `labels` (for token classification).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    [`LayoutLMv2Tokenizer`] runs end-to-end tokenization: punctuation splitting and wordpiece. It also turns the
    word-level bounding boxes into token-level bounding boxes.

    

Methods: __call__
    - save_vocabulary

## LayoutLMv2TokenizerFast


    Construct a "fast" LayoutLMv2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        cls_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`):
            The bounding box to use for the special [CLS] token.
        sep_token_box (`List[int]`, *optional*, defaults to `[1000, 1000, 1000, 1000]`):
            The bounding box to use for the special [SEP] token.
        pad_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`):
            The bounding box to use for the special [PAD] token.
        pad_token_label (`int`, *optional*, defaults to -100):
            The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch's
            CrossEntropyLoss.
        only_label_first_subword (`bool`, *optional*, defaults to `True`):
            Whether or not to only label the first subword, in case word labels are provided.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original LayoutLMv2).
    

Methods: __call__

## LayoutLMv2Processor


    Constructs a LayoutLMv2 processor which combines a LayoutLMv2 image processor and a LayoutLMv2 tokenizer into a
    single processor.

    [`LayoutLMv2Processor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv2ImageProcessor`] to resize document images to a fixed size, and optionally applies OCR to
    get words and normalized bounding boxes. These are then provided to [`LayoutLMv2Tokenizer`] or
    [`LayoutLMv2TokenizerFast`], which turns the words and bounding boxes into token-level `input_ids`,
    `attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
    into token-level `labels` for token classification tasks (such as FUNSD, CORD).

    Args:
        image_processor (`LayoutLMv2ImageProcessor`, *optional*):
            An instance of [`LayoutLMv2ImageProcessor`]. The image processor is a required input.
        tokenizer (`LayoutLMv2Tokenizer` or `LayoutLMv2TokenizerFast`, *optional*):
            An instance of [`LayoutLMv2Tokenizer`] or [`LayoutLMv2TokenizerFast`]. The tokenizer is a required input.
    

Methods: __call__

## LayoutLMv2Model

The bare LayoutLMv2 Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## LayoutLMv2ForSequenceClassification


    LayoutLMv2 Model with a sequence classification head on top (a linear layer on top of the concatenation of the
    final hidden state of the [CLS] token, average-pooled initial visual embeddings and average-pooled final visual
    embeddings, e.g. for document image classification tasks such as the
    [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


## LayoutLMv2ForTokenClassification


    LayoutLMv2 Model with a token classification head on top (a linear layer on top of the text part of the hidden
    states) e.g. for sequence labeling (information extraction) tasks such as
    [FUNSD](https://guillaumejaume.github.io/FUNSD/), [SROIE](https://rrc.cvc.uab.es/?ch=13),
    [CORD](https://github.com/clovaai/cord) and [Kleister-NDA](https://github.com/applicaai/kleister-nda).
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


## LayoutLMv2ForQuestionAnswering


    LayoutLMv2 Model with a span classification head on top for extractive question-answering tasks such as
    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the text part of the hidden-states output to
    compute `span start logits` and `span end logits`).
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

