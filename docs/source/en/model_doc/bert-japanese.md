<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BertJapanese

## Overview

The BERT models trained on Japanese text.

There are models with two different tokenization methods:

- Tokenize with MeCab and WordPiece. This requires some extra dependencies, [fugashi](https://github.com/polm/fugashi) which is a wrapper around [MeCab](https://taku910.github.io/mecab/).
- Tokenize into characters.

To use *MecabTokenizer*, you should `pip install transformers["ja"]` (or `pip install -e .["ja"]` if you install
from source) to install dependencies.

See [details on cl-tohoku repository](https://github.com/cl-tohoku/bert-japanese).

Example of using a model with MeCab and WordPiece tokenization:

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

>>> ## Input Japanese Text
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾輩 は 猫 で ある 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

Example of using a model with Character tokenization:

```python
>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

>>> ## Input Japanese Text
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾 輩 は 猫 で あ る 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

This model was contributed by [cl-tohoku](https://huggingface.co/cl-tohoku).

<Tip> 

This implementation is the same as BERT, except for tokenization method. Refer to [BERT documentation](bert) for 
API reference information.  

</Tip>


## BertJapaneseTokenizer


    Construct a BERT tokenizer for Japanese text.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer
    to: this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to a one-wordpiece-per-line vocabulary file.
        spm_file (`str`, *optional*):
            Path to [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm or .model
            extension) that contains the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether to lower case the input. Only has an effect when do_basic_tokenize=True.
        do_word_tokenize (`bool`, *optional*, defaults to `True`):
            Whether to do word tokenization.
        do_subword_tokenize (`bool`, *optional*, defaults to `True`):
            Whether to do subword tokenization.
        word_tokenizer_type (`str`, *optional*, defaults to `"basic"`):
            Type of word tokenizer. Choose from ["basic", "mecab", "sudachi", "jumanpp"].
        subword_tokenizer_type (`str`, *optional*, defaults to `"wordpiece"`):
            Type of subword tokenizer. Choose from ["wordpiece", "character", "sentencepiece",].
        mecab_kwargs (`dict`, *optional*):
            Dictionary passed to the `MecabTokenizer` constructor.
        sudachi_kwargs (`dict`, *optional*):
            Dictionary passed to the `SudachiTokenizer` constructor.
        jumanpp_kwargs (`dict`, *optional*):
            Dictionary passed to the `JumanppTokenizer` constructor.
    
