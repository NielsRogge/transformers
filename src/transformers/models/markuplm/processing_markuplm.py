# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for MarkupLM.
"""
from typing import Union

from .tokenization_markuplm import MarkupLMTokenizer
from .tokenization_markuplm_fast import MarkupLMTokenizerFast

import bs4
from bs4 import BeautifulSoup
import html
import torch


class MarkupLMProcessor:
    # This class can process one or multiple pages (in the form of html-code strings).
    # Nodes with no text are not taken into consideration in this implementation.
    # In other words, only the visible text on the page are processed, while the imgs, icons, js, are aborted.

    tags_dict = {'a': 0, 'abbr': 1, 'acronym': 2, 'address': 3, 'altGlyph': 4, 'altGlyphDef': 5, 'altGlyphItem': 6,
                 'animate': 7, 'animateColor': 8, 'animateMotion': 9, 'animateTransform': 10, 'applet': 11, 'area': 12,
                 'article': 13, 'aside': 14, 'audio': 15, 'b': 16, 'base': 17, 'basefont': 18, 'bdi': 19, 'bdo': 20,
                 'bgsound': 21, 'big': 22, 'blink': 23, 'blockquote': 24, 'body': 25, 'br': 26, 'button': 27,
                 'canvas': 28,
                 'caption': 29, 'center': 30, 'circle': 31, 'cite': 32, 'clipPath': 33, 'code': 34, 'col': 35,
                 'colgroup': 36, 'color-profile': 37, 'content': 38, 'cursor': 39, 'data': 40, 'datalist': 41, 'dd': 42,
                 'defs': 43, 'del': 44, 'desc': 45, 'details': 46, 'dfn': 47, 'dialog': 48, 'dir': 49, 'div': 50,
                 'dl': 51,
                 'dt': 52, 'ellipse': 53, 'em': 54, 'embed': 55, 'feBlend': 56, 'feColorMatrix': 57,
                 'feComponentTransfer': 58, 'feComposite': 59, 'feConvolveMatrix': 60, 'feDiffuseLighting': 61,
                 'feDisplacementMap': 62, 'feDistantLight': 63, 'feFlood': 64, 'feFuncA': 65, 'feFuncB': 66,
                 'feFuncG': 67,
                 'feFuncR': 68, 'feGaussianBlur': 69, 'feImage': 70, 'feMerge': 71, 'feMergeNode': 72,
                 'feMorphology': 73,
                 'feOffset': 74, 'fePointLight': 75, 'feSpecularLighting': 76, 'feSpotLight': 77, 'feTile': 78,
                 'feTurbulence': 79, 'fieldset': 80, 'figcaption': 81, 'figure': 82, 'filter': 83,
                 'font-face-format': 84,
                 'font-face-name': 85, 'font-face-src': 86, 'font-face-uri': 87, 'font-face': 88, 'font': 89,
                 'footer': 90,
                 'foreignObject': 91, 'form': 92, 'frame': 93, 'frameset': 94, 'g': 95, 'glyph': 96, 'glyphRef': 97,
                 'h1': 98, 'h2': 99, 'h3': 100, 'h4': 101, 'h5': 102, 'h6': 103, 'head': 104, 'header': 105,
                 'hgroup': 106,
                 'hkern': 107, 'hr': 108, 'html': 109, 'i': 110, 'iframe': 111, 'image': 112, 'img': 113, 'input': 114,
                 'ins': 115, 'kbd': 116, 'keygen': 117, 'label': 118, 'legend': 119, 'li': 120, 'line': 121,
                 'linearGradient': 122, 'link': 123, 'main': 124, 'map': 125, 'mark': 126, 'marker': 127,
                 'marquee': 128,
                 'mask': 129, 'math': 130, 'menu': 131, 'menuitem': 132, 'meta': 133, 'metadata': 134, 'meter': 135,
                 'missing-glyph': 136, 'mpath': 137, 'nav': 138, 'nobr': 139, 'noembed': 140, 'noframes': 141,
                 'noscript': 142, 'object': 143, 'ol': 144, 'optgroup': 145, 'option': 146, 'output': 147, 'p': 148,
                 'param': 149, 'path': 150, 'pattern': 151, 'picture': 152, 'plaintext': 153, 'polygon': 154,
                 'polyline': 155, 'portal': 156, 'pre': 157, 'progress': 158, 'q': 159, 'radialGradient': 160,
                 'rb': 161,
                 'rect': 162, 'rp': 163, 'rt': 164, 'rtc': 165, 'ruby': 166, 's': 167, 'samp': 168, 'script': 169,
                 'section': 170, 'select': 171, 'set': 172, 'shadow': 173, 'slot': 174, 'small': 175, 'source': 176,
                 'spacer': 177, 'span': 178, 'stop': 179, 'strike': 180, 'strong': 181, 'style': 182, 'sub': 183,
                 'summary': 184, 'sup': 185, 'svg': 186, 'switch': 187, 'symbol': 188, 'table': 189, 'tbody': 190,
                 'td': 191, 'template': 192, 'text': 193, 'textPath': 194, 'textarea': 195, 'tfoot': 196, 'th': 197,
                 'thead': 198, 'time': 199, 'title': 200, 'tr': 201, 'track': 202, 'tref': 203, 'tspan': 204, 'tt': 205,
                 'u': 206, 'ul': 207, 'use': 208, 'var': 209, 'video': 210, 'view': 211, 'vkern': 212, 'wbr': 213,
                 'xmp': 214}

    MAX_DEPTH = 50
    UNK_TAG_ID = len(tags_dict)
    PAD_TAG_ID = UNK_TAG_ID + 1
    MAX_WIDTH = 1000
    PAD_WIDTH = 1001

    PAD_XPATH_TAGS_SEQ = [PAD_TAG_ID] * MAX_DEPTH
    PAD_XPATH_SUBS_SEQ = [PAD_WIDTH] * MAX_DEPTH

    def __init__(self, tokenizer: Union[MarkupLMTokenizer, MarkupLMTokenizerFast]):
        self.tokenizer = tokenizer

    def xpath_tags_transfer(self, xpath_tags_seq_by_str):
        if len(xpath_tags_seq_by_str) > self.MAX_DEPTH:
            xpath_tags_seq_by_str = xpath_tags_seq_by_str[:self.MAX_DEPTH]
        mid = [self.tags_dict.get(i, self.UNK_TAG_ID) for i in xpath_tags_seq_by_str]
        mid += [self.PAD_TAG_ID] * (self.MAX_DEPTH - len(mid))
        return mid

    def xpath_subs_transfer(self, xpath_subs_seq_by_int):
        if len(xpath_subs_seq_by_int) > self.MAX_DEPTH:
            xpath_subs_seq_by_int = xpath_subs_seq_by_int[:self.MAX_DEPTH]
        mid = [min(i, self.MAX_WIDTH) for i in xpath_subs_seq_by_int]
        mid += [self.PAD_WIDTH] * (self.MAX_DEPTH - len(mid))
        return mid

    def xpath_soup(self, element):
        xpath_tags = []
        xpath_subscripts = []
        child = element if element.name else element.parent
        for parent in child.parents:  # type: bs4.element.Tag
            siblings = parent.find_all(child.name, recursive=False)
            xpath_tags.append(child.name)
            xpath_subscripts.append(
                0 if 1 == len(siblings) else next(i for i, s in enumerate(siblings, 1) if s is child))
            child = parent
        xpath_tags.reverse()
        xpath_subscripts.reverse()
        return xpath_tags, xpath_subscripts

    def get_three_from_single(self, html_string):
        html_code = BeautifulSoup(html_string, "html.parser")

        all_doc_strings = []
        string2xtag_seq = []
        string2xsubs_seq = []

        for element in html_code.descendants:
            if type(element) == bs4.element.NavigableString:
                if type(element.parent) != bs4.element.Tag:
                    continue

                text_in_this_tag = html.unescape(element).strip()
                if not text_in_this_tag:
                    continue

                all_doc_strings.append(text_in_this_tag)

                xpath_tags, xpath_subscripts = self.xpath_soup(element)
                string2xtag_seq.append(xpath_tags)
                string2xsubs_seq.append(xpath_subscripts)

        assert len(all_doc_strings) == len(string2xtag_seq)
        assert len(all_doc_strings) == len(string2xsubs_seq)

        return all_doc_strings, string2xtag_seq, string2xsubs_seq

    '''
    def process_single(self, html_string, max_length=None, return_type='pt'):

        all_doc_strings, string2xtag_seq, string2xsubs_seq = self.get_three_from_single(html_string)

        token_ids = []
        xpath_tags_seq = []
        xpath_subs_seq = []
        for i, doc_string in enumerate(all_doc_strings):
            token_ids_in_span = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(doc_string))
            token_ids += token_ids_in_span
            xpath_tags_seq += [self.xpath_tags_transfer(string2xtag_seq[i])] * len(token_ids_in_span)
            xpath_subs_seq += [self.xpath_tags_transfer(string2xsubs_seq[i])] * len(token_ids_in_span)

        if max_length is not None:
            # pad / truncation
            # the truncation is done at the end

            real_max_length = max_length - 2  # for cls and sep
            token_type_ids = [0] * max_length

            token_ids = token_ids[:real_max_length]
            xpath_tags_seq = xpath_tags_seq[:real_max_length]
            xpath_subs_seq = xpath_subs_seq[:real_max_length]

            curr_len = len(token_ids)

            token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id] + [
                self.tokenizer.pad_token_id] * (real_max_length - curr_len)
            xpath_tags_seq = [self.PAD_XPATH_TAGS_SEQ] + xpath_tags_seq + [self.PAD_XPATH_TAGS_SEQ] + [
                self.PAD_XPATH_TAGS_SEQ] * (real_max_length - curr_len)
            xpath_subs_seq = [self.PAD_XPATH_SUBS_SEQ] + xpath_subs_seq + [self.PAD_XPATH_SUBS_SEQ] + [
                self.PAD_XPATH_SUBS_SEQ] * (real_max_length - curr_len)
            attention_mask = [1] + [1] * curr_len + [1] + [0] * (real_max_length - curr_len)

        else:
            length = len(token_ids) + 2
            token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
            xpath_tags_seq = [self.PAD_XPATH_TAGS_SEQ] + xpath_tags_seq + [self.PAD_XPATH_TAGS_SEQ]
            xpath_subs_seq = [self.PAD_XPATH_SUBS_SEQ] + xpath_subs_seq + [self.PAD_XPATH_SUBS_SEQ]
            token_type_ids = [0] * length
            attention_mask = [1] * length

        if return_type == 'pt':
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            xpath_tags_seq = torch.tensor(xpath_tags_seq, dtype=torch.long)
            xpath_subs_seq = torch.tensor(xpath_subs_seq, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {"input_ids": token_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "xpath_tags_seq": xpath_tags_seq,
                "xpath_subs_seq": xpath_subs_seq}
    '''

    def process_multiple(self, html_strings, max_length=None, doc_stride=None, return_type='pt'):

        three_dict = {'all_doc_strings': [],
                      'string2xtag_seq': [],
                      'string2xsubs_seq': []}

        for html_string in html_strings:
            all_doc_strings, string2xtag_seq, string2xsubs_seq = self.get_three_from_single(html_string)
            three_dict['all_doc_strings'].append(all_doc_strings)
            three_dict['string2xtag_seq'].append(string2xtag_seq)
            three_dict['string2xsubs_seq'].append(string2xsubs_seq)

        tokenized_examples = self.tokenizer(
            three_dict["all_doc_strings"],
            truncation=True,
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            padding="max_length" if max_length is not None else False,
            is_split_into_words=True,
            return_tensors=return_type,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["xpath_tags_seq"] = []
        tokenized_examples["xpath_subs_seq"] = []

        for i, sample_index in enumerate(sample_mapping):
            word_ids = tokenized_examples.word_ids(i)
            xpath_tags_in_span = [self.PAD_XPATH_TAGS_SEQ if corr_word_id is None
                                  else self.xpath_tags_transfer(
                three_dict["string2xtag_seq"][sample_index][corr_word_id])
                                  for corr_word_id in word_ids]
            tokenized_examples["xpath_tags_seq"].append(xpath_tags_in_span)

            xpath_subs_in_span = [self.PAD_XPATH_SUBS_SEQ if corr_word_id is None
                                  else self.xpath_subs_transfer(
                three_dict["string2xsubs_seq"][sample_index][corr_word_id])
                                  for corr_word_id in word_ids]
            tokenized_examples["xpath_subs_seq"].append(xpath_subs_in_span)

        if return_type == 'pt':
            tokenized_examples["xpath_tags_seq"] = torch.tensor(tokenized_examples["xpath_tags_seq"], dtype=torch.long)
            tokenized_examples["xpath_subs_seq"] = torch.tensor(tokenized_examples["xpath_subs_seq"], dtype=torch.long)

        return tokenized_examples

    def __call__(self,
                 html_strings,  # a single string for a webpage, or a list of them
                 max_length=512,
                 doc_stride=128,
                 return_type='pt',
                 *args,
                 **kwargs
                 ):
        if isinstance(html_strings, str):
            return self.process_multiple([html_strings], max_length=max_length, doc_stride=doc_stride,
                                         return_type=return_type)
        elif isinstance(html_strings, list) and len(html_strings) >= 1 and isinstance(html_strings[0], str):
            return self.process_multiple(html_strings, max_length=max_length,
                                         doc_stride=doc_stride, return_type=return_type)
        else:
            raise ValueError("Not supported yet!")


if __name__ == '__main__':
    import os
    from .modeling_markuplm import MarkupLMModel

    page_name_1 = "page1.html"
    page_name_2 = "page2.html"
    page_name_3 = "page3.html"

    with open(page_name_1) as f:
        single_html_string = f.read()

    multi_html_strings = []

    with open(page_name_2) as f:
        multi_html_strings.append(f.read())
    with open(page_name_3) as f:
        multi_html_strings.append(f.read())

    model = MarkupLMModel.from_pretrained("microsoft/markuplm-base")
    tokenizer = MarkupLMTokenizerFast.from_pretrained("microsoft/markuplm-base", add_prefix_space=True)
    processor = MarkupLMProcessor(tokenizer)

    inputs = processor(single_html_string)
    print(inputs)
    output = model(**inputs)
    print(output)

    inputs = processor(multi_html_strings)
    print(inputs)
    output = model(**inputs)
    print(output)
