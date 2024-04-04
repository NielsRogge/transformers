# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for LLaVa.
"""


from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType


class LlavaProcessor(ProcessorMixin):
    r"""
    Constructs a LLaVa processor which wraps a CLIP image processor and a LLaMa tokenizer into a single processor.

    [`LlavaProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaProcessor.__call__`] and [`~LlavaProcessor.decode`] for more information.

    Args:

    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __call__(
        self,
        inputs: List[List[dict]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare multimodal inputs involving text and images for the model.

        Args:
            inputs (`List[List[dict]]`):
                Multimodal inputs of the following format:

                [
                    [
                        {"input": "image", "content": Image.open(..)},
                        {"input": "text", "content" : "Tell me a story"},
                    ],
                    [
                        {"input": "image", "content": Image.open(..)},
                        {"input": "text", "content" : "What can you do in Paris?"},
                ]

            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        text_inputs = []
        image_inputs = []
        for example in inputs:
            text_example = ""
            # here we have a single training example containing possibly interleaved modalities
            # we should create this:
            "USER: <image>\nThis is a cat\n<image>\nThis is a dog\n<image>\nWhat is this?\nASSISTANT:"
            for idx, item in enumerate(example):
                if item["input"] == "image":
                    image = item["content"]
                    image_inputs.append(image)
                elif item["input"] == "text":
                    text = item["content"]
                    # if the previous item was an image,
                    # we should append the special image token to the text_example
                    # TODO not sure this should be done by the processor
                    text_example += "<image>\n" + text + "\n" if example[idx - 1]["input"] == "image" else text + "\n"
                else:
                    raise ValueError(f"Unknown input type: {item['input']}")

            text_inputs.append(text_example)

        if len(text_inputs) > 0:
            # each text input should start with "USER:" and end with "ASSISTANT:"
            # TODO not sure this should be done by the processor
            # this should be part of the chat template, I assume
            text_inputs = ["USER: " + text + "ASSISTANT:" for text in text_inputs]

            text_inputs = self.tokenizer(
                text_inputs,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
            )
        else:
            text_inputs = {}

        if len(image_inputs) > 0:
            image_inputs = self.image_processor(image_inputs, return_tensors=return_tensors)
        else:
            image_inputs = {}

        return BatchFeature(data={**text_inputs, **image_inputs})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
