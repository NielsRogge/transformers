# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for Donut."""

from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageFeatureExtractionMixin,
    ImageInput,
    is_torch_tensor,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class DonutFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a Donut feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize_and_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to resize the shorter edge of the input to the minimum value of a certain `size`, and thumbnail the
            input to the given `size`.
        size (`Tuple(int)`, *optional*, defaults to [1920, 2560]):
            Resize the shorter edge of the input to the minimum value of the given size. Should be a tuple of (width,
            height). Only has an effect if `do_resize` is set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether or not to pad the input to `size`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`List[int]`, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize_and_thumbnail=True,
        size=[1920, 2560],
        resample=Image.BILINEAR,
        do_pad=True,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize_and_thumbnail = do_resize_and_thumbnail
        self.size = size
        self.resample = resample
        self.do_pad = do_pad
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def resize_and_thumbnail(self, image, size, resample):
        # resize the shorter edge of the image to `min(size)`
        image = self.resize(image, size=min(size), resample=resample, default_to_square=False)
        # create a thumbnail
        image.thumbnail((size[0], size[1]))

        return image

    def pad(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        delta_width = size[0] - image.width
        delta_height = size[1] - image.height

        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return ImageOps.expand(image, padding)

    def __call__(
        self, images: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (resizing + padding + normalization)
        if self.do_resize_and_thumbnail and self.size is not None:
            images = [
                self.resize_and_thumbnail(image=image, size=self.size, resample=self.resample) for image in images
            ]
        if self.do_pad and self.size is not None:
            images = [self.pad(image=image, size=self.size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
