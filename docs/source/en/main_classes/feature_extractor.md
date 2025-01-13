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

# Feature Extractor

A feature extractor is in charge of preparing input features for audio or vision models. This includes feature extraction from sequences, e.g., pre-processing audio files to generate Log-Mel Spectrogram features, feature extraction from images, e.g., cropping image files, but also padding, normalization, and conversion to NumPy, PyTorch, and TensorFlow tensors.


## FeatureExtractionMixin

feature_extraction_utils.FeatureExtractionMixin

    This is a feature extraction mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    
    - from_pretrained
    - save_pretrained

## SequenceFeatureExtractor


    This is a general feature extraction class for speech recognition.

    Args:
        feature_size (`int`):
            The feature dimension of the extracted features.
        sampling_rate (`int`):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`):
            The value that is used to fill the padding values / vectors.
    
    - pad

## BatchFeature


    Holds the output of the [`~SequenceFeatureExtractor.pad`] and feature extractor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`, *optional*):
            Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
            etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    

## ImageFeatureExtractionMixin

image_utils.ImageFeatureExtractionMixin

    Mixin that contain utilities for preparing image features.
    
