<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TVLT

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The TVLT model was proposed in [TVLT: Textless Vision-Language Transformer](https://arxiv.org/abs/2209.14156)
by Zineng Tang, Jaemin Cho, Yixin Nie, Mohit Bansal (the first three authors contributed equally). The Textless Vision-Language Transformer (TVLT) is a model that uses raw visual and audio inputs for vision-and-language representation learning, without using text-specific modules such as tokenization or automatic speech recognition (ASR). It can perform various audiovisual and vision-language tasks like retrieval, question answering, etc.

The abstract from the paper is the following:

*In this work, we present the Textless Vision-Language Transformer (TVLT), where homogeneous transformer blocks take raw visual and audio inputs for vision-and-language representation learning with minimal modality-specific design, and do not use text-specific modules such as tokenization or automatic speech recognition (ASR). TVLT is trained by reconstructing masked patches of continuous video frames and audio spectrograms (masked autoencoding) and contrastive modeling to align video and audio. TVLT attains performance comparable to its text-based counterpart on various multimodal tasks, such as visual question answering, image retrieval, video retrieval, and multimodal sentiment analysis, with 28x faster inference speed and only 1/3 of the parameters. Our findings suggest the possibility of learning compact and efficient visual-linguistic representations from low-level visual and audio signals without assuming the prior existence of text.*

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/tvlt_architecture.png"
alt="drawing" width="600"/>
</p>

<small> TVLT architecture. Taken from the <a href="[https://arxiv.org/abs/2102.03334](https://arxiv.org/abs/2209.14156)">original paper</a>. </small>

The original code can be found [here](https://github.com/zinengtang/TVLT). This model was contributed by [Zineng Tang](https://huggingface.co/ZinengTang).

## Usage tips

- TVLT is a model that takes both `pixel_values` and `audio_values` as input. One can use [`TvltProcessor`] to prepare data for the model.
  This processor wraps an image processor (for the image/video modality) and an audio feature extractor (for the audio modality) into one.
- TVLT is trained with images/videos and audios of various sizes: the authors resize and crop the input images/videos to 224 and limit the length of audio spectrogram to 2048. To make batching of videos and audios possible, the authors use a `pixel_mask` that indicates which pixels are real/padding and `audio_mask` that indicates which audio values are real/padding.
- The design of TVLT is very similar to that of a standard Vision Transformer (ViT) and masked autoencoder (MAE) as in [ViTMAE](vitmae). The difference is that the model includes embedding layers for the audio modality.
- The PyTorch version of this model is only available in torch 1.10 and higher.

## TvltConfig


    This is the configuration class to store the configuration of a [`TvltModel`]. It is used to instantiate a TVLT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the TVLT
    [ZinengTang/tvlt-base](https://huggingface.co/ZinengTang/tvlt-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        spectrogram_length (`int`, *optional*, defaults to 2048):
            The time length of each audio spectrogram.
        frequency_length (`int`, *optional*, defaults to 128):
            The frequency length of audio spectrogram.
        image_patch_size (`List[int]`, *optional*, defaults to `[16, 16]`):
            The size (resolution) of each image patch.
        audio_patch_size (`List[int]`, *optional*, defaults to `[16, 16]`):
            The size (resolution) of each audio patch.
        num_image_channels (`int`, *optional*, defaults to 3):
            The number of input image channels.
        num_audio_channels (`int`, *optional*, defaults to 1):
            The number of input audio channels.
        num_frames (`int`, *optional*, defaults to 8):
            The maximum number of frames for an input video.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        use_mean_pooling (`bool`, *optional*, defaults to `False`):
            Whether to mean pool the final hidden states instead of using the final hidden state of the [CLS] token.
        decoder_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the decoder.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the decoder.
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
        pixel_mask_ratio (`float`, *optional*, defaults to 0.75):
            Image patch masking ratio.
        audio_mask_ratio (`float`, *optional*, defaults to 0.15):
            Audio patch masking ratio.
        audio_mask_type (`str`, *optional*, defaults to `"frame-level"`):
            Audio patch masking type, choose between "frame-level" and "patch-level".
        task_matching (`bool`, *optional*, defaults to `True`):
            Whether to use vision audio matching task in pretraining.
        task_mae (`bool`, *optional*, defaults to `True`):
            Whether to use the masked auto-encoder (MAE) in pretraining.
        loss_type (`str`, *optional*, defaults to `"classification"`):
            Loss types including regression and classification.

    Example:

    ```python
    >>> from transformers import TvltConfig, TvltModel

    >>> # # Initializing a TVLT ZinengTang/tvlt-base style configuration
    >>> configuration = TvltConfig()

    >>> # # Initializing a model (with random weights) from the ZinengTang/tvlt-base style configuration
    >>> model = TvltModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## TvltProcessor


    Constructs a TVLT processor which wraps a TVLT image processor and TVLT feature extractor into a single processor.

    [`TvltProcessor`] offers all the functionalities of [`TvltImageProcessor`] and [`TvltFeatureExtractor`]. See the
    docstring of [`~TvltProcessor.__call__`] for more information.

    Args:
        image_processor (`TvltImageProcessor`):
            An instance of [`TvltImageProcessor`]. The image processor is a required input.
        feature_extractor (`TvltFeatureExtractor`):
            An instance of [`TvltFeatureExtractor`]. The feature extractor is a required input.
    

Methods: __call__

## TvltImageProcessor


    Constructs a TVLT image processor.

    This processor can be used to prepare either videos or images for the model by converting images to 1-frame videos.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the output image after resizing. The shortest edge of the image will be resized to
            `size["shortest_edge"]` while maintaining the aspect ratio of the original image. Can be overriden by
            `size` in the `preprocess` method.
        patch_size (`List[int]` *optional*, defaults to [16,16]):
            The patch size of image patch embedding.
        num_frames (`int` *optional*, defaults to 8):
            The maximum number of video frames.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop`
            parameter in the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to 1/255):
            Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
            in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    

Methods: preprocess

## TvltFeatureExtractor


    Constructs a TVLT audio feature extractor. This feature extractor can be used to prepare audios for the model.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        spectrogram_length (`Dict[str, int]` *optional*, defaults to 2048):
            The time length of each audio spectrogram.
        num_channels (`int` *optional*, defaults to 1):
            Number of audio channels.
        patch_size (`List[int]` *optional*, defaults to `[16, 16]`):
            The patch size of audio patch embedding.
        feature_size (`int`, *optional*, defaults to 128):
            The frequency length of audio spectrogram.
        sampling_rate (`int`, *optional*, defaults to 44100):
            The sampling rate at which the audio files should be digitalized expressed in Hertz (Hz).
        hop_length_to_sampling_rate (`int`, *optional*, defaults to 86):
            Hop length is length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
            For example, with sampling rate 44100, the hop length is 512, with 44100 / 512 = 86
        n_fft (`int`, *optional*, defaults to 2048):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    

Methods: __call__

## TvltModel

The bare TVLT Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TvltConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## TvltForPreTraining

The TVLT Model transformer with the decoder on top for self-supervised pre-training.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TvltConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## TvltForAudioVisualClassification


    Tvlt Model transformer with a classifier head on top (an MLP on top of the final hidden state of the [CLS] token)
    for audiovisual classification tasks, e.g. CMU-MOSEI Sentiment Analysis and Audio to Video Retrieval.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TvltConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
