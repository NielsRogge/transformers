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

# Perceiver

## Overview

The Perceiver IO model was proposed in [Perceiver IO: A General Architecture for Structured Inputs &
Outputs](https://arxiv.org/abs/2107.14795) by Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch,
Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M.
Botvinick, Andrew Zisserman, Oriol Vinyals, João Carreira.

Perceiver IO is a generalization of [Perceiver](https://arxiv.org/abs/2103.03206) to handle arbitrary outputs in
addition to arbitrary inputs. The original Perceiver only produced a single classification label. In addition to
classification labels, Perceiver IO can produce (for example) language, optical flow, and multimodal videos with audio.
This is done using the same building blocks as the original Perceiver. The computational complexity of Perceiver IO is
linear in the input and output size and the bulk of the processing occurs in the latent space, allowing us to process
inputs and outputs that are much larger than can be handled by standard Transformers. This means, for example,
Perceiver IO can do BERT-style masked language modeling directly using bytes instead of tokenized inputs.

The abstract from the paper is the following:

*The recently-proposed Perceiver model obtains good results on several domains (images, audio, multimodal, point
clouds) while scaling linearly in compute and memory with the input size. While the Perceiver supports many kinds of
inputs, it can only produce very simple outputs such as class scores. Perceiver IO overcomes this limitation without
sacrificing the original's appealing properties by learning to flexibly query the model's latent space to produce
outputs of arbitrary size and semantics. Perceiver IO still decouples model depth from data size and still scales
linearly with data size, but now with respect to both input and output sizes. The full Perceiver IO model achieves
strong results on tasks with highly structured output spaces, such as natural language and visual understanding,
StarCraft II, and multi-task and multi-modal domains. As highlights, Perceiver IO matches a Transformer-based BERT
baseline on the GLUE language benchmark without the need for input tokenization and achieves state-of-the-art
performance on Sintel optical flow estimation.*

Here's a TLDR explaining how Perceiver works:

The main problem with the self-attention mechanism of the Transformer is that the time and memory requirements scale
quadratically with the sequence length. Hence, models like BERT and RoBERTa are limited to a max sequence length of 512
tokens. Perceiver aims to solve this issue by, instead of performing self-attention on the inputs, perform it on a set
of latent variables, and only use the inputs for cross-attention. In this way, the time and memory requirements don't
depend on the length of the inputs anymore, as one uses a fixed amount of latent variables, like 256 or 512. These are
randomly initialized, after which they are trained end-to-end using backpropagation.

Internally, [`PerceiverModel`] will create the latents, which is a tensor of shape `(batch_size, num_latents,
d_latents)`. One must provide `inputs` (which could be text, images, audio, you name it!) to the model, which it will
use to perform cross-attention with the latents. The output of the Perceiver encoder is a tensor of the same shape. One
can then, similar to BERT, convert the last hidden states of the latents to classification logits by averaging along
the sequence dimension, and placing a linear layer on top of that to project the `d_latents` to `num_labels`.

This was the idea of the original Perceiver paper. However, it could only output classification logits. In a follow-up
work, PerceiverIO, they generalized it to let the model also produce outputs of arbitrary size. How, you might ask? The
idea is actually relatively simple: one defines outputs of an arbitrary size, and then applies cross-attention with the
last hidden states of the latents, using the outputs as queries, and the latents as keys and values.

So let's say one wants to perform masked language modeling (BERT-style) with the Perceiver. As the Perceiver's input
length will not have an impact on the computation time of the self-attention layers, one can provide raw bytes,
providing `inputs` of length 2048 to the model. If one now masks out certain of these 2048 tokens, one can define the
`outputs` as being of shape: `(batch_size, 2048, 768)`. Next, one performs cross-attention with the final hidden states
of the latents to update the `outputs` tensor. After cross-attention, one still has a tensor of shape `(batch_size,
2048, 768)`. One can then place a regular language modeling head on top, to project the last dimension to the
vocabulary size of the model, i.e. creating logits of shape `(batch_size, 2048, 262)` (as Perceiver uses a vocabulary
size of 262 byte IDs).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perceiver_architecture.jpg"
alt="drawing" width="600"/>

<small> Perceiver IO architecture. Taken from the <a href="https://arxiv.org/abs/2105.15203">original paper</a> </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found
[here](https://github.com/deepmind/deepmind-research/tree/master/perceiver).

<Tip warning={true}>

Perceiver does **not** work with `torch.nn.DataParallel` due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035)

</Tip>

## Resources

- The quickest way to get started with the Perceiver is by checking the [tutorial
  notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Perceiver).
- Refer to the [blog post](https://huggingface.co/blog/perceiver) if you want to fully understand how the model works and
is implemented in the library. Note that the models available in the library only showcase some examples of what you can do
with the Perceiver. There are many more use cases, including question answering, named-entity recognition, object detection,
audio classification, video classification, etc.
- [Text classification task guide](../tasks/sequence_classification)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Image classification task guide](../tasks/image_classification)

## Perceiver specific outputs

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverModelOutput

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverDecoderOutput

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverMaskedLMOutput

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverClassifierOutput

## PerceiverConfig


    This is the configuration class to store the configuration of a [`PerceiverModel`]. It is used to instantiate an
    Perceiver model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Perceiver
    [deepmind/language-perceiver](https://huggingface.co/deepmind/language-perceiver) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_latents (`int`, *optional*, defaults to 256):
            The number of latents.
        d_latents (`int`, *optional*, defaults to 1280):
            Dimension of the latent embeddings.
        d_model (`int`, *optional*, defaults to 768):
            Dimension of the inputs. Should only be provided in case [*PerceiverTextPreprocessor*] is used or no
            preprocessor is provided.
        num_blocks (`int`, *optional*, defaults to 1):
            Number of blocks in the Transformer encoder.
        num_self_attends_per_block (`int`, *optional*, defaults to 26):
            The number of self-attention layers per block.
        num_self_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each self-attention layer in the Transformer encoder.
        num_cross_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each cross-attention layer in the Transformer encoder.
        qk_channels (`int`, *optional*):
            Dimension to project the queries + keys before applying attention in the cross-attention and self-attention
            layers of the encoder. Will default to preserving the dimension of the queries if not specified.
        v_channels (`int`, *optional*):
            Dimension to project the values before applying attention in the cross-attention and self-attention layers
            of the encoder. Will default to preserving the dimension of the queries if not specified.
        cross_attention_shape_for_attention (`str`, *optional*, defaults to `"kv"`):
            Dimension to use when downsampling the queries and keys in the cross-attention layer of the encoder.
        self_attention_widening_factor (`int`, *optional*, defaults to 1):
            Dimension of the feed-forward layer in the cross-attention layer of the Transformer encoder.
        cross_attention_widening_factor (`int`, *optional*, defaults to 1):
            Dimension of the feed-forward layer in the self-attention layers of the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_query_residual (`float`, *optional*, defaults to `True`):
            Whether to add a query residual in the cross-attention layer of the encoder.
        vocab_size (`int`, *optional*, defaults to 262):
            Vocabulary size for the masked language modeling model.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that the masked language modeling model might ever be used with. Typically set
            this to something large just in case (e.g., 512 or 1024 or 2048).
        image_size (`int`, *optional*, defaults to 56):
            Size of the images after preprocessing, for [`PerceiverForImageClassificationLearned`].
        train_size (`List[int]`, *optional*, defaults to `[368, 496]`):
            Training size of the images for the optical flow model.
        num_frames (`int`, *optional*, defaults to 16):
            Number of video frames used for the multimodal autoencoding model.
        audio_samples_per_frame (`int`, *optional*, defaults to 1920):
            Number of audio samples per frame for the multimodal autoencoding model.
        samples_per_patch (`int`, *optional*, defaults to 16):
            Number of audio samples per patch when preprocessing the audio for the multimodal autoencoding model.
        output_shape (`List[int]`, *optional*, defaults to `[1, 16, 224, 224]`):
            Shape of the output (batch_size, num_frames, height, width) for the video decoder queries of the multimodal
            autoencoding model. This excludes the channel dimension.
        output_num_channels (`int`, *optional*, defaults to 512):
            Number of output channels for each modalitiy decoder.

    Example:

    ```python
    >>> from transformers import PerceiverModel, PerceiverConfig

    >>> # Initializing a Perceiver deepmind/language-perceiver style configuration
    >>> configuration = PerceiverConfig()

    >>> # Initializing a model from the deepmind/language-perceiver style configuration
    >>> model = PerceiverModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## PerceiverTokenizer


    Construct a Perceiver tokenizer. The Perceiver simply uses raw bytes utf-8 encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        bos_token (`str`, *optional*, defaults to `"[BOS]"`):
            The BOS token (reserved in the vocab, but not actually used).
        eos_token (`str`, *optional*, defaults to `"[EOS]"`):
            The end of sequence token (reserved in the vocab, but not actually used).

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The MASK token, useful for masked language modeling.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The CLS token (reserved in the vocab, but not actually used).
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from two sequences.

    

Methods: __call__

## PerceiverFeatureExtractor

No docstring available for PerceiverFeatureExtractor

Methods: __call__

## PerceiverImageProcessor


    Constructs a Perceiver image processor.

    Args:
        do_center_crop (`bool`, `optional`, defaults to `True`):
            Whether or not to center crop the image. If the input size if smaller than `crop_size` along any edge, the
            image will be padded with zeros and then center cropped. Can be overridden by the `do_center_crop`
            parameter in the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 256, "width": 256}`):
            Desired output size when applying center-cropping. Can be overridden by the `crop_size` parameter in the
            `preprocess` method.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image to `(size["height"], size["width"])`. Can be overridden by the `do_resize`
            parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by the `size` parameter in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Defines the resampling filter to use if resizing the image. Can be overridden by the `resample` parameter
            in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
            in the `preprocess` method.
        do_normalize:
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    

Methods: preprocess

## PerceiverTextPreprocessor

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverTextPreprocessor

## PerceiverImagePreprocessor

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverImagePreprocessor

## PerceiverOneHotPreprocessor

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverOneHotPreprocessor

## PerceiverAudioPreprocessor

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverAudioPreprocessor

## PerceiverMultimodalPreprocessor

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor

## PerceiverProjectionDecoder

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverProjectionDecoder

## PerceiverBasicDecoder

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverBasicDecoder

## PerceiverClassificationDecoder

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverClassificationDecoder

## PerceiverOpticalFlowDecoder

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverOpticalFlowDecoder

## PerceiverBasicVideoAutoencodingDecoder

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverBasicVideoAutoencodingDecoder

## PerceiverMultimodalDecoder

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder

## PerceiverProjectionPostprocessor

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverProjectionPostprocessor

## PerceiverAudioPostprocessor

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverAudioPostprocessor

## PerceiverClassificationPostprocessor

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverClassificationPostprocessor

## PerceiverMultimodalPostprocessor

Could not find docstring for models.perceiver.modeling_perceiver.PerceiverMultimodalPostprocessor

## PerceiverModel

The Perceiver: a scalable, fully attentional architecture.

    <Tip>

        Note that it's possible to fine-tune Perceiver on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        decoder (*DecoderType*, *optional*):
            Optional decoder to use to decode the latent representation of the encoder. Examples include
            *transformers.models.perceiver.modeling_perceiver.PerceiverBasicDecoder*,
            *transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder*,
            *transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder*.
        input_preprocessor (*PreprocessorType*, *optional*):
            Optional input preprocessor to use. Examples include
            *transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor*,
            *transformers.models.perceiver.modeling_perceiver.PerceiverAudioPreprocessor*,
            *transformers.models.perceiver.modeling_perceiver.PerceiverTextPreprocessor*,
            *transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor*.
        output_postprocessor (*PostprocessorType*, *optional*):
            Optional output postprocessor to use. Examples include
            *transformers.models.perceiver.modeling_perceiver.PerceiverImagePostprocessor*,
            *transformers.models.perceiver.modeling_perceiver.PerceiverAudioPostprocessor*,
            *transformers.models.perceiver.modeling_perceiver.PerceiverClassificationPostprocessor*,
            *transformers.models.perceiver.modeling_perceiver.PerceiverProjectionPostprocessor*,
            *transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalPostprocessor*.

        Note that you can define your own decoders, preprocessors and/or postprocessors to fit your use-case.


Methods: forward

## PerceiverForMaskedLM

Example use of Perceiver for masked language modeling.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PerceiverForSequenceClassification

Example use of Perceiver for text classification.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PerceiverForImageClassificationLearned


Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses learned position embeddings. In other words, this model is not given any privileged information about
the structure of images. As shown in the paper, this model can achieve a top-1 accuracy of 72.7 on ImageNet.

[`PerceiverForImageClassificationLearned`] uses [`~models.perceiver.modeling_perceiver.PerceiverImagePreprocessor`]
(with `prep_type="conv1x1"`) to preprocess the input images, and
[`~models.perceiver.modeling_perceiver.PerceiverClassificationDecoder`] to decode the latent representation of
[`PerceiverModel`] into classification logits.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PerceiverForImageClassificationFourier


Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses fixed 2D Fourier position embeddings. As shown in the paper, this model can achieve a top-1 accuracy of
79.0 on ImageNet, and 84.5 when pre-trained on a large-scale dataset (i.e. JFT).

[`PerceiverForImageClassificationLearned`] uses [`~models.perceiver.modeling_perceiver.PerceiverImagePreprocessor`]
(with `prep_type="pixels"`) to preprocess the input images, and
[`~models.perceiver.modeling_perceiver.PerceiverClassificationDecoder`] to decode the latent representation of
[`PerceiverModel`] into classification logits.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PerceiverForImageClassificationConvProcessing


Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses a 2D conv+maxpool preprocessing network. As shown in the paper, this model can achieve a top-1 accuracy
of 82.1 on ImageNet.

[`PerceiverForImageClassificationLearned`] uses [`~models.perceiver.modeling_perceiver.PerceiverImagePreprocessor`]
(with `prep_type="conv"`) to preprocess the input images, and
[`~models.perceiver.modeling_perceiver.PerceiverClassificationDecoder`] to decode the latent representation of
[`PerceiverModel`] into classification logits.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PerceiverForOpticalFlow


Example use of Perceiver for optical flow, for tasks such as Sintel and KITTI. [`PerceiverForOpticalFlow`] uses
[`~models.perceiver.modeling_perceiver.PerceiverImagePreprocessor`] (with *prep_type="patches"*) to preprocess the
input images, and [`~models.perceiver.modeling_perceiver.PerceiverOpticalFlowDecoder`] to decode the latent
representation of [`PerceiverModel`].

As input, one concatenates 2 subsequent frames along the channel dimension and extract a 3 x 3 patch around each pixel
(leading to 3 x 3 x 3 x 2 = 54 values for each pixel). Fixed Fourier position encodings are used to encode the position
of each pixel in the patch. Next, one applies the Perceiver encoder. To decode, one queries the latent representation
using the same encoding used for the input.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## PerceiverForMultimodalAutoencoding


Example use of Perceiver for multimodal (video) autoencoding, for tasks such as Kinetics-700.

[`PerceiverForMultimodalAutoencoding`] uses [`~models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor`] to
preprocess the 3 modalities: images, audio and class labels. This preprocessor uses modality-specific preprocessors to
preprocess every modality separately, after which they are concatenated. Trainable position embeddings are used to pad
each modality to the same number of channels to make concatenation along the time dimension possible. Next, one applies
the Perceiver encoder.

[`~models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder`] is used to decode the latent representation of
[`PerceiverModel`]. This decoder uses each modality-specific decoder to construct queries. The decoder queries are
created based on the inputs after preprocessing. However, autoencoding an entire video in a single forward pass is
computationally infeasible, hence one only uses parts of the decoder queries to do cross-attention with the latent
representation. This is determined by the subsampled indices for each modality, which can be provided as additional
input to the forward pass of [`PerceiverForMultimodalAutoencoding`].

[`~models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder`] also pads the decoder queries of the different
modalities to the same number of channels, in order to concatenate them along the time dimension. Next, cross-attention
is performed with the latent representation of [`PerceiverModel`].

Finally, [`~models.perceiver.modeling_perceiver.PerceiverMultiModalPostprocessor`] is used to turn this tensor into an
actual video. It first splits up the output into the different modalities, and then applies the respective
postprocessor for each modality.

Note that, by masking the classification label during evaluation (i.e. simply providing a tensor of zeros for the
"label" modality), this auto-encoding model becomes a Kinetics 700 video classifier.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
