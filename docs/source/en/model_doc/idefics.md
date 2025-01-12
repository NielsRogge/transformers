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

# IDEFICS

## Overview

The IDEFICS model was proposed in [OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents
](https://huggingface.co/papers/2306.16527
) by Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh

The abstract from the paper is the following:

*Large multimodal models trained on natural documents, which interleave images and text, outperform models trained on image-text pairs on various multimodal benchmarks that require reasoning over one or multiple images to generate a text. However, the datasets used to train these models have not been released, and the collection process has not been fully specified. We introduce the OBELICS dataset, an open web-scale filtered dataset of interleaved image-text documents comprising 141 million web pages extracted from Common Crawl, 353 million associated images, and 115 billion text tokens. We describe the dataset creation process, present comprehensive filtering rules, and provide an analysis of the dataset's content. To show the viability of OBELISC, we train an 80 billion parameters vision and language model on the dataset and obtain competitive performance on various multimodal benchmarks. We release the code to reproduce the dataset along with the dataset itself.*

This model was contributed by [HuggingFaceM4](https://huggingface.co/HuggingFaceM4). The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>). (TODO: don't have a public link yet).


<Tip warning={true}>

IDEFICS modeling code in Transformers is for finetuning and inferencing the pre-trained IDEFICS models.

To train a new IDEFICS model from scratch use the m4 codebase (a link will be provided once it's made public)

</Tip>


## IdeficsConfig


    This is the configuration class to store the configuration of a [`IdeficsModel`]. It is used to instantiate an
    Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Idefics-9B.

    e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        additional_vocab_size (`int`, *optional*, defaults to 0):
            Additional vocabulary size of the model, typically for the special "<img>" token. Additional vocab tokens
            are always trainable whereas regular vocab tokens can be frozen or not.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Idefics model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~IdeficsModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        alpha_initializer (`str`, *optional*, defaults to `"zeros"`):
            Initialization type for the alphas.
        alphas_initializer_range (`float`, *optional*, defaults to 0.0):
            The standard deviation of the truncated_normal_initializer for initializing the alphas in the Gated Cross
            Attention.
        alpha_type (`str`, *optional*, defaults to `"float"`):
            Whether the gating alphas should be vectors or single floats.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0)
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1)
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2)
            End of stream token id.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        cross_layer_interval (`int`, *optional*, default to 1)
            Interval for cross attention (from text to image) layers.
        qk_layer_norms (`bool`, *optional*, defaults to `False`): Whether to add layer norm after q and k
        freeze_text_layers (`bool`, *optional*, defaults to `True`): Whether to freeze text layers
        freeze_text_module_exceptions (`bool`, *optional*, defaults to `[]`):
            Exceptions to freezing text layers when `freeze_text_layers` is `True`
        freeze_lm_head (`bool`, *optional*, defaults to `False`): Whether to freeze lm head
        freeze_vision_layers (`bool`, *optional*, defaults to `True`):  Whether to freeze vision layers
        freeze_vision_module_exceptions (`bool`, *optional*, defaults to `[]`):
            Exceptions to freezing vision layers when `freeze_vision_layers` is `True`
        use_resampler (`bool`, *optional*, defaults to `False`): Whether to use the Resampler
        vision_config (`IdeficsVisionConfig`,  *optional*): Custom vision config or dict
        perceiver_config (`IdeficsPerceiverConfig`,  *optional*): Custom perceiver config or dict

    Example:

    ```python
    >>> from transformers import IdeficsModel, IdeficsConfig

    >>> # Initializing a Idefics idefics-9b style configuration
    >>> configuration = IdeficsConfig()

    >>> # Initializing a model from the idefics-9b style configuration
    >>> model = IdeficsModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## IdeficsModel

The bare LLaMA Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`IdeficsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Transformer decoder consisting of `config.num_hidden_layers` layers. Each layer is a [`IdeficsDecoderLayer`]

    Args:
        config: IdeficsConfig
    

Methods: forward

## IdeficsForVisionText2Text

No docstring available for IdeficsForVisionText2Text

Methods: forward

## TFIdeficsModel

No docstring available for TFIdeficsModel

Methods: call

## TFIdeficsForVisionText2Text

No docstring available for TFIdeficsForVisionText2Text

Methods: call

## IdeficsImageProcessor


    Constructs a Idefics image processor.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            Resize to image size
        image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        image_num_channels (`int`, *optional*, defaults to 3):
            Number of image channels.
    

Methods: preprocess

## IdeficsProcessor


    Constructs a IDEFICS processor which wraps a LLama tokenizer and IDEFICS image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`IdeficsImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`IdeficsImageProcessor`):
            An instance of [`IdeficsImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizerFast`):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
        image_size (`int`, *optional*, defaults to 224):
            Image size (assuming a square image)
        add_end_of_utterance_token (`str`, *optional*):
            The string representation of token representing end of utterance
    

Methods: __call__
