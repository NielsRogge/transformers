<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Chameleon

## Overview

The Chameleon model was proposed in [Chameleon: Mixed-Modal Early-Fusion Foundation Models
](https://arxiv.org/abs/2405.09818v1) by META AI Chameleon Team. Chameleon is a Vision-Language Model that use vector quantization to tokenize images which enables the model to generate multimodal output. The model takes images and texts as input, including an interleaved format, and generates textual response. Image generation module is not released yet.


The abstract from the paper is the following:

*We present Chameleon, a family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. We outline a stable training
approach from inception, an alignment recipe, and an architectural parameterization tailored for the
early-fusion, token-based, mixed-modal setting. The models are evaluated on a comprehensive range
of tasks, including visual question answering, image captioning, text generation, image generation, and
long-form mixed modal generation. Chameleon demonstrates broad and general capabilities, including
state-of-the-art performance in image captioning tasks, outperforms Llama-2 in text-only tasks while
being competitive with models such as Mixtral 8x7B and Gemini-Pro, and performs non-trivial image
generation, all in a single model. It also matches or exceeds the performance of much larger models,
including Gemini Pro and GPT-4V, according to human judgments on a new long-form mixed-modal
generation evaluation, where either the prompt or outputs contain mixed sequences of both images and
text. Chameleon marks a significant step forward in unified modeling of full multimodal documents*


<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/chameleon_arch.png"
alt="drawing" width="600"/>

<small> Chameleon incorporates a vector quantizer module to transform images into discrete tokens. That also enables image generation using an auto-regressive transformer. Taken from the <a href="https://arxiv.org/abs/2405.09818v1">original paper.</a> </small>

This model was contributed by [joaogante](https://huggingface.co/joaogante) and [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/facebookresearch/chameleon).


## Usage tips

- We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to set `processor.tokenizer.padding_side = "left"` before generating.

- Note that Chameleon was tuned for safety alignment. If the model is refusing to answer, consider asking a more concrete question, instead of an open question.

- Chameleon generates in chat format which means that the generated text will always be the "assistant's turn". You can enable a text completion generation by passing `return_for_text_completion=True` when calling the processor.

> [!NOTE]
> Chameleon implementation in Transformers uses a special image token to indicate where to merge image embeddings. For special image token we didn't add a new one but used one of the reserved tokens: `<reserved08707>`. You have to add `<image>` to your prompt in the place where the image should be embedded for correct generation.

## Usage example

### Single image inference

Chameleon is a gated model so make sure to have access and login to Hugging Face Hub using a token.
Here's how to load the model and perform inference in half-precision (`torch.bfloat16`):

```python
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
from PIL import Image
import requests

processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda")

# prepare image and text prompt
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
prompt = "What do you see in this image?<image>"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Multi image inference

Chameleon can perform inference with multiple images as input, where images either belong to the same prompt or different prompts (in batched inference). Here is how you can do it:

```python
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
from PIL import Image
import requests

processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16, device_map="cuda")

# Get three different images
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image_stop = Image.open(requests.get(url, stream=True).raw)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_cats = Image.open(requests.get(url, stream=True).raw)

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image_snowman = Image.open(requests.get(url, stream=True).raw)

# Prepare a batched prompt, where the first one is a multi-image prompt and the second is not
prompts = [
    "What do these images have in common?<image><image>",
    "<image>What is shown in this image?"
]

# We can simply feed images in the order they have to be used in the text prompt
# Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
inputs = processor(images=[image_stop, image_cats, image_snowman], text=prompts, padding=True, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

## Model optimization

### Quantization using Bitsandbytes

The model can be loaded in 8 or 4 bits, greatly reducing the memory requirements while maintaining the performance of the original model. First make sure to install bitsandbytes, `pip install bitsandbytes` and to have access to a GPU/accelerator that is supported by the library.

<Tip>

bitsandbytes is being refactored to support multiple backends beyond CUDA. Currently, ROCm (AMD GPU) and Intel CPU implementations are mature, with Intel XPU in progress and Apple Silicon support expected by Q4/Q1. For installation instructions and the latest backend updates, visit [this link](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend).

We value your feedback to help identify bugs before the full release! Check out [these docs](https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends) for more details and feedback links.

</Tip>

Simply change the snippet above with:

```python
from transformers import ChameleonForConditionalGeneration, BitsAndBytesConfig

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", quantization_config=quantization_config, device_map="cuda")
```

### Use Flash-Attention 2 and SDPA to further speed-up generation

The models supports both, Flash-Attention 2 and PyTorch's [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) which can be enables for optimization. SDPA is the default options when you load the model, If you want to switch for Flash Attention 2, first make sure to install flash-attn. Refer to the [original repository](https://github.com/Dao-AILab/flash-attention) regarding that package installation. Simply change the snippet above with:

```python
from transformers import ChameleonForConditionalGeneration

model_id = "facebook/chameleon-7b"
model = ChameleonForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2"
).to(0)
```

## ChameleonConfig


    This is the configuration class to store the configuration of a [`ChameleonModel`]. It is used to instantiate a
    chameleon model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [meta/chameleon-7B](https://huggingface.co/meta/chameleon-7B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the chameleon model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ChameleonModel`]; this includes text and image tokens.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Chameleon supports up to 4096 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/Localchameleon/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        model_parallel_size (`int`, *optional*, defaults to 1):
            Number of shards used when training the model. This will be used in qk layernorm because the original Chameleon inference
            doesn't do reduction in those layers and each rank has its own biases.
        swin_norm (`bool`, *optional*, defaults to `False`):
            Use Swin Transformer normalization.
        vq_config (`dict`, *optional*):
            ChameleonVQConfig instance containing the configuration for the VQ-VAE model.
        vocabulary_map (`dict`, *optional*):
            A dictionary containing the vocabulary map from the tokenizer. Used to obtain tokens from the image inputs.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.


    ```python
    >>> from transformers import ChameleonModel, ChameleonConfig

    >>> # Initializing a chameleon chameleon-7b style configuration
    >>> configuration = ChameleonConfig()

    >>> # Initializing a model from the chameleon-7b style configuration
    >>> model = ChameleonModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## ChameleonVQVAEConfig


    This is the configuration class to store the configuration of a [`ChameleonVQModel`]. It is used to instantiate a
    `ChameleonVQModel` according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a
    configuration with the defaults will yield a similar configuration to the VQModel of the
    [meta/chameleon-7B](https://huggingface.co/meta/chameleon-7B).

    Args:
        embed_dim (`int`, *optional*, defaults to 256):
            Dimensionality of each embedding vector.
        num_embeddings (`int`, *optional*, defaults to 8192):
            Number of codebook embeddings.
        double_latent (`bool`, *optional*, defaults to `False`):
            Whether to use double z channels.
        latent_channels (`int`, *optional*, defaults to 256):
            Number of channels for the latent space.
        resolution (`int`, *optional*, defaults to 512):
            Resolution of the input images.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        base_channels (`int`, *optional*, defaults to 128):
            Base channel count.
        channel_multiplier (`List[int]`, *optional*, defaults to `[1, 1, 2, 2, 4]`):
            Channel multipliers for each resolution.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Number of residual blocks.
        attn_resolutions (`List[int]`, *optional*):
            Resolutions to apply attention.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate.
        attn_type (`str`, *optional*, defaults to `"vanilla"`):
            Attention type used in VQ-GAN encoder. Can be "vanilla" or None.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    

## ChameleonProcessor


    Constructs a Chameleon processor which wraps a Chameleon image processor and a Chameleon tokenizer into a single
    processor.

    [`ChameleonProcessor`] offers all the functionalities of [`ChameleonImageProcessor`] and [`LlamaTokenizerFast`].
    See the [`~ChameleonProcessor.__call__`] and [`~ChameleonProcessor.decode`] for more information.

    Args:
        image_processor ([`ChameleonImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
        image_seq_length (`int`, *optional*, defaults to 1024):
            Sequence length of one image embedding.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The special token used to indicate image in the text.
    

## ChameleonImageProcessor


    Constructs a Chameleon image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 512}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to 1):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to {"height": 512, "width": 512}):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to 0.0078):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[1.0, 1.0, 1.0]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[1.0, 1.0, 1.0]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    

Methods: preprocess

## ChameleonVQVAE

The VQ-VAE model used in Chameleon for encoding/decoding images into discrete tokens.
    This model follows the "Make-a-scene: Scene-based text-to-image generation with human priors" paper from
    [ Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman](https://arxiv.org/abs/2203.13131).
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ChameleonVQVAEConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## ChameleonModel

The bare chameleon Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ChameleonConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ChameleonDecoderLayer`]

    Args:
        config: ChameleonConfig
    

Methods: forward

## ChameleonForConditionalGeneration

Chameleon Model with a head on top used for outputting logits for next token prediction.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ChameleonConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
