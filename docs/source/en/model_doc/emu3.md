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

# Emu3

## Overview

The Emu3 model was proposed in [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869) by Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, Zhongyuan Wang.

Emu3 is a multimodal LLM that uses vector quantization to tokenize images into discrete tokens. Discretized image tokens are later fused with text token ids for image and text generation. The model can additionally generate images by predicting image token ids. 


The abstract from the paper is the following:

*While next-token prediction is considered a promising path towards artificial general intelligence, it has struggled to excel in multimodal tasks, which are still dominated by diffusion models (e.g., Stable Diffusion) and compositional approaches (e.g., CLIP combined with LLMs). In this paper, we introduce Emu3, a new suite of state-of-the-art multimodal models trained solely with next-token prediction. By tokenizing images, text, and videos into a discrete space, we train a single transformer from scratch on a mixture of multimodal sequences. Emu3 outperforms several well-established task-specific models in both generation and perception tasks, surpassing flagship models such as SDXL and LLaVA-1.6, while eliminating the need for diffusion or compositional architectures. Emu3 is also capable of generating high-fidelity video via predicting the next token in a video sequence. We simplify complex multimodal model designs by converging on a singular focus: tokens, unlocking great potential for scaling both during training and inference. Our results demonstrate that next-token prediction is a promising path towards building general multimodal intelligence beyond language. We open-source key techniques and models to support further research in this direction.*

Tips:

- We advise users to set `processor.tokenizer.padding_side = "left"` before batched generation as it leads to more accurate results.

- Note that the model has been trained with a specific prompt format for chatting. Use `processor.apply_chat_template(my_conversation_dict)` to correctly format your prompts.

- Emu3 has two different checkpoints for image-generation and text-generation, make sure to use the correct checkpoint when loading the model. To generate an image, it is advised to use `prefix_constraints` so that the generated tokens are sampled only from possible image tokens. See more below for usage examples.

> [!TIP]
> Emu3 implementation in Transformers uses a special image token to indicate where to merge image embeddings. The special image token isn't new and uses one of the reserved tokens: `<|extra_0|>`. You have to add `<image>` to your prompt in the place where the image should be embedded for correct generation.


This model was contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/baaivision/Emu3).


## Usage example

### Text generation inference

Here's how to load the model and perform inference in half-precision (`torch.bfloat16`) to generate textual output from text or text and image inputs:

```python
from transformers import Emu3Processor, Emu3ForConditionalGeneration
import torch
from PIL import Image
import requests

processor = Emu3Processor.from_pretrained("Emu3-community/Emu3-Chat-hf")
model = Emu3ForConditionalGeneration.from_pretrained("Emu3-community/Emu3-Chat-hf", torch_dtype=torch.bfloat16, device_map="cuda")

# prepare image and text prompt
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
prompt = "What do you see in this image?<image>"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Image generation inference

Emu3 can also generate images from textual input. Here is how you can do it:

```python
processor = Emu3Processor.from_pretrained("Emu3-community/Emu3-Gen-hf")
model = Emu3ForConditionalGeneration.from_pretrained("Emu3-community/Emu3-Gen-hf", torch_dtype="bfloat16", device_map="auto", attn_implementation="flash_attention_2")


inputs = processor(
    text=["a portrait of young girl. masterpiece, film grained, best quality.", "a dog running under the rain"],
    padding=True,
    return_tensors="pt",
    return_for_image_generation=True,
)
inputs = inputs.to(device="cuda:0", dtype=torch.bfloat16)

neg_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
neg_inputs = processor(text=[neg_prompt] * 2, return_tensors="pt").to(device="cuda:0")

image_sizes = inputs.pop("image_sizes")
HEIGHT, WIDTH = image_sizes[0]
VISUAL_TOKENS = model.vocabulary_mapping.image_tokens

def prefix_allowed_tokens_fn(batch_id, input_ids):
    height, width = HEIGHT, WIDTH
    visual_tokens = VISUAL_TOKENS
    image_wrapper_token_id = torch.tensor([processor.tokenizer.image_wrapper_token_id], device=model.device)
    eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=model.device)
    eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=model.device)
    pad_token_id = torch.tensor([processor.tokenizer.pad_token_id], device=model.device)
    eof_token_id = torch.tensor([processor.tokenizer.eof_token_id], device=model.device)
    eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]

    position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
    offset = input_ids.shape[0] - position
    if offset % (width + 1) == 0:
        return (eol_token_id, )
    elif offset == (width + 1) * height + 1:
        return (eof_token_id, )
    elif offset == (width + 1) * height + 2:
        return (eoi_token_id, )
    elif offset == (width + 1) * height + 3:
        return (eos_token_id, )
    elif offset > (width + 1) * height + 3:
        return (pad_token_id, )
    else:
        return visual_tokens


out = model.generate(
    **inputs,
    max_new_tokens=50_000, # make sure to have enough tokens for one image
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    return_dict_in_generate=True,
    negative_prompt_ids=neg_inputs.input_ids, # indicate for Classifier-Free Guidance
    negative_prompt_attention_mask=neg_inputs.attention_mask,
)

image = model.decode_image_tokens(out.sequences[:, inputs.input_ids.shape[1]: ], height=HEIGHT, width=WIDTH)
images = processor.postprocess(list(image.float()), return_tensors="PIL.Image.Image") # internally we convert to np but it's not supported in bf16 precision
for i, image in enumerate(images['pixel_values']):
    image.save(f"result{i}.png")

```


## Emu3Config


    This is the configuration class to store the configuration of a [`Emu3Model`]. It is used to instantiate a
    emu3 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [Emu3-community/Emu3-Chat-hf](https://huggingface.co/Emu3-community/Emu3-Chat-hf).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vq_config (`Union[Dict, Emu3VQVAEConfig]`, *optional*):
            Emu3VQVAEConfig instance containing the configuration for the VQ-VAE model.
        text_config (`Union[Dict, Emu3TextConfig]``, *optional*):
            Emu3TextConfig instance containing the configuration for the language model.
        vocabulary_map (`dict`, *optional*):
            A dictionary containing the vocabulary map from the tokenizer. Used to obtain tokens from the image inputs.
    

## Emu3VQVAEConfig


    This is the configuration class to store the configuration of a [`Emu3VQVAE`]. It is used to instantiate an VQ-VAE
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a configuration to the VQ model presented in Emu3 paper.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        codebook_size (`int`, *optional*, defaults to 32768):
            Codebook size of the VQ model.
        embed_dim (`int`, *optional*, defaults to 4):
            Dimension of the quantized vector in codebook.
        latent_channels (`int`, *optional*, defaults to 4):
            Dimension of the output channel of encoder and the input channel of decoder
        double_latent (`bool`, *optional*, defaults to `False`):
            Whether double the output dim of the encoder.
        in_channels (`int`, *optional*, defaults to 3):
            Input channel of encoder.
        out_channels (`int`, *optional*, defaults to 3):
            Output channel of decoder.
        temporal_downsample_factor (`int`, *optional*, defaults to 4):
            Temporal downsample factor.
        base_channels (`int`, *optional*, defaults to 256):
            Basic channel number of the intermediate blocks.
        channel_multiplier (`List[int]`, *optional*, defaults to `[1, 2, 2, 4]`):
            Channel scaling factor of the intermediate blocks.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Residual block number in each stage.
        attn_resolutions (`List[int]`, *optional*, defaults to `[3]`):
            Stage indices to apply attention.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations in the attention layer.
        num_attention_heads (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Emu3VQVAE, Emu3VQVAEConfig

    >>> # Initializing a video VQ model of Emu3 configuration
    >>> configuration = Emu3VQVAEConfig()

    >>> # Initializing a model from the Emu3 VQ model style configuration
    >>> model = Emu3VQVAE(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## Emu3TextConfig


    This is the configuration class to store the configuration of a [`Emu3TextModel`]. It is used to instantiate a
    emu3 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [Emu3-community/Emu3-Chat-hf](https://huggingface.co/Emu3-community/Emu3-Chat-hf).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 184622):
            Vocabulary size of the Emu3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Emu3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 9216):
            The maximum sequence length that this model might ever be used with. Emu supports up to 9216 tokens,
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 151643):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 151849):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 151850):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.


    ```python
    >>> from transformers import Emu3Model, Emu3Config

    >>> # Initializing a Emu3-community/Emu3-Chat-hf style configuration
    >>> configuration = Emu3Config()

    >>> # Initializing a model from the Emu3-community/Emu3-Chat-hf style configuration
    >>> model = Emu3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## Emu3Processor


    Constructs a Emu3 processor which wraps a Emu3 image processor and a GPT2 tokenizer into a single
    processor.

    [`Emu3Processor`] offers all the functionalities of [`Emu3ImageProcessor`] and [`GPT2TokenizerFast`].
    See the [`~Emu3Processor.__call__`] and [`~Emu3Processor.decode`] for more information.

    Args:
        image_processor ([`Emu3ImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`Emu3TokenizerFast`]):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    

## Emu3ImageProcessor


    Constructs a Emu3 image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_pad (`bool`, *optional*, defaults to `True`):
                Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        min_pixels (`int`, *optional*, defaults to `512 * 512`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `1024 * 1024`):
            The max pixels of the image to resize the image.
        spatial_factor (`int`, *optional*, defaults to 8):
            The spatial downsample factor the image will be downsampled in feature extracting phase
    

Methods: preprocess

## Emu3VQVAE

The VQ-VAE model used in Emu3 for encoding/decoding images into discrete tokens.
    This model follows the "Make-a-scene: Scene-based text-to-image generation with human priors" paper from
    [ Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman](https://arxiv.org/abs/2203.13131).
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Emu3VQVAEConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## Emu3TextModel

The bare Emu3Text Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Emu3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Emu3TextDecoderLayer`]

    Args:
        config: Emu3TextConfig
    

Methods: forward

## Emu3ForCausalLM

No docstring available for Emu3ForCausalLM

Methods: forward

## Emu3ForConditionalGeneration

No docstring available for Emu3ForConditionalGeneration

Methods: forward
