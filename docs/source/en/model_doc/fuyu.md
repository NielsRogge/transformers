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

# Fuyu

## Overview

The Fuyu model was created by [ADEPT](https://www.adept.ai/blog/fuyu-8b), and authored by Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, Sağnak Taşırlar.

The authors introduced Fuyu-8B, a decoder-only multimodal model based on the classic transformers architecture, with query and key normalization. A linear encoder is added to create multimodal embeddings from image inputs.

By treating image tokens like text tokens and using a special image-newline character, the model knows when an image line ends. Image positional embeddings are removed. This avoids the need for different training phases for various image resolutions. With 8 billion parameters and licensed under CC-BY-NC, Fuyu-8B is notable for its ability to handle both text and images, its impressive context size of 16K, and its overall performance.

<Tip warning={true}>

The `Fuyu` models were trained using `bfloat16`, but the original inference uses `float16` The checkpoints uploaded on the hub use `torch_dtype = 'float16'` which will be
used by the `AutoModel` API to cast the checkpoints from `torch.float32` to `torch.float16`.

The `dtype` of the online weights is mostly irrelevant, unless you are using `torch_dtype="auto"` when initializing a model using `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`. The reason is that the model will first be downloaded ( using the `dtype` of the checkpoints online) then it will be cast to the default `dtype` of `torch` (becomes `torch.float32`). Users should specify the `torch_dtype` they want, and if they don't it will be `torch.float32`.

Finetuning the model in `float16` is not recommended and known to produce `nan`, as such the model should be fine-tuned in `bfloat16`.

</Tip>


Tips:

- To convert the model, you need to clone the original repository using `git clone https://github.com/persimmon-ai-labs/adept-inference`, then get the checkpoints:

```bash
git clone https://github.com/persimmon-ai-labs/adept-inference
wget path/to/fuyu-8b-model-weights.tar
tar -xvf fuyu-8b-model-weights.tar
python src/transformers/models/fuyu/convert_fuyu_weights_to_hf.py  --input_dir /path/to/downloaded/fuyu/weights/ --output_dir /output/path \
    --pt_model_path /path/to/fuyu_8b_release/iter_0001251/mp_rank_00/model_optim_rng.pt
    --ada_lib_path /path/to/adept-inference
```

For the chat model:
```bash
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
tar -xvf 8b_base_model_release.tar
```
Then, model can be loaded via:

```py
from transformers import FuyuConfig, FuyuForCausalLM
model_config = FuyuConfig()
model = FuyuForCausalLM(model_config).from_pretrained('/output/path')
```

Inputs need to be passed through a specific Processor to have the correct formats.
A processor requires an image_processor and a tokenizer. Hence, inputs can be loaded via:

```py
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.fuyu.processing_fuyu import FuyuProcessor
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor


tokenizer = AutoTokenizer.from_pretrained('adept-hf-collab/fuyu-8b')
image_processor = FuyuImageProcessor()


processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
text_prompt = "Generate a coco-style caption.\\n"

bus_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
bus_image_pil = Image.open(io.BytesIO(requests.get(bus_image_url).content))
inputs_to_model = processor(images=bus_image_pil, text=text_prompt)


```

This model was contributed by [Molbap](https://huggingface.co/Molbap).
The original code can be found [here](https://github.com/persimmon-ai-labs/adept-inference).

- Fuyu uses a `sentencepiece` based tokenizer, with a `Unigram` model. It supports bytefallback, which is only available in `tokenizers==0.14.0` for the fast tokenizer.
The `LlamaTokenizer` is used as it is a standard wrapper around sentencepiece.

- The authors suggest to use the following prompt for image captioning: `f"Generate a coco-style caption.\\n"`


## FuyuConfig


    This is the configuration class to store the configuration of a [`FuyuForCausalLM`]. It is used to instantiate an
    Fuyu model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 262144):
            Vocabulary size of the Fuyu model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FuyuForCausalLM`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 16384):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu2"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 16384):
            The maximum sequence length that this model might ever be used with.
        image_size (`int`, *optional*, defaults to 300):
            The input image size.
        patch_size (`int`, *optional*, defaults to 30):
            The input vision transformer encoding patch size.
        num_channels (`int`, *optional*, defaults to 3):
            The input image number of channels.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`. Whether to tie weight embeddings
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 25000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalFuyu/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        qk_layernorm (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the Queries and Keys after projecting the hidden states
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio after applying the MLP to the hidden states.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio after computing the attention scores.
        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
            Percentage of the query and keys which will have rotary embedding.

        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`Union[int, List[int]]`, *optional*, defaults to 2):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize the `language``[`Aut`].

    ```python
    >>> from transformers import FuyuConfig

    >>> # Initializing a Fuyu fuyu-7b style configuration
    >>> configuration = FuyuConfig()
    ```

## FuyuForCausalLM

Fuyu Model with a language modeling head on top for causal language model conditioned on image patches and text.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FuyuConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## FuyuImageProcessor


    This class should handle the image processing part before the main FuyuForCausalLM. In particular, it should
    handle:

    - Processing Images:
        Taking a batch of images as input. If the images are variable-sized, it resizes them based on the desired patch
        dimensions. The image output is always img_h, img_w of (1080, 1920)

        Then, it patches up these images using the patchify_image function.

    - Creating Image Input IDs:
        For each patch, a placeholder ID is given to identify where these patches belong in a token sequence. For
        variable-sized images, each line of patches is terminated with a newline ID.

    - Image Patch Indices:
        For each image patch, the code maintains an index where these patches should be inserted in a token stream.


    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image to `size`.
        size (`Dict[str, int]`, *optional*, defaults to `{"height": 1080, "width": 1920}`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to `size`.
        padding_value (`float`, *optional*, defaults to 1.0):
            The value to pad the image with.
        padding_mode (`str`, *optional*, defaults to `"constant"`):
            The padding mode to use when padding the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float`, *optional*, defaults to 0.5):
            The mean to use when normalizing the image.
        image_std (`float`, *optional*, defaults to 0.5):
            The standard deviation to use when normalizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to `1 / 255`):
            The factor to use when rescaling the image.
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 30, "width": 30}`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
    

Methods: __call__

## FuyuProcessor


    Constructs a Fuyu processor which wraps a Fuyu image processor and a Llama tokenizer into a single processor.

    [`FuyuProcessor`] offers all the functionalities of [`FuyuImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~FuyuProcessor.__call__`] and [`~FuyuProcessor.decode`] for more information.

    Args:
        image_processor ([`FuyuImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
    

Methods: __call__
