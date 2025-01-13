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

# Quantization

Quantization techniques reduce memory and computational costs by representing weights and activations with lower-precision data types like 8-bit integers (int8). This enables loading larger models you normally wouldn't be able to fit into memory, and speeding up inference. Transformers supports the AWQ and GPTQ quantization algorithms and it supports 8-bit and 4-bit quantization with bitsandbytes.

Quantization techniques that aren't supported in Transformers can be added with the [`HfQuantizer`] class.

<Tip>

Learn how to quantize models in the [Quantization](../quantization) guide.

</Tip>

## QuantoConfig


    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `quanto`.

    Args:
        weights (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are ("float8","int8","int4","int2")
        activations (`str`, *optional*):
            The target dtype for the activations after quantization. Supported values are (None,"int8","float8")
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    

## AqlmConfig


    This is a wrapper class about `aqlm` parameters.

    Args:
        in_group_size (`int`, *optional*, defaults to 8):
            The group size along the input dimension.
        out_group_size (`int`, *optional*, defaults to 1):
            The group size along the output dimension. It's recommended to always use 1.
        num_codebooks (`int`, *optional*, defaults to 1):
            Number of codebooks for the Additive Quantization procedure.
        nbits_per_codebook (`int`, *optional*, defaults to 16):
            Number of bits encoding a single codebook vector. Codebooks size is 2**nbits_per_codebook.
        linear_weights_not_to_quantize (`Optional[List[str]]`, *optional*):
            List of full paths of `nn.Linear` weight parameters that shall not be quantized.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    

## VptqConfig


    This is a wrapper class about `vptq` parameters.

    Args:
        enable_proxy_error (`bool`, *optional*, defaults to `False`): calculate proxy error for each layer
        config_for_layers (`Dict`, *optional*, defaults to `{}`): quantization params for each layer
        shared_layer_config (`Dict`, *optional*, defaults to `{}`): shared quantization params among layers
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    

## AwqConfig


    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `auto-awq` library awq quantization relying on auto_awq backend.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to.
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        zero_point (`bool`, *optional*, defaults to `True`):
            Whether to use zero point quantization.
        version (`AWQLinearVersion`, *optional*, defaults to `AWQLinearVersion.GEMM`):
            The version of the quantization algorithm to use. GEMM is better for big batch_size (e.g. >= 8) otherwise,
            GEMV is better (e.g. < 8 ). GEMM models are compatible with Exllama kernels.
        backend (`AwqBackendPackingMethod`, *optional*, defaults to `AwqBackendPackingMethod.AUTOAWQ`):
            The quantization backend. Some models might be quantized using `llm-awq` backend. This is useful for users
            that quantize their own models using `llm-awq` library.
        do_fuse (`bool`, *optional*, defaults to `False`):
            Whether to fuse attention and mlp layers together for faster inference
        fuse_max_seq_len (`int`, *optional*):
            The Maximum sequence length to generate when using fusing.
        modules_to_fuse (`dict`, *optional*, default to `None`):
            Overwrite the natively supported fusing scheme with the one specified by the users.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
            Note you cannot quantize directly with transformers, please refer to `AutoAWQ` documentation for quantizing HF models.
        exllama_config (`Dict[str, Any]`, *optional*):
            You can specify the version of the exllama kernel through the `version` key, the maximum sequence
            length through the `max_input_len` key, and the maximum batch size through the `max_batch_size` key.
            Defaults to `{"version": 2, "max_input_len": 2048, "max_batch_size": 8}` if unset.
    

## EetqConfig

    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `eetq`.

    Args:
        weights (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights. Supported value is only "int8"
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
    

## GPTQConfig


    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum` api for gptq quantization relying on auto_gptq backend.

    Args:
        bits (`int`):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`Union[List[str]]`, *optional*):
            The dataset used for quantization. You can provide your own dataset in a list of string or just use the
            original datasets used in GPTQ paper ['wikitext2','c4','c4-new']
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        damp_percent (`float`, *optional*, defaults to 0.1):
            The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.
        desc_act (`bool`, *optional*, defaults to `False`):
            Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly
            speed up inference but the perplexity may become slightly worse. Also known as act-order.
        sym (`bool`, *optional*, defaults to `True`):
            Whether to use symetric quantization.
        true_sequential (`bool`, *optional*, defaults to `True`):
            Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing
            the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes
            quantization using inputs that have passed through the previously quantized layers.
        use_cuda_fp16 (`bool`, *optional*, defaults to `False`):
            Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16.
        model_seqlen (`int`, *optional*):
            The maximum sequence length that the model can take.
        block_name_to_quantize (`str`, *optional*):
            The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
        module_name_preceding_first_block (`List[str]`, *optional*):
            The layers that are preceding the first Transformer block.
        batch_size (`int`, *optional*, defaults to 1):
            The batch size used when processing the dataset
        pad_token_id (`int`, *optional*):
            The pad token id. Needed to prepare the dataset when `batch_size` > 1.
        use_exllama (`bool`, *optional*):
            Whether to use exllama backend. Defaults to `True` if unset. Only works with `bits` = 4.
        max_input_length (`int`, *optional*):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input
            length. It is specific to the exllama backend with act-order.
        exllama_config (`Dict[str, Any]`, *optional*):
            The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults
            to `{"version": 1}` if unset.
        cache_block_outputs (`bool`, *optional*, defaults to `True`):
            Whether to cache block outputs to reuse as inputs for the succeeding block.
        modules_in_block_to_quantize (`List[List[str]]`, *optional*):
            List of list of module names to quantize in the specified block. This argument is useful to exclude certain linear modules from being quantized.
            The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially. If not set, we will quantize all linear layers.
            Example: `modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]]`.
            In this example, we will first quantize the q,k,v layers simultaneously since they are independent.
            Then, we will quantize `self_attn.o_proj` layer with the q,k,v layers quantized. This way, we will get
            better results since it reflects the real input `self_attn.o_proj` will get when the model is quantized.
    

## BitsAndBytesConfig


    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`List[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
            This sets the computational type which might be different than the input type. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        bnb_4bit_quant_storage (`torch.dtype` or str, *optional*, defaults to `torch.uint8`):
            This sets the storage type to pack the quanitzed 4-bit prarams.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    

## HfQuantizer

quantizers.base.HfQuantizer

    Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization.
    This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method
    yet.

    Attributes
        quantization_config (`transformers.utils.quantization_config.QuantizationConfigMixin`):
            The quantization config that defines the quantization parameters of your model that you want to quantize.
        modules_to_not_convert (`List[str]`, *optional*):
            The list of module names to not convert when quantizing the model.
        required_packages (`List[str]`, *optional*):
            The list of required pip packages to install prior to using the quantizer
        requires_calibration (`bool`):
            Whether the quantization method requires to calibrate the model before using it.
        requires_parameters_quantization (`bool`):
            Whether the quantization method requires to create a new Parameter. For example, for bitsandbytes, it is
            required to create a new xxxParameter in order to properly quantize the model.
    

## HiggsConfig


    HiggsConfig is a configuration class for quantization using the HIGGS method.

    Args:
        bits (int, *optional*, defaults to 4):
            Number of bits to use for quantization. Can be 2, 3 or 4. Default is 4.
        p (int, *optional*, defaults to 2):
            Quantization grid dimension. 1 and 2 are supported. 2 is always better in practice. Default is 2.
        modules_to_not_convert (`list`, *optional*, default to ["lm_head"]):
            List of linear layers that should not be quantized.
        hadamard_size (int, *optional*, defaults to 512):
            Hadamard size for the HIGGS method. Default is 512. Input dimension of matrices is padded to this value. Decreasing this below 512 will reduce the quality of the quantization.
        group_size (int, *optional*, defaults to 256):
            Group size for the HIGGS method. Can be 64, 128 or 256. Decreasing it barely affects the performance. Default is 256. Must be a divisor of hadamard_size.
    

## HqqConfig


    This is wrapper around hqq's BaseQuantizeConfig.

    Args:
        nbits (`int`, *optional*, defaults to 4):
            Number of bits. Supported values are (8, 4, 3, 2, 1).
        group_size (`int`, *optional*, defaults to 64):
            Group-size value. Supported values are any value that is divisble by weight.shape[axis]).
        view_as_float (`bool`, *optional*, defaults to `False`):
            View the quantized weight as float (used in distributed training) if set to `True`.
        axis (`Optional[int]`, *optional*):
            Axis along which grouping is performed. Supported values are 0 or 1.
        dynamic_config (dict, *optional*):
            Parameters for dynamic configuration. The key is the name tag of the layer and the value is a quantization config.
            If set, each layer specified by its id will use its dedicated quantization configuration.
        skip_modules (`List[str]`, *optional*, defaults to `['lm_head']`):
            List of `nn.Linear` layers to skip.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    

## FbgemmFp8Config


    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using fbgemm fp8 quantization.

    Args:
        activation_scale_ub (`float`, *optional*, defaults to 1200.0):
            The activation scale upper bound. This is used when quantizing the input activation.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
    

## CompressedTensorsConfig


    This is a wrapper class that handles compressed-tensors quantization config options.
    It is a wrapper around `compressed_tensors.QuantizationConfig`
    Args:
        config_groups (`typing.Dict[str, typing.Union[ForwardRef('QuantizationScheme'), typing.List[str]]]`, *optional*):
            dictionary mapping group name to a quantization scheme definition
        format (`str`, *optional*, defaults to `"dense"`):
            format the model is represented as. Set `run_compressed` True to execute model as the
            compressed format if not `dense`
        quantization_status (`QuantizationStatus`, *optional*, defaults to `"initialized"`):
            status of model in the quantization lifecycle, ie 'initialized', 'calibration', 'frozen'
        kv_cache_scheme (`typing.Union[QuantizationArgs, NoneType]`, *optional*):
            specifies quantization of the kv cache. If None, kv cache is not quantized.
        global_compression_ratio (`typing.Union[float, NoneType]`, *optional*):
            0-1 float percentage of model compression
        ignore (`typing.Union[typing.List[str], NoneType]`, *optional*):
            layer names or types to not quantize, supports regex prefixed by 're:'
        sparsity_config (`typing.Dict[str, typing.Any]`, *optional*):
            configuration for sparsity compression
        quant_method (`str`, *optional*, defaults to `"compressed-tensors"`):
            do not override, should be compressed-tensors
        run_compressed (`bool`, *optional*, defaults to `True`): alter submodules (usually linear) in order to
            emulate compressed model execution if True, otherwise use default submodule
    

## TorchAoConfig

This is a config class for torchao quantization/sparsity techniques.

    Args:
        quant_type (`str`):
            The type of quantization we want to use, currently supporting: `int4_weight_only`, `int8_weight_only` and `int8_dynamic_activation_int8_weight`.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
        kwargs (`Dict[str, Any]`, *optional*):
            The keyword arguments for the chosen type of quantization, for example, int4_weight_only quantization supports two keyword arguments
            `group_size` and `inner_k_tiles` currently. More API examples and documentation of arguments can be found in
            https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques

    Example:

    ```python
    quantization_config = TorchAoConfig("int4_weight_only", group_size=32)
    # int4_weight_only quant is only working with *torch.bfloat16* dtype right now
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    ```
    

## BitNetConfig

BitNetConfig(modules_to_not_convert: Optional[List] = None, **kwargs)
