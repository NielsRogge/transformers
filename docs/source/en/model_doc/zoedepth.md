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

# ZoeDepth

## Overview

The ZoeDepth model was proposed in [ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288) by Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, Matthias Müller. ZoeDepth extends the [DPT](dpt) framework for metric (also called absolute) depth estimation. ZoeDepth is pre-trained on 12 datasets using relative depth and fine-tuned on two domains (NYU and KITTI) using metric depth. A lightweight head is used with a novel bin adjustment design called metric bins module for each domain. During inference, each input image is automatically routed to the appropriate head using a latent classifier.

The abstract from the paper is the following:

*This paper tackles the problem of depth estimation from a single image. Existing work either focuses on generalization performance disregarding metric scale, i.e. relative depth estimation, or state-of-the-art results on specific datasets, i.e. metric depth estimation. We propose the first approach that combines both worlds, leading to a model with excellent generalization performance while maintaining metric scale. Our flagship model, ZoeD-M12-NK, is pre-trained on 12 datasets using relative depth and fine-tuned on two datasets using metric depth. We use a lightweight head with a novel bin adjustment design called metric bins module for each domain. During inference, each input image is automatically routed to the appropriate head using a latent classifier. Our framework admits multiple configurations depending on the datasets used for relative depth pre-training and metric fine-tuning. Without pre-training, we can already significantly improve the state of the art (SOTA) on the NYU Depth v2 indoor dataset. Pre-training on twelve datasets and fine-tuning on the NYU Depth v2 indoor dataset, we can further improve SOTA for a total of 21% in terms of relative absolute error (REL). Finally, ZoeD-M12-NK is the first model that can jointly train on multiple datasets (NYU Depth v2 and KITTI) without a significant drop in performance and achieve unprecedented zero-shot generalization performance to eight unseen datasets from both indoor and outdoor domains.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/zoedepth_architecture_bis.png"
alt="drawing" width="600"/>

<small> ZoeDepth architecture. Taken from the <a href="https://arxiv.org/abs/2302.12288">original paper.</a> </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/isl-org/ZoeDepth).

## Usage tips

- ZoeDepth is an absolute (also called metric) depth estimation model, unlike DPT which is a relative depth estimation model. This means that ZoeDepth is able to estimate depth in metric units like meters.

The easiest to perform inference with ZoeDepth is by leveraging the [pipeline API](../main_classes/pipelines.md):

```python
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> pipe = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")
>>> result = pipe(image)
>>> depth = result["depth"]
```

Alternatively, one can also perform inference using the classes:

```python
>>> from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
>>> model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():   
...     outputs = model(pixel_values)

>>> # interpolate to original size and visualize the prediction
>>> ## ZoeDepth dynamically pads the input image. Thus we pass the original image size as argument
>>> ## to `post_process_depth_estimation` to remove the padding and resize to original dimensions.
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     source_sizes=[(image.height, image.width)],
... )

>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
>>> depth = depth.detach().cpu().numpy() * 255
>>> depth = Image.fromarray(depth.astype("uint8"))
```

<Tip>
<p>In the <a href="https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/models/depth_model.py#L131">original implementation</a> ZoeDepth model performs inference on both the original and flipped images and averages out the results. The <code>post_process_depth_estimation</code> function can handle this for us by passing the flipped outputs to the optional <code>outputs_flipped</code> argument:</p>
<pre><code class="language-Python">&gt;&gt;&gt; with torch.no_grad():   
...     outputs = model(pixel_values)
...     outputs_flipped = model(pixel_values=torch.flip(inputs.pixel_values, dims=[3]))
&gt;&gt;&gt; post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     source_sizes=[(image.height, image.width)],
...     outputs_flipped=outputs_flipped,
... )
</code></pre>
</Tip>

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ZoeDepth.

- A demo notebook regarding inference with ZoeDepth models can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ZoeDepth). 🌎

## ZoeDepthConfig


    This is the configuration class to store the configuration of a [`ZoeDepthForDepthEstimation`]. It is used to instantiate an ZoeDepth
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ZoeDepth
    [Intel/zoedepth-nyu](https://huggingface.co/Intel/zoedepth-nyu) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*, defaults to `BeitConfig()`):
            The configuration of the backbone model.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.
        readout_type (`str`, *optional*, defaults to `"project"`):
            The readout type to use when processing the readout token (CLS token) of the intermediate hidden states of
            the ViT backbone. Can be one of [`"ignore"`, `"add"`, `"project"`].

            - "ignore" simply ignores the CLS token.
            - "add" passes the information from the CLS token to all other tokens by adding the representations.
            - "project" passes information to the other tokens by concatenating the readout to all other tokens before
              projecting the
            representation to the original feature dimension D using a linear layer followed by a GELU non-linearity.
        reassemble_factors (`List[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            The up/downsampling factors of the reassemble layers.
        neck_hidden_sizes (`List[str]`, *optional*, defaults to `[96, 192, 384, 768]`):
            The hidden sizes to project to for the feature maps of the backbone.
        fusion_hidden_size (`int`, *optional*, defaults to 256):
            The number of channels before fusion.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the heads.
        use_batch_norm_in_fusion_residual (`bool`, *optional*, defaults to `False`):
            Whether to use batch normalization in the pre-activate residual units of the fusion blocks.
        use_bias_in_fusion_residual (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the pre-activate residual units of the fusion blocks.
        num_relative_features (`int`, *optional*, defaults to 32):
            The number of features to use in the relative depth estimation head.
        add_projection (`bool`, *optional*, defaults to `False`):
            Whether to add a projection layer before the depth estimation head.
        bottleneck_features (`int`, *optional*, defaults to 256):
            The number of features in the bottleneck layer.
        num_attractors (`List[int], *optional*, defaults to `[16, 8, 4, 1]`):
            The number of attractors to use in each stage.
        bin_embedding_dim (`int`, *optional*, defaults to 128):
            The dimension of the bin embeddings.
        attractor_alpha (`int`, *optional*, defaults to 1000):
            The alpha value to use in the attractor.
        attractor_gamma (`int`, *optional*, defaults to 2):
            The gamma value to use in the attractor.
        attractor_kind (`str`, *optional*, defaults to `"mean"`):
            The kind of attractor to use. Can be one of [`"mean"`, `"sum"`].
        min_temp (`float`, *optional*, defaults to 0.0212):
            The minimum temperature value to consider.
        max_temp (`float`, *optional*, defaults to 50.0):
            The maximum temperature value to consider.
        bin_centers_type (`str`, *optional*, defaults to `"softplus"`):
            Activation type used for bin centers. Can be "normed" or "softplus". For "normed" bin centers, linear normalization trick
            is applied. This results in bounded bin centers. For "softplus", softplus activation is used and thus are unbounded.
        bin_configurations (`List[dict]`, *optional*, defaults to `[{'n_bins': 64, 'min_depth': 0.001, 'max_depth': 10.0}]`):
            Configuration for each of the bin heads.
            Each configuration should consist of the following keys:
            - name (`str`): The name of the bin head - only required in case of multiple bin configurations.
            - `n_bins` (`int`): The number of bins to use.
            - `min_depth` (`float`): The minimum depth value to consider.
            - `max_depth` (`float`): The maximum depth value to consider.
            In case only a single configuration is passed, the model will use a single head with the specified configuration.
            In case multiple configurations are passed, the model will use multiple heads with the specified configurations.
        num_patch_transformer_layers (`int`, *optional*):
            The number of transformer layers to use in the patch transformer. Only used in case of multiple bin configurations.
        patch_transformer_hidden_size (`int`, *optional*):
            The hidden size to use in the patch transformer. Only used in case of multiple bin configurations.
        patch_transformer_intermediate_size (`int`, *optional*):
            The intermediate size to use in the patch transformer. Only used in case of multiple bin configurations.
        patch_transformer_num_attention_heads (`int`, *optional*):
            The number of attention heads to use in the patch transformer. Only used in case of multiple bin configurations.

    Example:

    ```python
    >>> from transformers import ZoeDepthConfig, ZoeDepthForDepthEstimation

    >>> # Initializing a ZoeDepth zoedepth-large style configuration
    >>> configuration = ZoeDepthConfig()

    >>> # Initializing a model from the zoedepth-large style configuration
    >>> model = ZoeDepthForDepthEstimation(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## ZoeDepthImageProcessor


    Constructs a ZoeDepth image processor.

    Args:
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to apply pad the input.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overidden by `do_rescale` in
            `preprocess`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overidden by `rescale_factor` in `preprocess`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions. Can be overidden by `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 384, "width": 512}`):
            Size of the image after resizing. Size of the image after resizing. If `keep_aspect_ratio` is `True`,
            the image is resized by choosing the smaller of the height and width scaling factors and using it for both dimensions.
            If `ensure_multiple_of` is also set, the image is further resized to a size that is a multiple of this value.
            Can be overidden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Defines the resampling filter to use if resizing the image. Can be overidden by `resample` in `preprocess`.
        keep_aspect_ratio (`bool`, *optional*, defaults to `True`):
            If `True`, the image is resized by choosing the smaller of the height and width scaling factors and using it
            for both dimensions. This ensures that the image is scaled down as little as possible while still fitting
            within the desired output size. In case `ensure_multiple_of` is also set, the image is further resized to a
            size that is a multiple of this value by flooring the height and width to the nearest multiple of this value.
            Can be overidden by `keep_aspect_ratio` in `preprocess`.
        ensure_multiple_of (`int`, *optional*, defaults to 32):
            If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Works by flooring
            the height and width to the nearest multiple of this value.

            Works both with and without `keep_aspect_ratio` being set to `True`. Can be overidden by `ensure_multiple_of`
            in `preprocess`.
    

Methods: preprocess

## ZoeDepthForDepthEstimation


    ZoeDepth model with one or multiple metric depth estimation head(s) on top.
    
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward