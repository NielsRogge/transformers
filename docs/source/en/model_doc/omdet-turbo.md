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

# OmDet-Turbo

## Overview

The OmDet-Turbo model was proposed in [Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head](https://arxiv.org/abs/2403.06892) by Tiancheng Zhao, Peng Liu, Xuan He, Lu Zhang, Kyusong Lee. OmDet-Turbo incorporates components from RT-DETR and introduces a swift multimodal fusion module to achieve real-time open-vocabulary object detection capabilities while maintaining high accuracy. The base model achieves performance of up to 100.2 FPS and 53.4 AP on COCO zero-shot.

The abstract from the paper is the following:

*End-to-end transformer-based detectors (DETRs) have shown exceptional performance in both closed-set and open-vocabulary object detection (OVD) tasks through the integration of language modalities. However, their demanding computational requirements have hindered their practical application in real-time object detection (OD) scenarios. In this paper, we scrutinize the limitations of two leading models in the OVDEval benchmark, OmDet and Grounding-DINO, and introduce OmDet-Turbo. This novel transformer-based real-time OVD model features an innovative Efficient Fusion Head (EFH) module designed to alleviate the bottlenecks observed in OmDet and Grounding-DINO. Notably, OmDet-Turbo-Base achieves a 100.2 frames per second (FPS) with TensorRT and language cache techniques applied. Notably, in zero-shot scenarios on COCO and LVIS datasets, OmDet-Turbo achieves performance levels nearly on par with current state-of-the-art supervised models. Furthermore, it establishes new state-of-the-art benchmarks on ODinW and OVDEval, boasting an AP of 30.1 and an NMS-AP of 26.86, respectively. The practicality of OmDet-Turbo in industrial applications is underscored by its exceptional performance on benchmark datasets and superior inference speed, positioning it as a compelling choice for real-time object detection tasks.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/omdet_turbo_architecture.jpeg" alt="drawing" width="600"/>

<small> OmDet-Turbo architecture overview. Taken from the <a href="https://arxiv.org/abs/2403.06892">original paper</a>. </small>

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/om-ai-lab/OmDet).

## Usage tips

One unique property of OmDet-Turbo compared to other zero-shot object detection models, such as [Grounding DINO](grounding-dino), is the decoupled classes and prompt embedding structure that allows caching of text embeddings. This means that the model needs both classes and task as inputs, where classes is a list of objects we want to detect and task is the grounded text used to guide open-vocabulary detection. This approach limits the scope of the open-vocabulary detection and makes the decoding process faster.

[`OmDetTurboProcessor`] is used to prepare the classes, task and image triplet. The task input is optional, and when not provided, it will default to `"Detect [class1], [class2], [class3], ..."`. To process the results from the model, one can use `post_process_grounded_object_detection` from [`OmDetTurboProcessor`]. Notably, this function takes in the input classes, as unlike other zero-shot object detection models, the decoupling of classes and task embeddings means that no decoding of the predicted class embeddings is needed in the post-processing step, and the predicted classes can be matched to the inputted ones directly.

## Usage example

### Single image inference

Here's how to load the model and prepare the inputs to perform zero-shot object detection on a single image:

```python
>>> import torch
>>> import requests
>>> from PIL import Image

>>> from transformers import AutoProcessor, OmDetTurboForObjectDetection

>>> processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
>>> model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text_labels = ["cat", "remote"]
>>> inputs = processor(image, text=text_labels, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # convert outputs (bounding boxes and class logits)
>>> results = processor.post_process_grounded_object_detection(
...     outputs,
...     target_sizes=[(image.height, image.width)],
...     text_labels=text_labels,
...     threshold=0.3,
...     nms_threshold=0.3,
... )
>>> result = results[0]
>>> boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
>>> for box, score, text_label in zip(boxes, scores, text_labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
Detected remote with confidence 0.768 at location [39.89, 70.35, 176.74, 118.04]
Detected cat with confidence 0.72 at location [11.6, 54.19, 314.8, 473.95]
Detected remote with confidence 0.563 at location [333.38, 75.77, 370.7, 187.03]
Detected cat with confidence 0.552 at location [345.15, 23.95, 639.75, 371.67]
```

### Multi image inference

OmDet-Turbo can perform batched multi-image inference, with support for different text prompts and classes in the same batch:

```python
>>> import torch
>>> import requests
>>> from io import BytesIO
>>> from PIL import Image
>>> from transformers import AutoProcessor, OmDetTurboForObjectDetection

>>> processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
>>> model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")

>>> url1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image1 = Image.open(BytesIO(requests.get(url1).content)).convert("RGB")
>>> text_labels1 = ["cat", "remote"]
>>> task1 = "Detect {}.".format(", ".join(text_labels1))

>>> url2 = "http://images.cocodataset.org/train2017/000000257813.jpg"
>>> image2 = Image.open(BytesIO(requests.get(url2).content)).convert("RGB")
>>> text_labels2 = ["boat"]
>>> task2 = "Detect everything that looks like a boat."

>>> url3 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
>>> image3 = Image.open(BytesIO(requests.get(url3).content)).convert("RGB")
>>> text_labels3 = ["statue", "trees"]
>>> task3 = "Focus on the foreground, detect statue and trees."

>>> inputs = processor(
...     images=[image1, image2, image3],
...     text=[text_labels1, text_labels2, text_labels3],
...     task=[task1, task2, task3],
...     return_tensors="pt",
... )

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # convert outputs (bounding boxes and class logits)
>>> results = processor.post_process_grounded_object_detection(
...     outputs,
...     text_labels=[text_labels1, text_labels2, text_labels3],
...     target_sizes=[(image.height, image.width) for image in [image1, image2, image3]],
...     threshold=0.2,
...     nms_threshold=0.3,
... )

>>> for i, result in enumerate(results):
...     for score, text_label, box in zip(
...         result["scores"], result["text_labels"], result["boxes"]
...     ):
...         box = [round(i, 1) for i in box.tolist()]
...         print(
...             f"Detected {text_label} with confidence "
...             f"{round(score.item(), 2)} at location {box} in image {i}"
...         )
Detected remote with confidence 0.77 at location [39.9, 70.4, 176.7, 118.0] in image 0
Detected cat with confidence 0.72 at location [11.6, 54.2, 314.8, 474.0] in image 0
Detected remote with confidence 0.56 at location [333.4, 75.8, 370.7, 187.0] in image 0
Detected cat with confidence 0.55 at location [345.2, 24.0, 639.8, 371.7] in image 0
Detected boat with confidence 0.32 at location [146.9, 219.8, 209.6, 250.7] in image 1
Detected boat with confidence 0.3 at location [319.1, 223.2, 403.2, 238.4] in image 1
Detected boat with confidence 0.27 at location [37.7, 220.3, 84.0, 235.9] in image 1
Detected boat with confidence 0.22 at location [407.9, 207.0, 441.7, 220.2] in image 1
Detected statue with confidence 0.73 at location [544.7, 210.2, 651.9, 502.8] in image 2
Detected trees with confidence 0.25 at location [3.9, 584.3, 391.4, 785.6] in image 2
Detected trees with confidence 0.25 at location [1.4, 621.2, 118.2, 787.8] in image 2
Detected statue with confidence 0.2 at location [428.1, 205.5, 767.3, 759.5] in image 2

```

## OmDetTurboConfig


    This is the configuration class to store the configuration of a [`OmDetTurboForObjectDetection`].
    It is used to instantiate a OmDet-Turbo model according to the specified arguments, defining the model architecture
    Instantiating a configuration with the defaults will yield a similar configuration to that of the OmDet-Turbo
    [omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`PretrainedConfig`, *optional*):
            The configuration of the text backbone.
        backbone_config (`PretrainedConfig`, *optional*):
            The configuration of the vision backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use the timm for the vision backbone.
        backbone (`str`, *optional*, defaults to `"swin_tiny_patch4_window7_224"`):
            The name of the pretrained vision backbone to use. If `use_pretrained_backbone=False` a randomly initialized
            backbone with the same architecture `backbone` is used.
        backbone_kwargs (`dict`, *optional*):
            Additional kwargs for the vision backbone.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use a pretrained vision backbone.
        apply_layernorm_after_vision_backbone (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization on the feature maps of the vision backbone output.
        image_size (`int`, *optional*, defaults to 640):
            The size (resolution) of each image.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Whether to disable custom kernels.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for layer normalization.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for batch normalization.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        text_projection_in_dim (`int`, *optional*, defaults to 512):
            The input dimension for the text projection.
        text_projection_out_dim (`int`, *optional*, defaults to 512):
            The output dimension for the text projection.
        task_encoder_hidden_dim (`int`, *optional*, defaults to 1024):
            The feedforward dimension for the task encoder.
        class_embed_dim (`int`, *optional*, defaults to 512):
            The dimension of the classes embeddings.
        class_distance_type (`str`, *optional*, defaults to `"cosine"`):
            The type of of distance to compare predicted classes to projected classes embeddings.
            Can be `"cosine"` or `"dot"`.
        num_queries (`int`, *optional*, defaults to 900):
            The number of queries.
        csp_activation (`str`, *optional*, defaults to `"silu"`):
            The activation function of the Cross Stage Partial (CSP) networks of the encoder.
        conv_norm_activation (`str`, *optional*, defaults to `"gelu"`):
            The activation function of the ConvNormLayer layers of the encoder.
        encoder_feedforward_activation (`str`, *optional*, defaults to `"relu"`):
            The activation function for the feedforward network of the encoder.
        encoder_feedforward_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate following the activation of the encoder feedforward network.
        encoder_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate of the encoder multi-head attention module.
        hidden_expansion (`int`, *optional*, defaults to 1):
            The hidden expansion of the CSP networks in the encoder.
        vision_features_channels (`tuple(int)`, *optional*, defaults to `[256, 256, 256]`):
            The projected vision features channels used as inputs for the decoder.
        encoder_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of the encoder.
        encoder_in_channels (`List(int)`, *optional*, defaults to `[192, 384, 768]`):
            The input channels for the encoder.
        encoder_projection_indices (`List(int)`, *optional*, defaults to `[2]`):
            The indices of the input features projected by each layers.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            The number of attention heads for the encoder.
        encoder_dim_feedforward (`int`, *optional*, defaults to 2048):
            The feedforward dimension for the encoder.
        encoder_layers (`int`, *optional*, defaults to 1):
            The number of layers in the encoder.
        positional_encoding_temperature (`int`, *optional*, defaults to 10000):
            The positional encoding temperature in the encoder.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of feature levels for the multi-scale deformable attention module of the decoder.
        decoder_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of the decoder.
        decoder_num_heads (`int`, *optional*, defaults to 8):
            The number of heads for the decoder.
        decoder_num_layers (`int`, *optional*, defaults to 6):
            The number of layers for the decoder.
        decoder_activation (`str`, *optional*, defaults to `"relu"`):
            The activation function for the decoder.
        decoder_dim_feedforward (`int`, *optional*, defaults to 2048):
            The feedforward dimension for the decoder.
        decoder_num_points (`int`, *optional*, defaults to 4):
            The number of points sampled in the decoder multi-scale deformable attention module.
        decoder_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate for the decoder.
        eval_size (`Tuple[int, int]`, *optional*):
            Height and width used to computes the effective height and width of the position embeddings after taking
            into account the stride (see RTDetr).
        learn_initial_query (`bool`, *optional*, defaults to `False`):
            Whether to learn the initial query.
        cache_size (`int`, *optional*, defaults to 100):
            The cache size for the classes and prompts caches.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder-decoder model or not.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from the architecture. The values in kwargs will be saved as part of the configuration
            and can be used to control the model outputs.

    Examples:

    ```python
    >>> from transformers import OmDetTurboConfig, OmDetTurboForObjectDetection

    >>> # Initializing a OmDet-Turbo omlab/omdet-turbo-swin-tiny-hf style configuration
    >>> configuration = OmDetTurboConfig()

    >>> # Initializing a model (with random weights) from the omlab/omdet-turbo-swin-tiny-hf style configuration
    >>> model = OmDetTurboForObjectDetection(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## OmDetTurboProcessor


    Constructs a OmDet-Turbo processor which wraps a Deformable DETR image processor and an AutoTokenizer into a
    single processor.

    [`OmDetTurboProcessor`] offers all the functionalities of [`DetrImageProcessor`] and
    [`AutoTokenizer`]. See the docstring of [`~OmDetTurboProcessor.__call__`] and [`~OmDetTurboProcessor.decode`]
    for more information.

    Args:
        image_processor (`DetrImageProcessor`):
            An instance of [`DetrImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    

Methods: post_process_grounded_object_detection

## OmDetTurboForObjectDetection


    OmDetTurbo Model (consisting of a vision and a text backbone, and encoder-decoder architecture) outputting
    bounding boxes and classes scores for tasks such as COCO detection.
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OmDetTurboConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
