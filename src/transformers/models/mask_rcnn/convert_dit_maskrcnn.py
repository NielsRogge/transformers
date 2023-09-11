# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert DiT Mask R-CNN checkpoints from the unilm repository.

URL: https://github.com/microsoft/unilm/tree/master/dit/object_detection"""


import argparse
from pathlib import Path

import requests
import torch
import torchvision.transforms as T
from PIL import Image

from transformers import BeitConfig, MaskRCNNConfig, MaskRCNNForObjectDetection, MaskRCNNImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_beit_maskrcnn_config():
    backbone_config = BeitConfig(
        use_absolute_position_embeddings=True, add_fpn=True, out_features=["stage4", "stage6", "stage8", "stage12"]
    )
    config = MaskRCNNConfig(backbone_config=backbone_config)

    config.num_labels = 5

    # TODO id2label
    # repo_id = "huggingface/label-files"
    # filename = "coco-detection-mmdet-id2label.json"
    # id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # id2label = {int(k): v for k, v in id2label.items()}
    # config.id2label = id2label
    # config.label2id = {v: k for k, v in id2label.items()}

    return config


def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("backbone.cls_token", "backbone.embeddings.cls_token"))
    rename_keys.append(("backbone.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.pos_embed", "backbone.embeddings.position_embeddings"))

    # transformer encoder
    for i in range(config.backbone_config.num_hidden_layers):
        rename_keys.append((f"backbone.blocks.{i}.gamma_1", f"backbone.encoder.layer.{i}.lambda_1"))
        rename_keys.append((f"backbone.blocks.{i}.gamma_2", f"backbone.encoder.layer.{i}.lambda_2"))
        rename_keys.append((f"backbone.blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"backbone.blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"backbone.blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"backbone.blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"backbone.blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"backbone.blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"backbone.blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"backbone.blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.output.dense.bias"))
        rename_keys.append((f"backbone.blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"backbone.blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))

    return rename_keys


def rename_key_(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.backbone_config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"backbone.blocks.{i}.attn.qkv.weight")
        # next, add query, keys and values (in that order) to the state dict
        # note that only the query and value weights have a bias
        hidden_size = config.backbone_config.hidden_size
        v_bias = state_dict.pop(f"backbone.blocks.{i}.attn.v_bias")
        q_bias = state_dict.pop(f"backbone.blocks.{i}.attn.q_bias")
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = v_bias


def rename_key(name):
    if "depthwise_conv" in name:
        name = name.replace("depthwise_conv", "dwconv")
    if "pointwise_conv" in name:
        name = name.replace("pointwise_conv", "pwconv")

    # backbone layernorms
    if "backbone.layernorm0" in name:
        name = name.replace("backbone.layernorm0", "backbone.hidden_states_norms.stage1")
    if "backbone.layernorm1" in name:
        name = name.replace("backbone.layernorm1", "backbone.hidden_states_norms.stage2")
    if "backbone.layernorm2" in name:
        name = name.replace("backbone.layernorm2", "backbone.hidden_states_norms.stage3")
    if "backbone.layernorm3" in name:
        name = name.replace("backbone.layernorm3", "backbone.hidden_states_norms.stage4")

    # neck (simply remove "conv" attribute due to use of `ConvModule` in mmdet)
    if "lateral" in name or "fpn" in name or "mask_head" in name:
        if "conv.weight" in name:
            name = name.replace("conv.weight", "weight")
        if "conv.bias" in name:
            name = name.replace("conv.bias", "bias")

    return name


@torch.no_grad()
def convert_beit_maskrcnn_checkpoint(checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our MaskRCNN structure.
    """

    # define MaskRCNN configuration based on URL
    config = get_beit_maskrcnn_config()
    # load original state_dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val

    # rename some more keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key_(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = MaskRCNNForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    original_pixel_values = transform(image).unsqueeze(0)

    image_processor = MaskRCNNImageProcessor()
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    # verify image processor
    assert torch.allclose(pixel_values, original_pixel_values)

    model(pixel_values, output_hidden_states=True)

    # TODO verify outputs
    # expected_slice_logits = torch.tensor(
    #     [[-12.4785, -17.4976, -14.7001], [-10.9181, -16.7281, -13.2826], [-10.5053, -18.3817, -15.5554]],
    # )
    # expected_slice_boxes = torch.tensor(
    #     [[-0.8485, 0.6819, -1.1016], [1.4864, -0.1529, -1.2551], [0.0233, 0.4202, 0.2257]],
    # )
    # print("Logits:", outputs.logits[:3, :3])
    # assert torch.allclose(outputs.logits[:3, :3], expected_slice_logits, atol=1e-4)
    # assert torch.allclose(outputs.pred_boxes[:3, :3], expected_slice_boxes, atol=1e-4)
    # print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model and feature extractor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        model_name = "dit-maskrcnn"
        model.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/DiT Mask RCNN/result.pth",
        required=False,
        type=str,
        help="Path to the original MaskRCNN checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        required=False,
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub.",
    )

    args = parser.parse_args()
    convert_beit_maskrcnn_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
