# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch MaskRCNN model. """


import inspect
import random
import unittest

from huggingface_hub import hf_hub_download

from transformers import ConvNextConfig, MaskRCNNConfig
from transformers.testing_utils import require_torchvision, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin


if is_torch_available():
    import torch

    from transformers import MaskRCNNForObjectDetection
    from transformers.models.mask_rcnn.modeling_maskrcnn import (
        MASK_RCNN_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

if is_vision_available():
    from PIL import Image

    from transformers import MaskRCNNImageProcessor


class MaskRCNNModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        # backbone attributes
        num_channels=3,
        num_stages=4,
        hidden_sizes=[10, 20, 30, 40],
        depths=[2, 2, 3, 2],
        out_features=["stage1", "stage2", "stage3", "stage4"],
        # FPN attributes
        fpn_out_channels=10,
        # RPN attributes
        rpn_in_channels=10,
        rpn_feat_channels=10,
        # R-CNN attributes
        bbox_head_in_channels=10,
        bbox_head_fc_out_channels=10,
        bbox_roi_extractor_out_channels=10,
        mask_head_in_channels=10,
        mask_head_conv_out_channels=10,
        mask_roi_extractor_out_channels=10,
        is_training=True,
        use_labels=True,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_stages = num_stages
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.out_features = out_features
        self.fpn_out_channels = fpn_out_channels
        self.rpn_in_channels = rpn_in_channels
        self.rpn_feat_channels = rpn_feat_channels
        self.bbox_head_in_channels = bbox_head_in_channels
        self.bbox_head_fc_out_channels = bbox_head_fc_out_channels
        self.bbox_roi_extractor_out_channels = bbox_roi_extractor_out_channels
        self.mask_head_in_channels = mask_head_in_channels
        self.mask_head_conv_out_channels = mask_head_conv_out_channels
        self.mask_roi_extractor_out_channels = mask_roi_extractor_out_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        # we set a seed as the model internally uses NMS operations
        torch.manual_seed(0)
        pixel_values = torch.randn([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = []
            for _ in range(self.batch_size):
                # sample a number of objects
                number_of_objects = random.randint(0, 10)
                class_labels = torch.tensor([random.randint(0, self.num_labels - 1) for _ in range(number_of_objects)])
                boxes = torch.randn(number_of_objects, 4)
                masks = torch.randn(number_of_objects, self.image_size, self.image_size)
                target = {}
                target["class_labels"] = class_labels
                target["boxes"] = boxes
                target["masks"] = masks
                target["size"] = torch.tensor((self.image_size, self.image_size))
                labels.append(target)

        config = self.get_config()

        return config, pixel_values, labels

    def get_backbone_config(self):
        return ConvNextConfig(
            num_channels=self.num_channels,
            num_stages=self.num_stages,
            hidden_sizes=self.hidden_sizes,
            depths=self.depths,
            is_training=self.is_training,
            out_features=self.out_features,
        )

    def get_config(self):
        backbone_config = self.get_backbone_config()
        return MaskRCNNConfig(
            backbone_config=backbone_config,
            initializer_range=self.initializer_range,
            num_labels=self.num_labels,
            fpn_out_channels=self.fpn_out_channels,
            rpn_in_channels=self.rpn_in_channels,
            rpn_feat_channels=self.rpn_feat_channels,
            bbox_head_in_channels=self.bbox_head_in_channels,
            bbox_head_fc_out_channels=self.bbox_head_fc_out_channels,
            bbox_roi_extractor_out_channels=self.bbox_roi_extractor_out_channels,
            mask_head_in_channels=self.mask_head_in_channels,
            mask_head_conv_out_channels=self.mask_head_conv_out_channels,
            mask_roi_extractor_out_channels=self.mask_roi_extractor_out_channels,
        )

    def create_and_check_model_for_object_detection(self, config, pixel_values, labels):
        # we are setting a seed to make sure NMS returns the same number of proposals per image
        torch.manual_seed(2)

        model = MaskRCNNForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        # inference
        result = model(pixel_values)
        # expected logits shape: (num_proposals_per_image stacked on top of each other, num_labels + 1)
        self.parent.assertEqual(result.logits.shape, (387, self.num_labels + 1))
        # expected boxes shape: (num_proposals_per_image stacked on top of each other, num_labels * 4)
        self.parent.assertEqual(result.pred_boxes.shape, (387, self.num_labels * 4))

        # training
        result = model(pixel_values, labels=labels)
        self.parent.assertTrue(result.loss.item() > 0)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torchvision
class MaskRCNNModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Mask-RCNN does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (MaskRCNNForObjectDetection,) if is_torch_available() else ()

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False
    test_torchscript = False

    # special case for head models
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ in ["MaskRCNNForObjectDetection"]:
                labels = []
                for _ in range(self.model_tester.batch_size):
                    target = {}
                    target["class_labels"] = torch.ones(
                        size=(self.model_tester.num_labels,), device=torch_device, dtype=torch.long
                    )
                    target["boxes"] = torch.ones(
                        self.model_tester.num_labels, 4, device=torch_device, dtype=torch.float
                    )
                    target["masks"] = torch.ones(
                        self.model_tester.num_labels,
                        self.model_tester.image_size,
                        self.model_tester.image_size,
                        device=torch_device,
                        dtype=torch.float,
                    )
                    target["size"] = torch.tensor((self.model_tester.image_size, self.model_tester.image_size))
                    labels.append(target)
                inputs_dict["labels"] = labels

        return inputs_dict

    def setUp(self):
        self.model_tester = MaskRCNNModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MaskRCNNConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        return

    @unittest.skip(reason="Mask-RCNN does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Mask-RCNN does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Mask-RCNN does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model_for_object_detection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_for_object_detection(*config_and_inputs)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_stages = len(config.backbone_config.hidden_sizes)
            self.assertEqual(len(hidden_states), expected_num_stages + 1)

            # Mask-RCNN's feature maps are of shape (batch_size, num_channels, height, width)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size // 4, self.model_tester.image_size // 4],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @slow
    def test_model_from_pretrained(self):
        for model_name in MASK_RCNN_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = MaskRCNNForObjectDetection.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
@require_torchvision
@slow
class MaskRCNNModelIntegrationTest(unittest.TestCase):
    def test_inference_object_detection_head(self):
        # TODO update to appropriate organization + use from_pretrained for image processor
        processor = MaskRCNNImageProcessor()
        model = MaskRCNNForObjectDetection.from_pretrained("nielsr/convnext-tiny-maskrcnn").to(torch_device)

        image = prepare_img()
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values)

        # verify outputs
        self.assertListEqual(
            list(outputs.keys()),
            ["logits", "pred_boxes", "rois", "proposals", "fpn_hidden_states"],
        )
        expected_slice_logits = torch.tensor(
            [[-12.4785, -17.4976, -14.7001], [-10.9181, -16.7281, -13.2826], [-10.5053, -18.3817, -15.5554]],
            device=torch_device,
        )
        expected_slice_boxes = torch.tensor(
            [[-0.8485, 0.6819, -1.1016], [1.4864, -0.1529, -1.2551], [0.0233, 0.4202, 0.2257]],
            device=torch_device,
        )
        self.assertEquals(outputs.logits.shape, torch.Size([1000, 81]))
        self.assertTrue(torch.allclose(outputs.logits[:3, :3], expected_slice_logits))
        self.assertEquals(outputs.pred_boxes.shape, torch.Size([1000, 320]))
        self.assertTrue(torch.allclose(outputs.pred_boxes[:3, :3], expected_slice_boxes, atol=1e-4))

        # verify postprocessed results
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=[image.size[::-1]],
            scale_factors=[torch.tensor([1.6671875, 1.6666666, 1.6671875, 1.6666666])],
        )

        self.assertEqual(len(results), 1)
        expected_slice = torch.tensor([0.9981, 0.9959, 0.9874, 0.9110, 0.7234, 0.6268], device=torch_device)
        self.assertTrue(torch.allclose(results[0]["scores"], expected_slice, atol=1e-4))
        expected_slice = torch.tensor([15, 15, 65, 65, 57, 59], device=torch_device)
        self.assertTrue(torch.allclose(results[0]["labels"], expected_slice))
        self.assertEquals(results[0]["boxes"].shape, torch.Size([6, 4]))
        expected_slice = torch.tensor([17.9057, 55.4164, 318.9557], device=torch_device)
        self.assertTrue(torch.allclose(results[0]["boxes"][0, :3], expected_slice))

        # verify mask predictions
        detected_boxes = [result["boxes"] for result in results]
        scale_factors = [torch.tensor([1.6671875, 1.6666666, 1.6671875, 1.6666666])]
        mask_pred = model.roi_head.forward_test_mask(
            outputs.fpn_hidden_states, scale_factors, detected_boxes, rescale=True
        )

        self.assertEquals(mask_pred.shape, torch.Size([6, 80, 28, 28]))
        expected_slice = torch.tensor(
            [[-2.3380, -2.3863, -3.0293], [-2.4269, -2.1714, -2.8495], [-2.8431, -2.8594, -3.0908]],
            device=torch_device,
        )
        self.assertTrue(torch.allclose(mask_pred[0, 0, :3, :3], expected_slice, atol=1e-4))

        # verify postprocessed mask results
        mask_results = processor.post_process_instance_segmentation(
            results,
            mask_pred,
            target_sizes=[image.size[::-1]],
            scale_factors=scale_factors,
        )
        self.assertEquals(len(mask_results[0]), 80)
        self.assertEquals(mask_results[0][15][0].sum(), 52418)

    def test_training_object_detection_head(self):
        # make random mask reproducible
        # note that the same seed on CPU and on GPU doesn’t mean they spew the same random number sequences,
        # as they both have fairly different PRNGs (for efficiency reasons).
        # source: https://discuss.pytorch.org/t/random-seed-that-spans-across-devices/19735
        torch.manual_seed(2)

        # TODO update to appropriate organization
        model = MaskRCNNForObjectDetection.from_pretrained("nielsr/convnext-tiny-maskrcnn").to(torch_device)

        # TODO use image processor instead?
        local_path = hf_hub_download(repo_id="nielsr/init-files", filename="pixel_values.pt")
        img = torch.load(local_path).unsqueeze(0)

        labels = []
        local_path = hf_hub_download(repo_id="nielsr/init-files", filename="boxes.pt")
        target = {}
        target["boxes"] = torch.load(local_path).to(torch_device)
        local_path = hf_hub_download(repo_id="nielsr/init-files", filename="labels.pt")
        target["class_labels"] = torch.load(local_path).to(torch_device)
        local_path = hf_hub_download(repo_id="nielsr/init-files", filename="masks.pt")
        target["masks"] = torch.load(local_path).to(torch_device)
        target["size"] = torch.tensor(img.shape[1:]).to(torch_device)
        # labels["gt_bboxes_ignore"] = None
        labels.append(target)

        # forward pass
        with torch.no_grad():
            outputs = model(img.to(torch_device), labels=labels)
            losses = outputs.loss_dict

        # verify the losses
        expected_loss_gpu = {
            "loss_cls": torch.tensor(0.1876, device=torch_device),
            "loss_bbox": torch.tensor(0.1139, device=torch_device),
            "acc": torch.tensor([90.8203], device=torch_device),
            "loss_mask": torch.tensor([0.2180], device=torch_device),
        }
        for key, value in expected_loss_gpu.items():
            self.assertTrue(torch.allclose(losses[key], value, atol=1e-4))
