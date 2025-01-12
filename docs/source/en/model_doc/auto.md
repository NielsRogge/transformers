<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Auto Classes

In many cases, the architecture you want to use can be guessed from the name or the path of the pretrained model you
are supplying to the `from_pretrained()` method. AutoClasses are here to do this job for you so that you
automatically retrieve the relevant model given the name/path to the pretrained weights/config/vocabulary.

Instantiating one of [`AutoConfig`], [`AutoModel`], and
[`AutoTokenizer`] will directly create a class of the relevant architecture. For instance


```python
model = AutoModel.from_pretrained("google-bert/bert-base-cased")
```

will create a model that is an instance of [`BertModel`].

There is one class of `AutoModel` for each task, and for each backend (PyTorch, TensorFlow, or Flax).

## Extending the Auto Classes

Each of the auto classes has a method to be extended with your custom classes. For instance, if you have defined a
custom class of model `NewModel`, make sure you have a `NewModelConfig` then you can add those to the auto
classes like this:

```python
from transformers import AutoConfig, AutoModel

AutoConfig.register("new-model", NewModelConfig)
AutoModel.register(NewModelConfig, NewModel)
```

You will then be able to use the auto classes like you would usually do!

<Tip warning={true}>

If your `NewModelConfig` is a subclass of [`~transformers.PretrainedConfig`], make sure its
`model_type` attribute is set to the same key you use when registering the config (here `"new-model"`).

Likewise, if your `NewModel` is a subclass of [`PreTrainedModel`], make sure its
`config_class` attribute is set to the same class you use when registering the model (here
`NewModelConfig`).

</Tip>

## AutoConfig

Could not find docstring for AutoConfig

## AutoTokenizer

Could not find docstring for AutoTokenizer

## AutoFeatureExtractor

Could not find docstring for AutoFeatureExtractor

## AutoImageProcessor

Could not find docstring for AutoImageProcessor

## AutoProcessor

Could not find docstring for AutoProcessor

## Generic model classes

The following auto classes are available for instantiating a base model class without a specific head.

### AutoModel

Could not find docstring for AutoModel

### TFAutoModel

No docstring available for TFAutoModel

### FlaxAutoModel

No docstring available for FlaxAutoModel

## Generic pretraining classes

The following auto classes are available for instantiating a model with a pretraining head.

### AutoModelForPreTraining

Could not find docstring for AutoModelForPreTraining

### TFAutoModelForPreTraining

No docstring available for TFAutoModelForPreTraining

### FlaxAutoModelForPreTraining

No docstring available for FlaxAutoModelForPreTraining

## Natural Language Processing

The following auto classes are available for the following natural language processing tasks.

### AutoModelForCausalLM

Could not find docstring for AutoModelForCausalLM

### TFAutoModelForCausalLM

No docstring available for TFAutoModelForCausalLM

### FlaxAutoModelForCausalLM

No docstring available for FlaxAutoModelForCausalLM

### AutoModelForMaskedLM

Could not find docstring for AutoModelForMaskedLM

### TFAutoModelForMaskedLM

No docstring available for TFAutoModelForMaskedLM

### FlaxAutoModelForMaskedLM

No docstring available for FlaxAutoModelForMaskedLM

### AutoModelForMaskGeneration

Could not find docstring for AutoModelForMaskGeneration

### TFAutoModelForMaskGeneration

No docstring available for TFAutoModelForMaskGeneration

### AutoModelForSeq2SeqLM

Could not find docstring for AutoModelForSeq2SeqLM

### TFAutoModelForSeq2SeqLM

No docstring available for TFAutoModelForSeq2SeqLM

### FlaxAutoModelForSeq2SeqLM

No docstring available for FlaxAutoModelForSeq2SeqLM

### AutoModelForSequenceClassification

Could not find docstring for AutoModelForSequenceClassification

### TFAutoModelForSequenceClassification

No docstring available for TFAutoModelForSequenceClassification

### FlaxAutoModelForSequenceClassification

No docstring available for FlaxAutoModelForSequenceClassification

### AutoModelForMultipleChoice

Could not find docstring for AutoModelForMultipleChoice

### TFAutoModelForMultipleChoice

No docstring available for TFAutoModelForMultipleChoice

### FlaxAutoModelForMultipleChoice

No docstring available for FlaxAutoModelForMultipleChoice

### AutoModelForNextSentencePrediction

Could not find docstring for AutoModelForNextSentencePrediction

### TFAutoModelForNextSentencePrediction

No docstring available for TFAutoModelForNextSentencePrediction

### FlaxAutoModelForNextSentencePrediction

No docstring available for FlaxAutoModelForNextSentencePrediction

### AutoModelForTokenClassification

Could not find docstring for AutoModelForTokenClassification

### TFAutoModelForTokenClassification

No docstring available for TFAutoModelForTokenClassification

### FlaxAutoModelForTokenClassification

No docstring available for FlaxAutoModelForTokenClassification

### AutoModelForQuestionAnswering

Could not find docstring for AutoModelForQuestionAnswering

### TFAutoModelForQuestionAnswering

No docstring available for TFAutoModelForQuestionAnswering

### FlaxAutoModelForQuestionAnswering

No docstring available for FlaxAutoModelForQuestionAnswering

### AutoModelForTextEncoding

Could not find docstring for AutoModelForTextEncoding

### TFAutoModelForTextEncoding

No docstring available for TFAutoModelForTextEncoding

## Computer vision

The following auto classes are available for the following computer vision tasks.

### AutoModelForDepthEstimation

Could not find docstring for AutoModelForDepthEstimation

### AutoModelForImageClassification

Could not find docstring for AutoModelForImageClassification

### TFAutoModelForImageClassification

No docstring available for TFAutoModelForImageClassification

### FlaxAutoModelForImageClassification

No docstring available for FlaxAutoModelForImageClassification

### AutoModelForVideoClassification

Could not find docstring for AutoModelForVideoClassification

### AutoModelForKeypointDetection

Could not find docstring for AutoModelForKeypointDetection

### AutoModelForMaskedImageModeling

Could not find docstring for AutoModelForMaskedImageModeling

### TFAutoModelForMaskedImageModeling

No docstring available for TFAutoModelForMaskedImageModeling

### AutoModelForObjectDetection

Could not find docstring for AutoModelForObjectDetection

### AutoModelForImageSegmentation

Could not find docstring for AutoModelForImageSegmentation

### AutoModelForImageToImage

Could not find docstring for AutoModelForImageToImage

### AutoModelForSemanticSegmentation

Could not find docstring for AutoModelForSemanticSegmentation

### TFAutoModelForSemanticSegmentation

No docstring available for TFAutoModelForSemanticSegmentation

### AutoModelForInstanceSegmentation

Could not find docstring for AutoModelForInstanceSegmentation

### AutoModelForUniversalSegmentation

Could not find docstring for AutoModelForUniversalSegmentation

### AutoModelForZeroShotImageClassification

Could not find docstring for AutoModelForZeroShotImageClassification

### TFAutoModelForZeroShotImageClassification

No docstring available for TFAutoModelForZeroShotImageClassification

### AutoModelForZeroShotObjectDetection

Could not find docstring for AutoModelForZeroShotObjectDetection

## Audio

The following auto classes are available for the following audio tasks.

### AutoModelForAudioClassification

Could not find docstring for AutoModelForAudioClassification

### AutoModelForAudioFrameClassification

No docstring available for TFAutoModelForAudioClassification

### TFAutoModelForAudioFrameClassification

Could not find docstring for AutoModelForAudioFrameClassification

### AutoModelForCTC

Could not find docstring for AutoModelForCTC

### AutoModelForSpeechSeq2Seq

Could not find docstring for AutoModelForSpeechSeq2Seq

### TFAutoModelForSpeechSeq2Seq

No docstring available for TFAutoModelForSpeechSeq2Seq

### FlaxAutoModelForSpeechSeq2Seq

No docstring available for FlaxAutoModelForSpeechSeq2Seq

### AutoModelForAudioXVector

Could not find docstring for AutoModelForAudioXVector

### AutoModelForTextToSpectrogram

Could not find docstring for AutoModelForTextToSpectrogram

### AutoModelForTextToWaveform

Could not find docstring for AutoModelForTextToWaveform

## Multimodal

The following auto classes are available for the following multimodal tasks.

### AutoModelForTableQuestionAnswering

Could not find docstring for AutoModelForTableQuestionAnswering

### TFAutoModelForTableQuestionAnswering

No docstring available for TFAutoModelForTableQuestionAnswering

### AutoModelForDocumentQuestionAnswering

Could not find docstring for AutoModelForDocumentQuestionAnswering

### TFAutoModelForDocumentQuestionAnswering

No docstring available for TFAutoModelForDocumentQuestionAnswering

### AutoModelForVisualQuestionAnswering

Could not find docstring for AutoModelForVisualQuestionAnswering

### AutoModelForVision2Seq

Could not find docstring for AutoModelForVision2Seq

### TFAutoModelForVision2Seq

No docstring available for TFAutoModelForVision2Seq

### FlaxAutoModelForVision2Seq

No docstring available for FlaxAutoModelForVision2Seq

### AutoModelForImageTextToText

Could not find docstring for AutoModelForImageTextToText
