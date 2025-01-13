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

[[autodoc]] AutoConfig

## AutoTokenizer

[[autodoc]] AutoTokenizer

## AutoFeatureExtractor

[[autodoc]] AutoFeatureExtractor

## AutoImageProcessor

[[autodoc]] AutoImageProcessor

## AutoProcessor

[[autodoc]] AutoProcessor

## Generic model classes

The following auto classes are available for instantiating a base model class without a specific head.

### AutoModel

[[autodoc]] AutoModel

### TFAutoModel

No docstring available for TFAutoModel

### FlaxAutoModel

No docstring available for FlaxAutoModel

## Generic pretraining classes

The following auto classes are available for instantiating a model with a pretraining head.

### AutoModelForPreTraining

[[autodoc]] AutoModelForPreTraining

### TFAutoModelForPreTraining

No docstring available for TFAutoModelForPreTraining

### FlaxAutoModelForPreTraining

No docstring available for FlaxAutoModelForPreTraining

## Natural Language Processing

The following auto classes are available for the following natural language processing tasks.

### AutoModelForCausalLM

[[autodoc]] AutoModelForCausalLM

### TFAutoModelForCausalLM

No docstring available for TFAutoModelForCausalLM

### FlaxAutoModelForCausalLM

No docstring available for FlaxAutoModelForCausalLM

### AutoModelForMaskedLM

[[autodoc]] AutoModelForMaskedLM

### TFAutoModelForMaskedLM

No docstring available for TFAutoModelForMaskedLM

### FlaxAutoModelForMaskedLM

No docstring available for FlaxAutoModelForMaskedLM

### AutoModelForMaskGeneration

[[autodoc]] AutoModelForMaskGeneration

### TFAutoModelForMaskGeneration

No docstring available for TFAutoModelForMaskGeneration

### AutoModelForSeq2SeqLM

[[autodoc]] AutoModelForSeq2SeqLM

### TFAutoModelForSeq2SeqLM

No docstring available for TFAutoModelForSeq2SeqLM

### FlaxAutoModelForSeq2SeqLM

No docstring available for FlaxAutoModelForSeq2SeqLM

### AutoModelForSequenceClassification

[[autodoc]] AutoModelForSequenceClassification

### TFAutoModelForSequenceClassification

No docstring available for TFAutoModelForSequenceClassification

### FlaxAutoModelForSequenceClassification

No docstring available for FlaxAutoModelForSequenceClassification

### AutoModelForMultipleChoice

[[autodoc]] AutoModelForMultipleChoice

### TFAutoModelForMultipleChoice

No docstring available for TFAutoModelForMultipleChoice

### FlaxAutoModelForMultipleChoice

No docstring available for FlaxAutoModelForMultipleChoice

### AutoModelForNextSentencePrediction

[[autodoc]] AutoModelForNextSentencePrediction

### TFAutoModelForNextSentencePrediction

No docstring available for TFAutoModelForNextSentencePrediction

### FlaxAutoModelForNextSentencePrediction

No docstring available for FlaxAutoModelForNextSentencePrediction

### AutoModelForTokenClassification

[[autodoc]] AutoModelForTokenClassification

### TFAutoModelForTokenClassification

No docstring available for TFAutoModelForTokenClassification

### FlaxAutoModelForTokenClassification

No docstring available for FlaxAutoModelForTokenClassification

### AutoModelForQuestionAnswering

[[autodoc]] AutoModelForQuestionAnswering

### TFAutoModelForQuestionAnswering

No docstring available for TFAutoModelForQuestionAnswering

### FlaxAutoModelForQuestionAnswering

No docstring available for FlaxAutoModelForQuestionAnswering

### AutoModelForTextEncoding

[[autodoc]] AutoModelForTextEncoding

### TFAutoModelForTextEncoding

No docstring available for TFAutoModelForTextEncoding

## Computer vision

The following auto classes are available for the following computer vision tasks.

### AutoModelForDepthEstimation

[[autodoc]] AutoModelForDepthEstimation

### AutoModelForImageClassification

[[autodoc]] AutoModelForImageClassification

### TFAutoModelForImageClassification

No docstring available for TFAutoModelForImageClassification

### FlaxAutoModelForImageClassification

No docstring available for FlaxAutoModelForImageClassification

### AutoModelForVideoClassification

[[autodoc]] AutoModelForVideoClassification

### AutoModelForKeypointDetection

[[autodoc]] AutoModelForKeypointDetection

### AutoModelForMaskedImageModeling

[[autodoc]] AutoModelForMaskedImageModeling

### TFAutoModelForMaskedImageModeling

No docstring available for TFAutoModelForMaskedImageModeling

### AutoModelForObjectDetection

[[autodoc]] AutoModelForObjectDetection

### AutoModelForImageSegmentation

[[autodoc]] AutoModelForImageSegmentation

### AutoModelForImageToImage

[[autodoc]] AutoModelForImageToImage

### AutoModelForSemanticSegmentation

[[autodoc]] AutoModelForSemanticSegmentation

### TFAutoModelForSemanticSegmentation

No docstring available for TFAutoModelForSemanticSegmentation

### AutoModelForInstanceSegmentation

[[autodoc]] AutoModelForInstanceSegmentation

### AutoModelForUniversalSegmentation

[[autodoc]] AutoModelForUniversalSegmentation

### AutoModelForZeroShotImageClassification

[[autodoc]] AutoModelForZeroShotImageClassification

### TFAutoModelForZeroShotImageClassification

No docstring available for TFAutoModelForZeroShotImageClassification

### AutoModelForZeroShotObjectDetection

[[autodoc]] AutoModelForZeroShotObjectDetection

## Audio

The following auto classes are available for the following audio tasks.

### AutoModelForAudioClassification

[[autodoc]] AutoModelForAudioClassification

### AutoModelForAudioFrameClassification

No docstring available for TFAutoModelForAudioClassification

### TFAutoModelForAudioFrameClassification

[[autodoc]] AutoModelForAudioFrameClassification

### AutoModelForCTC

[[autodoc]] AutoModelForCTC

### AutoModelForSpeechSeq2Seq

[[autodoc]] AutoModelForSpeechSeq2Seq

### TFAutoModelForSpeechSeq2Seq

No docstring available for TFAutoModelForSpeechSeq2Seq

### FlaxAutoModelForSpeechSeq2Seq

No docstring available for FlaxAutoModelForSpeechSeq2Seq

### AutoModelForAudioXVector

[[autodoc]] AutoModelForAudioXVector

### AutoModelForTextToSpectrogram

[[autodoc]] AutoModelForTextToSpectrogram

### AutoModelForTextToWaveform

[[autodoc]] AutoModelForTextToWaveform

## Multimodal

The following auto classes are available for the following multimodal tasks.

### AutoModelForTableQuestionAnswering

[[autodoc]] AutoModelForTableQuestionAnswering

### TFAutoModelForTableQuestionAnswering

No docstring available for TFAutoModelForTableQuestionAnswering

### AutoModelForDocumentQuestionAnswering

[[autodoc]] AutoModelForDocumentQuestionAnswering

### TFAutoModelForDocumentQuestionAnswering

No docstring available for TFAutoModelForDocumentQuestionAnswering

### AutoModelForVisualQuestionAnswering

[[autodoc]] AutoModelForVisualQuestionAnswering

### AutoModelForVision2Seq

[[autodoc]] AutoModelForVision2Seq

### TFAutoModelForVision2Seq

No docstring available for TFAutoModelForVision2Seq

### FlaxAutoModelForVision2Seq

No docstring available for FlaxAutoModelForVision2Seq

### AutoModelForImageTextToText

[[autodoc]] AutoModelForImageTextToText
