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

AutoConfig

    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    

## AutoTokenizer

AutoTokenizer

    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the [`AutoTokenizer.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    

## AutoFeatureExtractor

AutoFeatureExtractor

    This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the
    library when created with the [`AutoFeatureExtractor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    

## AutoImageProcessor

AutoImageProcessor

    This is a generic image processor class that will be instantiated as one of the image processor classes of the
    library when created with the [`AutoImageProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    

## AutoProcessor

AutoProcessor

    This is a generic processor class that will be instantiated as one of the processor classes of the library when
    created with the [`AutoProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    

## Generic model classes

The following auto classes are available for instantiating a base model class without a specific head.

### AutoModel

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).


### TFAutoModel

No docstring available for TFAutoModel

### FlaxAutoModel

No docstring available for FlaxAutoModel

## Generic pretraining classes

The following auto classes are available for instantiating a model with a pretraining head.

### AutoModelForPreTraining

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForPreTraining

### TFAutoModelForPreTraining

No docstring available for TFAutoModelForPreTraining

### FlaxAutoModelForPreTraining

No docstring available for FlaxAutoModelForPreTraining

## Natural Language Processing

The following auto classes are available for the following natural language processing tasks.

### AutoModelForCausalLM

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForCausalLM

### TFAutoModelForCausalLM

No docstring available for TFAutoModelForCausalLM

### FlaxAutoModelForCausalLM

No docstring available for FlaxAutoModelForCausalLM

### AutoModelForMaskedLM

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForMaskedLM

### TFAutoModelForMaskedLM

No docstring available for TFAutoModelForMaskedLM

### FlaxAutoModelForMaskedLM

No docstring available for FlaxAutoModelForMaskedLM

### AutoModelForMaskGeneration

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForMaskGeneration

### TFAutoModelForMaskGeneration

No docstring available for TFAutoModelForMaskGeneration

### AutoModelForSeq2SeqLM

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForSeq2SeqLM

### TFAutoModelForSeq2SeqLM

No docstring available for TFAutoModelForSeq2SeqLM

### FlaxAutoModelForSeq2SeqLM

No docstring available for FlaxAutoModelForSeq2SeqLM

### AutoModelForSequenceClassification

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForSequenceClassification

### TFAutoModelForSequenceClassification

No docstring available for TFAutoModelForSequenceClassification

### FlaxAutoModelForSequenceClassification

No docstring available for FlaxAutoModelForSequenceClassification

### AutoModelForMultipleChoice

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForMultipleChoice

### TFAutoModelForMultipleChoice

No docstring available for TFAutoModelForMultipleChoice

### FlaxAutoModelForMultipleChoice

No docstring available for FlaxAutoModelForMultipleChoice

### AutoModelForNextSentencePrediction

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForNextSentencePrediction

### TFAutoModelForNextSentencePrediction

No docstring available for TFAutoModelForNextSentencePrediction

### FlaxAutoModelForNextSentencePrediction

No docstring available for FlaxAutoModelForNextSentencePrediction

### AutoModelForTokenClassification

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForTokenClassification

### TFAutoModelForTokenClassification

No docstring available for TFAutoModelForTokenClassification

### FlaxAutoModelForTokenClassification

No docstring available for FlaxAutoModelForTokenClassification

### AutoModelForQuestionAnswering

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForQuestionAnswering

### TFAutoModelForQuestionAnswering

No docstring available for TFAutoModelForQuestionAnswering

### FlaxAutoModelForQuestionAnswering

No docstring available for FlaxAutoModelForQuestionAnswering

### AutoModelForTextEncoding

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForTextEncoding

### TFAutoModelForTextEncoding

No docstring available for TFAutoModelForTextEncoding

## Computer vision

The following auto classes are available for the following computer vision tasks.

### AutoModelForDepthEstimation

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForDepthEstimation

### AutoModelForImageClassification

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForImageClassification

### TFAutoModelForImageClassification

No docstring available for TFAutoModelForImageClassification

### FlaxAutoModelForImageClassification

No docstring available for FlaxAutoModelForImageClassification

### AutoModelForVideoClassification

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForVideoClassification

### AutoModelForKeypointDetection

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForKeypointDetection

### AutoModelForMaskedImageModeling

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForMaskedImageModeling

### TFAutoModelForMaskedImageModeling

No docstring available for TFAutoModelForMaskedImageModeling

### AutoModelForObjectDetection

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForObjectDetection

### AutoModelForImageSegmentation

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForImageSegmentation

### AutoModelForImageToImage

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForImageToImage

### AutoModelForSemanticSegmentation

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForSemanticSegmentation

### TFAutoModelForSemanticSegmentation

No docstring available for TFAutoModelForSemanticSegmentation

### AutoModelForInstanceSegmentation

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForInstanceSegmentation

### AutoModelForUniversalSegmentation

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForUniversalSegmentation

### AutoModelForZeroShotImageClassification

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForZeroShotImageClassification

### TFAutoModelForZeroShotImageClassification

No docstring available for TFAutoModelForZeroShotImageClassification

### AutoModelForZeroShotObjectDetection

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForZeroShotObjectDetection

## Audio

The following auto classes are available for the following audio tasks.

### AutoModelForAudioClassification

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForAudioClassification

### AutoModelForAudioFrameClassification

No docstring available for TFAutoModelForAudioClassification

### TFAutoModelForAudioFrameClassification

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForAudioFrameClassification

### AutoModelForCTC

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForCTC

### AutoModelForSpeechSeq2Seq

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForSpeechSeq2Seq

### TFAutoModelForSpeechSeq2Seq

No docstring available for TFAutoModelForSpeechSeq2Seq

### FlaxAutoModelForSpeechSeq2Seq

No docstring available for FlaxAutoModelForSpeechSeq2Seq

### AutoModelForAudioXVector

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForAudioXVector

### AutoModelForTextToSpectrogram

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForTextToSpectrogram

### AutoModelForTextToWaveform

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForTextToWaveform

## Multimodal

The following auto classes are available for the following multimodal tasks.

### AutoModelForTableQuestionAnswering

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForTableQuestionAnswering

### TFAutoModelForTableQuestionAnswering

No docstring available for TFAutoModelForTableQuestionAnswering

### AutoModelForDocumentQuestionAnswering

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForDocumentQuestionAnswering

### TFAutoModelForDocumentQuestionAnswering

No docstring available for TFAutoModelForDocumentQuestionAnswering

### AutoModelForVisualQuestionAnswering

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForVisualQuestionAnswering

### AutoModelForVision2Seq

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForVision2Seq

### TFAutoModelForVision2Seq

No docstring available for TFAutoModelForVision2Seq

### FlaxAutoModelForVision2Seq

No docstring available for FlaxAutoModelForVision2Seq

### AutoModelForImageTextToText

AutoModel

    This is a generic model class that will be instantiated as one of the base model classes of the library when created
    with the [`~AutoModel.from_pretrained`] class method or the [`~AutoModel.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
ForImageTextToText
