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

# Pipelines

The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of
the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity
Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering. See the
[task summary](../task_summary) for examples of use.

There are two categories of pipeline abstractions to be aware about:

- The [`pipeline`] which is the most powerful object encapsulating all other pipelines.
- Task-specific pipelines are available for [audio](#audio), [computer vision](#computer-vision), [natural language processing](#natural-language-processing), and [multimodal](#multimodal) tasks.

## The pipeline abstraction

The *pipeline* abstraction is a wrapper around all the other available pipelines. It is instantiated as any other
pipeline but can provide additional quality of life.

Simple call on one item:

```python
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

If you want to use a specific model from the [hub](https://huggingface.co) you can ignore the task if the model on
the hub already defines it:

```python
>>> pipe = pipeline(model="FacebookAI/roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

To call a pipeline on many items, you can call it with a *list*.

```python
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

To iterate over full datasets it is recommended to use a `dataset` directly. This means you don't need to allocate
the whole dataset at once, nor do you need to do batching yourself. This should work just as fast as custom loops on
GPU. If it doesn't don't hesitate to create an issue.

```python
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")

# KeyDataset (only *pt*) will simply return the item in the dict returned by the dataset item
# as we're not interested in the *target* part of the dataset. For sentence pair use KeyPairDataset
for out in tqdm(pipe(KeyDataset(dataset, "file"))):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

For ease of use, a generator is also possible:


```python
from transformers import pipeline

pipe = pipeline("text-classification")


def data():
    while True:
        # This could come from a dataset, a database, a queue or HTTP request
        # in a server
        # Caveat: because this is iterative, you cannot use `num_workers > 1` variable
        # to use multiple threads to preprocess data. You can still have 1 thread that
        # does the preprocessing while the main runs the big inference
        yield "This is a test"


for out in pipe(data()):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```


    Utility factory method to build a [`Pipeline`].

    A pipeline consists of:

        - One or more components for pre-processing model inputs, such as a [tokenizer](tokenizer),
        [image_processor](image_processor), [feature_extractor](feature_extractor), or [processor](processors).
        - A [model](model) that generates predictions from the inputs.
        - Optional post-processing steps to refine the model's output, which can also be handled by processors.

    <Tip>
    While there are such optional arguments as `tokenizer`, `feature_extractor`, `image_processor`, and `processor`,
    they shouldn't be specified all at once. If these components are not provided, `pipeline` will try to load
    required ones automatically. In case you want to provide these components explicitly, please refer to a
    specific pipeline in order to get more details regarding what components are required.
    </Tip>

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`: will return a [`AudioClassificationPipeline`].
            - `"automatic-speech-recognition"`: will return a [`AutomaticSpeechRecognitionPipeline`].
            - `"depth-estimation"`: will return a [`DepthEstimationPipeline`].
            - `"document-question-answering"`: will return a [`DocumentQuestionAnsweringPipeline`].
            - `"feature-extraction"`: will return a [`FeatureExtractionPipeline`].
            - `"fill-mask"`: will return a [`FillMaskPipeline`]:.
            - `"image-classification"`: will return a [`ImageClassificationPipeline`].
            - `"image-feature-extraction"`: will return an [`ImageFeatureExtractionPipeline`].
            - `"image-segmentation"`: will return a [`ImageSegmentationPipeline`].
            - `"image-text-to-text"`: will return a [`ImageTextToTextPipeline`].
            - `"image-to-image"`: will return a [`ImageToImagePipeline`].
            - `"image-to-text"`: will return a [`ImageToTextPipeline`].
            - `"mask-generation"`: will return a [`MaskGenerationPipeline`].
            - `"object-detection"`: will return a [`ObjectDetectionPipeline`].
            - `"question-answering"`: will return a [`QuestionAnsweringPipeline`].
            - `"summarization"`: will return a [`SummarizationPipeline`].
            - `"table-question-answering"`: will return a [`TableQuestionAnsweringPipeline`].
            - `"text2text-generation"`: will return a [`Text2TextGenerationPipeline`].
            - `"text-classification"` (alias `"sentiment-analysis"` available): will return a
              [`TextClassificationPipeline`].
            - `"text-generation"`: will return a [`TextGenerationPipeline`]:.
            - `"text-to-audio"` (alias `"text-to-speech"` available): will return a [`TextToAudioPipeline`]:.
            - `"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
            - `"translation"`: will return a [`TranslationPipeline`].
            - `"translation_xx_to_yy"`: will return a [`TranslationPipeline`].
            - `"video-classification"`: will return a [`VideoClassificationPipeline`].
            - `"visual-question-answering"`: will return a [`VisualQuestionAnsweringPipeline`].
            - `"zero-shot-classification"`: will return a [`ZeroShotClassificationPipeline`].
            - `"zero-shot-image-classification"`: will return a [`ZeroShotImageClassificationPipeline`].
            - `"zero-shot-audio-classification"`: will return a [`ZeroShotAudioClassificationPipeline`].
            - `"zero-shot-object-detection"`: will return a [`ZeroShotObjectDetectionPipeline`].

        model (`str` or [`PreTrainedModel`] or [`TFPreTrainedModel`], *optional*):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from [`PreTrainedModel`] (for PyTorch) or
            [`TFPreTrainedModel`] (for TensorFlow).

            If not provided, the default for the `task` will be loaded.
        config (`str` or [`PretrainedConfig`], *optional*):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from [`PretrainedConfig`].

            If not provided, the default configuration file for the requested model will be used. That means that if
            `model` is given, its default configuration will be used. However, if `model` is not supplied, this
            `task`'s default model's config is used instead.
        tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from [`PreTrainedTokenizer`].

            If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
            However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
            will be loaded.
        feature_extractor (`str` or [`PreTrainedFeatureExtractor`], *optional*):
            The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained feature extractor inheriting from [`PreTrainedFeatureExtractor`].

            Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal
            models. Multi-modal models will also require a tokenizer to be passed.

            If not provided, the default feature extractor for the given `model` will be loaded (if it is a string). If
            `model` is not specified or not a string, then the default feature extractor for `config` is loaded (if it
            is a string). However, if `config` is also not given or not a string, then the default feature extractor
            for the given `task` will be loaded.
        image_processor (`str` or [`BaseImageProcessor`], *optional*):
            The image processor that will be used by the pipeline to preprocess images for the model. This can be a
            model identifier or an actual image processor inheriting from [`BaseImageProcessor`].

            Image processors are used for Vision models and multi-modal models that require image inputs. Multi-modal
            models will also require a tokenizer to be passed.

            If not provided, the default image processor for the given `model` will be loaded (if it is a string). If
            `model` is not specified or not a string, then the default image processor for `config` is loaded (if it is
            a string).
        processor (`str` or [`ProcessorMixin`], *optional*):
            The processor that will be used by the pipeline to preprocess data for the model. This can be a model
            identifier or an actual processor inheriting from [`ProcessorMixin`].

            Processors are used for multi-modal models that require multi-modal inputs, for example, a model that
            requires both text and image inputs.

            If not provided, the default processor for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default processor for `config` is loaded (if it is a string).
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        revision (`str`, *optional*, defaults to `"main"`):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
        use_fast (`bool`, *optional*, defaults to `True`):
            Whether or not to use a Fast tokenizer if possible (a [`PreTrainedTokenizerFast`]).
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        device (`int` or `str` or `torch.device`):
            Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which this
            pipeline will be allocated.
        device_map (`str` or `Dict[str, Union[int, str, torch.device]`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut). When `accelerate` library is present, set
            `device_map="auto"` to compute the most optimized `device_map` automatically (see
            [here](https://huggingface.co/docs/accelerate/main/en/package_reference/big_modeling#accelerate.cpu_offload)
            for more information).

            <Tip warning={true}>

            Do not use `device_map` AND `device` at the same time as they will conflict

            </Tip>

        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`).
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether or not to allow for custom code defined on the Hub in their own modeling, configuration,
            tokenization or even pipeline files. This option should only be set to `True` for repositories you trust
            and in which you have read the code, as it will execute code present on the Hub on your local machine.
        model_kwargs (`Dict[str, Any]`, *optional*):
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        [`Pipeline`]: A suitable pipeline for the task.

    Examples:

    ```python
    >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

    >>> # Sentiment analysis pipeline
    >>> analyzer = pipeline("sentiment-analysis")

    >>> # Question answering pipeline, specifying the checkpoint identifier
    >>> oracle = pipeline(
    ...     "question-answering", model="distilbert/distilbert-base-cased-distilled-squad", tokenizer="google-bert/bert-base-cased"
    ... )

    >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
    >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    >>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
    ```

## Pipeline batching

All pipelines can use batching. This will work
whenever the pipeline uses its streaming ability (so when passing lists or `Dataset` or `generator`).

```python
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
    # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
    # Exactly the same output as before, but the content are passed
    # as batches to the model
```

<Tip warning={true}>

However, this is not automatically a win for performance. It can be either a 10x speedup or 5x slowdown depending
on hardware, data and the actual model being used.

Example where it's mostly a speedup:

</Tip>

```python
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)


class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "This is a test"


dataset = MyDataset()

for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass
```

```
# On GTX 970
------------------------------
Streaming no batching
100%|██████████████████████████████████████████████████████████████████████| 5000/5000 [00:26<00:00, 187.52it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1205.95it/s]
------------------------------
Streaming batch_size=64
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2478.24it/s]
------------------------------
Streaming batch_size=256
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:01<00:00, 2554.43it/s]
(diminishing returns, saturated the GPU)
```

Example where it's most a slowdown:

```python
class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        if i % 64 == 0:
            n = 100
        else:
            n = 1
        return "This is a test" * n
```

This is a occasional very long sentence compared to the other. In that case, the **whole** batch will need to be 400
tokens long, so the whole batch will be [64, 400] instead of [64, 4], leading to the high slowdown. Even worse, on
bigger batches, the program simply crashes.


```
------------------------------
Streaming no batching
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 183.69it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 265.74it/s]
------------------------------
Streaming batch_size=64
100%|██████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.80it/s]
------------------------------
Streaming batch_size=256
  0%|                                                                                 | 0/1000 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/nicolas/src/transformers/test.py", line 42, in <module>
    for out in tqdm(pipe(dataset, batch_size=256), total=len(dataset)):
....
    q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
RuntimeError: CUDA out of memory. Tried to allocate 376.00 MiB (GPU 0; 3.95 GiB total capacity; 1.72 GiB already allocated; 354.88 MiB free; 2.46 GiB reserved in total by PyTorch)
```

There are no good (general) solutions for this problem, and your mileage may vary depending on your use cases. Rule of
thumb:

For users, a rule of thumb is:

- **Measure performance on your load, with your hardware. Measure, measure, and keep measuring. Real numbers are the
  only way to go.**
- If you are latency constrained (live product doing inference), don't batch.
- If you are using CPU, don't batch.
- If you are using throughput (you want to run your model on a bunch of static data), on GPU, then:

  - If you have no clue about the size of the sequence_length ("natural" data), by default don't batch, measure and
    try tentatively to add it, add OOM checks to recover when it will fail (and it will at some point if you don't
    control the sequence_length.)
  - If your sequence_length is super regular, then batching is more likely to be VERY interesting, measure and push
    it until you get OOMs.
  - The larger the GPU the more likely batching is going to be more interesting
- As soon as you enable batching, make sure you can handle OOMs nicely.

## Pipeline chunk batching

`zero-shot-classification` and `question-answering` are slightly specific in the sense, that a single input might yield
multiple forward pass of a model. Under normal circumstances, this would yield issues with `batch_size` argument.

In order to circumvent this issue, both of these pipelines are a bit specific, they are `ChunkPipeline` instead of
regular `Pipeline`. In short:


```python
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

Now becomes:


```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

This should be very transparent to your code because the pipelines are used in
the same way.

This is a simplified view, since the pipeline can handle automatically the batch to ! Meaning you don't have to care
about how many forward passes you inputs are actually going to trigger, you can optimize the `batch_size`
independently of the inputs. The caveats from the previous section still apply.

## Pipeline FP16 inference
Models can be run in FP16 which can be significantly faster on GPU while saving memory. Most models will not suffer noticeable performance loss from this. The larger the model, the less likely that it will.

To enable FP16 inference, you can simply pass `torch_dtype=torch.float16` or `torch_dtype='float16'` to the pipeline constructor. Note that this only works for models with a PyTorch backend. Your inputs will be converted to FP16 internally.

## Pipeline custom code

If you want to override a specific pipeline.

Don't hesitate to create an issue for your task at hand, the goal of the pipeline is to be easy to use and support most
cases, so `transformers` could maybe support your use case.


If you want to try simply you can:

- Subclass your pipeline of choice

```python
class MyPipeline(TextClassificationPipeline):
    def postprocess():
        # Your code goes here
        scores = scores * 100
        # And here


my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
# or if you use *pipeline* function, then:
my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)
```

That should enable you to do all the custom code you want.


## Implementing a pipeline

[Implementing a new pipeline](../add_new_pipeline)

## Audio

Pipelines available for audio tasks include the following.

### AudioClassificationPipeline


    Audio classification pipeline using any `AutoModelForAudioClassification`. This pipeline predicts the class of a
    raw waveform or an audio file. In case of an audio file, ffmpeg should be installed to support multiple audio
    formats.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="superb/wav2vec2-base-superb-ks")
    >>> classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    [{'score': 0.997, 'label': '_unknown_'}, {'score': 0.002, 'label': 'left'}, {'score': 0.0, 'label': 'yes'}, {'score': 0.0, 'label': 'down'}, {'score': 0.0, 'label': 'stop'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)


    This pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"audio-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=audio-classification).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        feature_extractor ([`SequenceFeatureExtractor`]):
            The feature extractor that will be used by the pipeline to encode data for the model. This object inherits from
            [`SequenceFeatureExtractor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### AutomaticSpeechRecognitionPipeline


    Pipeline that aims at extracting spoken text contained within some audio.

    The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
    to support multiple audio formats

    Example:

    ```python
    >>> from transformers import pipeline

    >>> transcriber = pipeline(model="openai/whisper-base")
    >>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    {'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        feature_extractor ([`SequenceFeatureExtractor`]):
            The feature extractor that will be used by the pipeline to encode waveform for the model.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        decoder (`pyctcdecode.BeamSearchDecoderCTC`, *optional*):
            [PyCTCDecode's
            BeamSearchDecoderCTC](https://github.com/kensho-technologies/pyctcdecode/blob/2fd33dc37c4111417e08d89ccd23d28e9b308d19/pyctcdecode/decoder.py#L180)
            can be passed for language model boosted decoding. See [`Wav2Vec2ProcessorWithLM`] for more information.
        chunk_length_s (`float`, *optional*, defaults to 0):
            The input length for in each chunk. If `chunk_length_s = 0` then chunking is disabled (default).

            <Tip>

            For more information on how to effectively use `chunk_length_s`, please have a look at the [ASR chunking
            blog post](https://huggingface.co/blog/asr-chunking).

            </Tip>

        stride_length_s (`float`, *optional*, defaults to `chunk_length_s / 6`):
            The length of stride on the left and right of each chunk. Used only with `chunk_length_s > 0`. This enables
            the model to *see* more context and infer letters better than without this context but the pipeline
            discards the stride bits at the end to make the final reconstitution as perfect as possible.

            <Tip>

            For more information on how to effectively use `stride_length_s`, please have a look at the [ASR chunking
            blog post](https://huggingface.co/blog/asr-chunking).

            </Tip>

        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed. If no framework is specified, will default to the one currently installed. If no framework is
            specified and both frameworks are installed, will default to the framework of the `model`, or to PyTorch if
            no model is provided.
        device (Union[`int`, `torch.device`], *optional*):
            Device ordinal for CPU/GPU supports. Setting this to `None` will leverage CPU, a positive will run the
            model on the associated CUDA device id.
        torch_dtype (Union[`int`, `torch.dtype`], *optional*):
            The data-type (dtype) of the computation. Setting this to `None` will use float32 precision. Set to
            `torch.float16` or `torch.bfloat16` to use half-precision in the respective dtypes.

    
    - __call__
    - all

### TextToAudioPipeline


    Text-to-audio generation pipeline using any `AutoModelForTextToWaveform` or `AutoModelForTextToSpectrogram`. This
    pipeline generates an audio file from an input text and optional other conditional inputs.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline(model="suno/bark-small")
    >>> output = pipe("Hey it's HuggingFace on the phone!")

    >>> audio = output["audio"]
    >>> sampling_rate = output["sampling_rate"]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    <Tip>

    You can specify parameters passed to the model by using [`TextToAudioPipeline.__call__.forward_params`] or
    [`TextToAudioPipeline.__call__.generate_kwargs`].

    Example:

    ```python
    >>> from transformers import pipeline

    >>> music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small", framework="pt")

    >>> # diversify the music generation by adding randomness with a high temperature and set a maximum music length
    >>> generate_kwargs = {
    ...     "do_sample": True,
    ...     "temperature": 0.7,
    ...     "max_new_tokens": 35,
    ... }

    >>> outputs = music_generator("Techno music with high melodic riffs", generate_kwargs=generate_kwargs)
    ```

    </Tip>

    This pipeline can currently be loaded from [`pipeline`] using the following task identifiers: `"text-to-speech"` or
    `"text-to-audio"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=text-to-speech).
    
    - __call__
    - all


### ZeroShotAudioClassificationPipeline


    Zero shot audio classification pipeline using `ClapModel`. This pipeline predicts the class of an audio when you
    provide an audio and a set of `candidate_labels`.

    <Tip warning={true}>

    The default `hypothesis_template` is : `"This is a sound of {}."`. Make sure you update it for your usage.

    </Tip>

    Example:
    ```python
    >>> from transformers import pipeline
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("ashraq/esc50")
    >>> audio = next(iter(dataset["train"]["audio"]))["array"]
    >>> classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
    >>> classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])
    [{'score': 0.9996, 'label': 'Sound of a dog'}, {'score': 0.0004, 'label': 'Sound of vaccum cleaner'}]
    ```


    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial) This audio
    classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-audio-classification"`. See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-audio-classification).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        feature_extractor ([`SequenceFeatureExtractor`]):
            The feature extractor that will be used by the pipeline to encode data for the model. This object inherits from
            [`SequenceFeatureExtractor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

## Computer vision

Pipelines available for computer vision tasks include the following.

### DepthEstimationPipeline

    Depth estimation pipeline using any `AutoModelForDepthEstimation`. This pipeline predicts the depth of an image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
    >>> output = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
    >>> # This is a tensor with the values being the depth expressed in meters for each pixel
    >>> output["predicted_depth"].shape
    torch.Size([1, 384, 384])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)


    This depth estimation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"depth-estimation"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=depth-estimation).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### ImageClassificationPipeline


    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an
    image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
    >>> classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.442, 'label': 'macaw'}, {'score': 0.088, 'label': 'popinjay'}, {'score': 0.075, 'label': 'parrot'}, {'score': 0.073, 'label': 'parodist, lampooner'}, {'score': 0.046, 'label': 'poll, poll_parrot'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-classification).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.
    - __call__
    - all

### ImageSegmentationPipeline


    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and
    their classes.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic")
    >>> segments = segmenter("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    >>> len(segments)
    2

    >>> segments[0]["label"]
    'bird'

    >>> segments[1]["label"]
    'bird'

    >>> type(segments[0]["mask"])  # This is a black and white mask showing where is the bird on the original image.
    <class 'PIL.Image.Image'>

    >>> segments[0]["mask"].size
    (768, 512)
    ```


    This image segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-segmentation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-segmentation).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### ImageToImagePipeline


    Image to Image pipeline using any `AutoModelForImageToImage`. This pipeline generates an image based on a previous
    image input.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests

    >>> from transformers import pipeline

    >>> upscaler = pipeline("image-to-image", model="caidas/swin2SR-classical-sr-x2-64")
    >>> img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    >>> img = img.resize((64, 64))
    >>> upscaled_img = upscaler(img)
    >>> img.size
    (64, 64)

    >>> upscaled_img.size
    (144, 144)
    ```

    This image to image pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-to-image"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=image-to-image).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### ObjectDetectionPipeline


    Object detection pipeline using any `AutoModelForObjectDetection`. This pipeline predicts bounding boxes of objects
    and their classes.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> detector = pipeline(model="facebook/detr-resnet-50")
    >>> detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}, {'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}]

    >>> # x, y  are expressed relative to the top left hand corner.
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"object-detection"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=object-detection).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### VideoClassificationPipeline


    Video classification pipeline using any `AutoModelForVideoClassification`. This pipeline predicts the class of a
    video.

    This video classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"video-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=video-classification).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### ZeroShotImageClassificationPipeline


    Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you
    provide an image and a set of `candidate_labels`.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="google/siglip-so400m-patch14-384")
    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["animals", "humans", "landscape"],
    ... )
    [{'score': 0.965, 'label': 'animals'}, {'score': 0.03, 'label': 'humans'}, {'score': 0.005, 'label': 'landscape'}]

    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["black and white", "photorealist", "painting"],
    ... )
    [{'score': 0.996, 'label': 'black and white'}, {'score': 0.003, 'label': 'photorealist'}, {'score': 0.0, 'label': 'painting'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-image-classification).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### ZeroShotObjectDetectionPipeline


    Zero shot object detection pipeline using `OwlViTForObjectDetection`. This pipeline predicts bounding boxes of
    objects when you provide an image and a set of `candidate_labels`.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
    >>> detector(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ...     candidate_labels=["cat", "couch"],
    ... )
    [{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]

    >>> detector(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["head", "bird"],
    ... )
    [{'score': 0.119, 'label': 'bird', 'box': {'xmin': 71, 'ymin': 170, 'xmax': 410, 'ymax': 508}}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-object-detection"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

## Natural Language Processing

Pipelines available for natural language processing tasks include the following.

### FillMaskPipeline


    Masked language modeling prediction pipeline using any `ModelWithLMHead`. See the [masked language modeling
    examples](../task_summary#masked-language-modeling) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> fill_masker = pipeline(model="google-bert/bert-base-uncased")
    >>> fill_masker("This is a simple [MASK].")
    [{'score': 0.042, 'token': 3291, 'token_str': 'problem', 'sequence': 'this is a simple problem.'}, {'score': 0.031, 'token': 3160, 'token_str': 'question', 'sequence': 'this is a simple question.'}, {'score': 0.03, 'token': 8522, 'token_str': 'equation', 'sequence': 'this is a simple equation.'}, {'score': 0.027, 'token': 2028, 'token_str': 'one', 'sequence': 'this is a simple one.'}, {'score': 0.024, 'token': 3627, 'token_str': 'rule', 'sequence': 'this is a simple rule.'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This mask filling pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"fill-mask"`.

    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library. See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=fill-mask).

    <Tip>

    This pipeline only works for inputs with exactly one token masked. Experimental: We added support for multiple
    masks. The returned values are raw model output, and correspond to disjoint probabilities where one might expect
    joint probabilities (See [discussion](https://github.com/huggingface/transformers/pull/10222)).

    </Tip>

    <Tip>

    This pipeline now supports tokenizer_kwargs. For example try:

    ```python
    >>> from transformers import pipeline

    >>> fill_masker = pipeline(model="google-bert/bert-base-uncased")
    >>> tokenizer_kwargs = {"truncation": True}
    >>> fill_masker(
    ...     "This is a simple [MASK]. " + "...with a large amount of repeated text appended. " * 100,
    ...     tokenizer_kwargs=tokenizer_kwargs,
    ... )
    ```


    </Tip>


    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
        top_k (`int`, *optional*, defaults to 5):
            The number of predictions to return.
        targets (`str` or `List[str]`, *optional*):
            When passed, the model will limit the scores to the passed targets instead of looking up in the whole
            vocab. If the provided targets are not in the model vocab, they will be tokenized and the first resulting
            token will be used (with a warning, and that might be slower).
        tokenizer_kwargs (`dict`, *optional*):
            Additional dictionary of keyword arguments passed along to the tokenizer.
    - __call__
    - all

### QuestionAnsweringPipeline


    Question Answering pipeline using any `ModelForQuestionAnswering`. See the [question answering
    examples](../task_summary#question-answering) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="deepset/roberta-base-squad2")
    >>> oracle(question="Where do I live?", context="My name is Wolfgang and I live in Berlin")
    {'score': 0.9191, 'start': 34, 'end': 40, 'answer': 'Berlin'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This question answering pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a question answering task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=question-answering).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### SummarizationPipeline


    Summarize news articles and other documents.

    This summarizing pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"summarization"`.

    The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is
    currently, '*bart-large-cnn*', '*google-t5/t5-small*', '*google-t5/t5-base*', '*google-t5/t5-large*', '*google-t5/t5-3b*', '*google-t5/t5-11b*'. See the up-to-date
    list of available models on [huggingface.co/models](https://huggingface.co/models?filter=summarization). For a list
    of available parameters, see the [following
    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Usage:

    ```python
    # use bart in pytorch
    summarizer = pipeline("summarization")
    summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)

    # use t5 in tf
    summarizer = pipeline("summarization", model="google-t5/t5-base", tokenizer="google-t5/t5-base", framework="tf")
    summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)
    ```
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### TableQuestionAnsweringPipeline


    Table Question Answering pipeline using a `ModelForTableQuestionAnswering`. This pipeline is only available in
    PyTorch.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="google/tapas-base-finetuned-wtq")
    >>> table = {
    ...     "Repository": ["Transformers", "Datasets", "Tokenizers"],
    ...     "Stars": ["36542", "4512", "3934"],
    ...     "Contributors": ["651", "77", "34"],
    ...     "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
    ... }
    >>> oracle(query="How many stars does the transformers repository have?", table=table)
    {'answer': 'AVERAGE > 36542', 'coordinates': [(0, 1)], 'cells': ['36542'], 'aggregator': 'AVERAGE'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This tabular question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"table-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=table-question-answering).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__

### TextClassificationPipeline


    Text classification pipeline using any `ModelForSequenceClassification`. See the [sequence classification
    examples](../task_summary#sequence-classification) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    >>> classifier("This movie is disgustingly good !")
    [{'label': 'POSITIVE', 'score': 1.0}]

    >>> classifier("Director tried too much.")
    [{'label': 'NEGATIVE', 'score': 0.996}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This text classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"sentiment-analysis"` (for classifying sequences according to positive or negative sentiments).

    If multiple classification labels are available (`model.config.num_labels >= 2`), the pipeline will run a softmax
    over the results. If there is a single label, the pipeline will run a sigmoid over the result. In case of regression
    tasks (`model.config.problem_type == "regression"`), will not apply any function on the output.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
    the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=text-classification).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
        return_all_scores (`bool`, *optional*, defaults to `False`):
            Whether to return all prediction scores or just the one of the predicted class.
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output. In case of regression tasks, will not
              apply any function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.
    - __call__
    - all

### TextGenerationPipeline


    Language generation pipeline using any `ModelWithLMHead`. This pipeline predicts the words that will follow a
    specified text prompt. When the underlying model is a conversational model, it can also accept one or more chats,
    in which case the pipeline will operate in chat mode and will continue the chat(s) by adding its response(s).
    Each chat takes the form of a list of dicts, where each dict contains "role" and "content" keys.

    Examples:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="openai-community/gpt2")
    >>> generator("I can't believe you did such a ", do_sample=False)
    [{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

    >>> # These parameters will return suggestions, and only the newly created text making it easier for prompting suggestions.
    >>> outputs = generator("My tart needs some", num_return_sequences=4, return_full_text=False)
    ```

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="HuggingFaceH4/zephyr-7b-beta")
    >>> # Zephyr-beta is a conversational model, so let's pass it a chat instead of a single string
    >>> generator([{"role": "user", "content": "What is the capital of France? Answer in one word."}], do_sample=False, max_new_tokens=2)
    [{'generated_text': [{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}, {'role': 'assistant', 'content': 'Paris'}]}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

    This language generation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"text-generation"`.

    The models that this pipeline can use are models that have been trained with an autoregressive language modeling
    objective. See the list of available [text completion models](https://huggingface.co/models?filter=text-generation)
    and the list of [conversational models](https://huggingface.co/models?other=conversational)
    on [huggingface.co/models].
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### Text2TextGenerationPipeline


    Pipeline for text to text generation using seq2seq models.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="mrm8488/t5-base-finetuned-question-generation-ap")
    >>> generator(
    ...     "answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
    ... )
    [{'generated_text': 'question: Who created the RuPERTa-base?'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

    This Text2TextGenerationPipeline pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"text2text-generation"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=text2text-generation). For a list of available
    parameters, see the [following
    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Usage:

    ```python
    text2text_generator = pipeline("text2text-generation")
    text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
    ```
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### TokenClassificationPipeline


    Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition
    examples](../task_summary#named-entity-recognition) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> token_classifier = pipeline(model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
    >>> sentence = "Je m'appelle jean-baptiste et je vis à montréal"
    >>> tokens = token_classifier(sentence)
    >>> tokens
    [{'entity_group': 'PER', 'score': 0.9931, 'word': 'jean-baptiste', 'start': 12, 'end': 26}, {'entity_group': 'LOC', 'score': 0.998, 'word': 'montréal', 'start': 38, 'end': 47}]

    >>> token = tokens[0]
    >>> # Start and end provide an easy way to highlight words in the original text.
    >>> sentence[token["start"] : token["end"]]
    ' jean-baptiste'

    >>> # Some models use the same idea to do part of speech.
    >>> syntaxer = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")
    >>> syntaxer("My name is Sarah and I live in London")
    [{'entity_group': 'PRON', 'score': 0.999, 'word': 'my', 'start': 0, 'end': 2}, {'entity_group': 'NOUN', 'score': 0.997, 'word': 'name', 'start': 3, 'end': 7}, {'entity_group': 'AUX', 'score': 0.994, 'word': 'is', 'start': 8, 'end': 10}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'sarah', 'start': 11, 'end': 16}, {'entity_group': 'CCONJ', 'score': 0.999, 'word': 'and', 'start': 17, 'end': 20}, {'entity_group': 'PRON', 'score': 0.999, 'word': 'i', 'start': 21, 'end': 22}, {'entity_group': 'VERB', 'score': 0.998, 'word': 'live', 'start': 23, 'end': 27}, {'entity_group': 'ADP', 'score': 0.999, 'word': 'in', 'start': 28, 'end': 30}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'london', 'start': 31, 'end': 37}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This token recognition pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=token-classification).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
        ignore_labels (`List[str]`, defaults to `["O"]`):
            A list of labels to ignore.
        grouped_entities (`bool`, *optional*, defaults to `False`):
            DEPRECATED, use `aggregation_strategy` instead. Whether or not to group the tokens corresponding to the
            same entity together in the predictions or not.
        stride (`int`, *optional*):
            If stride is provided, the pipeline is applied on all the text. The text is split into chunks of size
            model_max_length. Works only with fast tokenizers and `aggregation_strategy` different from `NONE`. The
            value of this argument defines the number of overlapping tokens between chunks. In other words, the model
            will shift forward by `tokenizer.model_max_length - stride` tokens each step.
        aggregation_strategy (`str`, *optional*, defaults to `"none"`):
            The strategy to fuse (or not) tokens based on the model prediction.

                - "none" : Will simply not do any aggregation and simply return raw results from the model
                - "simple" : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C,
                  I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{"word": ABC, "entity": "TAG"}, {"word": "D",
                  "entity": "TAG2"}, {"word": "E", "entity": "TAG2"}] Notice that two consecutive B tags will end up as
                  different entities. On word based languages, we might end up splitting words undesirably : Imagine
                  Microsoft being tagged as [{"word": "Micro", "entity": "ENTERPRISE"}, {"word": "soft", "entity":
                  "NAME"}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages
                  that support that meaning, which is basically tokens separated by a space). These mitigations will
                  only work on real words, "New york" might still be tagged with two different entities.
                - "first" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot
                  end up with different tags. Words will simply use the tag of the first token of the word when there
                  is ambiguity.
                - "average" : (works only on word based models) Will use the `SIMPLE` strategy except that words,
                  cannot end up with different tags. scores will be averaged first across tokens, and then the maximum
                  label is applied.
                - "max" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot
                  end up with different tags. Word entity will simply be the token with the maximum score.
    - __call__
    - all

### TranslationPipeline


    Translates from one language to another.

    This translation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"translation_xx_to_yy"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=translation).
    For a list of available parameters, see the [following
    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Usage:

    ```python
    en_fr_translator = pipeline("translation_en_to_fr")
    en_fr_translator("How old are you?")
    ```
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### ZeroShotClassificationPipeline


    NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification` trained on NLI (natural
    language inference) tasks. Equivalent of `text-classification` pipelines, but these models don't require a
    hardcoded number of potential classes, they can be chosen at runtime. It usually means it's slower but it is
    **much** more flexible.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pretrained model. Then, the logit for *entailment* is taken as the logit for the candidate
    label being valid. Any NLI model can be used, but the id of the *entailment* label must be included in the model
    config's :attr:*~transformers.PretrainedConfig.label2id*.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="facebook/bart-large-mnli")
    >>> oracle(
    ...     "I have a problem with my iphone that needs to be resolved asap!!",
    ...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    ... )
    {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

    >>> oracle(
    ...     "I have a problem with my iphone that needs to be resolved asap!!",
    ...     candidate_labels=["english", "german"],
    ... )
    {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['english', 'german'], 'scores': [0.814, 0.186]}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This NLI pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list
    of available models on [huggingface.co/models](https://huggingface.co/models?search=nli).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

## Multimodal

Pipelines available for multimodal tasks include the following.

### DocumentQuestionAnsweringPipeline


    Document Question Answering pipeline using any `AutoModelForDocumentQuestionAnswering`. The inputs/outputs are
    similar to the (extractive) question answering pipeline; however, the pipeline takes an image (and optional OCR'd
    words/boxes) as input instead of text context.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> document_qa = pipeline(model="impira/layoutlm-document-qa")
    >>> document_qa(
    ...     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    ...     question="What is the invoice number?",
    ... )
    [{'score': 0.425, 'answer': 'us-001', 'start': 16, 'end': 16}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This document question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"document-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a document question answering task.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=document-question-answering).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### FeatureExtractionPipeline


    Feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> extractor = pipeline(model="google-bert/bert-base-uncased", task="feature-extraction")
    >>> result = extractor("This is a simple test.", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, sequence_length, hidden_dimension] representing the input string.
    torch.Size([1, 8, 768])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This feature extraction pipeline can currently be loaded from [`pipeline`] using the task identifier:
    `"feature-extraction"`.

    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    [huggingface.co/models](https://huggingface.co/models).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        tokenize_kwargs (`dict`, *optional*):
                Additional dictionary of keyword arguments passed along to the tokenizer.
        return_tensors (`bool`, *optional*):
            If `True`, returns a tensor according to the specified framework, otherwise returns a list.
    - __call__
    - all

### ImageFeatureExtractionPipeline


    Image feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> extractor = pipeline(model="google/vit-base-patch16-224", task="image-feature-extraction")
    >>> result = extractor("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, sequence_lenth, hidden_dimension] representing the input image.
    torch.Size([1, 197, 768])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image feature extraction pipeline can currently be loaded from [`pipeline`] using the task identifier:
    `"image-feature-extraction"`.

    All vision models may be used for this pipeline. See a list of all models, including community-contributed models on
    [huggingface.co/models](https://huggingface.co/models).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
        image_processor_kwargs (`dict`, *optional*):
                Additional dictionary of keyword arguments passed along to the image processor e.g.
                {"size": {"height": 100, "width": 100}}
        pool (`bool`, *optional*, defaults to `False`):
            Whether or not to return the pooled output. If `False`, the model will return the raw hidden states.
    
    - __call__
    - all

### ImageToTextPipeline


    Image To Text pipeline using a `AutoModelForVision2Seq`. This pipeline predicts a caption for a given image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
    >>> captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'generated_text': 'two birds are standing next to each other '}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image to text pipeline can currently be loaded from pipeline() using the following task identifier:
    "image-to-text".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-to-text).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### ImageTextToTextPipeline


    Image-text-to-text pipeline using an `AutoModelForImageTextToText`. This pipeline generates text given an image and text.
    When the underlying model is a conversational model, it can also accept one or more chats,
    in which case the pipeline will operate in chat mode and will continue the chat(s) by adding its response(s).
    Each chat takes the form of a list of dicts, where each dict contains "role" and "content" keys.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline(task="image-text-to-text", model="Salesforce/blip-image-captioning-base")
    >>> pipe("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", text="A photo of")
    [{'generated_text': 'a photo of two birds'}]
    ```

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
    >>> messages = [
    >>>     {
    >>>         "role": "user",
    >>>         "content": [
    >>>             {
    >>>                 "type": "image",
    >>>                 "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    >>>             },
    >>>             {"type": "text", "text": "Describe this image."},
    >>>         ],
    >>>     },
    >>>     {
    >>>         "role": "assistant",
    >>>         "content": [
    >>>             {"type": "text", "text": "There is a dog and"},
    >>>         ],
    >>>     },
    >>> ]
    >>> pipe(text=messages, max_new_tokens=20, return_full_text=False)
    [{'input_text': [{'role': 'user',
        'content': [{'type': 'image',
        'url': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
        {'type': 'text', 'text': 'Describe this image.'}]},
    {'role': 'assistant',
        'content': [{'type': 'text', 'text': 'There is a dog and'}]}],
    'generated_text': ' a person in the image. The dog is sitting on the sand, and the person is sitting on'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image-text to text pipeline can currently be loaded from pipeline() using the following task identifier:
    "image-text-to-text".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-text-to-text).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        processor ([`ProcessorMixin`]):
            The processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`ProcessorMixin`]. Processor is a composite object that might contain `tokenizer`, `feature_extractor`, and
            `image_processor`.
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

### MaskGenerationPipeline


    Automatic mask generation for images using `SamForMaskGeneration`. This pipeline predicts binary masks for an
    image, given an image. It is a `ChunkPipeline` because you can seperate the points in a mini-batch in order to
    avoid OOM issues. Use the `points_per_batch` argument to control the number of points that will be processed at the
    same time. Default is `64`.

    The pipeline works in 3 steps:
        1. `preprocess`: A grid of 1024 points evenly separated is generated along with bounding boxes and point
           labels.
            For more details on how the points and bounding boxes are created, check the `_generate_crop_boxes`
            function. The image is also preprocessed using the `image_processor`. This function `yields` a minibatch of
            `points_per_batch`.

        2. `forward`: feeds the outputs of `preprocess` to the model. The image embedding is computed only once.
            Calls both `self.model.get_image_embeddings` and makes sure that the gradients are not computed, and the
            tensors and models are on the same device.

        3. `postprocess`: The most important part of the automatic mask generation happens here. Three steps
            are induced:
                - image_processor.postprocess_masks (run on each minibatch loop): takes in the raw output masks,
                  resizes them according
                to the image size, and transforms there to binary masks.
                - image_processor.filter_masks (on each minibatch loop): uses both `pred_iou_thresh` and
                  `stability_scores`. Also
                applies a variety of filters based on non maximum suppression to remove bad masks.
                - image_processor.postprocess_masks_for_amg applies the NSM on the mask to only keep relevant ones.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="facebook/sam-vit-base", task="mask-generation")
    >>> outputs = generator(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ... )

    >>> outputs = generator(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", points_per_batch=128
    ... )
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"mask-generation"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=mask-generation).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
        points_per_batch (*optional*, int, default to 64):
            Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU
            memory.
        output_bboxes_mask (`bool`, *optional*, default to `False`):
            Whether or not to output the bounding box predictions.
        output_rle_masks (`bool`, *optional*, default to `False`):
            Whether or not to output the masks in `RLE` format
    - __call__
    - all

### VisualQuestionAnsweringPipeline


    Visual Question Answering pipeline using a `AutoModelForVisualQuestionAnswering`. This pipeline is currently only
    available in PyTorch.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="dandelin/vilt-b32-finetuned-vqa")
    >>> image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"
    >>> oracle(question="What is she wearing ?", image=image_url)
    [{'score': 0.948, 'answer': 'hat'}, {'score': 0.009, 'answer': 'fedora'}, {'score': 0.003, 'answer': 'clothes'}, {'score': 0.003, 'answer': 'sun hat'}, {'score': 0.002, 'answer': 'nothing'}]

    >>> oracle(question="What is she wearing ?", image=image_url, top_k=1)
    [{'score': 0.948, 'answer': 'hat'}]

    >>> oracle(question="Is this a person ?", image=image_url, top_k=1)
    [{'score': 0.993, 'answer': 'yes'}]

    >>> oracle(question="Is this a man ?", image=image_url, top_k=1)
    [{'score': 0.996, 'answer': 'no'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This visual question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifiers: `"visual-question-answering", "vqa"`.

    The models that this pipeline can use are models that have been fine-tuned on a visual question answering task. See
    the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=visual-question-answering).
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
    - __call__
    - all

## Parent class: `Pipeline`


    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance [`FeatureExtractionPipeline`] (`'feature-extraction'`) output large tensor object
    as nested-lists. In order to avoid dumping such large structure as textual data we provide the `binary_output`
    constructor argument. If set to `True`, the output will be stored in the pickle format.
    
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        feature_extractor ([`SequenceFeatureExtractor`]):
            The feature extractor that will be used by the pipeline to encode data for the model. This object inherits from
            [`SequenceFeatureExtractor`].
        image_processor ([`BaseImageProcessor`]):
            The image processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`BaseImageProcessor`].
        processor ([`ProcessorMixin`]):
            The processor that will be used by the pipeline to encode data for the model. This object inherits from
            [`ProcessorMixin`]. Processor is a composite object that might contain `tokenizer`, `feature_extractor`, and
            `image_processor`.
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as
            the raw output data e.g. text.
