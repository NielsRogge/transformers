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

# Models

The base classes [`PreTrainedModel`], [`TFPreTrainedModel`], and
[`FlaxPreTrainedModel`] implement the common methods for loading/saving a model either from a local
file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace's AWS
S3 repository).

[`PreTrainedModel`] and [`TFPreTrainedModel`] also implement a few methods which
are common among all the models to:

- resize the input token embeddings when new tokens are added to the vocabulary
- prune the attention heads of the model.

The other methods that are common to each model are defined in [`~modeling_utils.ModuleUtilsMixin`]
(for the PyTorch models) and [`~modeling_tf_utils.TFModuleUtilsMixin`] (for the TensorFlow models) or
for text generation, [`~generation.GenerationMixin`] (for the PyTorch models),
[`~generation.TFGenerationMixin`] (for the TensorFlow models) and
[`~generation.FlaxGenerationMixin`] (for the Flax/JAX models).


## PreTrainedModel


    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **load_tf_weights** (`Callable`) -- A python *method* for loading a TensorFlow checkpoint in a PyTorch model,
          taking as arguments:

            - **model** ([`PreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
            - **config** ([`PreTrainedConfig`]) -- An instance of the configuration associated to the model.
            - **path** (`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **is_parallelizable** (`bool`) -- A flag indicating whether this model supports model parallelization.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    
    - push_to_hub
    - all

Custom models should also include a `_supports_assign_param_buffer`, which determines if superfast init can apply
on the particular model. Signs that your model needs this are if `test_save_and_load_from_pretrained` fails. If so,
set this to `False`.

## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

## TFPreTrainedModel

Could not find docstring for TFPreTrainedModel
    - push_to_hub
    - all

## TFModelUtilsMixin

[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

## FlaxPreTrainedModel

Could not find docstring for FlaxPreTrainedModel
    - push_to_hub
    - all

## Pushing to the Hub

[[autodoc]] utils.PushToHubMixin

## Sharded checkpoints

[[autodoc]] modeling_utils.load_sharded_checkpoint
