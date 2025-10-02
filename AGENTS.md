# AGENTS.md Guide for Hugging Face Transformers

This AGENTS.md file provides guidance for code agents working with this codebase.

## Environment setup

When contributing to the Transformers library, prefer [`uv`](https://docs.astral.sh/uv/guides/install-python/) for Python package management since it is much faster than `pip`. First install it as explained in the [installation guide](https://docs.astral.sh/uv/getting-started/installation/), e.g. using pip:

```bash
pip install uv
```

Next, let `uv` create a virtual environment and install an editable version the Transformers library within it as follows:

```bash
uv venv
uv pip install -e ".[dev-torch]"
```

This command installes an editable (`-e`) version of the Transformers library along with required `dev` dependencies which includes PyTorch. 
Make sure to always work within the virtual environment in order to use the necessary dependencies like `ruff` and `pytest`.

Find more details at `docs/source/en/contributing.md`.

## Core Project Structure

- `/src/transformers`: This contains the core source code for the library
  - `/models`: Code for individual models. Models inherit from base classes in the root `/src/transformers` directory.
- `/tests`: This contains the core test classes for the library. These are usually inherited rather than directly run.
  - `/models`: Tests for individual models. Model tests inherit from common tests in the root `/tests` directory.
- `/docs`: This contains the documentation for the library, including guides, tutorials, and API references.

## Copying and inheritance

Many models in the codebase share similar code, but the philosophy of Transformers is that each model implementation should be self-contained and independent. Each model is implemented in a standalone `modeling_xxx.py` file which does not rely on inheritance. This allows people to easily debug a single model implementation without having to traverse many files.

However, to make it easier for contributors to add new models, the "modular" system was introduced. Modular allows contributors to use inheritance by implementing a `modular_xxx.py` file. These files are not meant to be used directly. Instead, style tools like `make fix-copies` and `make fixup` automatically generate a complete standalone modeling file, like `modeling_bert.py`, from the modular file like `modular_bert.py`. If a model has a modular file, the modeling file should never be edited directly! Instead, changes should be made in the modular file, and then you should run `make fixup` to update the modeling file automatically. When adding new models, you should prefer the `modular` style. Find a complete guide on modular at docs/source/en/modular_transformers.md.

Besides that, there's the "Copied from" syntax. Functions or entire classes can have a comment at the top like this: `# Copied from transformers.models.llama.modeling_llama.rotate_half` or `# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->MT5`. These comments are actively checked by the style tools, and copies will automatically be updated when the base code is updated. If you need to update a copied function, you should either update the base function and use `make fixup` to propagate the change to all copies, or simply remove the `# Copied from` comment if that is inappropriate.

## Preprocessor classes

Each model in the Transformers library has one or more corresponding preprocessor classes which allow to preprocess inputs (such as text, images or video) for the model. Currently, the following preprocessor classes are available:

- tokenizers: used to convert text into so-called `input_ids`. See `docs/source/en/main_classes/tokenizer.md` for details.
- image processors: used to convert images into so-called `pixel_values`. See `docs/source/en/main_classes/image_processor.md` for details.
- video processors: used to convert videos into similarly called `pixel_values`. See `docs/source/en/main_classes/video_processor.md` for details.
- feature extractors: used to convert audio into so-called `input_features`. See `docs/source/en/main_classes/feature_extractor.md` for details.
- processors: used to combine several processor classes for multimodal models. For example, the Qwen2-VL is a vision-language model, hence it requires both a tokenizer and an image processor. The `Qwen2VLProcessor` class combines both processor classes into one. See `docs/source/en/processors.md` for details.

Tokenizers and image processors both come in two flavors, a slow and a fast one. 
- fast tokenizers use a Rust backend from the `tokenizers` library.
- fast image processors use the `torchvision` backend.

Typically one prefers the fast implementation.

## Auto mappings

Oftentimes, a new model can simply reuse an existing preprocessor class, like a tokenizer, image processor or multimodal processor. In that case, there's no need for the new model to duplicate that logic, but instead simply add a mapping to the corresponding "auto" file.

For example, when Qwen3-VL came out, one simply added the line `("qwen3_vl", ("Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast")),` to `src/transformers/models/auto/image_processing_auto.py` so that people can use the `AutoImageProcessor` class which will load an instance of `Qwen2VLImageProcessorFast` behind the scenes (since Qwen3-VL reuses the same image processor as Qwen2-VL).

In case the model introduces some new logic which is not yet present in any of the existing preprocessors, one typically adds a new one. For example, if a text model introduces a new way of tokenization, one would add a new tokenizer at `src/transformers/models/[name]/tokenization_[name]_fast.py`.

Additionally, in case a model class follows the same API as other classes, one also adds it to the corresponding auto mapping at `src/transformers/models/auto/modeling_auto.py`. For example, the Qwen2-VL and Qwen3-VL classes can both be loaded using the `AutoModelForImageTextToText` class.

## Coding Conventions for Hugging Face Transformers

- PRs should be as brief as possible. Bugfix PRs in particular can often be only one or two lines long, and do not need large comments, docstrings or new functions in this case. Aim to minimize the size of the diff.
- When writing tests, they should be added to an existing file. The only exception is for PRs to add a new model, when a new test directory should be created for that model.
- Code style is enforced in the CI. You can install the style tools with `pip install -e .[quality]`. You can then run `make fixup` to apply style and consistency fixes to your code.

## Transformers-cli add-new-model-like

When adding a new model, make sure to run the `add-new-model-like` command which requires you to fill in some variables which will then bootstrap a template for you to implement:

```bash
transformers add-new-model-like
```

This command is implemented at `src/transformers/commands/add_new_model_like.py`.

## Conversion script

When converting checkpoints of a given model from the original Github repository to the Transformers API, one typically implements a so-called conversion script at `src/transformers/models/[name]/convert_name_to_hf.py`. This script not only converts the weights by remapping the keys and values of the state dictionary, it also verifies whether the outputs of the Transformers model are exactly the same as the original implementation on the same dummy inputs. For example, the `src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py` conversion script was used to convert the weights of DINOv3, a vision model by Meta, from the original implementation to the Transformers format.

Conversion of weights is typically done by forwarding the same dummy input (e.g. a cats image in case of a vision model) through both the original implementation and the Transformers implementation, and then verifying the outputs at each layer of the neural network in a bottom-up fashion, starting with the embedding layer, then the position embedding layer, and so on. One can use print statements or `torch.testing.assert_close` to verify the values. A conversion is successful in case the outputs of the model are exactly the same between both implementations (up to a certain absolute tolerance or `atol`).

The script should allow for flexibility to convert each of the released checkpoints by providing a `--model_name` or `--model_id` flag. Additionally, an option to push the converted weights to the hub is provided via a `--push_to_hub` flag.

## Testing

Tests live in the `tests` folder. In order to run tests, you may need to install dependencies. You can do this with `pip install -e .[testing]`. You will probably also need to `pip install torch accelerate` if your environment does not already have them.

Tests for models live in the `tests/models` folder. Each model has its own test files implemented. Some models only have a test file for their model implementation, e.g. `tests/models/[name]/test_modeling_[name].py`, others also have test files for their preprocessors, e.g. `test_processing_[name].py` or `test_tokenization_[name].py`. Models which don't have their preprocessors tested are models which simply re-use existing preprocessors from other models. This is defined in the `src/transformers/models/auto` files.

### Running tests

In case you want to run a single test file, you can use the following command:

```bash
pytest tests/models/[name]/test_modeling_[name].py
```

In case you want to run all tests of a given model, you can run the following command:

```bash
pytest tests/models/[name]
```

In case your changes affect code in other classes like tokenizers or processors, you should run those tests instead, like `test_processing_[name].py` or `test_tokenization_[name].py`.

To run an individual test method, you can use the following command (example given for BERT):

```bash
pytest tests/models/bert/test_modeling_bert.py::BertModelTest::test_model_as_decoder_with_3d_input_mask
```

### Slow tests

Some tests take a long time to run, for example when they involve loading pre-trained weights from the hub. In that case, they have the @slow annotator. Integration tests are examples of slow tests, as they verify the outputs of the model on some dummy or fixed inputs. Slow tests can be run using the `RUN_SLOW=1` environment variable, for example:

```bash
RUN_SLOW=yes pytest tests/models/bert/test_modeling_bert.py::BertModelIntegrationTest
```

After making changes, you should usually run `make fixup` to ensure any copies and modular files are updated, and then test all affected models. This includes both
the model you made the changes in and any other models that were updated by `make fixup`. 