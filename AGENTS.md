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
uv pip install -e ".[dev]"
```

Make sure to always work within the virtual environment in order to use the necessary dependencies like `ruff` and `pytest`.

Find more details at `docs/source/en/contributing.md`.

## Core Project Structure

- `/src/transformers`: This contains the core source code for the library
  - `/models`: Code for individual models. Models inherit from base classes in the root `/src/transformers` directory.
- `/tests`: This contains the core test classes for the library. These are usually inherited rather than directly run.
  - `/models`: Tests for individual models. Model tests inherit from common tests in the root `/tests` directory.
- `/docs`: This contains the documentation for the library, including guides, tutorials, and API references.

## Coding Conventions for Hugging Face Transformers

- PRs should be as brief as possible. Bugfix PRs in particular can often be only one or two lines long, and do not need large comments, docstrings or new functions in this case. Aim to minimize the size of the diff.
- When writing tests, they should be added to an existing file. The only exception is for PRs to add a new model, when a new test directory should be created for that model.
- Code style is enforced in the CI. You can install the style tools with `pip install -e .[quality]`. You can then run `make fixup` to apply style and consistency fixes to your code.

## Copying and inheritance

Many models in the codebase share similar code, but the philosophy of Transformers is that each model implementation should be self-contained and independent. Each model is implemented in a standalone `modeling_xxx.py` file which does not rely on inheritance. This allows people to easily debug a single model implementation without having to traverse many files.

However, to make it easier for contributors to add new models, the "modular" system was introduced. Modular allows contributors to use inheritance by implementing a `modular_xxx.py` file. These files are not meant to be used directly. Instead, style tools like `make fix-copies` and `make fixup` automatically generate a complete standalone modeling file, like `modeling_bert.py`, from the modular file like `modular_bert.py`. If a model has a modular file, the modeling file should never be edited directly! Instead, changes should be made in the modular file, and then you should run `make fixup` to update the modeling file automatically. When adding new models, you should prefer the `modular` style. Find a complete guide on modular at docs/source/en/modular_transformers.md.

Besides that, there's the "Copied from" syntax. Functions or entire classes can have a comment at the top like this: `# Copied from transformers.models.llama.modeling_llama.rotate_half` or `# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->MT5`. These comments are actively checked by the style tools, and copies will automatically be updated when the base code is updated. If you need to update a copied function, you should either update the base function and use `make fixup` to propagate the change to all copies, or simply remove the `# Copied from` comment if that is inappropriate.
- "Modular" files. These files briefly define models by composing them using inheritance from other models. They are not meant to be used directly. 

## Testing

After making changes, you should usually run `make fixup` to ensure any copies and modular files are updated, and then test all affected models. This includes both
the model you made the changes in and any other models that were updated by `make fixup`. Tests can be run with `pytest tests/models/[name]/test_modeling_[name].py`
If your changes affect code in other classes like tokenizers or processors, you should run those tests instead, like `test_processing_[name].py` or `test_tokenization_[name].py`.

In order to run tests, you may need to install dependencies. You can do this with `pip install -e .[testing]`. You will probably also need to `pip install torch accelerate` if your environment does not already have them.