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

# Callbacks

Callbacks are objects that can customize the behavior of the training loop in the PyTorch
[`Trainer`] (this feature is not yet implemented in TensorFlow) that can inspect the training loop
state (for progress reporting, logging on TensorBoard or other ML platforms...) and take decisions (like early
stopping).

Callbacks are "read only" pieces of code, apart from the [`TrainerControl`] object they return, they
cannot change anything in the training loop. For customizations that require changes in the training loop, you should
subclass [`Trainer`] and override the methods you need (see [trainer](trainer) for examples).

By default, `TrainingArguments.report_to` is set to `"all"`, so a [`Trainer`] will use the following callbacks.

- [`DefaultFlowCallback`] which handles the default behavior for logging, saving and evaluation.
- [`PrinterCallback`] or [`ProgressCallback`] to display progress and print the
  logs (the first one is used if you deactivate tqdm through the [`TrainingArguments`], otherwise
  it's the second one).
- [`~integrations.TensorBoardCallback`] if tensorboard is accessible (either through PyTorch >= 1.4
  or tensorboardX).
- [`~integrations.WandbCallback`] if [wandb](https://www.wandb.com/) is installed.
- [`~integrations.CometCallback`] if [comet_ml](https://www.comet.com/site/) is installed.
- [`~integrations.MLflowCallback`] if [mlflow](https://www.mlflow.org/) is installed.
- [`~integrations.NeptuneCallback`] if [neptune](https://neptune.ai/) is installed.
- [`~integrations.AzureMLCallback`] if [azureml-sdk](https://pypi.org/project/azureml-sdk/) is
  installed.
- [`~integrations.CodeCarbonCallback`] if [codecarbon](https://pypi.org/project/codecarbon/) is
  installed.
- [`~integrations.ClearMLCallback`] if [clearml](https://github.com/allegroai/clearml) is installed.
- [`~integrations.DagsHubCallback`] if [dagshub](https://dagshub.com/) is installed.
- [`~integrations.FlyteCallback`] if [flyte](https://flyte.org/) is installed.
- [`~integrations.DVCLiveCallback`] if [dvclive](https://dvc.org/doc/dvclive) is installed.

If a package is installed but you don't wish to use the accompanying integration, you can change `TrainingArguments.report_to` to a list of just those integrations you want to use (e.g. `["azure_ml", "wandb"]`). 

The main class that implements callbacks is [`TrainerCallback`]. It gets the
[`TrainingArguments`] used to instantiate the [`Trainer`], can access that
Trainer's internal state via [`TrainerState`], and can take some actions on the training loop via
[`TrainerControl`].


## Available Callbacks

Here is the list of the available [`TrainerCallback`] in the library:

[[autodoc]] integrations.CometCallback
    - setup


    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    


    A bare [`TrainerCallback`] that just prints the logs.
    


    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    


    A [`TrainerCallback`] that handles early stopping.

    Args:
        early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
        early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    

[[autodoc]] integrations.TensorBoardCallback

[[autodoc]] integrations.WandbCallback
    - setup

[[autodoc]] integrations.MLflowCallback
    - setup

[[autodoc]] integrations.AzureMLCallback

[[autodoc]] integrations.CodeCarbonCallback

[[autodoc]] integrations.NeptuneCallback

[[autodoc]] integrations.ClearMLCallback

[[autodoc]] integrations.DagsHubCallback

[[autodoc]] integrations.FlyteCallback

[[autodoc]] integrations.DVCLiveCallback
    - setup

## TrainerCallback


    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:

    Args:
        args ([`TrainingArguments`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some decisions.
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            The model being trained.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for encoding the data. This is deprecated in favour of `processing_class`.
        processing_class ([`PreTrainedTokenizer` or `BaseImageProcessor` or `ProcessorMixin` or `FeatureExtractionMixin`]):
            The processing class used for encoding the data. Can be a tokenizer, a processor, an image processor or a feature extractor.
        optimizer (`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for training.
        eval_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for evaluation.
        metrics (`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event `on_evaluate`.
        logs  (`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event `on_log`.

    The `control` object is the only one that can be changed by the callback, in which case the event that changes it
    should return the modified version.

    The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
    simple [`~transformers.PrinterCallback`].

    Example:

    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)
    ```

Here is an example of how to register a custom callback with the PyTorch [`Trainer`]:

```python
class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MyCallback],  # We can either pass the callback class this way or an instance of it (MyCallback())
)
```

Another way to register a callback is to call `trainer.add_callback()` as follows:

```python
trainer = Trainer(...)
trainer.add_callback(MyCallback)
# Alternatively, we can pass an instance of the callback class
trainer.add_callback(MyCallback())
```

## TrainerState


    A class containing the [`Trainer`] inner state that will be saved along the model and optimizer when checkpointing
    and passed to the [`TrainerCallback`].

    <Tip>

    In all this class, one step is to be understood as one update step. When using gradient accumulation, one update
    step may require several forward and backward passes: if you use `gradient_accumulation_steps=n`, then one update
    step requires going through *n* batches.

    </Tip>

    Args:
        epoch (`float`, *optional*):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        global_step (`int`, *optional*, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (`int`, *optional*, defaults to 0):
            The number of update steps to do during the current training.
        logging_steps (`int`, *optional*, defaults to 500):
            Log every X updates steps
        eval_steps (`int`, *optional*):
            Run an evaluation every X steps.
        save_steps (`int`, *optional*, defaults to 500):
            Save checkpoint every X updates steps.
        train_batch_size (`int`, *optional*):
            The batch size for the training dataloader. Only needed when
            `auto_find_batch_size` has been used.
        num_input_tokens_seen (`int`, *optional*, defaults to 0):
            When tracking the inputs tokens, the number of tokens seen during training (number of input tokens, not the
            number of prediction tokens).
        total_flos (`float`, *optional*, defaults to 0):
            The total number of floating operations done by the model since the beginning of training (stored as floats
            to avoid overflow).
        log_history (`List[Dict[str, float]]`, *optional*):
            The list of logs done since the beginning of training.
        best_metric (`float`, *optional*):
            When tracking the best model, the value of the best metric encountered so far.
        best_model_checkpoint (`str`, *optional*):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_local_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
            several machines) main process.
        is_world_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the global main process (when training in a distributed fashion on several
            machines, this is only going to be `True` for one process).
        is_hyper_param_search (`bool`, *optional*, defaults to `False`):
            Whether we are in the process of a hyper parameter search using Trainer.hyperparameter_search. This will
            impact the way data will be logged in TensorBoard.
        stateful_callbacks (`List[StatefulTrainerCallback]`, *optional*):
            Callbacks attached to the `Trainer` that should have their states be saved or restored.
            Relevent callbacks should implement a `state` and `from_state` function.
    

## TrainerControl


    A class that handles the [`Trainer`] control flow. This class is used by the [`TrainerCallback`] to activate some
    switches in the training loop.

    Args:
        should_training_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the training should be interrupted.

            If `True`, this variable will not be set back to `False`. The training will just stop.
        should_epoch_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the current epoch should be interrupted.

            If `True`, this variable will be set back to `False` at the beginning of the next epoch.
        should_save (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be saved at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_evaluate (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be evaluated at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_log (`bool`, *optional*, defaults to `False`):
            Whether or not the logs should be reported at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
    
