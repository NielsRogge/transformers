<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Mamba 2

## Overview

The Mamba2 model was proposed in [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) by Tri Dao and Albert Gu. It is a State Space Model similar to Mamba 1, with better performances in a simplified architecture. 


The abstract from the paper is the following:

*While Transformers have been the main architecture behind deep learning's success in language modeling, state-space models (SSMs) such as Mamba have recently been shown to match or outperform Transformers at small to medium scale. We show that these families of models are actually quite closely related, and develop a rich framework of theoretical connections between SSMs and variants of attention, connected through various decompositions of a well-studied class of structured semiseparable matrices. Our state space duality (SSD) framework allows us to design a new architecture (Mamba-2) whose core layer is an a refinement of Mamba's selective SSM that is 2-8X faster, while continuing to be competitive with Transformers on language modeling.*

Tips:

This version should support all implementations of Mamba 2, and in particular [Mamba-2 codestral](https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1) from Mistral AI. In particular, mamba 2 codestral was released with a number of `groups` equal to 8, which can be thought intuitively as similar to the number of kv heads in an attention-based model. 
This model has two different forward passes, `torch_forward` or `cuda_kernels_forward`. The latter uses the original cuda kernels if they are found in your environment, and is slower on the prefill i.e. requires a "warmup run" due to high cpu overhead, see [here](https://github.com/state-spaces/mamba/issues/389#issuecomment-2171755306) and [also here](https://github.com/state-spaces/mamba/issues/355#issuecomment-2147597457). Without compilation, the `torch_forward` implementation is faster by a factor 3 to 4. Further, there are no positional embeddings in this model, but there is an `attention_mask` and a specific logic to mask out hidden states in two places in the case of batched generation, see [here](https://github.com/state-spaces/mamba/issues/66#issuecomment-1863563829) as well. Due to this, in addition to the reimplementation of mamba2 kernels, batched generation and cached generation are expected to have slight discrepancies. Further, the results given by the cuda kernels or the torch forward are expected to be slightly different. The SSM algorithm heavily relies on tensor contractions, which have matmul equivalents but the order of operations is slightly different, making the difference greater at smaller precisions. 
Another note, shutdown of hidden states corresponding to padding tokens is done in 2 places and mostly has been tested with left-padding. Right-padding will propagate noise down the line and is not guaranteed to yield satisfactory results. `tokenizer.padding_side = "left"` ensures you are using the correct padding side.

This model was contributed by [Molbap](https://huggingface.co/Molbap), with tremendous help from [Anton Vlasjuk](https://github.com/vasqu).
The original code can be found [here](https://github.com/state-spaces/mamba).


# Usage

### A simple generation example: 
```python 
from transformers import Mamba2Config, Mamba2ForCausalLM, AutoTokenizer
import torch
model_id = 'mistralai/Mamba-Codestral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_id, revision='refs/pr/9', from_slow=True, legacy=False)
model = Mamba2ForCausalLM.from_pretrained(model_id, revision='refs/pr/9')
input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

Here's a draft script for finetuning: 
```python 
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, Mamba2ForCausalLM, TrainingArguments
model_id = 'mistralai/Mamba-Codestral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_id, revision='refs/pr/9', from_slow=True, legacy=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" #enforce padding side left

model = Mamba2ForCausalLM.from_pretrained(model_id, revision='refs/pr/9')
dataset = load_dataset("Abirate/english_quotes", split="train")
# Without CUDA kernels, batch size of 2 occupies one 80GB device
# but precision can be reduced.
# Experiments and trials welcome!
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)
lora_config =  LoraConfig(
        r=8,
        target_modules=["embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="quote",
)
trainer.train()
```


## Mamba2Config


    This is the configuration class to store the configuration of a [`Mamba2Model`]. It is used to instantiate a MAMBA2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MAMBA2
    [state-spaces/mamba2-2.8b](https://huggingface.co/state-spaces/mamba2-2.8b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_heads (`int`, *optional*, defaults to 128):
            Number of heads for the evolution matrices of mamba 2.
        head_dim (`int`, *optional*, defaults to 64):
            Dimension of each head.
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the MAMBA2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Mamba2Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        state_size (`int`, *optional*, defaults to 128): shape of the state space latents.
        num_hidden_layers (`int`, *optional*, defaults to 64):
            Number of hidden layers in the model.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end of sentence token in the vocabulary.
        expand (`int`, *optional*, defaults to 2): Expanding factor used to determine the intermediate size.
        conv_kernel (`int`, *optional*, defaults to 4): Size of the convolution kernel.
        n_groups (`int`, *optional*, defaults to 8):
            Number of groups for the evolution matrices of mamba 2.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the mixer block.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.1):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
        time_step_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj.bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        time_step_limit (`tuple`, *optional*, defaults to `(0.0, inf)`):
            Accepted range of time step values.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether or not to rescale `out_proj` weights when initializing.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the cache should be used.
        rms_norm (`bool`, *optional*, defaults to `True`):
            Whether to use RMS norm or not.
        chunk_size (`int`, *optional*, defaults to 256):
            Size of the chunks that will comprise the sequence.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embeddings or not.


    Example:

    ```python
    >>> from transformers import Mamba2Config, Mamba2Model

    >>> # Initializing a Mamba2 configuration
    >>> configuration = Mamba2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Mamba2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## Mamba2Model

The bare MAMBA2 Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Mamba2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward

## Mamba2LMHeadModel


    The MAMBA2 Model transformer with a language modeling head on top (linear layer with weights not tied to the input
    embeddings).
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Mamba2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
