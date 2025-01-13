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

# Utilities for Generation

This page lists all the utility functions used by [`~generation.GenerationMixin.generate`].

## Generate Outputs

The output of [`~generation.GenerationMixin.generate`] is an instance of a subclass of
[`~utils.ModelOutput`]. This output is a data structure containing all the information returned
by [`~generation.GenerationMixin.generate`], but that can also be used as tuple or dictionary.

Here's an example:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

The `generation_output` object is a [`~generation.GenerateDecoderOnlyOutput`], as we can
see in the documentation of that class below, it means it has the following attributes:

- `sequences`: the generated sequences of tokens
- `scores` (optional): the prediction scores of the language modelling head, for each generation step
- `hidden_states` (optional): the hidden states of the model, for each generation step
- `attentions` (optional): the attention weights of the model, for each generation step

Here we have the `scores` since we passed along `output_scores=True`, but we don't have `hidden_states` and
`attentions` because we didn't pass `output_hidden_states=True` or `output_attentions=True`.

You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you
will get `None`. Here for instance `generation_output.scores` are all the generated prediction scores of the
language modeling head, and `generation_output.attentions` is `None`.

When using our `generation_output` object as a tuple, it only keeps the attributes that don't have `None` values.
Here, for instance, it has two elements, `loss` then `logits`, so

```python
generation_output[:2]
```

will return the tuple `(generation_output.sequences, generation_output.scores)` for instance.

When using our `generation_output` object as a dictionary, it only keeps the attributes that don't have `None`
values. Here, for instance, it has two keys that are `sequences` and `scores`.

We document here all output types.


### PyTorch

[[autodoc]] generation.GenerateDecoderOnlyOutput

[[autodoc]] generation.GenerateEncoderDecoderOutput

[[autodoc]] generation.GenerateBeamDecoderOnlyOutput

[[autodoc]] generation.GenerateBeamEncoderDecoderOutput

### TensorFlow

[[autodoc]] generation.TFGreedySearchEncoderDecoderOutput

[[autodoc]] generation.TFGreedySearchDecoderOnlyOutput

[[autodoc]] generation.TFSampleEncoderDecoderOutput

[[autodoc]] generation.TFSampleDecoderOnlyOutput

[[autodoc]] generation.TFBeamSearchEncoderDecoderOutput

[[autodoc]] generation.TFBeamSearchDecoderOnlyOutput

[[autodoc]] generation.TFBeamSampleEncoderDecoderOutput

[[autodoc]] generation.TFBeamSampleDecoderOnlyOutput

[[autodoc]] generation.TFContrastiveSearchEncoderDecoderOutput

[[autodoc]] generation.TFContrastiveSearchDecoderOnlyOutput

### FLAX

[[autodoc]] generation.FlaxSampleOutput

[[autodoc]] generation.FlaxGreedySearchOutput

[[autodoc]] generation.FlaxBeamSearchOutput

## LogitsProcessor

A [`LogitsProcessor`] can be used to modify the prediction scores of a language model head for
generation.

### PyTorch


    [`LogitsProcessor`] enforcing alternated generation between the two codebooks of Bark.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [Bark](https://huggingface.co/docs/transformers/en/model_doc/bark)'s fine submodel. See the model documentation
    for examples.

    </Tip>

    Args:
        input_start_len (`int`):
            The length of the initial input sequence.
        semantic_vocab_size (`int`):
            Vocabulary size of the semantic part, i.e number of tokens associated to the semantic vocabulary.
        codebook_size (`int`):
            Number of tokens associated to the codebook.
    
    - __call__


    [`LogitsProcessor`] for classifier free guidance (CFG). The scores are split over the batch dimension,
    where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
    correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
    weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

    See [the paper](https://arxiv.org/abs/2306.05284) for more information.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen)

    </Tip>

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.

    Examples:

    ```python
    >>> from transformers import AutoProcessor, MusicgenForConditionalGeneration

    >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    >>> inputs = processor(
    ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    ...     padding=True,
    ...     return_tensors="pt",
    ... )
    >>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
    ```
    
    - __call__


    [`LogitsProcessor`] that works similarly to [`NoRepeatNGramLogitsProcessor`], but applied exclusively to prevent
    the repetition of n-grams present in the prompt.

    It was designed to promote chattiness in a language model, by preventing the generation of n-grams present in
    previous conversation rounds.

    Args:
        encoder_ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur within the encoder input ids.
        encoder_input_ids (`int`):
            The encoder_input_ids that should not be repeated within the decoder ids.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("Alice: I love cats. What do you love?\nBob:", return_tensors="pt")

    >>> # With greedy decoding, we see Bob repeating Alice's opinion. If Bob was a chatbot, it would be a poor one.
    >>> outputs = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice: I love cats. What do you love?
    Bob: I love cats. What do you

    >>> # With this logits processor, we can prevent Bob from repeating Alice's opinion.
    >>> outputs = model.generate(**inputs, encoder_no_repeat_ngram_size=2)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice: I love cats. What do you love?
    Bob: My cats are very cute.
    ```
    
    - __call__


    [`LogitsProcessor`] that works similarly to [`RepetitionPenaltyLogitsProcessor`], but with an *inverse* penalty
    that is applied to the tokens present in the prompt. In other words, a penalty above 1.0 increases the odds of
    selecting tokens that were present in the prompt.

    It was designed to avoid hallucination in input-grounded tasks, like summarization. Although originally intended
    for encoder-decoder models, it can also be used with decoder-only models like LLMs.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 rewards prompt tokens. Between 0.0
            and 1.0 penalizes prompt tokens.
        encoder_input_ids (`torch.LongTensor`):
            The encoder_input_ids that should be repeated within the decoder ids.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer(["Alice and Bob. The third member's name was"], return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    Alice and Bob. The third member's name was not mentioned.

    >>> # With the `encoder_repetition_penalty` argument we can trigger this logits processor in `generate`, which can
    >>> # promote the use of prompt tokens ("Bob" in this example)
    >>> gen_out = model.generate(**inputs, encoder_repetition_penalty=1.2)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    Alice and Bob. The third member's name was Bob. The third member's name was Bob.
    ```
    
    - __call__


    [`LogitsProcessor`] that performs epsilon-sampling, i.e. restricting to tokens with `prob >= epsilon`. Takes the
    largest min_tokens_to_keep tokens if no tokens satisfy this constraint. See [Truncation Sampling as Language Model
    Desmoothing](https://arxiv.org/abs/2210.15191) for more information.

    Args:
        epsilon (`float`):
            If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With epsilon sampling, the output gets restricted to high-probability tokens. Note that this is similar to
    >>> # Top P sampling, which restricts tokens based on their cumulative probability.
    >>> # Pro tip: The paper recomends using `epsilon_cutoff` values between 3e-4 and 9e-4
    >>> outputs = model.generate(**inputs, do_sample=True, epsilon_cutoff=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    
    - __call__


    [`LogitsProcessor`] that performs eta-sampling, a technique to filter out tokens with probabilities below a dynamic
    cutoff value, `eta`, which is calculated based on a combination of the hyperparameter `epsilon` and the entropy of
    the token probabilities, i.e. `eta := min(epsilon, sqrt(epsilon * e^-entropy(probabilities)))`. Takes the largest
    min_tokens_to_keep tokens if no tokens satisfy this constraint. It addresses the issue of poor quality in long
    samples of text generated by neural language models leading to more coherent and fluent text. See [Truncation
    Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more information. Note: `do_sample`
    must be set to `True` for this `LogitsProcessor` to work.


    Args:
        epsilon (`float`):
            A float value in the range (0, 1). Hyperparameter used to calculate the dynamic cutoff value, `eta`. The
            suggested values from the paper ranges from 3e-4 to 4e-3 depending on the size of the model.
        filter_value (`float`, *optional*, defaults to -inf):
            All values that are found to be below the dynamic cutoff value, `eta`, are set to this float value. This
            parameter is useful when logits need to be modified for very low probability tokens that should be excluded
            from generation entirely.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Specifies the minimum number of tokens that must be kept for generation, regardless of their probabilities.
            For example, if `min_tokens_to_keep` is set to 1, at least one token will always be kept for generation,
            even if all tokens have probabilities below the cutoff `eta`.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.

    Examples:
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With eta sampling, the output gets restricted to high-probability tokens. You can see it as a dynamic form of
    >>> # epsilon sampling that adapts its cutoff probability based on the entropy (high entropy = lower cutoff).
    >>> # Pro tip: The paper recomends using `eta_cutoff` values between 3e-4 to 4e-3
    >>> outputs = model.generate(**inputs, do_sample=True, eta_cutoff=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    
    - __call__


    [`LogitsProcessor`] that exponentially increases the score of the `eos_token_id` after `start_index` has been
    reached. This allows generating shorter sequences without having a hard cutoff, allowing the `eos_token` to be
    predicted in a meaningful position.

    Args:
        exponential_decay_length_penalty (`tuple(int, float)`):
            This tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty
            starts and `decay_factor` represents the factor of exponential decay
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        input_ids_seq_length (`int`):
            The length of the input sequence.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    >>> text = "Just wanted to let you know, I"
    >>> inputs = tokenizer(text, return_tensors="pt")

    >>> # Let's consider that we want short sentences, so we limit `max_length=30`. However, we observe that the answer
    >>> # tends to end abruptly.
    >>> set_seed(1)
    >>> outputs = model.generate(**inputs, do_sample=True, temperature=0.9, max_length=30, pad_token_id=50256)
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which was
    published in 2010. Although

    >>> # To promote the appearance of the EOS token at the right time, we add the `exponential_decay_length_penalty =
    >>> # (start_index, decay_factor)`. Instead of cutting at max_tokens, the output comes to an end before and usually
    >>> # with more meaning. What happens is that starting from `start_index` the EOS token score will be increased
    >>> # by `decay_factor` exponentially. However, if you set a high decay factor, you may also end up with abruptly
    >>> # ending sequences.
    >>> set_seed(1)
    >>> outputs = model.generate(
    ...     **inputs,
    ...     do_sample=True,
    ...     temperature=0.9,
    ...     max_length=30,
    ...     pad_token_id=50256,
    ...     exponential_decay_length_penalty=(15, 1.6),
    ... )
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network
    which<|endoftext|>

    >>> # With a small decay factor, you will have a higher chance of getting a meaningful sequence.
    >>> set_seed(1)
    >>> outputs = model.generate(
    ...     **inputs,
    ...     do_sample=True,
    ...     temperature=0.9,
    ...     max_length=30,
    ...     pad_token_id=50256,
    ...     exponential_decay_length_penalty=(15, 1.01),
    ... )
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which was
    published in 2010.<|endoftext|>
    ```
    
    - __call__


    [`LogitsProcessor`] that enforces the specified token as the first generated token. Used with encoder-decoder
    models.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    >>> inputs = tokenizer("Translate from English to German: I love cats.", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=10)
    >>> print(tokenizer.batch_decode(outputs)[0])
    <pad> Ich liebe Kitty.</s>

    >>> # We can use `forced_bos_token_id` to force the start of generation with an encoder-decoder model
    >>> # (including forcing it to end straight away with an EOS token)
    >>> outputs = model.generate(**inputs, max_new_tokens=10, forced_bos_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(outputs)[0])
    <pad></s>
    ```
    
    - __call__


    [`LogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2, 3", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=10)
    >>> print(tokenizer.batch_decode(outputs)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8

    >>> # `forced_eos_token_id` ensures the generation ends with a EOS token
    >>> outputs = model.generate(**inputs, max_new_tokens=10, forced_eos_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(outputs)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7,<|endoftext|>
    ```
    
    - __call__


    [`LogitsProcessor`] that enforces diverse beam search.

    Note that this logits processor is only effective for [`PreTrainedModel.group_beam_search`]. See [Diverse Beam
    Search: Decoding Diverse Solutions from Neural Sequence Models](https://arxiv.org/pdf/1610.02424.pdf) for more
    details.

    Traditional beam search often generates very similar sequences across different beams.
    `HammingDiversityLogitsProcessor` addresses this by penalizing beams that generate tokens already chosen by other
    beams in the same time step.

    Args:
        diversity_penalty (`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. A higher `diversity_penalty` will enforce greater diversity among the beams. Adjusting
            this value can help strike a balance between diversity and natural likelihood.
        num_beams (`int`):
            Number of beams for beam search. 1 means no beam search.
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    >>> import torch

    >>> # Initialize the model and tokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

    >>> # A long text about the solar system
    >>> text = (
    ...     "The Solar System is a gravitationally bound system comprising the Sun and the objects that orbit it, "
    ...     "either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight "
    ...     "planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System "
    ...     "bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant "
    ...     "interstellar molecular cloud."
    ... )
    >>> inputs = tokenizer("summarize: " + text, return_tensors="pt")

    >>> # Generate diverse summary
    >>> outputs_diverse = model.generate(
    ...     **inputs,
    ...     num_beam_groups=2,
    ...     diversity_penalty=10.0,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summaries_diverse = tokenizer.batch_decode(outputs_diverse, skip_special_tokens=True)

    >>> # Generate non-diverse summary
    >>> outputs_non_diverse = model.generate(
    ...     **inputs,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summary_non_diverse = tokenizer.batch_decode(outputs_non_diverse, skip_special_tokens=True)

    >>> # With `diversity_penalty`, the resulting beams are much more diverse
    >>> print(summary_non_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the Solar System formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.']

    >>> print(summaries_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets. the rest of the objects are smaller objects, such as the five dwarf planets and small solar system bodies.']
    ```
    
    - __call__


    [`LogitsProcessor`] that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
    the logits processor should only be used if necessary since it can slow down the generation method.

    This logits processor has no `generate` example, as there shouldn't be a correct combination of flags that warrants
    its use.
    
    - __call__


    [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> import torch

    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2, 3", return_tensors="pt")

    >>> # By default, the scores are not normalized -- the sum of their exponentials is NOT a normalized probability
    >>> # distribution, summing to 1
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(torch.allclose(torch.sum(torch.exp(outputs.scores[-1])), torch.Tensor((1.000,)), rtol=1e-4))
    False

    >>> # Normalizing them may have a positive impact on beam methods, or when using the scores on your application
    >>> outputs = model.generate(**inputs, renormalize_logits=True, return_dict_in_generate=True, output_scores=True)
    >>> print(torch.allclose(torch.sum(torch.exp(outputs.scores[-1])), torch.Tensor((1.000,)), rtol=1e-4))
    True
    ```
    
    - __call__

Abstract base class for all logit processors that can be applied during generation.
    - __call__

Abstract base class for all logit processors that can be applied during generation.List
    - __call__


    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0. Note that, for decoder-only models
    like most LLMs, the length includes the prompt.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("A number:", return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting `min_length` to a value smaller than the uncontrolled output length has no impact
    >>> gen_out = model.generate(**inputs, min_length=3)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting a larger `min_length` will force the model to generate beyond its natural ending point, which is not
    >>> # necessarily incorrect
    >>> gen_out = model.generate(**inputs, min_length=10)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one thousand, nine hundred and ninety-four
    ```
    
    - __call__


    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
    Contrarily to [`MinLengthLogitsProcessor`], this processor ignores the prompt.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the
            input length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer(["A number:"], return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting `min_new_tokens` will force the model to generate beyond its natural ending point, which is not
    >>> # necessarily incorrect
    >>> gen_out = model.generate(**inputs, min_new_tokens=2)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one thousand
    ```
    
    - __call__


    [`LogitsProcessor`] that performs min-p, i.e. keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter becomes more agressive in the presence of
    high-probability tokens, which is a sign of a confident output that we shouldn't deviate from.

    Often used together with [`TemperatureLogitsWarper`]. Used as an alternative to [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    Created by @menhguin and @kalomaze (github handles). Code adapted from [this external PR](https://github.com/oobabooga/text-generation-webui/pull/4449/files)

    Args:
        min_p (`float`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in
            the 0.99-0.8 range (use the opposite of normal `top_p` values).
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With `min_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `min_p` in the 0.01-0.2 range.
    >>> outputs = model.generate(**inputs, do_sample=True, min_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    
    - __call__


    [`LogitsProcessor`] that enforces that specified sequences will never be selected.

    <Tip>

    In order to get the token ids of the words that should not appear in the generated text, make sure to set
    `add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words,
    add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers,
    as fast tokenizers' prefixing behaviours come from `pre tokenizers`. Read more
    [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated.
        eos_token_id (`Union[int, List[int], torch.Tensor]`, *optional*):
            The id(s) of the *end-of-sequence* token.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

    >>> output_ids = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a mess.

    >>> # Now let's take the bad words out. Please note that the tokenizer is initialized differently
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=True)


    >>> def get_tokens_as_list(word_list):
    ...     "Converts a sequence of words into a list of tokens"
    ...     tokens_list = []
    ...     for word in word_list:
    ...         tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
    ...         tokens_list.append(tokenized_word)
    ...     return tokens_list


    >>> bad_words_ids = get_tokens_as_list(word_list=["mess"])
    >>> output_ids = model.generate(
    ...     inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, pad_token_id=tokenizer.eos_token_id
    ... )
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a surprise.
    ```
    
    - __call__


    N-grams are groups of "n" consecutive words, characters, or tokens taken from a sequence of text. Given the
    sentence: "She runs fast", the bi-grams (n=2) would be ("she", "runs") and ("runs", "fast"). In text generation,
    avoiding repetitions of word sequences provides a more diverse output. This [`LogitsProcessor`] enforces no
    repetition of n-grams by setting the scores of banned tokens to negative infinity which eliminates those tokens
    from consideration when further processing the scores. Note that, for decoder-only models like most LLMs, the
    prompt is also considered to obtain the n-grams.
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    <Tip>

    Use n-gram penalties with care. For instance, penalizing 2-grams (bigrams) in an article about the city of New York
    might lead to undesirable outcomes where the city's name appears only once in the entire text.
    [Reference](https://huggingface.co/blog/how-to-generate)

    </Tip>

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    >>> inputs = tokenizer(["Today I"], return_tensors="pt")

    >>> output = model.generate(**inputs)
    >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
    Today I’m not sure if I’m going to be able to do it.

    >>> # Now let's add ngram size using `no_repeat_ngram_size`. This stops the repetitions ("I’m") in the output.
    >>> output = model.generate(**inputs, no_repeat_ngram_size=2)
    >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
    Today I’m not sure if I can get a better understanding of the nature of this issue
    ```
    
    - __call__


    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("Alice and Bob", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=5)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice and Bob are friends

    >>> # We can contrain it with `prefix_allowed_tokens_fn` to force a certain behavior based on a prefix.
    >>> # For instance, we can force an entire entity to be generated when its beginning is detected.
    >>> entity = tokenizer(" Bob Marley", return_tensors="pt").input_ids[0]  # 3 tokens
    >>> def prefix_allowed_tokens_fn(batch_id, input_ids):
    ...     '''
    ...     Attempts to generate 'Bob Marley' when 'Bob' is detected.
    ...     In this case, `batch_id` is not used, but you can set rules for each batch member.
    ...     '''
    ...     if input_ids[-1] == entity[0]:
    ...         return [entity[1].item()]
    ...     elif input_ids[-2] == entity[0] and input_ids[-1] == entity[1]:
    ...         return [entity[2].item()]
    ...     return list(range(tokenizer.vocab_size))  # If no match, allow all tokens

    >>> outputs = model.generate(**inputs, max_new_tokens=5, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice and Bob Marley
    ```
    
    - __call__


    [`LogitsProcessor`] that prevents the repetition of previous tokens through a penalty. This penalty is applied at
    most once per token. Note that, for decoder-only models like most LLMs, the considered tokens include the prompt.

    In the original [paper](https://arxiv.org/pdf/1909.05858.pdf), the authors suggest the use of a penalty of around
    1.2 to achieve a good balance between truthful generation and lack of repetition. To penalize and reduce
    repetition, use `penalty` values above 1.0, where a higher value penalizes more strongly. To reward and encourage
    repetition, use `penalty` values between 0.0 and 1.0, where a lower value rewards more strongly.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 penalizes previously generated
            tokens. Between 0.0 and 1.0 rewards previously generated tokens.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> # Initializing the model and tokenizer for it
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    >>> inputs = tokenizer(["I'm not going to"], return_tensors="pt")

    >>> # This shows a normal generate without any specific parameters
    >>> summary_ids = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    I'm not going to be able to do that. I'm going to be able to do that

    >>> # This generates a penalty for repeated tokens
    >>> penalized_ids = model.generate(**inputs, repetition_penalty=1.1)
    >>> print(tokenizer.batch_decode(penalized_ids, skip_special_tokens=True)[0])
    I'm not going to be able to do that. I'll just have to go out and play
    ```
    
    - __call__


    [`LogitsProcessor`] that applies an additive bias on sequences. The bias is applied to the last token of a sequence
    when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than
    one token, consider using beam methods (to gracefully work around partially completed sequences that have a
    negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

    <Tip>

    In order to get the token ids of the sequences that you want to bias, make sure to set `add_prefix_space=True` when
    initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The
    `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers' prefixing behaviours
    come from `pre tokenizers`. Read more [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        sequence_bias (`List[List[Union[List[int], float]]]`):
            List of lists that maps a sequence of tokens to its bias term (e.g. `[[[10, 45], -2.0],
            [[64], -7.5]]`). Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias
            will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be
            completed (in the token selection step after this processor is applied).

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

    >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Trump Jr

    >>> # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=True)


    >>> def get_tokens(word):
    ...     return tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]


    >>> # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
    >>> sequence_bias = [get_tokens("Trump"), -10.0]
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Donald,

    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Rumsfeld,

    >>> # We can also add a positive bias to nudge the model towards specific tokens or continuations
    >>> sequence_bias = [get_tokens("Donald Duck"), 10.0]
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Duck.
    ```
    
    - __call__


    [`SuppressTokensAtBeginLogitsProcessor`] supresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are
    not generated at the beginning. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:

    ```python
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # Whisper has `begin_suppress_tokens` set by default (= `[220, 50256]`). 50256 is the EOS token, so this means
    >>> # it can't generate and EOS token in the first iteration, but it can in the others.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(outputs.scores[0][0, 50256])
    tensor(-inf)
    >>> print(outputs.scores[-1][0, 50256])  # in other places we can see some probability mass for EOS
    tensor(29.9010)

    >>> # If we disable `begin_suppress_tokens`, we can generate EOS in the first iteration.
    >>> outputs = model.generate(
    ...     **inputs, return_dict_in_generate=True, output_scores=True, begin_suppress_tokens=None
    ... )
    >>> print(outputs.scores[0][0, 50256])
    tensor(11.2027)
    ```
    
    - __call__


    This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so
    that they are not generated. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:

    ```python
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # Whisper has a long list of suppressed tokens. For instance, in this case, the token 1 is suppressed by default.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(outputs.scores[1][0, 1])  # 1 (and not 0) is the first freely generated token
    tensor(-inf)

    >>> # If we disable `suppress_tokens`, we can generate it.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, suppress_tokens=None)
    >>> print(outputs.scores[1][0, 1])
    tensor(6.0678)
    ```
    
    - __call__


    Logits processor that implements watermarking techniques for text generation models.
    This class facilitates the application of SynthID text watermarking, a method for embedding imperceptible signals
    into generated text to aid in detecting synthetic content. It operates by subtly manipulating the probabilities of
    token selection during text generation in a manner that can be reliably recovered later for verification.

    Key Features:
    * **State Management:** Maintains internal state to track token sequences and generate watermarking keys
    dynamically.

    * **Key Generation:** Computes hashes based on token sequences and watermarking parameters to create unique keys
    for each position.

    * **G-Value Sampling:** Employs a pre-computed sampling table to sample watermarking values (g-values) based on
    the generated keys.

    * **Score Adjustment:** Applies calculated g-values to modify token probabilities during generation, embedding the
    watermark.

    * **Context Repetition Handling:** Incorporates logic to avoid watermarking tokens in repeated contexts,
    preserving naturalness.

    * **EOS Token Masking:** Supports masking end-of-sentence tokens to prevent their inclusion in watermarking
    calculations.

    * **Utility Functions:** Provides functions to compute g-values directly, check for context repetition, create
    EOS token masks, and estimate expected mean g-values.

    Refer to paper url: https://www.nature.com/articles/s41586-024-08025-4 for more details around this.

    Args:
        ngram_len (`int`):
            Ngram length.
        keys (`List[int]`):
            A sequence of watermarking keys, one for each depth.
        sampling_table_size (`int`):
            Size of the sampling table.
        sampling_table_seed (`int`):
            Random seed to generate the sampling table.
        context_history_size (`int`):
            Size of the tensor to keep track of seen contexts.
        device (`torch.device`):
            Device to use.
        skip_first_ngram_calls (`bool`, *optional*, defaults to `False`):
            Whether to skip first ngram calls.
        debug_mode (`bool`, optional, *optional*, defaults to `False`):
            Logits are modified to uniform one got before watermarking modification is applied. This is to test the
            implementation.

    Examples:
    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

    >>> tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b', padding_side="left")
    >>> model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')

    >>> # SynthID Text configuration
    >>> watermarking_config = SynthIDTextWatermarkingConfig(
    ...     keys=[654, 400, 836, 123, 340, 443, 597, 160, 57],
    ...     ngram_len=5,
    ... )

    >>> # Generation with watermarking
    >>> tokenized_prompts = tokenizer(["Once upon a time, "], return_tensors="pt", padding=True)
    >>> output_sequences = model.generate(
    ...     **tokenized_prompts, watermarking_config=watermarking_config, do_sample=True, max_new_tokens=10
    ... )
    >>> watermarked_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    ```
    
    - __call__


    [`LogitsProcessor`] for temperature (exponential scaling output probability distribution), which effectively means
    that it can control the randomness of the predicted tokens. Often used together with [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    <Tip>

    Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won't have
    any effect.

    </Tip>

    Args:
        temperature (`float`):
            Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
            randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
            token.

    Examples:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)  # for reproducibility

    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

    >>> # With temperature=1.0, the default, we consistently get random outputs due to random sampling.
    >>> generate_kwargs = {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is one of these companies that is going to take a',
    "Hugging Face Company is a brand created by Brian A. O'Neil"]

    >>> # However, with temperature close to 0, it approximates greedy decoding strategies (invariant)
    >>> generate_kwargs["temperature"] = 0.0001
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is a company that has been around for over 20 years',
    'Hugging Face Company is a company that has been around for over 20 years']
    ```
    
    - __call__


    [`LogitsProcessor`] that performs top-k, i.e. restricting to the k highest probability elements. Often used
    together with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: A, B, C, D", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: A, B, C, D, E — S — O, P — R

    >>> # With `top_k` sampling, the output gets restricted the k most likely tokens.
    >>> # Pro tip: In practice, LLMs use `top_k` in the 5-50 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_k=2)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: A, B, C, D, E, F, G, H, I
    ```
    
    - __call__


    [`LogitsProcessor`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    Often used together with [`TemperatureLogitsWarper`] and [`TopKLogitsWarper`].

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With `top_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `top_p` in the 0.9-0.95 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    
    - __call__


    [`LogitsProcessor`] that performs typical decoding. Inspired on how humans use language, it prioritizes tokens
    whose log probability is close to the entropy of the token probability distribution. This means that the most
    likely tokens may be discarded in the process.

    See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information.

    Args:
        mass (`float`, *optional*, defaults to 0.9):
            Value of typical_p between 0 and 1 inclusive, defaults to 0.9.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("1, 2, 3", return_tensors="pt")

    >>> # We can see that greedy decoding produces a sequence of numbers
    >>> outputs = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,

    >>> # For this particular seed, we can see that sampling produces nearly the same low-information (= low entropy)
    >>> # sequence
    >>> set_seed(18)
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    1, 2, 3, 4, 5, 6, 7, 8, 9 and 10

    >>> # With `typical_p` set, the most obvious sequence is no longer produced, which may be good for your problem
    >>> set_seed(18)
    >>> outputs = model.generate(
    ...     **inputs, do_sample=True, typical_p=0.1, return_dict_in_generate=True, output_scores=True
    ... )
    >>> print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0])
    1, 2, 3 and 5

    >>> # We can see that the token corresponding to "4" (token 934) in the second position, the most likely token
    >>> # as seen with greedy decoding, was entirely blocked out
    >>> print(outputs.scores[1][0, 934])
    tensor(-inf)
    ```
    
    - __call__


    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (`float`):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale != 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality. A value smaller than 1 has the opposite effect, while
            making the negative prompt provided with negative_prompt_ids (if any) act as a positive prompt.
        model (`PreTrainedModel`):
            The model computing the unconditional scores. Supposedly the same as the one computing the conditional
            scores. Both models must use the same tokenizer.
        unconditional_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to
            the last token of the prompt.
        unconditional_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention mask for unconditional_ids.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to cache key/values during the negative prompt forward pass.


    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=1.5)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

    >>> # with a negative prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

    >>> # with a positive prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    "Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
    ```
    
    - __call__



    [`LogitsProcessor`] that modifies the logits for the generation of timestamps in the transcription. When the input
    tokens are at a specific threshold, the processor sets the scores to negative infinity. The processor makes sure
    that timestamp tokens appear in pairs, by masking out the logits that would break this pairing pattern. This is
    done to maintain the consistency and structure of generated timestamps. It also ensures that when the predicted
    probability of sampling any of the timestamp token is greater than any individual non-timestamp token, those
    non-timestamp logits are set to negative infinity. This is done to ensure the generation of timestamps over other
    potential tokens.


    See [the paper](https://arxiv.org/abs/2212.04356) for more information.

    Args:
        generate_config (`GenerateConfig`):
            The generate config used to generate the output. The following parameters are required:
                eos_token_id (`int`, *optional*, defaults to 50257):
                    The id of the *end-of-sequence* token.
                no_timestamps_token_id (`int`, *optional*, defaults to 50363):
                    The id of the `"<|notimestamps|>"` token.
                max_initial_timestamp_index (`int`, *optional*, defaults to 1):
                    Used to set the maximum value of the initial timestamp. This is used to prevent the model from
                    predicting timestamps that are too far in the future.
        begin_index (`Optional`, *optional*): Token index of the first token that is generated by the model.
        _detect_timestamp_from_logprob (`bool`, *optional*): Whether timestamps can be predicted from logprobs over all timestamps.

    Examples:
    ``` python
    >>> import torch
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration, GenerationConfig
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[3]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features

    >>> #Displaying timestamps
    >>> generated_ids = model.generate(inputs=input_features, return_timestamps=True)
    >>> transcription = processor.batch_decode(generated_ids, decode_with_timestamps=True)[0]
    >>> print("Transcription:", transcription)
    Transcription: <|startoftranscript|><|0.00|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can<|6.44|><|6.44|> discover in it but little of rocky Ithaca.<|9.44|><|endoftext|>


    >>> #No timestamps & change EOS:
    >>> #This allows the user to select a specific token to terminate the sequence on, in this case it's the word "can"(460)
    >>> model.generation_config.eos_token_id = 460
    >>> generated_ids = model.generate(inputs=input_features,return_timestamps=False)
    >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    >>> print("Transcription:", transcription)
    Transcription:  He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can
    ```
    
    - __call__


    Logits processor for watermarking generated text. The processor modifies model output scores by adding a small bias to
    randomized set of "green" tokens before generating the next token. "Green" tokens selection process depends on the
    `seeding_scheme` used. The code was based on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

    The text generated by this `LogitsProcessor` can be detected using `WatermarkDetector`. See [`~WatermarkDetector.__call__`] for details,

    See [the paper](https://arxiv.org/abs/2306.04634) for more information.

    Args:
        vocab_size (`int`):
            The model tokenizer's vocab_size. Used to calculate "green" tokens ratio.
        device (`str`):
            The device where model is allocated.
        greenlist_ratio (`float`, optional, *optional*, defaults to 0.25):
            The ratio of "green" tokens used to the vocabulary size. Defaults to 0.25.
        bias (`float`, optional, *optional*, defaults to 2.0):
            The bias added to the selected "green" tokens' logits. Consider lowering the
            `bias` if the text generation quality degrades. Recommended values are in the
            range of [0.5, 2.0]. Defaults to 2.0.
        hashing_key (`int`, optional, *optional*, defaults to 15485863):
            Key used for hashing. If you deploy this watermark, we advise using another private key.
            Defaults to 15485863 (the millionth prime).
        seeding_scheme (`str`, optional, *optional*, defaults to `"lefthash"`):
            The seeding scheme used for selecting "green" tokens. Accepts values:
                - "lefthash" (default): "green" tokens selection depend on the last token (Algorithm 2 from paper)
                - "selfhash": "green" tokens selection depends on the current token itself (Algorithm 3 from paper)
                    The downside of this scheme is that it considers all possible next tokens and can be slower than "lefthash".
            The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.
        context_width (`int`, *optional*, defaults to 1):
            The number of previous tokens to use when setting the seed.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkingConfig

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["Alice and Bob are"], return_tensors="pt")

    >>> # normal generation
    >>> out = model.generate(inputs["input_ids"], max_length=20, do_sample=False)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Alice and Bob are both in the same room.\n\n"I\'m not sure if you\'re'

    >>> # watermarked generation
    >>> watermarking_config = WatermarkingConfig(bias=2.5, context_width=2, seeding_scheme="selfhash")
    >>> out = model.generate(inputs["input_ids"], watermarking_config=watermarking_config, max_length=20, do_sample=False)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Alice and Bob are both still alive and well and the story is pretty much a one-hour adventure'

    >>> # to detect watermarked text use the WatermarkDetector class
    >>> from transformers import WatermarkDetector
    >>> detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config= watermarking_config)
    >>> detection_preds = detector(out)
    >>> detection_preds
    array([ True])
    ```
    
    - __call__


### TensorFlow

[[autodoc]] TFForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] TFForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] TFForceTokensLogitsProcessor
    - __call__

[[autodoc]] TFLogitsProcessor
    - __call__

[[autodoc]] TFLogitsProcessorList
    - __call__

[[autodoc]] TFLogitsWarper
    - __call__

[[autodoc]] TFMinLengthLogitsProcessor
    - __call__

[[autodoc]] TFNoBadWordsLogitsProcessor
    - __call__

[[autodoc]] TFNoRepeatNGramLogitsProcessor
    - __call__

[[autodoc]] TFRepetitionPenaltyLogitsProcessor
    - __call__

[[autodoc]] TFSuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] TFSuppressTokensLogitsProcessor
    - __call__

[[autodoc]] TFTemperatureLogitsWarper
    - __call__

[[autodoc]] TFTopKLogitsWarper
    - __call__

[[autodoc]] TFTopPLogitsWarper
    - __call__

### FLAX

[[autodoc]] FlaxForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] FlaxForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] FlaxForceTokensLogitsProcessor
    - __call__

[[autodoc]] FlaxLogitsProcessor
    - __call__

[[autodoc]] FlaxLogitsProcessorList
    - __call__

[[autodoc]] FlaxLogitsWarper
    - __call__

[[autodoc]] FlaxMinLengthLogitsProcessor
    - __call__

[[autodoc]] FlaxSuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] FlaxSuppressTokensLogitsProcessor
    - __call__

[[autodoc]] FlaxTemperatureLogitsWarper
    - __call__

[[autodoc]] FlaxTopKLogitsWarper
    - __call__

[[autodoc]] FlaxTopPLogitsWarper
    - __call__

[[autodoc]] FlaxWhisperTimeStampLogitsProcessor
    - __call__

## StoppingCriteria

A [`StoppingCriteria`] can be used to change when to stop generation (other than EOS token). Please note that this is exclusively available to our PyTorch implementations.

Abstract base class for all stopping criteria that can be applied during generation.

    If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
    output_scores=True` to `generate`.
    
    - __call__

Abstract base class for all stopping criteria that can be applied during generation.

    If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
    output_scores=True` to `generate`.
    List
    - __call__


    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
        max_position_embeddings (`int`, *optional*):
            The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
    
    - __call__


    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    `initial_time`.

    Args:
        max_time (`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (`float`, *optional*, defaults to `time.time()`):
            The start of the generation allowed time.
    
    - __call__


    This class can be used to stop generation whenever specific string sequences are generated. It preprocesses
    the strings together with the tokenizer vocab to find positions where tokens can validly complete the stop strings.

    Generation is stopped as soon as a token is generated that completes any of the stop strings.
    We want to catch any instance in which the stop string would be present in the decoded output, which means
    we must also catch cases with "overhangs" off one or both ends. To make this more concrete, for the stop string
    "stop", any of the following token sequences would trigger the match:

    - ["st", "op"]
    - ["stop"]
    - ["st", "opera"]
    - ["sto", "pper"]
    - ["las", "topper"]
    - ["s", "to", "pped"]

    Note that a match will only be triggered if the stop string is at the end of the generated sequence. In other
    words, these sequences will not trigger a match:

    - ["stop", "at"]
    - ["st", "op", "at"]
    - ["st", "opera", "tion"]

    The reason these are not a match is that the stop string does not overlap with the final token. If you can remove
    one or more tokens from the end of the sequence without destroying the stop string, then this criterion will not
    match that stop string. This is by design; because this check is run after each token is generated, we can't miss a
    valid stop string if one is generated, but we don't want to halt generation just because the stop string exists
    somewhere in the past input_ids.

    How is the match actually performed, though? We do it in quite a confusing way, because we want the entire match
    process to be compilable with Torch or XLA, which means we cannot use standard string methods. However, it is possible,
    with some work, to do string matching with pure tensor operations. We'll begin by describing the algorithm we use
    with standard string operations, and then at the end we'll explain how this is converted to pure tensor operations.

    The key to the algorithm is an observation: Because the stop string must overlap with the end of the token sequence, we can start at
    the end of the sequence and work backwards. Specifically, we check that there is an overlap between the start of
    the final token and the end of the stop_string, or to put it another way, stop_string[-i:] == token[:i] for
    some i > 0. If you look at the positive examples above, you'll see the last token in all of them fulfills this
    property:

    - ["st", "op"] (overlap is "op", overlap length == 2)
    - ["stop"]  (overlap is "stop", overlap length == 4)
    - ["st", "opera"]  (overlap is "op", overlap length == 2)
    - ["sto", "pper"]  (overlap is "p", overlap length == 1)
    - ["las", "topper"]  (overlap is "top", overlap length == 3)
    - ["s", "to", "pped"]  (overlap is "p", overlap length == 1)

    It's impossible to construct a matching sequence that does not have this property (feel free to verify this
    yourself). However, although this overlap between the start of the final token and the end of the stop string is
    necessary for a match, it is not sufficient. We also need to check that the rest of the token sequence is
    consistent with the stop string.

    How do we do that? Let's use ["s", "to", "pped"] as an example. We know that the final token, "pped", has an
    overlap of 1 with the stop string, "stop". We then go back to the previous token, "to". Since we have already
    matched 1 character from the stop string, the remainder to check is "sto". We check that the next token "to"
    matches the end of the remainder, which it does. We have now matched 3 characters from the stop string, and the
    remainder to match is "s". We go back to the previous token again, which is also "s". This is a match, and so
    we have matched the entire stop string.

    How does it work when the tokens run off the start of the stop string, though? Let's consider the example of
    ["las", "topper"]. The final token, "topper", has an overlap of 3 with the stop string, "stop". Therefore,
    the remaining stop string to match is "s". We go back to the previous token, "las". Because the remainder to
    match is just "s", with length 1, we consider only the final 1 character from the token, which is "s". This
    matches the stop string, and so the entire string is matched.

    How do we compute these matches with tensor operations, though? Simply: we efficiently precompute the necessary
    information for all tokens! For every token, we compute:
    - Its overlap with the end of the stop string, if any
    - The positions inside the stop string where the token matches, including matches that run off the start.
    - The total length of the token

    For example, for the token "pped", we would compute an end overlap of 1, no internal matching positions,
    and a length of 4. For the token "to", we would compute no end overlap, a single internal matching position
    of 1 (counting from the end), and a length of 2. For the token "s", we would compute no end overlap,
    a single internal matching position of 3 (again counting from the end) and a length of 1.

    As long as we have this information, we can execute the algorithm above without any string comparison
    operations. We simply perform the following steps:
    - Check if the final token has an end-overlap with the start string
    - Continue backwards, keeping track of how much of the stop string we've matched so far
    - At each point, check if the next token has the current position as one of its valid positions
    - Continue until either a match fails, or we completely match the whole stop string

    Again, consider ["s", "to", "pped"] as an example. "pped" has an end overlap of 1, so we can begin a match.
    We have matched 1 character so far, so we check that the next token "to", has 1 as a valid position (again,
    counting from the end). It does, so we add the length of "to" to our position tracker. We have now matched
    3 characters, so we check that the next token "s" has 3 as a valid position. It does, so we add its length
    to the position tracker. The position tracker is now 4, which is the length of the stop string. We have matched the
    entire stop string.

    In the second case, ["las", "topper"], "topper" has an end overlap of 3, so we can begin a match. We have
    matched 3 characters so far, so we check that the next token "las" has 3 as a valid position. It does, because we
    allow tokens to match positions that run off the start of the stop string. We add its length to the position
    tracker. The position tracker is now 6, which is greater than the length of the stop string! Don't panic, though -
    this also counts as a match of the stop string. We have matched the entire stop string.


    Args:
        tokenizer (`PreTrainedTokenizer`):
            The model's associated tokenizer (necessary to extract vocab and tokenize the termination sequences)
        stop_strings (`Union[str, List[str]]`):
            A list of strings that should end generation. If a string is passed, it will be treated like a
            list with a single element.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    >>> model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    >>> inputs = tokenizer("The biggest states in the USA by land area:", return_tensors="pt")

    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    The biggest states in the USA by land area:
    - Alaska
    - Texas
    - California

    >>> # Passing one or more stop strings will halt generation after those strings are emitted
    >>> # Note that generating with stop strings requires you to pass the tokenizer too
    >>> gen_out = model.generate(**inputs, stop_strings=["Texas"], tokenizer=tokenizer)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    The biggest states in the USA by land area:
    - Alaska
    - Texas
    ```
    
    - __call__


    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
    
    - __call__

## Constraints

A [`Constraint`] can be used to force the generation to include specific tokens or sequences in the output. Please note that this is exclusively available to our PyTorch implementations.

Abstract base class for all constraints that can be applied during generation.
    It must define how the constraint can be satisfied.

    All classes that inherit Constraint must follow the requirement that

    ```py
    completed = False
    while not completed:
        _, completed = constraint.update(constraint.advance())
    ```

    will always terminate (halt).
    


    [`Constraint`] enforcing that an ordered sequence of tokens is included in the output.

    Args:
        token_ids (`List[int]`):
            The id of the token that must be generated by the output.
    


    A special [`Constraint`] that is fulfilled by fulfilling just one of several constraints.

    Args:
        nested_token_ids (`List[List[int]]`):
            A list of words, where each word is a list of ids. This constraint is fulfilled by generating just one from
            the list of words.
    

Abstract base class for all constraints that can be applied during generation.
    It must define how the constraint can be satisfied.

    All classes that inherit Constraint must follow the requirement that

    ```py
    completed = False
    while not completed:
        _, completed = constraint.update(constraint.advance())
    ```

    will always terminate (halt).
    ListState

## BeamSearch


    Abstract base class for all beam scorers that are used for [`~PreTrainedModel.beam_search`] and
    [`~PreTrainedModel.beam_sample`].
    
    - process
    - finalize


    [`BeamScorer`] implementing standard beam search decoding.

    Adapted in part from [Facebook's XLM beam search
    code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

    Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
    implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformers.BeamSearchScorer.finalize`].
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    
    - process
    - finalize


    [`BeamScorer`] implementing constrained beam search decoding.


    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        constraints (`List[Constraint]`):
            A list of positive constraints represented as `Constraint` objects that must be fulfilled in the generation
            output. For more information, the documentation of [`Constraint`] should be read.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformers.BeamSearchScorer.finalize`].
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    
    - process
    - finalize

## Streamers


    Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)

        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    


    Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that benefit from acessing the generated text in a non-blocking way (e.g. in an interactive
    Gradio demo).

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        >>> from threading import Thread

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextIteratorStreamer(tok)

        >>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        >>> generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        >>> thread = Thread(target=model.generate, kwargs=generation_kwargs)
        >>> thread.start()
        >>> generated_text = ""
        >>> for new_text in streamer:
        ...     generated_text += new_text
        >>> generated_text
        'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
        ```
    


    Streamer that stores print-ready text in a queue, to be used by a downstream application as an async iterator.
    This is useful for applications that benefit from acessing the generated text asynchronously (e.g. in an
    interactive Gradio demo).

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Raises:
        TimeoutError: If token generation time exceeds timeout value.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, AsyncTextIteratorStreamer
        >>> from threading import Thread
        >>> import asyncio

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")

        >>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        >>> async def main():
        ...     # Important: AsyncTextIteratorStreamer must be initialized inside a coroutine!
        ...     streamer = AsyncTextIteratorStreamer(tok)
        ...     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        ...     thread = Thread(target=model.generate, kwargs=generation_kwargs)
        ...     thread.start()
        ...     generated_text = ""
        ...     async for new_text in streamer:
        ...         generated_text += new_text
        >>>     print(generated_text)
        >>> asyncio.run(main())
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    

## Caches


    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    
    - update


    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    Config
	- update


    Configuration class for quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        residual_length (`Optional[int]`, *optional*, defaults to 128):
            Length of the residual cache which will always be stored in original presicion.
            Defaults to 128.
        compute_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The defualt dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to perform computations, should be same as the model's device.
    
	- validate


    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    
    - update
    - get_seq_length
    - reorder_cache
    - to_legacy_cache
    - from_legacy_cache


    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`
    
    - update
    - get_seq_length


    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

    Example:

        ```python
        >>> # Run pip install quanto first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache, QuantizedCacheConfig

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4)
        >>> past_key_values = QuantoQuantizedCache(cache_config=cache_config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        QuantoQuantizedCache()
        ```
    


    Quantized Cache class that uses `HQQ` as a backend to perform quantization. Current implementation supports `int2`, `int4`, `int8` dtypes.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

    Example:

        ```python
        >>> # Run pip install hqq first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HQQQuantizedCache, QuantizedCacheConfig

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4, axis_key=1, axis_value=1)
        >>> past_key_values = HQQQuantizedCache(cache_config=cache_config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HQQQuantizedCache()
        ```
    


    A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
    generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
    tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_sink_tokens (`int`):
            The number of sink tokens. See the original paper for more information.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        SinkCache()
        ```
    
    - update
    - get_seq_length
    - reorder_cache


    A drop-in replacement for DynamicCache that conserves GPU memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    In addition to the default CUDA stream, where all forward() computations happen,
    this class uses another stream, the prefetch stream, which it creates itself.
    Since scheduling of operations on separate streams happens independently, this class uses
    the prefetch stream to asynchronously prefetch the KV cache of layer k+1 when layer k is executing.
    The movement of the layer k-1 cache to the CPU is handled by the default stream as a simple way to
    ensure the eviction is scheduled after all computations on that cache are finished.
    
    - update
    - prefetch_layer
    - evict_previous_layer


    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used. If you are manually setting the batch size, make sure to take into account the number of beams if you are running beam search
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        StaticCache()
        ```
    
    - update
    - get_seq_length
    - reset


    Static cache class to be used with `torch.compile(model)` that offloads to the CPU or
    another device.

    Args:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize
            the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`Union[str, torch.device]`):
            The device on which the cache should be initialized. Should be the same as the
            layer device.
        dtype (`torch.dtype`, *optional*):
            The default `dtype` to use when initializing the cache.
        offload_device (`Union[str, torch.device]`, *optional*, defaults to `cpu`):
            The device to offload to. Defaults to CPU.
        layer_device_map (`Dict[int, Union[str, torch.device, int]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Attributes:
        key_cache (`List[torch.Tensor]`):
            Off-loaded key cache tensors. First one will be on device, where-as the others are
            off-loaded.
        value_cache (`List[torch.Tensor]`):
            Off-loaded value cache tensors. First one will be on device, where-as the others are
            off-loaded.
        max_batch_size (`int`):
            The maximum batch size with which this cache can be used.
        max_cache_len (`int`):
            The maximum sequence length with which this cache can be used.
        device (`torch.device`):
            The device on which the cache is used.
        offload_device (`torch.device`):
            The device used to offload to.
        dtype (`torch.dtype`):
            The `dtype` used to initializing the cache.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, OffloadedStaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = OffloadedStaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> past_kv_length = outputs.past_key_values # access cache filled with key/values from generation
        ```
    
    - update
    - get_seq_length
    - reset


    Hybrid Cache class to be used with `torch.compile` for Gemma2 models that alternate between a local sliding window attention
    and global attention in every other layer. Under the hood, Hybrid Cache leverages ["SlidingWindowCache"] for sliding window attention
    and ["StaticCache"] for global attention. For more information, see the documentation of each subcomponeent cache class.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (torch.dtype, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HybridCache

        >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

        >>> inputs = tokenizer(text="My name is Gemma", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = HybridCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HybridCache()
        ```
    
    - update
    - get_seq_length
    - reset


    Sliding Window Cache class to be used with `torch.compile` for models like Mistral that support sliding window attention.
    Every time when we try to update the cache, we compute the `indices` based on `cache_position >= self.config.sliding_window - 1`,
    if true(which means the cache can not hold all the old key value states and new states together because of the sliding window constraint),
    we need to do a cycle shift based on `indices` to replace the oldest states by the new key value states passed in.

    The `to_shift` is only true once we are above sliding_window. Thus with `sliding_window==64`:

    indices = (slicing + to_shift[-1].int()-1) % self.config.sliding_window
    tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,  0])

    We overwrite the cache using these, then we always write at cache_position (clamped to `sliding_window`)

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SlidingWindowCache

        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

        >>> inputs = tokenizer(text="My name is Mistral", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = SlidingWindowCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        SlidingWindowCache()
        ```
    
    - update
    - reset


    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
        >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

        >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

        >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
        >>> self_attention_cache = DynamicCache()
        >>> cross_attention_cache = DynamicCache()
        >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        EncoderDecoderCache()
        ```

    
    - get_seq_length
    - to_legacy_cache
    - from_legacy_cache
    - reset
    - reorder_cache


    Cache for mamba model which does not have attention mechanism and key value states.

    Arguments:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.

    Attributes:
        dtype: (`torch.dtype`):
            The default `dtype` used to initializing the cache.
        intermediate_size: (`int`):
            Model's intermediate_size taken from config.
        ssm_state_size: (`int`):
            Model's state_size taken from config.
        conv_kernel_size: (`int`):
            Model's convolution kernel size taken from config
        conv_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, conv_kernel_size]` that holds convolutional states.
        ssm_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, ssm_state_size]` that holds ssm states

    Example:

        ```python
        >>> from transformers import AutoTokenizer, MambaForCausalLM, MambaCache

        >>> model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

        >>> inputs = tokenizer(text="My name is Mamba", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = MambaCache(config=model.config, batch_size=1, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values
        MambaCache()
        ```
    
    - update_conv_state
    - update_ssm_state
    - reset

## Watermark Utils


    Class that holds arguments for watermark generation and should be passed into `GenerationConfig` during `generate`.
    See [this paper](https://arxiv.org/abs/2306.04634) for more details on the arguments.

    Accepts the following keys:
        - greenlist_ratio (`float`):
            Used for watermarking. The ratio of "green" tokens used to the vocabulary size. Defaults to 0.25.
        - bias (`float`):
            Used with watermarking. The bias added to the selected "green" tokens' logits. Defaults to 2.0.
        - hashing_key (`int`):
            Hashing key used for watermarking. Defaults to 15485863 (the millionth prime).
        - seeding_scheme (`str`):
            Algorithm to use for watermarking. Accepts values:
                - "lefthash" (default): "green" tokens selection depend on the last token (Algorithm 2 from the paper)
                - "selfhash": "green" tokens selection depends on the current token itself (Algorithm 3 from the paper)
                    The downside of this scheme is that it considers all possible next tokens and can be slower than "lefthash".
        - context_width(`int`):
            The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.
    
    - __call__


    Detector for detection of watermark generated text. The detector needs to be given the exact same settings that were
    given during text generation to replicate the watermark greenlist generation and so detect the watermark. This includes
    the correct device that was used during text generation, the correct watermarking arguments and the correct tokenizer vocab size.
    The code was based on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

    See [the paper](https://arxiv.org/abs/2306.04634) for more information.

    Args:
        model_config (`PretrainedConfig`):
            The model config that will be used to get model specific arguments used when generating.
        device (`str`):
            The device which was used during watermarked text generation.
        watermarking_config (Union[`WatermarkingConfig`, `Dict`]):
            The exact same watermarking config and arguments used when generating text.
        ignore_repeated_ngrams (`bool`, *optional*, defaults to `False`):
            Whether to count every unique ngram only once or not.
        max_cache_size (`int`, *optional*, defaults to 128):
            The max size to be used for LRU caching of seeding/sampling algorithms called for every token.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig

    >>> model_id = "openai-community/gpt2"
    >>> model = AutoModelForCausalLM.from_pretrained(model_id)
    >>> tok = AutoTokenizer.from_pretrained(model_id)
    >>> tok.pad_token_id = tok.eos_token_id
    >>> tok.padding_side = "left"

    >>> inputs = tok(["This is the beginning of a long story", "Alice and Bob are"], padding=True, return_tensors="pt")
    >>> input_len = inputs["input_ids"].shape[-1]

    >>> # first generate text with watermark and without
    >>> watermarking_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
    >>> out_watermarked = model.generate(**inputs, watermarking_config=watermarking_config, do_sample=False, max_length=20)
    >>> out = model.generate(**inputs, do_sample=False, max_length=20)

    >>> # now we can instantiate the detector and check the generated text
    >>> detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=watermarking_config)
    >>> detection_out_watermarked = detector(out_watermarked, return_dict=True)
    >>> detection_out = detector(out, return_dict=True)
    >>> detection_out_watermarked.prediction
    array([ True,  True])

    >>> detection_out.prediction
    array([False,  False])
    ```
    
    - __call__


    This is the configuration class to store the configuration of a [`BayesianDetectorModel`]. It is used to
    instantiate a Bayesian Detector model according to the specified arguments.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        watermarking_depth (`int`, *optional*):
            The number of tournament layers.
        base_rate (`float1`, *optional*, defaults to 0.5):
            Prior probability P(w) that a text is watermarked.
    


    Bayesian classifier for watermark detection.

    This detector uses Bayes' rule to compute a watermarking score, which is the sigmoid of the log of ratio of the
    posterior probabilities P(watermarked|g_values) and P(unwatermarked|g_values). Please see the section on
    BayesianScore in the paper for further details.
    Paper URL: https://www.nature.com/articles/s41586-024-08025-4

    Note that this detector only works with non-distortionary Tournament-based watermarking using the Bernoulli(0.5)
    g-value distribution.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BayesianDetectorConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    
    - forward


    Class that holds arguments for watermark generation and should be passed into `GenerationConfig` during `generate`.
    See [this paper](https://www.nature.com/articles/s41586-024-08025-4) for more details on the arguments.

    Args:
        ngram_len (`int`):
            Ngram length.
        keys (`List[int]`):
            A sequence of watermarking keys, one for each depth.
        context_history_size (`int`, *optional*, defaults to 1024):
            Size of the tensor to keep track of seen contexts.
        sampling_table_seed (`int`, *optional*, defaults to 0):
            Random seed to generate the sampling table.
        sampling_table_size (`int`, *optional*, defaults to 65536):
            Size of the sampling table.
        skip_first_ngram_calls (`bool`, *optional*, defaults to `False`):
            Whether to skip first ngram calls.
        debug_mode (`bool`, optional, *optional*, defaults to `False`):
            Logits are modified to uniform one got before watermarking modification is applied. This is to test the
            implementation.

    Examples:
    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

    >>> tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b', padding_side="left")
    >>> model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')

    >>> # SynthID Text configuration
    >>> watermarking_config = SynthIDTextWatermarkingConfig(
    ...     keys=[654, 400, 836, 123, 340, 443, 597, 160, 57],
    ...     ngram_len=5,
    ... )

    >>> # Generation with watermarking
    >>> tokenized_prompts = tokenizer(["Once upon a time, "], return_tensors="pt", padding=True)
    >>> output_sequences = model.generate(
    ...     **tokenized_prompts, watermarking_config=watermarking_config, do_sample=True, max_new_tokens=10
    ... )
    >>> watermarked_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    ```
    


    SynthID text watermark detector class.

    This class has to be initialized with the trained bayesian detector module check script
    in examples/synthid_text/detector_training.py for example in training/saving/loading this
    detector module. The folder also showcases example use case of this detector.

    Parameters:
        detector_module ([`BayesianDetectorModel`]):
            Bayesian detector module object initialized with parameters.
            Check examples/research_projects/synthid_text/detector_training.py for usage.
        logits_processor (`SynthIDTextWatermarkLogitsProcessor`):
            The logits processor used for watermarking.
        tokenizer (`Any`):
            The tokenizer used for the model.

    Examples:
    ```python
    >>> from transformers import (
    ...     AutoTokenizer, BayesianDetectorModel, SynthIDTextWatermarkLogitsProcessor, SynthIDTextWatermarkDetector
    ... )

    >>> # Load the detector. See examples/research_projects/synthid_text for training a detector.
    >>> detector_model = BayesianDetectorModel.from_pretrained("joaogante/dummy_synthid_detector")
    >>> logits_processor = SynthIDTextWatermarkLogitsProcessor(
    ...     **detector_model.config.watermarking_config, device="cpu"
    ... )
    >>> tokenizer = AutoTokenizer.from_pretrained(detector_model.config.model_name)
    >>> detector = SynthIDTextWatermarkDetector(detector_model, logits_processor, tokenizer)

    >>> # Test whether a certain string is watermarked
    >>> test_input = tokenizer(["This is a test input"], return_tensors="pt")
    >>> is_watermarked = detector(test_input.input_ids)
    ```
    
    - __call__

## Compile Utils


    Class that holds arguments relative to `torch.compile` behavior, when using automatic compilation in `generate`.
    See [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) for more details on the arguments.

    Args:
        fullgraph (`bool`, *optional*, defaults to `True`):
            If `True`, requires that the whole forward be capturable in a single graph.
        dynamic (`bool` or `None`, *optional*):
            Whether to try to use dynamic shape graphs.
        backend (`str` or `Callable`, *optional*, defaults to `"inductor"`):
            Backend to be used.
        mode (`str`, *optional*, defaults to `"reduce-overhead"`):
            Controls balance between performance and overhead.
        options (`dict`, *optional*):
            A dictionary of options to pass to the backend.

    Examples:
    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig

    >>> tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
    >>> model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b').cuda()

    >>> # Automatic compile configuration, used with static cache
    >>> compile_config = CompileConfig(dynamic=True)

    >>> # Generation with static cache and compile config
    >>> input = tokenizer.encode("Hello there, how", return_tensors="pt").cuda()
    >>> output = model.generate(
    ...     input, do_sample=False, max_new_tokens=300, cache_implementation="static", compile_config=compile_config
    ... )
    >>> output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    ```
    
    - __call__

