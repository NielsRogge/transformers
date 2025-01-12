<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# Jukebox

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The Jukebox model was proposed in [Jukebox: A generative model for music](https://arxiv.org/pdf/2005.00341.pdf)
by Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford,
Ilya Sutskever. It introduces a generative music model which can produce minute long samples that can be conditioned on
an artist, genres and lyrics.

The abstract from the paper is the following:

*We introduce Jukebox, a model that generates music with singing in the raw audio domain. We tackle the long context of raw audio using a multiscale VQ-VAE to compress it to discrete codes, and modeling those using autoregressive Transformers. We show that the combined model at scale can generate high-fidelity and diverse songs with coherence up to multiple minutes. We can condition on artist and genre to steer the musical and vocal style, and on unaligned lyrics to make the singing more controllable. We are releasing thousands of non cherry-picked samples, along with model weights and code.*

As shown on the following figure, Jukebox is made of 3 `priors` which are decoder only models. They follow the architecture described in [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509), modified to support longer context length.
First, a autoencoder is used to encode the text lyrics. Next, the first (also called `top_prior`) prior attends to the last hidden states extracted from the lyrics encoder. The priors are linked to the previous priors respectively via an `AudioConditioner` module. The`AudioConditioner` upsamples the outputs of the previous prior to raw tokens at a certain audio frame per second resolution.
The metadata such as *artist, genre and timing* are passed to each prior, in the form of a start token and positional embedding for the timing data.  The hidden states are mapped to the closest codebook vector from the VQVAE in order to convert them to raw audio.

![JukeboxModel](https://gist.githubusercontent.com/ArthurZucker/92c1acaae62ebf1b6a951710bdd8b6af/raw/c9c517bf4eff61393f6c7dec9366ef02bdd059a3/jukebox.svg)

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ).
The original code can be found [here](https://github.com/openai/jukebox).

## Usage tips

- This model only supports inference. This is for a few reasons, mostly because it requires a crazy amount of memory to train. Feel free to open a PR and add what's missing to have a full integration with the hugging face trainer!
- This model is very slow, and takes 8h to generate a minute long audio using the 5b top prior on a V100 GPU. In order automaticallay handle the device on which the model should execute, use `accelerate`.
- Contrary to the paper, the order of the priors goes from `0` to `1` as it felt more intuitive : we sample starting from `0`.
- Primed sampling (conditioning the sampling on raw audio) requires more memory than ancestral sampling and should be used with `fp16` set to `True`.

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ).
The original code can be found [here](https://github.com/openai/jukebox).

## JukeboxConfig


    This is the configuration class to store the configuration of a [`JukeboxModel`].

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a configuration with the defaults will
    yield a similar configuration to that of
    [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox-1b-lyrics) architecture.


    The downsampling and stride are used to determine downsampling of the input sequence. For example, downsampling =
    (5,3), and strides = (2, 2) will downsample the audio by 2^5 = 32 to get the first level of codes, and 2**8 = 256
    to get the second level codes. This is mostly true for training the top level prior and the upsamplers.

    Args:
        vqvae_config (`JukeboxVQVAEConfig`, *optional*):
            Configuration for the `JukeboxVQVAE` model.
        prior_config_list (`List[JukeboxPriorConfig]`, *optional*):
            List of the configs for each of the `JukeboxPrior` of the model. The original architecture uses 3 priors.
        nb_priors (`int`, *optional*, defaults to 3):
            Number of prior models that will sequentially sample tokens. Each prior is conditional auto regressive
            (decoder) model, apart from the top prior, which can include a lyric encoder. The available models were
            trained using a top prior and 2 upsampler priors.
        sampling_rate (`int`, *optional*, defaults to 44100):
            Sampling rate of the raw audio.
        timing_dims (`int`, *optional*, defaults to 64):
            Dimensions of the JukeboxRangeEmbedding layer which is equivalent to traditional positional embedding
            layer. The timing embedding layer converts the absolute and relative position in the currently sampled
            audio to a tensor of length `timing_dims` that will be added to the music tokens.
        min_duration (`int`, *optional*, defaults to 0):
            Minimum duration of the audios to generate
        max_duration (`float`, *optional*, defaults to 600.0):
            Maximum duration of the audios to generate
        max_nb_genres (`int`, *optional*, defaults to 5):
            Maximum number of genres that can be used to condition a single sample.
        metadata_conditioning (`bool`, *optional*, defaults to `True`):
            Whether or not to use metadata conditioning, corresponding to the artist, the genre and the min/maximum
            duration.

    Example:

    ```python
    >>> from transformers import JukeboxModel, JukeboxConfig

    >>> # Initializing a Jukebox configuration
    >>> configuration = JukeboxConfig()

    >>> # Initializing a model from the configuration
    >>> model = JukeboxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    

## JukeboxPriorConfig


        This is the configuration class to store the configuration of a [`JukeboxPrior`]. It is used to instantiate a
        `JukeboxPrior` according to the specified arguments, defining the model architecture. Instantiating a
        configuration with the defaults will yield a similar configuration to that of the top level prior from the
        [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox
    -1b-lyrics) architecture.

        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.



    Args:
        act_fn (`str`, *optional*, defaults to `"quick_gelu"`):
            Activation function.
        alignment_head (`int`, *optional*, defaults to 2):
            Head that is responsible of the alignment between lyrics and music. Only used to compute the lyric to audio
            alignment
        alignment_layer (`int`, *optional*, defaults to 68):
            Index of the layer that is responsible of the alignment between lyrics and music. Only used to compute the
            lyric to audio alignment
        attention_multiplier (`float`, *optional*, defaults to 0.25):
            Multiplier coefficient used to define the hidden dimension of the attention layers. 0.25 means that
            0.25*width of the model will be used.
        attention_pattern (`str`, *optional*, defaults to `"enc_dec_with_lyrics"`):
            Which attention pattern to use for the decoder/
        attn_dropout (`int`, *optional*, defaults to 0):
            Dropout probability for the post-attention layer dropout in the decoder.
        attn_res_scale (`bool`, *optional*, defaults to `False`):
            Whether or not to scale the residuals in the attention conditioner block.
        blocks (`int`, *optional*, defaults to 64):
            Number of blocks used in the `block_attn`. A sequence of length seq_len is factored as `[blocks, seq_len //
            blocks]` in the `JukeboxAttention` layer.
        conv_res_scale (`int`, *optional*):
            Whether or not to scale the residuals in the conditioner block. Since the top level prior does not have a
            conditioner, the default value is to None and should not be modified.
        num_layers (`int`, *optional*, defaults to 72):
            Number of layers of the transformer architecture.
        emb_dropout (`int`, *optional*, defaults to 0):
            Embedding dropout used in the lyric decoder.
        encoder_config (`JukeboxPriorConfig`, *optional*) :
            Configuration of the encoder which models the prior on the lyrics.
        encoder_loss_fraction (`float`, *optional*, defaults to 0.4):
            Multiplication factor used in front of the lyric encoder loss.
        hidden_size (`int`, *optional*, defaults to 2048):
            Hidden dimension of the attention layers.
        init_scale (`float`, *optional*, defaults to 0.2):
            Initialization scales for the prior modules.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether or not the prior is an encoder-decoder model. In case it is not, and `nb_relevant_lyric_tokens` is
            greater than 0, the `encoder` args should be specified for the lyric encoding.
        mask (`bool`, *optional*, defaults to `False`):
            Whether or not to mask the previous positions in the attention.
        max_duration (`int`, *optional*, defaults to 600):
            Maximum supported duration of the generated song in seconds.
        max_nb_genres (`int`, *optional*, defaults to 1):
            Maximum number of genres that can be used to condition the model.
        merged_decoder (`bool`, *optional*, defaults to `True`):
            Whether or not the decoder and the encoder inputs are merged. This is used for the separated
            encoder-decoder architecture
        metadata_conditioning (`bool`, *optional*, defaults to `True)`:
            Whether or not to condition on the artist and genre metadata.
        metadata_dims (`List[int]`, *optional*, defaults to `[604, 7898]`):
            Number of genres and the number of artists that were used to train the embedding layers of the prior
            models.
        min_duration (`int`, *optional*, defaults to 0):
            Minimum duration of the generated audio on which the model was trained.
        mlp_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier coefficient used to define the hidden dimension of the MLP layers. 0.25 means that 0.25*width of
            the model will be used.
        music_vocab_size (`int`, *optional*, defaults to 2048):
            Number of different music tokens. Should be similar to the `JukeboxVQVAEConfig.nb_discrete_codes`.
        n_ctx (`int`, *optional*, defaults to 6144):
            Number of context tokens for each prior. The context tokens are the music tokens that are attended to when
            generating music tokens.
        n_heads (`int`, *optional*, defaults to 2):
                Number of attention heads.
        nb_relevant_lyric_tokens (`int`, *optional*, defaults to 384):
            Number of lyric tokens that are used when sampling a single window of length `n_ctx`
        res_conv_depth (`int`, *optional*, defaults to 3):
            Depth of the `JukeboxDecoderConvBock` used to upsample the previously sampled audio in the
            `JukeboxMusicTokenConditioner`.
        res_conv_width (`int`, *optional*, defaults to 128):
            Width of the `JukeboxDecoderConvBock` used to upsample the previously sampled audio in the
            `JukeboxMusicTokenConditioner`.
        res_convolution_multiplier (`int`, *optional*, defaults to 1):
            Multiplier used to scale the `hidden_dim` of the `JukeboxResConv1DBlock`.
        res_dilation_cycle (`int`, *optional*):
            Dilation cycle used to define the `JukeboxMusicTokenConditioner`. Usually similar to the ones used in the
            corresponding level of the VQVAE. The first prior does not use it as it is not conditioned on upper level
            tokens.
        res_dilation_growth_rate (`int`, *optional*, defaults to 1):
            Dilation grow rate used between each convolutionnal block of the `JukeboxMusicTokenConditioner`
        res_downs_t (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
            Downsampling rates used in the audio conditioning network
        res_strides_t (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Striding used in the audio conditioning network
        resid_dropout (`int`, *optional*, defaults to 0):
            Residual dropout used in the attention pattern.
        sampling_rate (`int`, *optional*, defaults to 44100):
            Sampling rate used for training.
        spread (`int`, *optional*):
            Spread used in the `summary_spread_attention` pattern
        timing_dims (`int`, *optional*, defaults to 64):
            Dimension of the timing embedding.
        zero_out (`bool`, *optional*, defaults to `False`):
            Whether or not to zero out convolution weights when initializing.
    

## JukeboxVQVAEConfig


    This is the configuration class to store the configuration of a [`JukeboxVQVAE`]. It is used to instantiate a
    `JukeboxVQVAE` according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VQVAE from
    [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox-1b-lyrics) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        act_fn (`str`, *optional*, defaults to `"relu"`):
            Activation function of the model.
        nb_discrete_codes (`int`, *optional*, defaults to 2048):
            Number of codes of the VQVAE.
        commit (`float`, *optional*, defaults to 0.02):
            Commit loss multiplier.
        conv_input_shape (`int`, *optional*, defaults to 1):
            Number of audio channels.
        conv_res_scale (`bool`, *optional*, defaults to `False`):
            Whether or not to scale the residuals of the `JukeboxResConv1DBlock`.
        embed_dim (`int`, *optional*, defaults to 64):
            Embedding dimension of the codebook vectors.
        hop_fraction (`List[int]`, *optional*, defaults to `[0.125, 0.5, 0.5]`):
            Fraction of non-intersecting window used when continuing the sampling process.
        levels (`int`, *optional*, defaults to 3):
            Number of hierarchical levels that used in the VQVAE.
        lmu (`float`, *optional*, defaults to 0.99):
            Used in the codebook update, exponential moving average coefficient. For more detail refer to Appendix A.1
            of the original [VQVAE paper](https://arxiv.org/pdf/1711.00937v2.pdf)
        multipliers (`List[int]`, *optional*, defaults to `[2, 1, 1]`):
            Depth and width multipliers used for each level. Used on the `res_conv_width` and `res_conv_depth`
        res_conv_depth (`int`, *optional*, defaults to 4):
            Depth of the encoder and decoder block. If no `multipliers` are used, this is the same for each level.
        res_conv_width (`int`, *optional*, defaults to 32):
            Width of the encoder and decoder block. If no `multipliers` are used, this is the same for each level.
        res_convolution_multiplier (`int`, *optional*, defaults to 1):
            Scaling factor of the hidden dimension used in the `JukeboxResConv1DBlock`.
        res_dilation_cycle (`int`, *optional*):
            Dilation cycle value used in the `JukeboxResnet`. If an int is used, each new Conv1 block will have a depth
            reduced by a power of `res_dilation_cycle`.
        res_dilation_growth_rate (`int`, *optional*, defaults to 3):
            Resnet dilation growth rate used in the VQVAE (dilation_growth_rate ** depth)
        res_downs_t (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
            Downsampling rate for each level of the hierarchical VQ-VAE.
        res_strides_t (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Stride used for each level of the hierarchical VQ-VAE.
        sample_length (`int`, *optional*, defaults to 1058304):
            Provides the max input shape of the VQVAE. Is used to compute the input shape of each level.
        init_scale (`float`, *optional*, defaults to 0.2):
            Initialization scale.
        zero_out (`bool`, *optional*, defaults to `False`):
            Whether or not to zero out convolution weights when initializing.
    

## JukeboxTokenizer


    Constructs a Jukebox tokenizer. Jukebox can be conditioned on 3 different inputs :
        - Artists, unique ids are associated to each artist from the provided dictionary.
        - Genres, unique ids are associated to each genre from the provided dictionary.
        - Lyrics, character based tokenization. Must be initialized with the list of characters that are inside the
        vocabulary.

    This tokenizer does not require training. It should be able to process a different number of inputs:
    as the conditioning of the model can be done on the three different queries. If None is provided, defaults values will be used.:

    Depending on the number of genres on which the model should be conditioned (`n_genres`).
    ```python
    >>> from transformers import JukeboxTokenizer

    >>> tokenizer = JukeboxTokenizer.from_pretrained("openai/jukebox-1b-lyrics")
    >>> tokenizer("Alan Jackson", "Country Rock", "old town road")["input_ids"]
    [tensor([[   0,    0,    0, 6785,  546,   41,   38,   30,   76,   46,   41,   49,
               40,   76,   44,   41,   27,   30]]), tensor([[  0,   0,   0, 145,   0]]), tensor([[  0,   0,   0, 145,   0]])]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    If nothing is provided, the genres and the artist will either be selected randomly or set to None

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to:
    this superclass for more information regarding those methods.

    However the code does not allow that and only supports composing from various genres.

    Args:
        artists_file (`str`):
            Path to the vocabulary file which contains a mapping between artists and ids. The default file supports
            both "v2" and "v3"
        genres_file (`str`):
            Path to the vocabulary file which contain a mapping between genres and ids.
        lyrics_file (`str`):
            Path to the vocabulary file which contains the accepted characters for the lyrics tokenization.
        version (`List[str]`, `optional`, default to `["v3", "v2", "v2"]`) :
            List of the tokenizer versions. The `5b-lyrics`'s top level prior model was trained using `v3` instead of
            `v2`.
        n_genres (`int`, `optional`, defaults to 1):
            Maximum number of genres to use for composition.
        max_n_lyric_tokens (`int`, `optional`, defaults to 512):
            Maximum number of lyric tokens to keep.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    

Methods: save_vocabulary

## JukeboxModel

The bare JUKEBOX Model used for music generation. 4 sampling techniques are supported : `primed_sample`, `upsample`,
    `continue_sample` and `ancestral_sample`. It does not have a `forward` method as the training is not end to end. If
    you want to fine-tune the model, it is recommended to use the `JukeboxPrior` class and train each prior
    individually.
    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config (`JukeboxConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: ancestral_sample
    - primed_sample
    - continue_sample
    - upsample
    - _sample

## JukeboxPrior


    The JukeboxPrior class, which is a wrapper around the various conditioning and the transformer. JukeboxPrior can be
    seen as language models trained on music. They model the next `music token` prediction task. If a (lyric) `encoderù
    is defined, it also models the `next character` prediction on the lyrics. Can be conditionned on timing, artist,
    genre, lyrics and codes from lower-levels Priors.

    Args:
        config (`JukeboxPriorConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        level (`int`, *optional*):
            Current level of the Prior. Should be in range `[0,nb_priors]`.
        nb_priors (`int`, *optional*, defaults to 3):
            Total number of priors.
        vqvae_encoder (`Callable`, *optional*):
            Encoding method of the VQVAE encoder used in the forward pass of the model. Passing functions instead of
            the vqvae module to avoid getting the parameters.
        vqvae_decoder (`Callable`, *optional*):
            Decoding method of the VQVAE decoder used in the forward pass of the model. Passing functions instead of
            the vqvae module to avoid getting the parameters.
    

Methods: sample
    - forward

## JukeboxVQVAE

The Hierarchical VQ-VAE model used in Jukebox. This model follows the Hierarchical VQVAE paper from [Will Williams, Sam
Ringer, Tom Ash, John Hughes, David MacLeod, Jamie Dougherty](https://arxiv.org/abs/2002.08111).

    

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config (`JukeboxConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
    - encode
    - decode
