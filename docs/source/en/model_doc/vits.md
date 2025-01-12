<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# VITS

## Overview

The VITS model was proposed in [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) by Jaehyeon Kim, Jungil Kong, Juhee Son.

VITS (**V**ariational **I**nference with adversarial learning for end-to-end **T**ext-to-**S**peech) is an end-to-end 
speech synthesis model that predicts a speech waveform conditional on an input text sequence. It is a conditional variational 
autoencoder (VAE) comprised of a posterior encoder, decoder, and conditional prior.

A set of spectrogram-based acoustic features are predicted by the flow-based module, which is formed of a Transformer-based
text encoder and multiple coupling layers. The spectrogram is decoded using a stack of transposed convolutional layers,
much in the same style as the HiFi-GAN vocoder. Motivated by the one-to-many nature of the TTS problem, where the same text 
input can be spoken in multiple ways, the model also includes a stochastic duration predictor, which allows the model to 
synthesise speech with different rhythms from the same input text. 

The model is trained end-to-end with a combination of losses derived from variational lower bound and adversarial training. 
To improve the expressiveness of the model, normalizing flows are applied to the conditional prior distribution. During 
inference, the text encodings are up-sampled based on the duration prediction module, and then mapped into the 
waveform using a cascade of the flow module and HiFi-GAN decoder. Due to the stochastic nature of the duration predictor,
the model is non-deterministic, and thus requires a fixed seed to generate the same speech waveform.

The abstract from the paper is the following:

*Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.*

This model can also be used with TTS checkpoints from [Massively Multilingual Speech (MMS)](https://arxiv.org/abs/2305.13516) 
as these checkpoints use the same architecture and a slightly modified tokenizer.

This model was contributed by [Matthijs](https://huggingface.co/Matthijs) and [sanchit-gandhi](https://huggingface.co/sanchit-gandhi). The original code can be found [here](https://github.com/jaywalnut310/vits).

## Usage examples

Both the VITS and MMS-TTS checkpoints can be used with the same API. Since the flow-based model is non-deterministic, it 
is good practice to set a seed to ensure reproducibility of the outputs. For languages with a Roman alphabet, 
such as English or French, the tokenizer can be used directly to pre-process the text inputs. The following code example 
runs a forward pass using the MMS-TTS English checkpoint:

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

set_seed(555)  # make deterministic

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0]
```

The resulting waveform can be saved as a `.wav` file:

```python
import scipy

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=waveform)
```

Or displayed in a Jupyter Notebook / Google Colab:

```python
from IPython.display import Audio

Audio(waveform, rate=model.config.sampling_rate)
```

For certain languages with a non-Roman alphabet, such as Arabic, Mandarin or Hindi, the [`uroman`](https://github.com/isi-nlp/uroman) 
perl package is required to pre-process the text inputs to the Roman alphabet.

You can check whether you require the `uroman` package for your language by inspecting the `is_uroman` attribute of 
the pre-trained `tokenizer`:

```python
from transformers import VitsTokenizer

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
print(tokenizer.is_uroman)
```
If the is_uroman attribute is `True`, the tokenizer will automatically apply the `uroman` package to your text inputs, but you need to install uroman if not already installed using:  
```
pip install --upgrade uroman
```
Note: Python version required to use `uroman` as python package should be >= `3.10`. 
You can use the tokenizer as usual without any additional preprocessing steps:
```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import os
import subprocess

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")
model = VitsModel.from_pretrained("facebook/mms-tts-kor")
text = "이봐 무슨 일이야"
inputs = tokenizer(text=text, return_tensors="pt")

set_seed(555)  # make deterministic
with torch.no_grad():
   outputs = model(inputs["input_ids"])

waveform = outputs.waveform[0]
```
If you don't want to upgrade to python >= `3.10`, then you can use the `uroman` perl package to pre-process the text inputs to the Roman alphabet.
To do this, first clone the uroman repository to your local machine and set the bash variable `UROMAN` to the local path:


```bash
git clone https://github.com/isi-nlp/uroman.git
cd uroman
export UROMAN=$(pwd)
```

You can then pre-process the text input using the following code snippet. You can either rely on using the bash variable 
`UROMAN` to point to the uroman repository, or you can pass the uroman directory as an argument to the `uromanize` function:

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import os
import subprocess

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")
model = VitsModel.from_pretrained("facebook/mms-tts-kor")

def uromanize(input_string, uroman_path):
    """Convert non-Roman strings to Roman using the `uroman` perl package."""
    script_path = os.path.join(uroman_path, "bin", "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Execute the perl command
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

    # Return the output as a string and skip the new-line character at the end
    return stdout.decode()[:-1]

text = "이봐 무슨 일이야"
uromanized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

inputs = tokenizer(text=uromanized_text, return_tensors="pt")

set_seed(555)  # make deterministic
with torch.no_grad():
   outputs = model(inputs["input_ids"])

waveform = outputs.waveform[0]
```

## VitsConfig


    This is the configuration class to store the configuration of a [`VitsModel`]. It is used to instantiate a VITS
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the VITS
    [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 38):
            Vocabulary size of the VITS model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method of [`VitsModel`].
        hidden_size (`int`, *optional*, defaults to 192):
            Dimensionality of the text encoder layers.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer encoder.
        window_size (`int`, *optional*, defaults to 4):
            Window size for the relative positional embeddings in the attention layers of the Transformer encoder.
        use_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the key, query, value projection layers in the Transformer encoder.
        ffn_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        ffn_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the 1D convolution layers used by the feed-forward network in the Transformer encoder.
        flow_size (`int`, *optional*, defaults to 192):
            Dimensionality of the flow layers.
        spectrogram_bins (`int`, *optional*, defaults to 513):
            Number of frequency bins in the target spectrogram.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        use_stochastic_duration_prediction (`bool`, *optional*, defaults to `True`):
            Whether to use the stochastic duration prediction module or the regular duration predictor.
        num_speakers (`int`, *optional*, defaults to 1):
            Number of speakers if this is a multi-speaker model.
        speaker_embedding_size (`int`, *optional*, defaults to 0):
            Number of channels used by the speaker embeddings. Is zero for single-speaker models.
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the HiFi-GAN upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 2, 2]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the HiFi-GAN upsampling network.
            The length of `upsample_rates` defines the number of convolutional layers and has to match the length of
            `upsample_kernel_sizes`.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[16, 16, 4, 4]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the HiFi-GAN upsampling
            network. The length of `upsample_kernel_sizes` defines the number of convolutional layers and has to match
            the length of `upsample_rates`.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the HiFi-GAN
            multi-receptive field fusion (MRF) module.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            HiFi-GAN multi-receptive field fusion (MRF) module.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        depth_separable_channels (`int`, *optional*, defaults to 2):
            Number of channels to use in each depth-separable block.
        depth_separable_num_layers (`int`, *optional*, defaults to 3):
            Number of convolutional layers to use in each depth-separable block.
        duration_predictor_flow_bins (`int`, *optional*, defaults to 10):
            Number of channels to map using the unonstrained rational spline in the duration predictor model.
        duration_predictor_tail_bound (`float`, *optional*, defaults to 5.0):
            Value of the tail bin boundary when computing the unconstrained rational spline in the duration predictor
            model.
        duration_predictor_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the 1D convolution layers used in the duration predictor model.
        duration_predictor_dropout (`float`, *optional*, defaults to 0.5):
            The dropout ratio for the duration predictor model.
        duration_predictor_num_flows (`int`, *optional*, defaults to 4):
            Number of flow stages used by the duration predictor model.
        duration_predictor_filter_channels (`int`, *optional*, defaults to 256):
            Number of channels for the convolution layers used in the duration predictor model.
        prior_encoder_num_flows (`int`, *optional*, defaults to 4):
            Number of flow stages used by the prior encoder flow model.
        prior_encoder_num_wavenet_layers (`int`, *optional*, defaults to 4):
            Number of WaveNet layers used by the prior encoder flow model.
        posterior_encoder_num_wavenet_layers (`int`, *optional*, defaults to 16):
            Number of WaveNet layers used by the posterior encoder model.
        wavenet_kernel_size (`int`, *optional*, defaults to 5):
            Kernel size of the 1D convolution layers used in the WaveNet model.
        wavenet_dilation_rate (`int`, *optional*, defaults to 1):
            Dilation rates of the dilated 1D convolutional layers used in the WaveNet model.
        wavenet_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the WaveNet layers.
        speaking_rate (`float`, *optional*, defaults to 1.0):
            Speaking rate. Larger values give faster synthesised speech.
        noise_scale (`float`, *optional*, defaults to 0.667):
            How random the speech prediction is. Larger values create more variation in the predicted speech.
        noise_scale_duration (`float`, *optional*, defaults to 0.8):
            How random the duration prediction is. Larger values create more variation in the predicted durations.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the output audio waveform is digitalized expressed in hertz (Hz).

    Example:

    ```python
    >>> from transformers import VitsModel, VitsConfig

    >>> # Initializing a "facebook/mms-tts-eng" style configuration
    >>> configuration = VitsConfig()

    >>> # Initializing a model (with random weights) from the "facebook/mms-tts-eng" style configuration
    >>> model = VitsModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

## VitsTokenizer


    Construct a VITS tokenizer. Also supports MMS-TTS.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        language (`str`, *optional*):
            Language identifier.
        add_blank (`bool`, *optional*, defaults to `True`):
            Whether to insert token id 0 in between the other tokens.
        normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the input text by removing all casing and punctuation.
        phonemize (`bool`, *optional*, defaults to `True`):
            Whether to convert the input text into phonemes.
        is_uroman (`bool`, *optional*, defaults to `False`):
            Whether the `uroman` Romanizer needs to be applied to the input text prior to tokenizing.
    

Methods: __call__
    - save_vocabulary

## VitsModel

The complete VITS model, for text-to-speech synthesis.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VitsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.


Methods: forward
