import torch
import torch.nn as nn
import torch.nn.functional as F
import cached_conv as cc
import math
from torchaudio.transforms import MFCC

from typing import Tuple, List

def _make_mlp(in_size: int, hidden_layers: int, hidden_size: int) -> cc.CachedSequential:
  """
  Constructs a multi-layer perceptron.
  Args:
  - in_size: int, the input layer size
  - hidden_layers: int, the number of hidden layers
  - hidden_size: int, the size of each hidden layer
  Returns:
  - mlp: cc.CachedSequential, the multi-layer perceptron
  """
  sizes = [in_size]
  sizes.extend(hidden_layers * [hidden_size])

  return _make_sequential(sizes)


def _make_sequential(sizes: List[int]):
  """
  Constructs a sequential model.
  Args:
  - sizes: List[int], the sizes of the layers
  Returns:
  - mlp: cc.CachedSequential, the sequential model
  """
  layers = []
  for i in range(len(sizes)-1):
    layers.append(nn.Linear(sizes[i], sizes[i+1]))
    layers.append(nn.LayerNorm(sizes[i+1]))
    layers.append(nn.LeakyReLU())

  return nn.Sequential(*layers)

def _scaled_sigmoid(x: torch.Tensor):
  """
  Custom activation function for the output layer. It is a scaled sigmoid function,
  guaranteeing that the output is always positive.
  Args:
    - x: torch.Tensor, the input tensor
  Returns:
    - y: torch.Tensor, the output tensor
  """
  return 2*torch.pow(torch.sigmoid(x), math.log(10)) + 1e-18

def _is_batch_size_one(x: torch.Tensor):
  """
  Check if the batch size of a tensor is one.
  Args:
    - x: torch.Tensor, the input tensor
  Returns:
    - bool, True if the batch size is one, False otherwise
  """
  return x.shape[0] == 1


class VariationalEncoder(nn.Module):
  def __init__(self,
               sample_rate: int = 44100,
               layer_sizes: List[int] = [128, 64, 32],
               latent_size: int = 16,
               downsample_factor: int = 32,
               n_mfcc: int = 30,
               streaming: bool = False):
    """
    Arguments:
      - sample_rate: int, the sample rate of the input audio
      - layer_sizes: List[int], the sizes of the layers in the bottleneck
      - latent_size: int, the size of the output latent space
      - resampling_factor: int, the factor by which to downsample the mfccs
      - n_mfcc : int, the number of mfccs to extract
      - streaming: bool, streaming mode (realtime)
    """
    super().__init__()
    self.streaming = streaming

    self.downsample_factor = downsample_factor
    self.mfcc = MFCC(sample_rate = sample_rate, n_mfcc = n_mfcc)

    self.normalization = nn.LayerNorm(n_mfcc)

    self.gru = nn.GRU(n_mfcc, layer_sizes[0], batch_first = True)
    self.register_buffer('hidden_state', torch.zeros(1, 1, layer_sizes[0]), persistent=False)

    self.bottleneck = _make_sequential(layer_sizes)

    self.mu_logvar_out = nn.Linear(layer_sizes[-1], 2*latent_size)


  def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass of the encoder.
    Arguments:
      - audio: torch.Tensor, the input audio tensor [batch_size, n_samples]
    Returns:
      - mu, logvar: Tuple[torch.Tensor, torch.Tensor], the latent space tensor
    """
    # Extract MFCCs
    mfcc = self.mfcc(audio)

    # Expand the MFCCs to match the audio length
    mfcc = F.interpolate(mfcc, size = audio.shape[-1], mode = 'nearest')

    # Downsample the MFCCs
    x = F.interpolate(mfcc, scale_factor = 1/self.downsample_factor, mode = 'linear')

    # Reshape to [batch_size, signal_length, n_mfcc]
    x = x.permute(0, 2, 1)

    # Normalize the input
    x = self.normalization(x)

    # Pass through the GRU layer
    if self.streaming and _is_batch_size_one(x):
      x, hx = self.gru(x, self.hidden_state)
      self.hidden_state.copy_(hx)
    else:
      x, _ = self.gru(x)

    # Pass through bottleneck
    x = self.bottleneck(x)

    # Pass through the dense layer
    z = self.mu_logvar_out(x)

    mu, logvar = z.chunk(2, dim = -1)

    return mu, logvar

  def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparametrization trick.
    Arguments:
      - mu: torch.Tensor, the mean of the latent space
      - logvar: torch.Tensor, the log variance of the latent space
    Returns:
      - z: torch.Tensor, the latent space tensor
    """
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

  def reparametrize_alter(self, mean: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Value]:
    """
    Reparametrize the latent variable z.
    Args:
      - z: torch.Tensor[batch_size, latent_size], the latent variable
    Returns:
      - z: torch.Tensor[batch_size, latent_size], the reparametrized latent variable
      - kl: torch.Tensor[batch_size, 1], the KL divergence
    """
    std = F.softplus(scale) + 1e-4
    var = std * std
    logvar = torch.log(var)

    z = torch.randn_like(mean) * std + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return z, kl


class VariationalDecoder(nn.Module):
  def __init__(self,
               n_bands: int = 512,
               n_sines: int = 500,
               latent_size: int = 16,
               layer_sizes: List[int] = [32, 64, 128],
               output_mlp_layers: int = 3,
               streaming: bool = False):
    """
    Arguments:
      - n_bands: int, the number of noise bands
      - latent_size: int, the size of the latent space
      - layer_sizes: List[int], the sizes of the layers in the bottleneck
      - output_mlp_layers: int, the number of layers in the output MLP
      - streaming: bool, streaming mode (realtime)
    """
    super().__init__()

    self.n_bands = n_bands
    self.n_sines = n_sines
    self.streaming = streaming

    # MLP mapping from the latent space
    self.input_bottleneck = _make_sequential([latent_size] + layer_sizes)

    hidden_size = layer_sizes[-1]

    # Intermediate GRU layer
    self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
    self.register_buffer('hidden_state', torch.zeros(1, 1, hidden_size), persistent=False)

    # Intermediary 3-layer MLP
    self.inter_mlp = _make_mlp(hidden_size, output_mlp_layers, hidden_size)

    # Output layer predicting noiseband amplitudes, and sine frequencies and amplitudes
    self.output_params = nn.Linear(hidden_size, n_bands + n_sines * 2)


  def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass of the decoder.
    Arguments:
      - z: torch.Tensor, the latent space tensor
    Returns:
      - noiseband_amps: torch.Tensor, the predicted noiseband amplitudes
      - sine_freqs: torch.Tensor, the predicted sine frequencies
      - sine_amps: torch.Tensor, the predicted sine amplitudes
    """
    # Pass through the input MLP
    x = self.input_bottleneck(z)

    # Pass through the GRU layer
    if self.streaming and _is_batch_size_one(z):
      x, hx = self.gru(x, self.hidden_state)
      self.hidden_state.copy_(hx)
    else:
      x, _ = self.gru(x)

    # Pass through the intermediary MLP
    x = self.inter_mlp(x)

    # Pass through the output layer
    output = _scaled_sigmoid(self.output_params(x))

    noiseband_amps = output[..., :self.n_bands].permute(0, 2, 1)
    sine_freqs = output[..., self.n_bands:self.n_bands + self.n_sines].permute(0, 2, 1)
    sine_amps = output[..., self.n_bands + self.n_sines:].permute(0, 2, 1)

    return noiseband_amps, sine_freqs, sine_amps
