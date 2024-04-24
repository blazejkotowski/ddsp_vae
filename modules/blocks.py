import torch
import torch.nn as nn
import torch.nn.functional as F
import cached_conv as cc
import math
from torchaudio.transforms import MFCC

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

  layers = []
  for i in range(len(sizes)-1):
    layers.append(nn.Linear(sizes[i], sizes[i+1]))
    layers.append(nn.LayerNorm(sizes[i+1]))
    layers.append(nn.LeakyReLU())

  return cc.CachedSequential(*layers)

def _scaled_sigmoid(self, x: torch.Tensor):
  """
  Custom activation function for the output layer. It is a scaled sigmoid function,
  guaranteeing that the output is always positive.
  Args:
    - x: torch.Tensor, the input tensor
  Returns:
    - y: torch.Tensor, the output tensor
  """
  return 2*torch.pow(torch.sigmoid(x), math.log(10)) + 1e-18

class VariationalEncoder(nn.Module):
  def __init__(self, hidden_size: int, sample_rate: int = 44100, latent_size: int = 16, downsample_factor: int = 32):
    """
    Arguments:
      - hidden_size: int, the size of the hidden state of the GRU
      - sample_rate: int, the sample rate of the input audio
      - latent_size: int, the size of the output latent space
      - resampling_factor: int, the factor by which to downsample the mfccs
    """
    super().__init__()

    self.downsample_factor = downsample_factor
    self.mfcc = MFCC(sample_rate = sample_rate, n_mfcc = 40)
    self.normalization = nn.LayerNorm(hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)
    self.dense_out = nn.Linear(hidden_size, latent_size)


  def forward(self, audio: torch.Tensor):
    """
    Forward pass of the encoder.
    Arguments:
      - audio: torch.Tensor, the input audio tensor [batch_size, n_samples]
    Returns:
      - z: torch.Tensor, the latent space tensor
    """
    # Extract MFCCs
    breakpoint()
    mfcc = self.mfcc(audio)

    # Expand the MFCCs to match the audio length
    mfcc = F.interpolate(mfcc, size = audio.shape[-1], mode = 'nearest')

    # Downsample the MFCCs
    x = F.interpolate(mfcc, scale_factor = 1/self.downsample_factor, mode = 'linear')

    # Normalize the input
    x = self.normalization(x)

    # Pass through the GRU layer
    x, hx = self.gru(x) # TODO: Cache and use the hidden state

    # Pass through the dense layer
    z = self.dense_out(x)

    return z



class VariationalDecoder(nn.Module):
  def __init__(self, latent_size: int, hidden_layers: int, hidden_size: int, n_bands: int):
    """
    Arguments:
      - latent_size: int, the size of the latent space
      - hidden_layers: int, the number of hidden layers in the MLP
      - hidden_size: int, the size of each hidden layer in the MLP
      - n_bands: int, the number of noise bands
    """
    super().__init__()

    # MLP mapping from the latent space
    self.input_mlp  = _make_mlp(latent_size, hidden_layers, hidden_size)

    # Intermediate GRU layer
    self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    # Intermediary 3-layer MLP
    self.inter_mlp = _make_mlp(hidden_size, hidden_layers, hidden_size)

    # Output layer predicting amplitudes
    self.output_amps = nn.Linear(hidden_size, n_bands)


  def forward(self, z: torch.Tensor):
    """
    Forward pass of the decoder.
    Arguments:
      - z: torch.Tensor, the latent space tensor
    Returns:
      - amplitudes: torch.Tensor, the predicted amplitudes
    """
    # Pass through the input MLP
    x = self.input_mlp(z)

    # Pass through the GRU layer
    x, hx = self.gru(x) # TODO: Cache and use the hidden state

    # Pass through the intermediary MLP
    x = self.inter_mlp(x)

    # Pass through the output layer
    amplitudes = _scaled_sigmoid(self.output_amps(x))

    return amplitudes
