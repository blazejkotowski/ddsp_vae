import lightning as L
from lightning.pytorch.utilities import grad_norm

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import auraloss

from modules.filterbank import FilterBank

import math

from typing import List

class NoiseBandNet(L.LightningModule):
  """
  A neural network that learns how to resynthesise signal, predicting amplitudes of
  precalculated, loopable noise bands.

  Args:
    - m_filters: int, the number of filters in the filterbank
    - hidden_size: int, the size of the hidden layers of the neural network
    - n_control_params: int, the number of control parameters to be used
    - samplerate : int, the sampling rate of the input signal
    - resampling_factor: int, internal up / down sampling factor for control signal and noisebands
    - learning_rate: float, the learning rate for the optimizer
  """
  def __init__(self,
               m_filters: int = 2048,
               samplerate: int = 44100,
               hidden_size: int = 128,
               n_control_params: int = 2,
               resampling_factor: int = 32,
               learning_rate: float = 1e-3):
    super().__init__()

    self._filterbank = FilterBank(
      m_filters=m_filters,
      fs=samplerate
    )

    self._resampling_factor = resampling_factor

    # Define the neural network
    ## Parallel connection of the control parameters to the dedicated MLPs
    self.control_param_mlps = nn.ModuleList([self._make_mlp(1, 1, hidden_size) for _ in range(n_control_params)])

    ## Intermediate GRU layer
    self.gru = nn.GRU((hidden_size) * n_control_params, hidden_size, batch_first=True)

    ## Intermediary 3-layer MLP
    self.inter_mlp = self._make_mlp(hidden_size + n_control_params, 3, hidden_size)

    ## Output layer predicting amplitudes
    self.output_amps = nn.Linear(hidden_size, len(self._noisebands))

    # Define the loss
    self.loss = self._construct_loss_function()
    self._learning_rate = learning_rate


  def forward(self, control_params: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the network.
    Args:
      - control_params: List[torch.Tensor[batch_size, signal_length, 1]], a list of control parameters
    Returns:
      - signal: torch.Tensor, the synthesized signal
    """
    # predict the amplitudes of the noise bands
    amps = self._predict_amplitudes(control_params)

    # synthesize the signal
    signal = self._synthesize(amps)
    return signal


  def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
    """
    Compute the loss for a batch of data

    Args:
      batch:
        Tuple[
            torch.Tensor[batch_size, n_signal],
            torch.Tensor[params_number, batch_size, n_signal]
          ], audio, control_params
      batch_idx: int, index of the batch (unused)
    Returns:
      loss: torch.Tensor[batch_size, 1], tensor of loss
    """
    x_audio, control_params = batch

    # Downsample the control params by resampling factor
    control_params = [F.interpolate(c, scale_factor=1/self._resampling_factor, mode='linear') for c in control_params]

    # Predict the audio
    y_audio = self.forward(control_params)

    # Compute return the loss
    loss = self.loss(y_audio, x_audio)
    print(f"Loss: {loss.item()}")
    return loss


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self._learning_rate)


  # def on_before_optimizer_step(self, optimizer):
  #     # Compute the 2-norm for each layer
  #     # If using mixed precision, the gradients are already unscaled here
  #     norms = grad_norm(self.control_param_mlps, norm_type=2)
  #     print(norms)
  #     norms = grad_norm(self.gru, norm_type=2)
  #     print(norms)
  #     norms = grad_norm(self.inter_mlp, norm_type=2)
  #     print(norms)
  #     norms = grad_norm(self.output_amps, norm_type=2)
  #     print(norms)


  def _predict_amplitudes(self, control_params: torch.Tensor) -> torch.Tensor:
    """
    Predict noiseband amplitudes given the control parameters.
    Args:
      - control_params: List[torch.Tensor[batch_size, signal_length, 1]], a list of control parameters
    Returns:
      - amps: torch.Tensor, the predicted amplitudes of the noise bands
    """
    control_params = [c.permute(0, 2, 1) for c in control_params]

    # pass through the control parameter MLPs
    x = [mlp(param) for param, mlp in zip(control_params, self.control_param_mlps)] # out: [control_params_number, batch_size, signal_length, hidden_size]

    # concatenate both mlp outputs together
    x = torch.cat(x, dim=-1) # out: [batch_size, signal_length, hidden_size * control_params_number]

    # pass concatenated control parameter outputs through GRU
    # GRU returns (output, final_hidden_state) tuple. We are interested in the output.
    x = self.gru(x)[0] # out: [batch_size, signal_length, hidden_size]

    # append the control params to the GRU output
    for c in control_params:
      x = torch.cat([x, c], dim=-1) # out: [batch_size, signal_length, hidden_size + control_params_number]

    # pass through the intermediary MLP
    x = self.inter_mlp(x) # out: (batch_size, signal_length, hidden_size)

    # pass through the output layer and custom activation
    amps = self._scaled_sigmoid(self.output_amps(x)).permute(0, 2, 1) # out: [batch_size, n_bands, signal_length]

    print("Amps contain nan?", torch.any(torch.isnan(amps)))
    return amps


  def _synthesize(self, amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
      - amplitudes: torch.Tensor[batch_size, n_bands, sig_length], the predicted amplitudes of the noise bands
    Returns:
      - signal: torch.Tensor[batch_size, sig_length], the synthesized signal
    """
    # upsample the amplitudes
    upsampled_amplitudes = F.interpolate(amplitudes, scale_factor=self._resampling_factor, mode='linear')

    # fit the noisebands into the amplitudes
    repeats = upsampled_amplitudes.shape[-1] // self._noisebands.shape[-1] + 1
    looped_bands = self._noisebands.repeat(1, repeats) # repeat
    looped_bands = looped_bands[:, :upsampled_amplitudes.shape[-1]] # trim
    looped_bands = looped_bands.to(upsampled_amplitudes.device, dtype=torch.float32)

    # synthesize the signal
    signal = (upsampled_amplitudes * looped_bands).sum(1, keepdim=True)
    is_nan = torch.any(torch.isnan(signal))
    print(f"Did synthesised signal produce NaNs? {is_nan}")

    # signal = torch.sum(upsampled_amplitudes * looped_bands, dim=1, keepdim=True)
    return signal


  def _scaled_sigmoid(self, x: torch.Tensor):
    """
    Custom activation function for the output layer. It is a scaled sigmoid function,
    guaranteeing that the output is always positive.
    Args:
      - x: torch.Tensor, the input tensor
    Returns:
      - y: torch.Tensor, the output tensor
    """
    # return 2*torch.pow(torch.sigmoid(x), np.log(10)) + 1e-18
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-18


  @property
  def _noisebands(self) -> List[np.ndarray]:
    """Delegate the noisebands to the filterbank object."""
    return self._filterbank.noisebands


  @staticmethod
  def _construct_loss_function():
    """
    Construct the loss function for the model: a multi-resolution STFT loss
    """
    fft_sizes = np.array([8192, 4096, 2048, 1024, 512, 128, 32])
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[8192, 4096, 2048, 1024, 512, 128, 32],
                                                hop_sizes=[8192//4, 4096//4, 2048//4, 1024//4, 512//4, 128//4, 32//4],
                                                win_lengths=[8192, 4096, 2048, 1024, 512, 128, 32])


  @staticmethod
  def _make_mlp(in_size: int, hidden_layers: int, hidden_size: int) -> nn.Sequential:
    """
    Constructs a multi-layer perceptron.
    Args:
    - in_size: int, the input layer size
    - hidden_layers: int, the number of hidden layers
    - hidden_size: int, the size of each hidden layer
    Returns:
    - mlp: nn.Sequential, the multi-layer perceptron
    """
    sizes = [in_size]
    sizes.extend(hidden_layers * [hidden_size])

    layers = []
    for i in range(len(sizes)-1):
      layers.append(nn.Linear(sizes[i], sizes[i+1]))
      layers.append(nn.LayerNorm(sizes[i+1]))
      layers.append(nn.LeakyReLU())

    return nn.Sequential(*layers)
