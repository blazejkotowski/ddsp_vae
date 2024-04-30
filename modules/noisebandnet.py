import lightning as L
from lightning.pytorch.utilities import grad_norm

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import auraloss
import math

from modules.filterbank import FilterBank
from modules.blocks import VariationalEncoder, VariationalDecoder

from typing import List, Tuple, Optional

class NoiseBandNet(L.LightningModule):
  """
  A neural network that learns how to resynthesise signal, predicting amplitudes of
  precalculated, loopable noise bands.

  Args:
    - m_filters: int, the number of filters in the filterbank
    - hidden_size: int, the size of the hidden layers of the neural network
    - hidden_layers: int, the number of hidden layers of the neural network
    - latent_size: int, number of latent dimensions
    - samplerate : int, the sampling rate of the input signal
    - resampling_factor: int, internal up / down sampling factor for control signal and noisebands
    - learning_rate: float, the learning rate for the optimizer
    - torch_device: str, the device to run the model on
    - streaming: bool, whether to run the model in streaming mode
    - beta: float, the β parameter for the β-VAE loss
  """
  def __init__(self,
               m_filters: int = 2048,
               samplerate: int = 44100,
               hidden_size: int = 128,
               hidden_layers: int = 3,
               latent_size: int = 16,
               resampling_factor: int = 32,
               learning_rate: float = 1e-3,
               torch_device: str = 'cpu',
               streaming: bool = False,
               beta: float = 1.0):
    super().__init__()
    # Save hyperparameters in the checkpoints
    self.save_hyperparameters()

    self._filterbank = FilterBank(
      m_filters=m_filters,
      fs=samplerate
    )
    self.resampling_factor = resampling_factor
    self.latent_size = latent_size
    self.samplerate = samplerate
    self.beta = beta

    self._torch_device = torch_device
    self._noisebands_shift = 0

    # Define the neural network
    ## Encoder to extract latents from the input audio signal
    self.encoder = VariationalEncoder(
      hidden_size=hidden_size,
      sample_rate=samplerate,
      latent_size=latent_size,
      streaming=streaming
    )

    ## Decoder to predict the amplitudes of the noise bands
    self.decoder = VariationalDecoder(
      latent_size=latent_size,
      hidden_layers=hidden_layers,
      hidden_size=hidden_size,
      n_bands=self._filterbank.noisebands.shape[0],
      streaming=streaming
    )

    # Define the loss
    self.recons_loss = self._construct_mrstft_loss()

    self._learning_rate = learning_rate


  def forward(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the network.
    Args:
      - audio: torch.Tensor[batch_size, n_signal], the input audio signal
    Returns:
      - signal: torch.Tensor, the synthesized signal
    """
    # encode the audio signal
    mu, logvar = self.encoder(audio)
    z = self.encoder.reparametrize(mu, logvar)

    # predict the amplitudes of the noise bands
    amps = self.decoder(z)

    # synthesize the signal
    signal = self._synthesize(amps)
    return signal

  def training_step(self, x_audio: torch.Tensor, batch_idx: int) -> torch.Tensor:
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
    mu, logvar = self.encoder(x_audio)
    z = self.encoder.reparametrize(mu, logvar)

    # predict the amplitudes of the noise bands
    amps = self.decoder(z)

    # synthesize the signal
    y_audio = self._synthesize(amps)

    # Compute the reconstruction loss
    recons_loss = self.recons_loss(y_audio, x_audio)

    # Compute the KLD loss
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1))

    # Compute the total loss using β parameter
    loss = recons_loss + self.beta * kld_loss

    self.log("recons_loss", recons_loss, prog_bar=True, logger=True)
    self.log("kld_loss", self.beta*kld_loss, prog_bar=True, logger=True)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return loss

  # TODO: Generate the validationa audio and add to tensorboard
  # def on_validation_epoch_end(self, )


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self._learning_rate)


  def _synthesize(self, amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
      - amplitudes: torch.Tensor[batch_size, n_bands, sig_length], the predicted amplitudes of the noise bands
    Returns:
      - signal: torch.Tensor[batch_size, sig_length], the synthesized signal
    """
    # upsample the amplitudes
    upsampled_amplitudes = F.interpolate(amplitudes, scale_factor=float(self.resampling_factor), mode='linear')

    # shift the noisebands to maintain the continuity of the noise signal
    noisebands = torch.roll(self._noisebands, shifts=-self._noisebands_shift, dims=-1)

    if self.training:
      # roll the noisebands randomly to avoid overfitting to the noise values
      # check whether model is training
      noisebands = torch.roll(noisebands, shifts=int(torch.randint(0, noisebands.shape[-1], size=(1,))), dims=-1)

    # fit the noisebands into the mplitudes
    repeats = math.ceil(upsampled_amplitudes.shape[-1] / noisebands.shape[-1])
    looped_bands = noisebands.repeat(1, repeats) # repeat
    looped_bands = looped_bands[:, :upsampled_amplitudes.shape[-1]] # trim
    looped_bands = looped_bands.to(upsampled_amplitudes.device, dtype=torch.float32)

    # Save the noisebands shift for the next iteration
    self._noisebands_shift = (self._noisebands_shift + upsampled_amplitudes.shape[-1]) % self._noisebands.shape[-1]

    # synthesize the signal
    signal = torch.sum(upsampled_amplitudes * looped_bands, dim=1, keepdim=True)
    return signal


  @property
  def _noisebands(self):
    """Delegate the noisebands to the filterbank object."""
    return self._filterbank.noisebands


  @torch.jit.ignore
  def _construct_mrstft_loss(self):
    """
    Construct the loss function for the model: a multi-resolution STFT loss
    """
    fft_sizes = np.array([8192, 4096, 2048, 1024, 512, 128, 32])
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[8192, 4096, 2048, 1024, 512, 128, 32],
                                                hop_sizes=fft_sizes//4,
                                                win_lengths=fft_sizes)

