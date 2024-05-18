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

from modules.synths import SineSynth

from typing import List, Tuple, Optional

class NoiseBandNet(L.LightningModule):
  """
  A neural network that learns how to resynthesise signal, predicting amplitudes of
  precalculated, loopable noise bands.

  Args:
    - m_filters: int, the number of filters in the filterbank
    - latent_size: int, number of latent dimensions
    - samplerate : int, the sampling rate of the input signal
    - encoder_ratios: List[int], the capacity ratios for encoder layers
    - decoder_ratios: List[int], the capacity ratios for decoder layers
    - capacity: int, the capacity of the model
    - resampling_factor: int, internal up / down sampling factor for control signal and noisebands
    - learning_rate: float, the learning rate for the optimizer
    - torch_device: str, the device to run the model on
    - streaming: bool, whether to run the model in streaming mode
    - kld_weight: float, the weight for the KLD loss
  """
  def __init__(self,
               m_filters: int = 2048,
               n_sines: int = 500,
               samplerate: int = 44100,
               encoder_ratios: List[int] = [8, 4, 2],
               decoder_ratios: List[int] = [2, 4, 8],
               capacity: int = 64,
               latent_size: int = 16,
               resampling_factor: int = 32,
               learning_rate: float = 1e-3,
               torch_device: str = 'cpu',
               streaming: bool = False,
               kld_weight: float = 0.00025):
    super().__init__()
    # Save hyperparameters in the checkpoints
    self.save_hyperparameters()

    # Noise bands filterbank
    self._filterbank = FilterBank(
      m_filters=m_filters,
      fs=samplerate
    )

    # Sine synthesiser
    self._sine_synth = SineSynth(fs=samplerate, n_sines=n_sines, streaming=streaming)

    self.resampling_factor = resampling_factor
    self.latent_size = latent_size
    self.samplerate = samplerate
    self.beta = 0
    self.kld_weight = kld_weight

    self._encoder_ratios = encoder_ratios
    self._decoder_ratios = decoder_ratios
    self._capacity = capacity
    self._torch_device = torch_device
    self._noisebands_shift = 0

    # Define the neural network
    ## Encoder to extract latents from the input audio signal
    self.encoder = VariationalEncoder(
      layer_sizes=(np.array(encoder_ratios)*capacity).tolist(),
      sample_rate=samplerate,
      latent_size=latent_size,
      streaming=streaming,
    )

    ## Decoder to predict the amplitudes of the noise bands
    self.decoder = VariationalDecoder(
      latent_size=latent_size,
      layer_sizes=(np.array(decoder_ratios)*capacity).tolist(),
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
    z, _ = self.encoder.reparametrize_alter(mu, logvar)

    # predict the amplitudes of the noise bands
    synth_params = self.decoder(z)

    # synthesize the signal
    signal = self._synthesize(*synth_params)
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
    # mu, logvar = self.encoder(x_audio)
    # z = self.encoder.reparametrize(mu, logvar)

    mu, scale = self.encoder(x_audio)
    z, kld_loss = self.encoder.reparametrize_alter(mu, scale)

    # predict the amplitudes of the noise bands
    synth_params = self.decoder(z)

    # synthesize the signal
    y_audio = self._synthesize(*synth_params)

    # Compute the reconstruction loss
    recons_loss = self.recons_loss(y_audio, x_audio)

    # Compute the KLD loss
    # kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

    # Compute the total loss using β parameter
    loss = recons_loss + self.kld_weight * self.beta * kld_loss

    self.log("recons_loss", recons_loss, prog_bar=True, logger=True)
    self.log("kld_loss", self.beta*kld_loss, prog_bar=True, logger=True)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    self.log("beta", self.beta, prog_bar=True, logger=True)
    return loss

  # TODO: Generate the validationa audio and add to tensorboard
  # def on_validation_epoch_end(self, )


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self._learning_rate)


  def _synthesize(self, noiseband_amps: torch.Tensor, sine_freqs: torch.Tensor, sine_amps: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
      - noiseband_amps: torch.Tensor[batch_size, n_bands, sig_length], the predicted amplitudes of the noise bands
      - sine_freqs: torch.Tensor[batch_size, n_sines, sig_length], the predicted frequencies of the sines
      - sine_amps: torch.Tensor[batch_size, n_sines, sig_length], the predicted amplitudes of the sines
    Returns:
      - signal: torch.Tensor[batch_size, sig_length], the synthesized signal
    """
    noisebands = self._synthesize_noisebands(noiseband_amps)
    sines = self._sine_synth.generate(sine_freqs, sine_amps)
    return torch.sum(torch.hstack([noisebands, sines]), dim=1, keepdim=True)


  def _synthesize_noisebands(self, amplitudes: torch.Tensor) -> torch.Tensor:
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

