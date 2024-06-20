import lightning as L
import torch

import numpy as np
import auraloss

from ddsp.blocks import VariationalEncoder, Decoder
from ddsp.synths import SineSynth, NoiseBandSynth
from ddsp.extractors import SpectralCentroidExtractor, LoudnessExtractor # ValenceArousalExtractor

from typing import List, Tuple, Dict

class DDSP(L.LightningModule):
  """
  A neural network that learns how to resynthesise signal, predicting amplitudes of
  precalculated, loopable noise bands.

  Args:
    - n_filters: int, the number of filters in the filterbank
    - n_sines: int, the number of sines to synthesise
    - latent_size: int, number of latent dimensions
    - fs : int, the sampling rate of the input signal
    - encoder_ratios: List[int], the capacity ratios for encoder layers
    - decoder_ratios: List[int], the capacity ratios for decoder layers
    - capacity: int, the capacity of the model
    - resampling_factor: int, internal up / down sampling factor for control signal and noisebands
    - learning_rate: float, the learning rate for the optimizer
    - streaming: bool, whether to run the model in streaming mode
    - kld_weight: float, the weight for the KLD loss
  """
  def __init__(self,
               n_filters: int = 2048,
               n_sines: int = 500,
               latent_size: int = 16,
               fs: int = 44100,
               encoder_ratios: List[int] = [8, 4, 2],
               decoder_ratios: List[int] = [2, 4, 8],
               capacity: int = 64,
               resampling_factor: int = 32,
               learning_rate: float = 1e-3,
               kld_weight: float = 0.001,
               ar_weight: float = 1,
               streaming: bool = False):
    super().__init__()
    # Save hyperparameters in the checkpoints
    self.save_hyperparameters()
    self.fs = fs
    self.latent_size = latent_size
    self.resampling_factor = resampling_factor

    # Noisebands synthesiserg
    self._noisebands_synth = NoiseBandSynth(n_filters=n_filters, fs=fs, resampling_factor=self.resampling_factor)

    # Sine synthesiser
    self._sine_synth = SineSynth(n_sines=n_sines, fs=fs, resampling_factor=self.resampling_factor, streaming=streaming)

    # ELBO regularization params
    self._beta = 0
    self._kld_weight = kld_weight

    # Define the neural network
    ## Encoder to extract latents from the input audio signal
    self.encoder = VariationalEncoder(
      layer_sizes=(np.array(encoder_ratios)*capacity).tolist(),
      sample_rate=fs,
      latent_size=latent_size,
      streaming=streaming,
    )

    ## Decoder to predict the amplitudes of the noise bands
    self.decoder = Decoder(
      latent_size=latent_size,
      layer_sizes=(np.array(decoder_ratios)*capacity).tolist(),
      n_bands=n_filters,
      streaming=streaming,
      n_sines=n_sines,
    )

    # Define the loss
    self._recons_loss = self._construct_mrstft_loss()

    # Learning rate
    self._learning_rate = learning_rate

    # Validation inputs and outputs
    self._last_validation_in = None
    self._last_validation_out = None
    self._validation_index = 1

    # Attribute regularization
    self._ar_feature_extractors = [LoudnessExtractor(resampling_factor=self.resampling_factor),
                                SpectralCentroidExtractor(resampling_factor=self.resampling_factor)]

    self._ar_weight = ar_weight


  def forward(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the network.
    Args:
      - audio: torch.Tensor[batch_size, n_signal], the input audio signal
    Returns:
      - signal: torch.Tensor, the synthesized signal
    """
    # signal, _ = self._autoencode(audio, compute_loss=False)
    mu, scale = self.encoder(audio)

    # Reparametrization trick
    z, _ = self.encoder.reparametrize(mu, scale)

    # Predict the parameters of the synthesiser and synthesize
    synth_params = self.decoder(z)
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
    _, losses = self._autoencode(x_audio)

    self.log("recons_loss", losses["recons_loss"], prog_bar=True, logger=True)
    self.log("kld_loss", losses["kld_loss"], prog_bar=True, logger=True)
    self.log("train_loss", losses["loss"], prog_bar=True, logger=True)
    self.log("ar_loss", losses["ar_loss"], prog_bar=True, logger=True)
    self.log("beta", self._beta, prog_bar=True, logger=True)

    return losses["loss"]


  def validation_step(self, x_audio: torch.Tensor, batch_idx: int) -> torch.Tensor:
    """Compute the loss for validation data"""
    y_audio, losses = self._autoencode(x_audio)

    loss = losses["ref_loss"]

    self.log("val_loss", loss, prog_bar=False, logger=True)

    if self._last_validation_in is None:
      self._last_validation_in = x_audio
      self._last_validation_out = y_audio.squeeze(1)

    return y_audio


  def _autoencode(self, x_audio: torch.Tensor, compute_loss: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Autoencode the audio signal
    Args:
      - x_audio: torch.Tensor[batch_size, n_signal], the input audio signal
      - compute_loss: bool, whether to compute the loss
    Returns:
      - y_audio: torch.Tensor[batch_size, n_signal], the autoencoded audio signal
      - losses: Dict[str, torch.Tensor], the losses computed during the autoencoding
    """
    # Initialise for torchscript
    kld_loss = torch.zeros(1)
    ar_loss = torch.zeros(1)
    recons_loss = torch.zeros(1)
    loss = torch.zeros(1)

    # Encode the audio signal
    mu, scale = self.encoder(x_audio)

    # Reparametrization trick
    z, kld_loss = self.encoder.reparametrize(mu, scale)

    # Predict the parameters of the synthesiser
    synth_params = self.decoder(z)

    # Synthesize the output signal
    y_audio = self._synthesize(*synth_params)

    if compute_loss:
      # Compute the reconstruction loss
      recons_loss = self._recons_loss(y_audio, x_audio)

      # Compute the argument regularization loss
      ar_loss = self._ar_weight * self._attribute_regularization(z, y_audio)

      # Reference loss disregarding the β parameter
      ref_loss = recons_loss + self._kld_weight * kld_loss + ar_loss

      # Compute the total loss with β parameter
      loss = recons_loss + self._kld_weight * self._beta * kld_loss + ar_loss

    # Construct losses dictionary
    losses = {
      "recons_loss": recons_loss,
      "kld_loss": self._kld_weight * kld_loss,
      "ar_loss": ar_loss,
      "ref_loss": ref_loss,
      "loss": loss
    }

    return y_audio, losses



  def on_validation_epoch_end(self):
    """At the end of the validation epoch, log the validation audio"""
    if self._last_validation_out is not None:
      device = self._last_validation_out.device
    else:
      device = 'cpu'

    audio = torch.FloatTensor(0).to(device) # Concatenated audio
    silence = torch.zeros(1, int(self.fs/2)).to(device) # 0.5s silence
    for input, output in zip(self._last_validation_in, self._last_validation_out):
      audio = torch.cat((audio, input.unsqueeze(0), silence, output.unsqueeze(0), silence.repeat(1, 3)), dim=-1)

    audio = audio.clip_(-1, 1) # Clip the audio to stay in range
    self.logger.experiment.add_audio("audio_validation", audio, self._validation_index, self.fs)

    self._last_validation_in = None
    self._last_validation_out = None
    self._validation_index += 1


  def configure_optimizers(self):
    """Configure the optimizer for the model"""
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
    sines = self._sine_synth(sine_freqs, sine_amps)
    noisebands = self._noisebands_synth(noiseband_amps)
    return torch.sum(torch.hstack([noisebands, sines]), dim=1, keepdim=True)

  @torch.jit.ignore
  def _attribute_regularization(self, z: torch.Tensor, y_audio: torch.Tensor) -> torch.Tensor:
    """
    Compute the attribute regularization loss.

    source: Pati, A. Lerch  A, 2020, "Attribute-based Regularization of Latent
    Spaces for Variational Auto-Encoders" https://arxiv.org/abs/2004.05485

    Args:
      - z: torch.Tensor[batch_size, latent_size], the latent variables
      - y_audio: torch.Tensor[batch_size, n_signal], the output audio signal
    Returns:
      - arg_reg_loss: torch.Tensor[1], the argument regularization loss
    """
    # Initial loss
    loss = torch.zeros(1, device = z.device)

    # Ignore if the weight is 0 or no feature extractors are defined
    if self._ar_weight == 0 or len(self._ar_feature_extractors) == 0:
      return loss

    # MAE Attribute regularization loss
    mae = lambda latent_distance, attribute_distance: torch.mean(torch.abs(torch.tanh(latent_distance) - torch.sign(attribute_distance)))
    # self-distance matrix between the pair of values in a 1d tensor
    distance_matrix = lambda x : x.unsqueeze(0) - x.unsqueeze(1)

    # Calculate the attributes of the audio signal
    attributes = [extractor(y_audio.detach().cpu().squeeze()).to(y_audio) for extractor in self._ar_feature_extractors]

    # Regularize the latent dimensions according to attributes
    for dimension_id, attribute_values in enumerate(attributes):
      # distance for latent values
      latent_values = z[..., dimension_id]
      latent_distance = distance_matrix(latent_values)

      # distance for selected attribute values
      attribute_distance = distance_matrix(attribute_values)

      # Compute the mae loss between the latent distance and attribute distance
      loss = loss + mae(latent_distance, attribute_distance)

    # normalize loss
    loss /= len(attributes)

    return loss


  @torch.jit.ignore
  def _construct_mrstft_loss(self):
    """Construct the loss function for the model: a multi-resolution STFT loss"""
    fft_sizes = np.array([8192, 4096, 2048, 1024, 512, 128, 32])
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[8192, 4096, 2048, 1024, 512, 128, 32],
                                                hop_sizes=fft_sizes//4,
                                                win_lengths=fft_sizes)

