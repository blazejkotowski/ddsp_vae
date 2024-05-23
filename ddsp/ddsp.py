import lightning as L
import torch

import numpy as np
import auraloss

from ddsp.blocks import VariationalEncoder, Decoder
from ddsp.synths import SineSynth, NoiseBandSynth

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
               kld_weight: float = 0.00025,
               streaming: bool = False):
    super().__init__()
    # Save hyperparameters in the checkpoints
    self.save_hyperparameters()
    self.fs = fs

    # Noisebands synthesiserg
    self._noisebands_synth = NoiseBandSynth(n_filters=n_filters, fs=fs, resampling_factor=resampling_factor)

    # Sine synthesiser
    self._sine_synth = SineSynth(n_sines=n_sines, fs=fs, resampling_factor=resampling_factor, streaming=streaming)

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
    self._validation_inputs = []
    self._validation_outputs = []
    self._validation_index = 1


  def forward(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the network.
    Args:
      - audio: torch.Tensor[batch_size, n_signal], the input audio signal
    Returns:
      - signal: torch.Tensor, the synthesized signal
    """
    signal, _ = self._autoencode(audio)
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
    self.log("beta", self._beta, prog_bar=True, logger=True)

    return losses["loss"]


  def validation_step(self, x_audio: torch.Tensor, batch_idx: int) -> torch.Tensor:
    """Compute the loss for validation data"""
    y_audio, losses = self._autoencode(x_audio)

    loss = losses["recons_loss"]

    self.log("val_loss", loss, prog_bar=False, logger=True)

    self._validation_inputs.append(x_audio)
    self._validation_outputs.append(y_audio.squeeze(1))

    return y_audio


  def _autoencode(self, x_audio: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Autoencode the audio signal
    Args:
      - x_audio: torch.Tensor[batch_size, n_signal], the input audio signal
    Returns:
      - y_audio: torch.Tensor[batch_size, n_signal], the autoencoded audio signal
      - losses: Dict[str, torch.Tensor], the losses computed during the autoencoding
    """
    # Encode the audio signal
    mu, scale = self.encoder(x_audio)

    # Reparametrization trick
    z, kld_loss = self.encoder.reparametrize(mu, scale)

    # Predict the parameters of the synthesiser
    synth_params = self.decoder(z)

    # Synthesize the output signal
    y_audio = self._synthesize(*synth_params)

    # Compute the reconstruction loss
    recons_loss = self._recons_loss(y_audio, x_audio)

    # Compute the total loss using β parameter
    loss = recons_loss + self._kld_weight * self._beta * kld_loss

    # Construct losses dictionary
    losses = {
      "recons_loss": recons_loss,
      "kld_loss": kld_loss,
      "loss": loss
    }

    return y_audio, losses


  def on_validation_epoch_end(self):
    """At the end of the validation epoch, log the validation audio"""
    if len(self._validation_outputs) > 0:
      device = self._validation_outputs[0].device
    else:
      device = 'cpu'

    audio = torch.FloatTensor(0, device=device) # Concatenated audio
    silence = torch.zeros(1, int(self.fs/2), device=device) # 0.5s silence
    for inputs, outputs in zip(self._validation_inputs, self._validation_outputs):
      for input, output in zip(inputs, outputs):
        audio = torch.cat((audio, input.unsqueeze(0), silence, output.unsqueeze(0), silence.repeat(1, 3)), dim=-1)

    self.logger.experiment.add_audio("audio_validation", audio, self._validation_index, self.fs)

    self._validation_index += 1
    self._validation_inputs.clear()
    self._validation_outputs.clear()


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
  def _construct_mrstft_loss(self):
    """Construct the loss function for the model: a multi-resolution STFT loss"""
    fft_sizes = np.array([8192, 4096, 2048, 1024, 512, 128, 32])
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[8192, 4096, 2048, 1024, 512, 128, 32],
                                                hop_sizes=fft_sizes//4,
                                                win_lengths=fft_sizes)

