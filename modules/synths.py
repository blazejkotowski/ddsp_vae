from torch import nn
import torch
import numpy as np
import torchaudio
import torch.nn.functional as F

class SineSynth(nn.Module):
  """
  Mixture of sinweaves synthesiser.

  Arguments:
    - fs: int, the sampling rate of the input signal
    - n_sines: int, the number of sinewaves to synthesise
    - resampling_factor: int, the internal up / down sampling factor for the sinewaves
  """
  def __init__(self,
               fs: int = 44100,
               n_sines: int = 1000,
               resampling_factor: int = 32):
    super().__init__()
    self.fs = fs
    self.n_sines = n_sines
    self.resampling_factor = resampling_factor
    self.phases = None

  def generate(self, frequencies: torch.Tensor, amplitudes: torch.Tensor):
    """
    Generates a mixture of sinewaves with the given frequencies and amplitudes per sample.

    Arguments:
      - frequencies: torch.Tensor[batch_size, n_sines, n_samples], the frequencies of the sinewaves
      - amplitudes: torch.Tensor[batch_size, n_sines, n_samples], the amplitudes of the sinewaves
    """
    batch_size = frequencies.shape[0]
    if self.phases is None:
      self.phases = torch.zeros(batch_size, self.n_sines)

    # Upsample from the internal sampling rate to the target sampling rate
    frequencies = F.interpolate(frequencies, scale_factor=self.resampling_factor, mode='linear')
    amplitudes = F.interpolate(amplitudes, scale_factor=self.resampling_factor, mode='linear')

    # Calculate the phase increments
    omegas = frequencies * 2 * np.pi / self.fs

    # Calculate the phases at points
    phases = torch.cumsum(omegas, axis=-1)
    phases = (phases.permute(2, 0, 1) + self.phases).permute(1, 2, 0)

    # Copy the last phases to maintain continuity
    self.phases.copy_(phases[: ,: , -1])

    # Generate the sinewaves
    sines = amplitudes * torch.sin(phases)

    # Sum the sinewaves
    signal = torch.sum(sines, dim=1, keepdim=True)
    return signal


  def _test(self, batch_size: int = 1, n_changes: int = 5, duration: float = 0.5, audiofile: str = 'sinewaves.wav'):
    # Generate a test signal of randomised sine frequencies and amplitudes
    freqs = torch.rand(batch_size, self.n_sines, n_changes) * 5000 + 40
    amps = torch.rand(batch_size, self.n_sines, n_changes) / self.n_sines

    freqs = F.interpolate(freqs, scale_factor=self.fs*duration*n_changes/self.resampling_factor, mode='nearest')
    amps = F.interpolate(amps, scale_factor=self.fs*duration*n_changes/self.resampling_factor, mode='nearest')

    signal = self.generate(freqs, amps)

    batch_size = signal.shape[0]
    for i in range(batch_size):
      torchaudio.save(f"{i}-{audiofile}", signal[i], self.fs)
