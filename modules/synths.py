from torch import nn
import torch
import math
import torchaudio
import torch.nn.functional as F

from modules.memory_optimizations import multiply_and_sum_tensors
class SineSynth(nn.Module):
  """
  Mixture of sinweaves synthesiser.

  Arguments:
    - fs: int, the sampling rate of the input signal
    - n_sines: int, the number of sinewaves to synthesise
    - resampling_factor: int, the internal up / down sampling factor for the sinewaves
    - streaming: bool, whether to run the model in streaming mode
  """
  def __init__(self,
               fs: int = 44100,
               n_sines: int = 1000,
               resampling_factor: int = 32,
               streaming: bool = False):
    super().__init__()
    self.fs = fs
    self.n_sines = n_sines
    self.resampling_factor = resampling_factor
    self.phases = None
    self.streaming = streaming

  def generate(self, frequencies: torch.Tensor, amplitudes: torch.Tensor):
    """
    Generates a mixture of sinewaves with the given frequencies and amplitudes per sample.

    Arguments:
      - frequencies: torch.Tensor[batch_size, n_sines, n_samples], the frequencies of the sinewaves
      - amplitudes: torch.Tensor[batch_size, n_sines, n_samples], the amplitudes of the sinewaves
    """
    batch_size = frequencies.shape[0]

    # We only need to initialise phases buffer if we are in streaming mode
    if self.streaming and (self.phases is None or self.phases.shape[0] != batch_size):
      self.phases = torch.zeros(batch_size, self.n_sines)

    # Upsample from the internal sampling rate to the target sampling rate
    frequencies = F.interpolate(frequencies, scale_factor=float(self.resampling_factor), mode='linear')
    amplitudes = F.interpolate(amplitudes, scale_factor=float(self.resampling_factor), mode='linear')

    # Calculate the phase increments
    omegas = frequencies * 2 * math.pi / self.fs

    # Calculate the phases at points, in place
    phases = omegas.cumsum_(dim=-1)
    phases = phases % (2 * math.pi)

    if self.streaming:
      # Shift the phases by the last phase from last generation
      # breakpoint()
      phases = (phases.permute(2, 0, 1) + self.phases).permute(1, 2, 0)

      # Copy the last phases for next iteration
      self.phases.copy_(phases[: ,: , -1] % (2 * math.pi))

    # Generate and sum the sinewaves
    signal = multiply_and_sum_tensors(amplitudes, torch.sin(phases))
    # signal = torch.sum(amplitudes * torch.sin(phases), dim=1, keepdim=True)
    return signal


  def _test(self, batch_size: int = 1, n_changes: int = 5, duration: float = 0.5, audiofile: str = 'sinewaves.wav'):
    # Generate a test signal of randomised sine frequencies and amplitudes
    freqs = torch.rand(batch_size, self.n_sines, n_changes) * 5000 + 40
    amps = torch.rand(batch_size, self.n_sines, n_changes) / self.n_sines

    freqs = F.interpolate(freqs, scale_factor=self.fs*duration/n_changes/self.resampling_factor, mode='nearest')
    amps = F.interpolate(amps, scale_factor=self.fs*duration/n_changes/self.resampling_factor, mode='nearest')

    freq_chunks = freqs.chunk(100, dim=-1)
    amp_chunks = amps.chunk(100, dim=-1)

    signal = torch.Tensor()
    for freq, amp in zip(freq_chunks, amp_chunks):
      signal = torch.cat((signal, self.generate(freq, amp)), dim=-1)

    batch_size = signal.shape[0]
    for i in range(batch_size):
      torchaudio.save(f"{i}-{audiofile}", signal[i], self.fs)
