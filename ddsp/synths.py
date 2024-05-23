from torch import nn
import torch
import math
import torchaudio
import torch.nn.functional as F

from ddsp.filterbank import FilterBank

class BaseSynth(nn.Module):
  """
  Base class for synthesizers.

  Arguments:
    - fs: int, the sampling rate of the input signal
    - resampling_factor: int, the internal up / down sampling factor for the signal
  """
  def __init__(self, fs: int = 44100, resampling_factor: int = 32):
    super().__init__()
    self._fs = fs
    self._resampling_factor = resampling_factor

  def __call__(self, *args, **kwargs):
    raise NotImplementedError

class NoiseBandSynth(BaseSynth):
  """
  A synthesiser that generates a mixture noise bands from amplitudes.

  Arguments:
    - n_filters: int, the number of filters in the filterbank
    - fs: int, the sampling rate of the input signal
    - resampling_factor: int, the internal up / down sampling factor for the signal
  """

  def __init__(self, n_filters: int = 2048, fs: int = 44100, resampling_factor: int = 32):
    super().__init__()
    self._resampling_factor = resampling_factor


    self._filterbank = FilterBank(
      n_filters=n_filters,
      fs=fs
    )

    # Shift of the noisebands between inferences, to maintain continuity
    self._noisebands_shift = 0


  def __call__(self, amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
      - amplitudes: torch.Tensor[batch_size, n_bands, sig_length], the predicted amplitudes of the noise bands
    Returns:
      - signal: torch.Tensor[batch_size, sig_length], the synthesized signal
    """
    # upsample the amplitudes
    upsampled_amplitudes = F.interpolate(amplitudes, scale_factor=float(self._resampling_factor), mode='linear')

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

    # Synthesize the signal
    signal = torch.sum(upsampled_amplitudes * looped_bands, dim=1, keepdim=True)
    return signal

  @property
  def _noisebands(self):
    """Delegate the noisebands to the filterbank object."""
    return self._filterbank.noisebands

class SineSynth(BaseSynth):
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
    self._fs = fs
    self._n_sines = n_sines
    self._resampling_factor = resampling_factor
    self._phases = None
    self._streaming = streaming

  def __call__(self, frequencies: torch.Tensor, amplitudes: torch.Tensor):
    """
    Generates a mixture of sinewaves with the given frequencies and amplitudes per sample.

    Arguments:
      - frequencies: torch.Tensor[batch_size, n_sines, n_samples], the frequencies of the sinewaves
      - amplitudes: torch.Tensor[batch_size, n_sines, n_samples], the amplitudes of the sinewaves
    """
    batch_size = frequencies.shape[0]

    # We only need to initialise phases buffer if we are in streaming mode
    if self._streaming and (self._phases is None or self._phases.shape[0] != batch_size):
      self._phases = torch.zeros(batch_size, self._n_sines)

    # Upsample from the internal sampling rate to the target sampling rate
    frequencies = F.interpolate(frequencies, scale_factor=float(self._resampling_factor), mode='linear')
    amplitudes = F.interpolate(amplitudes, scale_factor=float(self._resampling_factor), mode='linear')

    # Calculate the phase increments
    omegas = frequencies * 2 * math.pi / self._fs

    # Calculate the phases at points, in place
    phases = omegas.cumsum_(dim=-1)
    phases = phases % (2 * math.pi)

    if self._streaming:
      # Shift the phases by the last phase from last generation
      # breakpoint()
      phases = (phases.permute(2, 0, 1) + self._phases).permute(1, 2, 0)

      # Copy the last phases for next iteration
      self._phases.copy_(phases[: ,: , -1] % (2 * math.pi))

    # Generate and sum the sinewaves
    signal = torch.sum(amplitudes * torch.sin(phases), dim=1, keepdim=True)
    return signal


  def _test(self, batch_size: int = 1, n_changes: int = 5, duration: float = 0.5, audiofile: str = 'sinewaves.wav'):
    # Generate a test signal of randomised sine frequencies and amplitudes
    freqs = torch.rand(batch_size, self._n_sines, n_changes) * 5000 + 40
    amps = torch.rand(batch_size, self._n_sines, n_changes) / self._n_sines

    freqs = F.interpolate(freqs, scale_factor=self._fs*duration/n_changes/self._resampling_factor, mode='nearest')
    amps = F.interpolate(amps, scale_factor=self._fs*duration/n_changes/self._resampling_factor, mode='nearest')

    freq_chunks = freqs.chunk(100, dim=-1)
    amp_chunks = amps.chunk(100, dim=-1)

    signal = torch.Tensor()
    for freq, amp in zip(freq_chunks, amp_chunks):
      signal = torch.cat((signal, self.generate(freq, amp)), dim=-1)

    batch_size = signal.shape[0]
    for i in range(batch_size):
      torchaudio.save(f"{i}-{audiofile}", signal[i], self._fs)
