from torch import nn
import torch
import numpy as np
import scipy.signal as signal

from typing import List, Any, Optional

class FilterBank(nn.Module):
  """
  A bank of filters defined by a set of parameters.
  The bank consists of a lowpass filter, a specified number of bandpass filters and a highpass filter.

  Args:
    - n_filters: int, the number of filters in the bank
    - lin_filters_ratio: float, the ratio of linear filters covering the lower frequency range
    - lin_filters_cutoff_ratio: float, the ratio of the lower frequency range covered by the linear filters
    - lowpass_cutoff: float, the cutoff frequency of the lowpass filter
    - transition_width: float, the width of the transition band of the kaiser filters
    - stopband_attenuation: float, the attenuation in the stopband of the kaiser filters, in dB
    - fs: int, the sampling frequency of the input signal
  """

  def __init__(self,
               n_filters: int = 2048,
               lin_filters_ratio: float = 1/4,
               lin_filters_cutoff_ratio: float = 1/8,
               lowpass_cutoff: float = 20,
               transition_width: float = 0.2,
               stopband_attenuation: float = 50.0,
               max_freq: Optional[float] = None,
               fs: int = 44100,
               device='cuda'):
    super().__init__()
    self._n_filters = n_filters - 2 # lowpass and highpass filters are added in default at the limits of the spectrum
    self._stopband_attenuation = stopband_attenuation
    self._fs = fs
    self._transition_width = transition_width
    self._lin_filters_ratio = lin_filters_ratio
    self._lin_filters_cutoff_ratio = lin_filters_cutoff_ratio
    self._lowpass_cutoff = lowpass_cutoff
    self._max_freq = max_freq

    self._filters = self._build_filterbank()

    self.noisebands = torch.from_numpy(np.array(self._bake_noisebands())).to(device)


  def _build_filterbank(self):
    """
    Builds the filterbank
    """
    # Nyquist frequency
    nyqfreq = self._fs/2
    if self._max_freq is not None:
      nyqfreq = self._max_freq

    # Compute the linear filter bands
    linear_filters_number = int(self._n_filters * self._lin_filters_ratio)
    linear_cutoff_freq = int(self._lin_filters_cutoff_ratio * nyqfreq)
    lin_boundaries = np.linspace(self._lowpass_cutoff, linear_cutoff_freq, linear_filters_number+1)

    # Compute the logarithmic filter bands, omitting the last boundary since it's present in the linear bands
    # endpoint set to False since we don't want the nyquist frequency to be included in bandpass filters
    log_filters_number = self._n_filters - linear_filters_number
    log_boundaries = np.geomspace(linear_cutoff_freq, nyqfreq, log_filters_number+1, endpoint=False)[1:]

    # Create the bands
    boundaries = np.concatenate((lin_boundaries, log_boundaries))
    bands = np.column_stack((boundaries[:-1], boundaries[1:]))

    # Construct the filterbank
    filters = [self._make_filter(bands[0,0], 'lowpass')] # first filter - lowpass
    filters.extend([self._make_filter(band, 'bandpass') for band in bands]) # middle filters - bandpass
    filters.append(self._make_filter(bands[-1,1], 'highpass')) # last filter - highpass

    return filters


  def _bake_noisebands(self) -> List[np.ndarray]:
    """
    "Bakes" the noise, meaning: creates loopable white noise filtered by the filterbank.
    """
    noisebands = []
    maxlen = np.max([len(h) for h in self._filters]).astype(int)
    n_signal = np.power(2, np.ceil(np.log2(maxlen))).astype(int)

    np.random.seed(666) # deterministic seed

    for h in self._filters:
      # pad h with the zeros from left
      h = np.pad(h, (0, n_signal-len(h)))

      # center the filter around 0
      h = np.roll(h, len(h)//2)

      # calcualte fft of the filter to get the frequency response
      bigH = np.fft.rfft(h)

      # get the mangnitude only
      magH = np.abs(bigH)

      # generate filtered white noise by randomising the phase of the frequency response
      # the phase needs to be 0 at the beginning and the end to make signal loopable
      phase = np.exp(1j*np.pi*np.random.uniform(-1, 1, len(magH)))
      phase[0] = phase[-1] = 0
      bigY = magH * phase

      # inverse fft to get the time domain signal
      y = np.fft.irfft(bigY)

      # append to noisebank
      noisebands.append(y)

    # normalize the amplitude
    max_sample = np.max([np.max(np.abs(y)) for y in noisebands])
    noisebands = [y/max_sample for y in noisebands]

    return noisebands


  def _make_filter(self, band: Any, typ: str = 'bandpass') -> np.ndarray:
    """
    Creates a filter with a given band
    - band: List[int, int] for the bandpass filter and int for lowpass and highpass
    - typ: str, the type of the filter, 'bandpass', 'lowpass', 'highpass'
    """
    # Calcualte bandwidth depending on the type of the filter
    if typ == 'bandpass':
      bandwidth = (band[1] - band[0])
    elif typ == 'lowpass':
      bandwidth = band
    elif typ == 'highpass':
      bandwidth = self._fs/2 - band

    # calculate the filter order and beta parameter
    width = (bandwidth/(self._fs/2)) * self._transition_width
    numtaps, beta = signal.kaiserord(ripple=self._stopband_attenuation, width=width)
    # make the numtapps odd
    if numtaps % 2 == 0:
      numtaps += 1

    # create the filter
    pass_zero = False # bandpass filter
    if typ == 'lowpass':
      pass_zero = True

    # return the filter (h)
    return signal.firwin(numtaps=numtaps, cutoff=band, window=('kaiser', beta), scale=True, fs=self._fs, pass_zero=pass_zero)


