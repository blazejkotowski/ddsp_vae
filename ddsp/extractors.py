import torch
import librosa

import torch.nn.functional as F

from essentia.standard import TensorflowPredictVGGish, TensorflowPredict2D


class BaseExtractor(object):
  """
  Base class for feature extractors
  """

  def __init__(self, resampling_factor: int):
    """
    Args:
      - resampling_factor: int, the factor to resample the extracted features
    """
    self._resampling_factor = resampling_factor


  def __call__(self, audio: torch.Tensor, *args) -> torch.Tensor:
    """
    Args:
      - audio: torch.Tensor[batch_size, n_samples], the input audio tensor
      - args: Tuple, additional arguments
    Returns:
      - features: torch.Tensor[batch_size, n_samples, n_features], the extracted features
    """
    features = self._calculate(audio, *args)
    return F.interpolate(features.unsqueeze(1), scale_factor=float(1/self._resampling_factor), mode='linear').squeeze(1)


  def _calculate(self, audio: torch.Tensor, *args) -> torch.Tensor:
    """
    Implementation of the feature extractor
    """
    raise NotImplementedError



class PitchExtractor(BaseExtractor):
  """
  Extracts the pitch from an audio signal
  """

  def __init__(self, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'), *args, **kwargs):
    super(PitchExtractor, self).__init__(*args, **kwargs)

    self._fmin = fmin
    self._fmax = fmax

  def _calculate(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the pitch extractor with librosa. Extracts the pitch in Hz and returns
    and interpolated tensor with values for each audio sample.

    Args:
      - audio: torch.Tensor[batch_size, n_samples], the input audio tensor
    Returns:
      - pitches: torch.Tensor[batch_size, n_samples, 1], the extracted pitch
    """
    # Processes the batch of audio parallely
    pitches = librosa.yin(audio.numpy(), fmin=self._fmin, fmax=self._fmax)
    pitches = torch.tensor(pitches, dtype=torch.float32)
    pitches = F.interpolate(pitches.unsqueeze(0), size=audio.shape[-1], mode='linear').squeeze(0)

    return pitches.unsqueeze(-1)


class SpectralCentroidExtractor(BaseExtractor):
  """
  Extracts the spectral centroid from an audio signal
  """
  def __init__(self, *args, **kwargs):
    super(SpectralCentroidExtractor, self).__init__(*args, **kwargs)

  def _calculate(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the spectral centroid extractor with librosa. Extracts the spectral centroid
    and returns and interpolated tensor with values for each audio sample.

    Args:
      - audio: torch.Tensor[batch_size, n_samples], the input audio tensor
    Returns:
      - spectral_centroid: torch.Tensor[batch_size, n_samples, 1], the extracted spectral centroid
    """
    # Processes the batch of audio parallely
    spectral_centroid = librosa.feature.spectral_centroid(y=audio.squeeze().numpy())
    spectral_centroid = torch.tensor(spectral_centroid, dtype=torch.float32)
    spectral_centroid = F.interpolate(spectral_centroid, size=audio.shape[-1], mode='linear')

    return spectral_centroid.squeeze()


class LoudnessExtractor(BaseExtractor):
  """
  Extracts the loudness from an audio signal
  """
  def __init__(self, *args, **kwargs):
    super(LoudnessExtractor, self).__init__(*args, **kwargs)

  def _calculate(self, audio: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the loudness extractor with librosa. Extracts the loudness
    and returns and interpolated tensor with values for each audio sample.

    Args:
      - audio: torch.Tensor[batch_size, n_samples], the input audio tensor
    Returns:
      - loudness: torch.Tensor[batch_size, n_samples, 1], the extracted loudness
    """
    # Processes the batch of audio parallely
    loudness = librosa.feature.rms(audio.numpy())
    loudness = torch.tensor(loudness, dtype=torch.float32)
    loudness = F.interpolate(loudness.unsqueeze(0), size=audio.shape[-1], mode='linear').squeeze(0)

    return loudness.unsqueeze(-1)

class ValenceArousalExtractor(BaseExtractor):
  """
  Extracts emotional valence and arousal with the use of essentia models
  """

  def _calculate(self, audio: torch.Tensor) -> torch.Tensor:
    embedding_model = TensorflowPredict2D(graphFilename='../vendor_models/vggish/audioset-vggish-3.pb', output="model/vggish/embeddings")
    embeddings = embedding_model(audio)

    model = TensorflowPredict2D(graphFilename="../vendor_models/emomusic-audioset-vggish-2.pb", output="model/Identity")
    pred = model(embeddings)

    breakpoint()

    return F.interpolate(pred, size=audio.shape[-1], mode='linear').squeeze(0)
