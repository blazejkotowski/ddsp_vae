import torch
import librosa
import torch.nn.functional as F

from .base_extractor import BaseExtractor
from .utils import normalize_feature, smoothen_feature

class LibrosaFeatureExtractor(BaseExtractor):
  """
  Extracts the loudness from an audio signal
  """
  FN_SPECTRAL_CENTROID = 'spectral_centroid'
  FN_ZERO_CROSSING = 'zero_crossing_rate'
  FN_LOUDNESS = 'rms'
  FN_SPECTRAL_FLATNESS = 'spectral_flatness'
  FN_SPECTRAL_ROLLOFF = 'spectral_rolloff'
  FN_SPECTRAL_BANDWIDTH = 'spectral_bandwidth'

  def __init__(self, feature_fn_name: str, smoothing_kernel_size: int = 257, postprocess: bool = False,  *args, **kwargs):
    """
    Args:
      - feature_fn_name: str, the name of the librosa feature extraction function to use
      - args: additional arguments for the base extractor
      - postprocess: bool, whether to apply postprocessing:
        > clamping between .05 and .95 percentiles,
        > normalization between [0, 1]
        > and smoothing to the extracted features
      - kwargs: additional keyword arguments for the base extractor
    """
    super(LibrosaFeatureExtractor, self).__init__(*args, **kwargs)
    self._feature_fn = getattr(librosa.feature, feature_fn_name)
    self._smoothing_kernel_size = smoothing_kernel_size
    self._postprocess = postprocess


  def _calculate(self, audio: torch.Tensor, *args) -> torch.Tensor:
    """
    Implementation of the loudness extractor with librosa. Extracts the loudness
    and returns and interpolated tensor with values for each audio sample.

    Args:
      - audio: torch.Tensor [T_audio] or [B, T_audio], the input audio tensor
    Returns:
      - features: torch.Tensor [T_audio, C] or [B, T_audio, C], the extracted features at audio rate
    """
    x = audio
    squeeze_back = False
    if x.ndim == 1:
      x = x.unsqueeze(0)  # [1, T]
      squeeze_back = True

    feats = []
    for i in range(x.shape[0]):
      xi = x[i].detach().cpu().numpy()
      fi = self._feature_fn(y=xi, center=False)
      # fi shape usually [C, frames] or [1, frames]
      # Force CPU: this one-time cache-build upsamples features to the FULL audio length, which for
      # long concatenated files (e.g. hours of VCTK = ~700M samples) OOMs the GPU under train.py's
      # global cuda default device. The result is cached, so CPU is correct (and downstream utils
      # inherit this tensor's device, keeping the whole chain off-GPU).
      fi = torch.tensor(fi, dtype=torch.float32, device='cpu')
      if fi.ndim == 1:
        fi = fi.unsqueeze(0)  # [1, frames]
      # interpolate to audio length along time
      fi = F.interpolate(fi.unsqueeze(0), size=x.shape[-1], mode='linear', align_corners=False).squeeze(0)  # [C, T]
      # postprocess per-channel
      fi = normalize_feature(fi, low=5.0, high=95.0, dim=-1)
      if self._smoothing_kernel_size > 1:
        fi = smoothen_feature(fi, window_size=self._smoothing_kernel_size)
      feats.append(fi.transpose(0, 1))  # [T, C]

    feat_batch = torch.stack(feats, dim=0)  # [B, T, C]
    if squeeze_back:
      feat_batch = feat_batch.squeeze(0)  # [T, C]
    return feat_batch
