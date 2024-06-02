from torch.utils.data import Dataset
from torch import jit
import torch

import librosa as li
from glob import glob
import os

from typing import Tuple

from ddsp import AudioDataset

torch.set_default_dtype(torch.float32)

class PriorDataset(Dataset):
  def __init__(self,
               encoding_model_path: str,
               audio_dataset_path: str,
               sequence_length: int,
               sampling_rate: int = 44100,
               device: str = None):
    """
    Arguments:
      - encoding_model_path: str, the path to the encoding model
      - audio_dataset_path: str, the path to the dataset
      - sequence_length: int, the length of the preceding latent code sequence, in samples
      - sampling_rate: int, the sampling rate of the audio
      - device: str, the device to use. None will use the originally saved device. [None, 'cuda', 'cpu'].
    """
    self._device = device
    self._sequence_length = sequence_length
    self._sampling_rate = sampling_rate

    vae_model = jit.load(encoding_model_path, map_location=device).pretrained.to(device)
    self._resampling_factor = vae_model.resampling_factor

    self._encoder = vae_model.encoder
    self._encoder.streaming = False

    self._audio = self._load_audio_dataset(audio_dataset_path)
    self._encodings = {}

  def __len__(self) -> int:
    return len(self._audio) - (self._resampling_factor * self._sequence_length)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the latent code for a given index.
    Arguments:
      - idx: int, the index of the audio
    Returns:
      - x: torch.Tensor[n_frames, n_latents], the preceding latent code sequence
      - y: torch.Tensor[n_latents], the target latent code
    """
    sample_start = idx
    sample_end = sample_start + (self._sequence_length * self._resampling_factor)
    audio_sample = self._audio[sample_start:sample_end]

    if idx not in self._encodings:
       with torch.no_grad():
        mu, scale = self._encoder(audio_sample.unsqueeze(0))
        mu_scale = torch.cat([mu, scale], dim = -1)
        # enc, _ = self._encoder.reparametrize(mu, scale)
        self._encodings[idx] = mu_scale.squeeze(0)

    encoding = self._encodings[idx]

    return encoding[:-1], encoding[-1]


  def _load_audio_dataset(self, path: str) -> torch.Tensor:
    """
    Load and concat entire dataset into single tensor.

    Arguments:
      - path: str, path to the dataset
    Returns:
      - audio: torch.Tensor, the audio tensor
    """
    audio = torch.tensor([])
    for filepath in glob(os.path.join(path, '**', '*.wav'), recursive=True):
      x = self._load_audio_file(filepath)
      audio = torch.concat([audio, torch.from_numpy(x)], dim = 0)
    return audio

  def _load_audio_file(self, path: str):
    """
    Load an audio file from a path.
    """
    audio, _ = li.load(path, sr = self._sampling_rate, mono = True)
    return audio

