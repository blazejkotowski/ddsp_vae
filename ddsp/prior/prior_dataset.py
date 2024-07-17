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

    audio = self._load_audio_dataset(audio_dataset_path)
    self._encodings = self._normalize(self._encode_audio_dataset(audio))


  def __len__(self) -> int:
    return len(self._encodings) - self._sequence_length - 1


  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the latent code for a given index.
    Arguments:
      - idx: int, the index of the audio
    Returns:
      - x: torch.Tensor[n_frames, n_latents], the preceding latent code sequence
      - y: torch.Tensor[n_latents], the target latent code
    """

    encoding = self._encodings[idx:idx+self._sequence_length+1]
    return encoding


  def _normalize(self, x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the latent codes.
    Arguments:
      - x: torch.Tensor, the latent codes
    Returns:
      - x: torch.Tensor, the normalized latent codes
    """
    breakpoint()
    return (x - x.mean(dim = 0)) / x.var(dim = 0)

  def _encode_audio_dataset(self, audio):
    """
    Encode the entire audio dataset into latent codes.
    Arguments:
      - audio: torch.Tensor, the audio tensor
      - sequence_length: int, the length of the preceding latent code sequence, in samples
    Returns:
      - encodings: Dict[int, torch.Tensor], the encoded latent codes
    """
    print("Encoding audio dataset...")
    chunk_length = 44100 # 1 s
    encodings = torch.tensor([], dtype=torch.float32, device=self._device)

    for i in range(len(audio) // chunk_length + 1):
      sample_start = i * chunk_length
      sample_end = sample_start + chunk_length
      audio_sample = audio[sample_start:sample_end]
      with torch.no_grad():
        mu, scale = self._encoder(audio_sample.unsqueeze(0))
        mu_scale = torch.cat([mu, scale], dim = -1).squeeze(0)
        encodings = torch.cat([encodings, mu_scale], dim = 0)

    return encodings


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
    return audio.to(self._device)


  def _load_audio_file(self, path: str):
    """
    Load an audio file from a path.
    """
    audio, _ = li.load(path, sr = self._sampling_rate, mono = True)
    return audio

