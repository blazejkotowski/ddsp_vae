import torch
from torch.utils.data import Dataset
import librosa as li
import os
from glob import glob

class AudioDataset(Dataset):
  def __init__(self, dataset_path: str, n_signal: int, sampling_rate: int = 44100):
    """
    Arguments:
      - dataset_path: str, the path to the dataset
      - n_signal: int, the size of the audio chunks, in samples
      - sampling_rate: int, the sampling rate of the audio
    """
    self._n_signal = n_signal
    self._sampling_rate = sampling_rate

    self._audio = self._load_dataset(dataset_path)
    self._dataset_length = len(self._audio) // self._n_signal

  def __len__(self):
    return self._dataset_length

  def __getitem__(self, idx):
    idx = idx % self._dataset_length

    sample_start = idx * self._n_signal
    sample_end = sample_start + self._n_signal

    if sample_end > len(self._audio):
      audio = torch.cat([self._audio[sample_start:], self._audio[:sample_end - len(self._audio)]])
    else:
      audio = self._audio[sample_start:sample_end]

    return audio

  def _load_dataset(self, path: str) -> torch.Tensor:
    """
    Load and concat entire dataset into single tensor.

    Arguments:
      - path: str, path to the dataset
    Returns:
      - audio: torch.Tensor, the audio tensor
    """
    audio = torch.tensor([])
    for filepath in glob(os.path.join(path, '**', '*.wav'), recursive=True):
      x = self._load_file(filepath)
      audio = torch.concat([audio, torch.from_numpy(x)], dim = 0)
    return audio

  def _load_file(self, path: str):
    """
    Load an audio file from a path.
    """
    audio, _ = li.load(path, sr = self._sampling_rate, mono = True)
    return audio


