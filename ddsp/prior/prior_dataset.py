from torch.utils.data import Dataset
from torch import jit
import torch

from typing import Tuple

from ddsp import AudioDataset


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
    self._sequence_length = sequence_length
    self._sampling_rate = sampling_rate

    self._audio_dataset = AudioDataset(audio_dataset_path, sequence_length + 1)
    self._encoder = jit.load(encoding_model_path, map_location=device).pretrained.encoder
    self._encoder.streaming = False

    self._encodings = {}

  def __len__(self) -> int:
    return len(self._audio_dataset)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the latent code for a given index.
    Arguments:
      - idx: int, the index of the audio
    Returns:
      - x: torch.Tensor[n_frames, n_latents], the preceding latent code sequence
      - y: torch.Tensor[n_latents], the target latent code
    """
    print(f"__getitem__: getting item at index {idx}")
    audio = self._audio_dataset[idx]
    audio = audio.unsqueeze(0)

    if idx not in self._encodings:
       mu, scale = self._encoder(audio)
       enc, _ = self._encoder.reparametrize(mu, scale)
       self._encodings[idx] = enc.squeeze(0)

    encoding = self._encodings[idx]

    print("__getitem__: Got the encoding")
    return encoding[:-1], encoding[-1]
