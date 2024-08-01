from torch.utils.data import Dataset
from torch import jit
import torch

import librosa as li
from glob import glob
import os

from typing import Tuple, List

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

    audio_tensors = self._load_audio_dataset(audio_dataset_path)
    self._encodings = self._encode_audio_dataset(audio_tensors)

    # self._encodings, self.normalization_dict = self._normalize(encodings)


  def __len__(self) -> int:
    return len(self._encodings)


  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the latent code for a given index.
    Arguments:
      - idx: int, the index of the audio
    Returns:
      - x: torch.Tensor[n_frames, n_latents], the preceding latent code sequence
      - y: torch.Tensor[n_latents], the target latent code
    """
    return self._encodings[idx]


  def _normalize(self, x: List[torch.Tensor]) -> torch.Tensor:
    """
    Normalize the latent codes.
    Arguments:
      - x: List[torch.Tensor], the sequences of latent codes
    Returns:
      - x: torch.Tensor, the normalized latent codes
    """
    all_x = torch.cat(x, dim = 0)
    mean = all_x.mean(dim=0).detach().cpu().numpy()
    var = all_x.var(dim=0).detach().cpu().numpy()

    normalization_dict = {'mean': mean, 'var': var}

    # normalized = [torch.from_numpy((item.cpu().numpy() - mean) / var).to(self._device) for item in x]
    normalized = x

    return normalized, normalization_dict


  def _encode_audio_dataset(self, audio_tensors: List[torch.Tensor]):
    """
    Encode the entire audio dataset into latent codes.
    Arguments:
      - audio_tensors: List[torch.Tensor], the audio tensors
    Returns:
      - encodings: Dict[int, torch.Tensor], the encoded latent codes
    """
    print("Encoding audio dataset...")
    encodings = []

    for audio in audio_tensors:
      with torch.no_grad():
        mu, scale = self._encoder(audio.unsqueeze(0))
        mu_scale = torch.cat([mu, scale], dim = -1).squeeze(0)

        # mu_scale = mu.squeeze(0) # try only mu
        # mu_scale = mu_scale[..., :1] # try only one (the first) latent variable

        # Overlapping, shifting window chunks
        for i in range(mu_scale.size(0) - (self._sequence_length*2)):
          encodings.append(mu_scale[i:i+self._sequence_length*2])

        # # Non-overlapping chunks
        # for chunk in mu_scale.split(self._sequence_length+1):
        #   if chunk.size(0) == self._sequence_length+1:
        #     encodings.append(chunk)

    return encodings


  def _load_audio_dataset(self, path: str) -> List[torch.Tensor]:
    """
    Load and concat entire dataset into single tensor.

    Arguments:
      - path: str, path to the dataset
    Returns:
      - audio: torch.Tensor, the audio tensor
    """
    audio_tensors = []
    for filepath in glob(os.path.join(path, '**', '*.wav'), recursive=True):
      x = self._load_audio_file(filepath)
      audio = torch.from_numpy(x)
      if audio.size(0) >= self._resampling_factor:
        audio_tensors.append(audio.to(self._device))

    return audio_tensors


  def _load_audio_file(self, path: str):
    """
    Load an audio file from a path.
    """
    audio, _ = li.load(path, sr = self._sampling_rate, mono = True)
    return audio



import torch
from torch.utils.data import Dataset

class DummyMultivariateSequenceDataset(Dataset):
  def __init__(self, num_features, seq_length, n_examples):
    """
    Args:
        num_features (int): Number of features in each data point.
        seq_length (int): Length of each sequence.
        n_examples (int): Number of examples in the dataset.
    """
    self.num_features = num_features
    self.seq_length = seq_length
    self.n_examples = n_examples

    # Generate the synthetic data
    self.data = self._generate_data()

  def _generate_data(self):
    # Generate random data for features
    data = torch.randn(self.n_examples, self.seq_length + 1, self.num_features)

    # Ensure that the data is easy to predict: make the target (next step) a linear combination of the input
    for i in range(self.seq_length):
        data[:, i + 1, :] = data[:, i, :] * 0.5 + torch.randn(self.n_examples, self.num_features) * 0.1

    return data

  def __len__(self):
    return self.n_examples

  def __getitem__(self, idx):
    return self.data[idx, :self.seq_length +1, :]
