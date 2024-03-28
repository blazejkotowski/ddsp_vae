import torchaudio
import torch
import argparse
import os

import torch.nn.functional as F

from modules import NoiseBandNet

from typing import List, Tuple

def random_control_params(audio_duration: float,
                          param_ranges: List[Tuple[float,float]],
                          fs: int,
                          resampling_factor: int = 32) -> List[torch.Tensor]:
  """
  Generates random control parameters sampling from normal distribution

  Args:
    audio_duration: float
      The duration of the generated audio in seconds
    param_ranges: List[Tuple[float, float]]
      A list of tuples containing the minimum and maximum values for each control parameter
    fs: int
      Sampling rate
    resmapling_factor: int
      Resampling factor of the NoiseBandNet

  Returns:
    control_params: List[torch.Tensor[1, signal_length, 1]]
      A list of control parameters
  """
  control_params = []
  for param_range in param_ranges:
    n_samples = int(fs*audio_duration)
    param = torch.randn(1, 1, n_samples) * (param_range[1] - param_range[0]) + param_range[0]
    param = F.interpolate(param, scale_factor=1/resampling_factor, mode='linear')
    control_params.append(param)

  return control_params


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_checkpoint', type=str, help='Path to the model checkpoint')
  parser.add_argument('--audio_duration', type=float, help='Duration of the generation in seconds')
  parser.add_argument('--fs', type=int, default=44100, help='Sampling rate')
  parser.add_argument('--save_path', type=str, help='Output filename')
  config = parser.parse_args()


  control_params = random_control_params(
    audio_duration = config.audio_duration,
    fs = config.fs,
    param_ranges=[
      (0.071135, 0.16186), # loudness
      (312.75, 7570.21) # spectral centroid
    ]
  )

  nbn = NoiseBandNet.load_from_checkpoint(config.model_checkpoint)
  audio = nbn(control_params).squeeze(0).detach()

  torchaudio.save(config.save_path, audio, config.fs)




