import torchaudio
import torch
import argparse
import random
import numpy as np

import torch.nn.functional as F

from modules import NoiseBandNet

from typing import List, Tuple

def random_walk(n) -> np.ndarray:
    current_value = random.uniform(0, 1)
    walk = [current_value]
    base_step_size = 1e-3
    sin_freq = 0.2
    fs=44100

    # generate step size curve
    # points = [random.uniform(2, 4) for _ in range(10)]
    # step_sizes = np.interp

    for i in range(1, n):
        step_size = base_step_size * (np.sin(i/fs*sin_freq*2*np.pi)+1)/2*5 # modulate the step size by a sinewave
        step = random.uniform(-1, 1) * step_size
        current_value += step

        # Bounce the value back if range is exceeded
        if current_value < 0:
            current_value = -current_value
        elif current_value > 1:
            current_value -= (current_value-1)*2

        walk.append(current_value)

    return np.array(walk, dtype=np.float32)


def random_control_params(audio_duration: float,
                          num_params: int,
                          fs: int,
                          resampling_factor: int = 32,
                          stereo: bool = False) -> List[torch.Tensor]:
  """
  Generates random control parameters sampling from normal distribution

  Args:
    audio_duration: float
      The duration of the generated audio in seconds
    num_params: int
      Numbear of control parameters
    fs: int
      Sampling rate
    resmapling_factor: int
      Resampling factor of the NoiseBandNet
    stereo: bool
      Whether to generate control params for 2 channels

  Returns:
    control_params: List[torch.Tensor[1, signal_length, 1]]
      A list of control parameters
  """
  control_params = []
  for _ in range(num_params):
    n_samples = int(fs*audio_duration)
    # param = torch.randn(1, 1, n_samples) * (param_range[1] - param_range[0]) + param_range[0] # Gaussian distribution
    # param = torch.FloatTensor(1, 1, n_samples).uniform_(param_range[0], param_range[1]) # Uniform distribution
    param = torch.from_numpy(random_walk(n_samples)).unsqueeze(0).unsqueeze(0) # Random walk
    param = F.interpolate(param, scale_factor=1/resampling_factor, mode='linear')
    if stereo:
       modifier = torch.from_numpy(random_walk(n_samples)).unsqueeze(0).unsqueeze(0)*0.2-0.1
       modifier = F.interpolate(modifier, scale_factor=1/resampling_factor, mode='linear')
      #  second_channel = param + torch.FloatTensor(param.shape).uniform_(-0.2, 0.2)
       second_channel = param + modifier
       param = torch.vstack([param, second_channel])
    control_params.append(param)

  return control_params


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_checkpoint', type=str, help='Path to the model checkpoint')
  parser.add_argument('--audio_duration', type=float, help='Duration of the generation in seconds')
  parser.add_argument('--fs', type=int, default=44100, help='Sampling rate')
  parser.add_argument('--num_params', type=int, default=2, help='Number of control parameters')
  parser.add_argument('--save_path', type=str, help='Output filename')
  parser.add_argument('--faux_stereo', type=bool, default=False, help='Generates faux stereo by slightly parameters of the other channel')
  config = parser.parse_args()


  control_params = random_control_params(
    audio_duration = config.audio_duration,
    num_params = config.num_params,
    fs = config.fs,
    stereo = config.faux_stereo
  )

  control_params[0] *= 0.45 # scale loudness down
  # control_params[1] = control_params[1]*0.3 + 0.7 # shift spectral centroid up
  control_params[1] *= 0.5 + 0.5 # scale spectral centroid down
  nbn = NoiseBandNet.load_from_checkpoint(config.model_checkpoint)
  nbn.eval()

  audio = nbn(control_params).squeeze(1).detach()

  torchaudio.save(config.save_path, audio, config.fs)




