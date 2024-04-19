
import torchaudio
import torch
import argparse
import random
import numpy as np
import time

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

def random_sequence(n_samples, mode='uniform') -> torch.Tensor:
   if mode == 'uniform':
    return torch.FloatTensor(1, 1, n_samples).uniform_(0, 1)
   elif mode == 'normal':
    return torch.randn(1, 1, n_samples) / 2 + 0.5
   elif mode == 'walk':
    return torch.from_numpy(random_walk(n_samples)).unsqueeze(0).unsqueeze(0)

def random_control_params(audio_duration: float,
                          num_params: int,
                          fs: int,
                          resampling_factor: int = 32,
                          stereo: bool = False,
                          mode: str = 'walk') -> List[torch.Tensor]:
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
    param = random_sequence(n_samples, mode=mode)
    param = F.interpolate(param, scale_factor=1/resampling_factor, mode='linear')
    if stereo:
       modifier = random_sequence(n_samples, mode=mode)*0.2-0.1
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
    stereo = config.faux_stereo,
    mode = 'walk'
  )

  # control_params[0] *= 0.45 # scale loudness down
  # control_params[0] = control_params[0]*0 + 0.4 # set loudness to constant
  # control_params[1] = control_params[1]*0.3 + 0.7 # shift spectral centroid up
  # control_params[1] *= 0.5 + 0.5 # scale spectral centroid down
  # control_params[2] = control_params[2]*0 + 1 # set flatness to 0
  # Set all params to constant 0.5
  for i in range(len(control_params)):
    control_params[i] = control_params[i]*0 + 0.05

  control_params[0] = control_params[0]*0 + 0.15

  control_params[2] = control_params[2]*0


  nbn = NoiseBandNet.load_from_checkpoint(config.model_checkpoint)
                                          # n_control_params=3) # this is for ybalferran model
  nbn.eval()

  start_time = time.time()
  # Generate audio in chunks
  # # Calculations for downsampeld signal
  n_signal = control_params[0].shape[-1]
  n_noiseband = int(nbn._noisebands.shape[-1] // nbn.resampling_factor)
  n_chunk = n_noiseband # Actual lenght of loopable noiseband

  n_chunk = int(8192 / 32) # Arbitrary

  chunks = np.ceil(n_signal / n_chunk).astype(int)

  print(f"Concatenating {chunks} chunks.")
  audio = None
  for i in range(chunks):
    if (i+1)*n_chunk > n_signal:
      current_control = [c[:, :, i*n_chunk:] for c in control_params]
    else:
      current_control = [c[:, :, i*n_chunk:(i+1)*n_chunk] for c in control_params]

    if audio is None:
      audio = nbn(current_control).squeeze(1).detach()
    else:
      audio = torch.cat([audio, nbn(current_control).squeeze(1).detach()], dim=-1)

  # # Generate all at once
  # audio = nbn(control_params).squeeze(1).detach()

  elapsed_time = time.time() - start_time
  audio_duration = audio.shape[-1] / config.fs
  print(f"It took {elapsed_time:.2f}s to generate a {audio_duration:.2f}s audio")

  torchaudio.save(config.save_path, audio, config.fs)




