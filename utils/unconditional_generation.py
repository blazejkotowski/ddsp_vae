import argparse
import torch
import torchaudio

import numpy as np

from matplotlib import pyplot as plt

from ddsp.prior import Prior
from ddsp import DDSP
from ddsp.utils import find_checkpoint

torch.set_grad_enabled(False)

def eq(ten1, ten2, precision=1e-6):
  """Checks equivalence of two tensors"""
  return torch.all(torch.abs(ten1 - ten2) < precision)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--prior_dir', type=str, required=True)
  parser.add_argument('--ddsp_path', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  parser.add_argument('--duration', type=float, default=20.0)
  parser.add_argument('--fs', type=int, default=44100)
  parser.add_argument('--warmup_audio_file', type=str, default=None)
  config = parser.parse_args()

  checkpoint = find_checkpoint(config.prior_dir)
  prior = Prior.load_from_checkpoint(checkpoint).eval()
  print(f"Loaded prior from {checkpoint}. Prior in streaming mode: {prior._streaming}")

  window_size = prior.sequence_length
  latent_size = prior.latent_size

  ddsp = torch.jit.load(config.ddsp_path).pretrained
  resampling_factor = ddsp.resampling_factor

  muscale_sequence = []

  # warmup initialization (provided audio file)
  if config.warmup_audio_file:
    audio, _ = torchaudio.load(config.warmup_audio_file)
    mu, scale = ddsp.encoder(audio[..., :44100*15])
    muscale = torch.cat([mu, scale], dim=-1)[:, -1, :].unsqueeze(0)
    muscale_sequence.append(muscale)
    breakpoint()
  else:
    init_latents = torch.zeros(1, window_size, latent_size)
    last_muscale = prior(init_latents)[:, -1, :]
    muscale_sequence.append(last_muscale.unsqueeze(0))
    last_window = torch.cat([init_latents[:, 1:, :], last_muscale.unsqueeze(0)], dim=1)

  # Generation of the latent code sequence in a loop
  n_samples = int(config.duration * config.fs)
  print(f"Generating {n_samples} in {n_samples // resampling_factor} steps with resampling factor {resampling_factor}")
  for _ in range(n_samples // resampling_factor):
    current_muscale = prior(last_window)[:, -1, :]
    muscale_sequence.append(current_muscale)
    last_window = torch.cat([last_window[:, 1:, :], current_muscale.unsqueeze(0)], dim=1)
    if eq(last_muscale, current_muscale):
      print(f"Converged at length of {len(muscale_sequence)} samples, diverging")
      current_muscale += torch.rand_like(current_muscale) * 1e-3
      break
    last_muscale = current_muscale

  # Concatenate the latent code sequence into a single tensor and remove the batch dimension
  muscale_sequence = torch.cat(muscale_sequence, dim=0)

  print(f"Generated {muscale_sequence.shape[0]} samples.")
  mu, scale = muscale_sequence.chunk(2, dim=-1)

  plt.plot(mu.numpy(), label='mu')
  # plt.plot(scale.numpy(), label='scale')
  plt.show()

  print(f"difference between first and last: {torch.abs(muscale_sequence[0] - muscale_sequence[-1])}")
  np.set_printoptions(precision=4, suppress=True)
  print(f"mu:\n\tmean: {mu.mean(dim=0).numpy()}\n\tsdev: {mu.std(dim=0).numpy()}\nscale:\n\tmean: {scale.mean(dim=0).numpy()}\n\tstdev: {scale.std(dim=0).numpy()}")

  z, _ = ddsp.encoder.reparametrize(mu.unsqueeze(0), scale.unsqueeze(0))
  breakpoint()

  params = ddsp.decoder(z)
  audio = ddsp.synthesize(params)

  torchaudio.save(config.output, audio.squeeze(0), config.fs)
