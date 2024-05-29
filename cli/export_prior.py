import lightning as L
import torch

import argparse
import nn_tilde

from ddsp.utils import find_checkpoint
from ddsp.prior import Prior

torch.set_grad_enabled(False)

class ScriptedPrior(nn_tilde.Module):
  def __init__(self, prior: Prior, vae_model: nn_tilde.Module):
    super().__init__()
    self._prior = prior

    self._latent_size = prior.latent_size

    self._vae_model = vae_model

    self.register_buffer('_previous_step', torch.zeros(1, 1, self._latent_size))

    self.register_method('forward',
                         in_channels=self._latent_size//2,
                         in_ratio=1,
                         out_channels=self._latent_size//2,
                         out_ratio=0.5,
                         input_labels=[f'Latent Input {i}' for i in range(self._latent_size//2)],
                         output_labels=[f'Latent Output {i}' for i in range(self._latent_size//2)],
                         test_method=True)

  def forward(self, x):
    x = x.permute(0, 2, 1)
    # x = self.previous_step
    y = self._prior(x)
    self._previous_step.copy_(y[-1])

    mu, scale = y.chunk(2, dim=-1)
    latents, _ = self._vae_model.pretrained.encoder.reparametrize(mu, scale)

    breakpoint()
    return latents

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--streaming', type=bool, default=False, help='Enable streaming mode')
  parser.add_argument('--checkpoint_directory', type=str, help='Path to the training directory', required=True)
  parser.add_argument('--vae_model', type=str, help='Path to the VAE model', required=True)
  parser.add_argument('--out_path', type=str, help='Path to save the scripted model', required=True)
  config = parser.parse_args()

  vae_model = torch.jit.load(config.vae_model).eval()

  checkpoint = find_checkpoint(config.checkpoint_directory)
  prior = Prior.load_from_checkpoint(checkpoint, streaming=True)
  prior._trainer = L.Trainer() # workaround

  scripted = ScriptedPrior(prior, vae_model)
  scripted.export_to_ts(config.out_path)
