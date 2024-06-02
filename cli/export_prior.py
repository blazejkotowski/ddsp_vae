import lightning as L
import torch

import argparse
import nn_tilde

import torch.nn.functional as F

from ddsp.utils import find_checkpoint
from ddsp.prior import Prior

torch.set_grad_enabled(True)

class ScriptedPrior(nn_tilde.Module):
  def __init__(self, prior: Prior, vae_model: nn_tilde.Module):
    super().__init__()
    self._prior = prior

    self._latent_size = prior.latent_size
    self._resampling_factor = vae_model.pretrained.resampling_factor

    self._vae_model = vae_model

    self.register_buffer('_previous_step', torch.zeros(1, self._latent_size))

    self.register_method('forward',
                         in_channels=1,
                         in_ratio=1,
                         out_channels=self._latent_size//2,
                         out_ratio=1,
                         input_labels=['Temperature'],
                         output_labels=[f'Latent Output {i}' for i in range(self._latent_size//2)],
                         test_method=True)

  @torch.jit.export
  def forward(self, input: torch.Tensor) -> torch.Tensor:
    """
    Returns:
      - latents: torch.Tensor[batch_size, latent_size, seq_len], the predicted sequence of latents
    """
    input = input.permute(0, 2, 1) # => [batch_size, seq_len, input_size]
    batch_size, seq_len, _ = input.shape

    # In case batch size has changed
    if self._previous_step.shape[0] < batch_size:
      self._previous_step = self._previous_step.repeat(batch_size, 1)
    elif self._previous_step.shape[0] > batch_size:
      self._previous_step = self._previous_step[:batch_size]

    sequence = []
    x = self._prior(self._previous_step.unsqueeze(1))

    # for _ in range(seq_len):
    for _ in range(seq_len // self._resampling_factor):
      y = self._prior(x)
      sequence.append(y)
      self._previous_step.copy_(y.squeeze(1))
      x = y

    sequence = torch.cat(sequence, dim=1)
    mu, scale = sequence.chunk(2, dim=-1)
    latents, _ = self._vae_model.pretrained.encoder.reparametrize(mu, scale)

    # latents = latents.permute(0, 2, 1)

    latents = F.interpolate(latents.permute(0, 2, 1), size=seq_len, mode='linear').detach().to(device='cpu', dtype=torch.float32)
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
  prior = Prior.load_from_checkpoint(checkpoint, streaming=True).eval()
  prior._trainer = L.Trainer() # workaround

  scripted = ScriptedPrior(prior, vae_model)
  scripted.export_to_ts(config.out_path)
