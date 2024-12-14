import nn_tilde
import argparse
import torch
import lightning as L
import cached_conv as cc

import time

from ddsp.utils import find_checkpoint

from ddsp import DDSP
from ddsp.prior import Prior

torch.enable_grad(False)
torch.set_printoptions(threshold=10000)

class ScriptedDDSP(nn_tilde.Module):
  def __init__(self,
               pretrained: DDSP,
               prior_model: Prior = None):
    super().__init__()

    self.pretrained = pretrained

    if prior_model is None:
      prior_model = FakePrior()
    else:
      prior_model = PriorWrapper(prior_model)

    self.prior_model = prior_model

    # # # Calculate the input ratio
    # x_len = 2**14
    # x = torch.zeros(1, 1, x_len) for _ in range(self.pretrained.n_control_params)
    # y, _ = self.pretrained(x)
    # in_ratio = y.shape[-1] / x_len
    # print(f"in_ratio: {in_ratio}")

    # self.register_buffer("prior_buffer", torch.randn(1, self.prior_model._max_len, self.prior_model._latent_size))

    self.register_method(
      "forward",
      in_channels = 1,
      in_ratio = 1,
      out_channels = 1,
      out_ratio = 1,
      input_labels=['(signal) Audio Input'],
      output_labels=['(signal) Audio Output'],
      test_method=True,
    )

    self.register_method(
      "decode",
      in_channels = self.pretrained.latent_size,
      in_ratio = self.pretrained.resampling_factor,
      out_channels = 1, # number of output audio channels
      out_ratio = 1,
      input_labels=[f'(signal) Latent Dimension {i}' for i in range(1, self.pretrained.latent_size+1)],
      output_labels=['(signal) Audio Output'],
      test_method=True,
    )

    self.register_method(
      "encode",
      in_channels = 1,
      in_ratio = 1,
      out_channels = self.pretrained.latent_size,
      out_ratio = self.pretrained.resampling_factor,
      input_labels=['(signal) Audio Input'],
      output_labels=[f'(signal) Latent Dimension {i}' for i in range(1, self.pretrained.latent_size+1)],
      test_method=True
    )

    if not isinstance(self.prior_model, FakePrior):
      self.register_method(
        "prior",
        in_channels=self.pretrained.latent_size,
        in_ratio=self.pretrained.resampling_factor,
        out_channels=self.pretrained.latent_size,
        out_ratio=self.pretrained.resampling_factor,
      )

  @torch.jit.export
  def decode(self, latents: torch.Tensor):
    synth_params = self.pretrained.decoder(latents.permute(0, 2, 1))
    audio = self.pretrained._synthesize(*synth_params)
    return audio.float()

  @torch.jit.export
  def encode(self, audio: torch.Tensor):
    mu, scale = self.pretrained.encoder(audio.squeeze(1))
    # latents = self.pretrained.encoder.reparametrize(mu, logvar)
    latents, _ = self.pretrained.encoder.reparametrize(mu, scale)
    return latents.permute(0, 2, 1).float()

  @torch.jit.export
  def forward(self, audio: torch.Tensor):
    return self.pretrained(audio.squeeze(1)).float()

  @torch.jit.export
  def prior(self, x: torch.Tensor):
    return self.prior_model(x)



class FakePrior(torch.nn.Module):
  def forward(self, x: torch.Tensor):
    return torch.zeros_like(x)


class PriorWrapper(torch.nn.Module):
  def __init__(self, prior: Prior):
    super().__init__()

    self.prior = prior

    self.max_len = self.prior._max_len
    self.init_primer_len = self.max_len // 4
    self.current_buffer_len = self.init_primer_len
    self.register_buffer("prior_buffer", torch.randn(1, self.max_len, self.prior._latent_size))
    # self.prior_buffer = torch.randn(1, self.max_len, self.prior._latent_size)


  def append_to_buffer(self, x: torch.Tensor):
    """
    Appends the input tensor to the prior buffer, if the buffer is full,
    it is reset to the initial primer length.

    Args:
      x, torch.Tensor[batch_size, seq_len, latent_size]
    """
    x = x[:1, ...] # only first in batch
    seq_len = x.shape[1]

    if self.current_buffer_len + seq_len > self.max_len:
      self.prior_buffer[:, :self.init_primer_len-seq_len, :] = self.prior_buffer[:, -self.init_primer_len+seq_len:, :].clone()
      self.prior_buffer[:, self.init_primer_len:self.init_primer_len+seq_len, :] = x
      self.current_buffer_len = self.init_primer_len
    else:
      self.prior_buffer[:, self.current_buffer_len:self.current_buffer_len+seq_len, :] = x
      self.current_buffer_len += seq_len
  

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x, torch.Tensor[batch_size, latent_size, seq_len]
    """
    # self.append_to_buffer(x.permute(0, 2, 1))

    steps = x.shape[-1]

    output = torch.zeros(1, steps, self.prior._latent_size)
    local_buffer = self.prior_buffer.clone()
    current_len = self.current_buffer_len

    for i in range(steps):
      prime = local_buffer[:, :current_len, :]
      logits = self.prior(prime)
      latent = self.prior.sample(logits, temperature=1.0)[:, -1:, :]

      local_buffer[:, current_len:current_len+1, :] = latent
      output[:, i, :] = latent[:, 0, :]

      current_len += 1

      if current_len >= self.max_len:
        local_buffer[:, :self.init_primer_len, :] = local_buffer[:, -self.init_primer_len:, :].clone()
        current_len = self.init_primer_len
    
    if x.size(0) > 1:
      output = output.repeat_interleave(x.size(0), dim=0)

    self.append_to_buffer(output)

    return output.permute(0, 2, 1).float()

class ONNXDDSP(torch.nn.Module):
  def __init__(self,
               pretrained: DDSP):
    super().__init__()

    self.pretrained = pretrained

  def decode(self, latents: torch.Tensor):
    synth_params = self.pretrained.decoder(latents.permute(0, 2, 1))
    audio = self.pretrained._synthesize(*synth_params)
    return audio

  def encode(self, audio: torch.Tensor):
    mu, scale = self.pretrained.encoder(audio.squeeze(1))
    latents, _ = self.pretrained.encoder.reparametrize(mu, scale)
    return latents.permute(0, 2, 1)

  def forward(self, audio: torch.Tensor):
    return self.pretrained(audio.squeeze(1))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_directory', type=str, help='Path to the model training')
  parser.add_argument('--prior_directory', type=str, default=None, help='Path to the prior model training')
  parser.add_argument('--output_path', type=str, help='Directory to save the autoencoded audio')
  parser.add_argument('--streaming', type=bool, default=True, help='Whether to use streaming mode')
  parser.add_argument('--type', default='best', help='Type of model to export', choices=['best', 'last'])
  config = parser.parse_args()

  cc.use_cached_conv(config.streaming)

  checkpoint_path = find_checkpoint(config.model_directory, typ=config.type)
  print(f"exporting model from checkpoint: {checkpoint_path}")

  format = config.output_path.split('.')[-1]
  if format not in ['ts', 'onnx']:
    raise ValueError(f'Invalid format: {format}, supported formats are: ts, onnx')

  prior = None
  if config.prior_directory is not None:
    prior_checkpoint_path = find_checkpoint(config.prior_directory, typ=config.type)
    print("exporting prior model from checkpoint: ", prior_checkpoint_path)
    prior = Prior.load_from_checkpoint(prior_checkpoint_path, strict=False).to('cpu')
    prior.eval()
    for k in prior._normalization_dict.keys():
      prior._normalization_dict[k] = prior._normalization_dict[k].to('cpu')

    prior._trainer = L.Trainer()

  ddsp = DDSP.load_from_checkpoint(checkpoint_path, strict=False, streaming=True, device='cpu').to('cpu')
  if format == 'onnx':
    ddsp.eval()
    scripted = ONNXDDSP(ddsp).to('cpu')
    torch.onnx.dynamo_export(
      scripted,
      torch.zeros(1, 1, 2**14),
    ).save(config.output_path)
  elif format == 'ts':
    ddsp._trainer = L.Trainer() # ugly workaround
    ddsp._recons_loss = None # for the torchscript
    ddsp.eval()


    scripted = ScriptedDDSP(ddsp, prior).to('cpu')
    scripted.export_to_ts(config.output_path)

    print("Model exported to: ", config.output_path)
