import nn_tilde
import argparse
import torch
import lightning as L
import cached_conv as cc

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
    self.prior_model = prior_model

    # # # Calculate the input ratio
    # x_len = 2**14
    # x = torch.zeros(1, 1, x_len) for _ in range(self.pretrained.n_control_params)
    # y, _ = self.pretrained(x)
    # in_ratio = y.shape[-1] / x_len
    # print(f"in_ratio: {in_ratio}")

    self.init_primer_length = self.prior_model._max_len // 4
    self.prior_buffer_length = self.init_primer_length
    # self.register_attribute("prior_buffer_length", self.init_primer_length)
    self.prior_buffer = torch.randn(1, self.prior_model._max_len, self.prior_model._latent_size)
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

    if self.prior_model is not None:
      self.register_method(
        "prior",
        in_channels=1,
        in_ratio=self.pretrained.resampling_factor,
        out_channels=self.pretrained.latent_size,
        out_ratio=self.pretrained.resampling_factor,
        # test_buffer_size=self.pretrained.resampling_factor,
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
    steps = x.shape[-1]

    latents = torch.zeros(x.shape[0], steps, self.pretrained.latent_size)

    for i in range(steps):
      # Get the sequence for the prior buffer
      prime = self.prior_buffer[:, :self.prior_buffer_length, :]

      # Predict the next latent code
      y = self.prior_model(prime)[:, -1, :]

      # Update the prior buffer
      self.prior_buffer[:, self.prior_buffer_length, :] = y
      self.prior_buffer_length += 1

      # "Reset" the buffer if it's full
      if self.prior_buffer_length >= self.prior_model._max_len:
        self.prior_buffer_length = self.init_primer_length
        
        # copy the end of the prior buffer to the beginning
        self.prior_buffer[:, :self.prior_buffer_length, :] = self.prior_buffer[:, -self.init_primer_length:, :]

      mu, scale = y.unsqueeze(0).chunk(2, dim=-1) # => [batch_size, seq_len, latent_size]
      latent, _ = self.pretrained.encoder.reparametrize(mu, scale)

      if x.size(0) > 1:
        latent = latent.repeat_interleave(x.size(0), dim=0)

      latents[:, i, :] = latent[:, 0, :]
      
    return latents.permute(0, 2, 1).float() # [batch_size, latent_size, seq_len]


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
    prior = Prior.load_from_checkpoint(prior_checkpoint_path, strict=False).to('cpu')
    prior.eval()

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
