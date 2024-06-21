import nn_tilde
import argparse
import torch
import lightning as L
import cached_conv as cc

from ddsp.utils import find_checkpoint

from ddsp import DDSP

torch.enable_grad(False)
torch.set_printoptions(threshold=10000)

class ScriptedDDSP(nn_tilde.Module):
  def __init__(self,
               pretrained: DDSP):
    super().__init__()

    self.pretrained = pretrained

    # # # Calculate the input ratio
    # x_len = 2**14
    # x = torch.zeros(1, 1, x_len) for _ in range(self.pretrained.n_control_params)
    # y, _ = self.pretrained(x)
    # in_ratio = y.shape[-1] / x_len
    # print(f"in_ratio: {in_ratio}")

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

  @torch.jit.export
  def decode(self, latents: torch.Tensor):
    synth_params = self.pretrained.decoder(latents.permute(0, 2, 1))
    audio = self.pretrained._synthesize(*synth_params)
    return audio

  @torch.jit.export
  def encode(self, audio: torch.Tensor):
    mu, scale = self.pretrained.encoder(audio.squeeze(1))
    # latents = self.pretrained.encoder.reparametrize(mu, logvar)
    latents, _ = self.pretrained.encoder.reparametrize(mu, scale)
    return latents.permute(0, 2, 1)

  @torch.jit.export
  def forward(self, audio: torch.Tensor):
    return self.pretrained(audio.squeeze(1))


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
  parser.add_argument('--output_path', type=str, help='Directory to save the autoencoded audio')
  parser.add_argument('--streaming', type=bool, default=True, help='Whether to use streaming mode')
  parser.add_argument('--type', default='best', help='Type of model to export', choices=['best', 'last'])
  config = parser.parse_args()

  cc.use_cached_conv(config.streaming)

  checkpoint_path = find_checkpoint(config.model_directory, typ=config.type)

  format = config.output_path.split('.')[-1]
  if format not in ['ts', 'onnx']:
    raise ValueError(f'Invalid format: {format}, supported formats are: ts, onnx')

  ddsp = DDSP.load_from_checkpoint(checkpoint_path, strict=False, streaming=True).to('cpu')
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
    scripted = ScriptedDDSP(ddsp).to('cpu')
    scripted.export_to_ts(config.output_path)
