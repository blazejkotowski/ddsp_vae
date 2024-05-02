import nn_tilde
import argparse
import torch
import lightning as L
import cached_conv as cc

from modules import NoiseBandNet

torch.enable_grad(False)
torch.set_printoptions(threshold=10000)

class ScriptedNoiseBandNet(nn_tilde.Module):
  def __init__(self,
               pretrained: NoiseBandNet):
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
    amps = self.pretrained.decoder(latents.permute(0, 2, 1))
    audio = self.pretrained._synthesize(amps)
    return audio

  @torch.jit.export
  def encode(self, audio: torch.Tensor):
    mu, logvar = self.pretrained.encoder(audio.squeeze(1))
    latents = self.pretrained.encoder.reparametrize(mu, logvar)
    return latents.permute(0, 2, 1)

  @torch.jit.export
  def forward(self, audio: torch.Tensor):
    return self.pretrained(audio.squeeze(1))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_checkpoint', type=str, help='Path to the model checkpoint')
  parser.add_argument('--output_path', type=str, help='Directory to save the autoencoded audio')
  parser.add_argument('--streaming', type=bool, default=True, help='Whether to use streaming mode')
  config = parser.parse_args()

  cc.use_cached_conv(config.streaming)

  nbn = NoiseBandNet.load_from_checkpoint(config.model_checkpoint, strict=False, streaming=True).to('cpu')
  nbn._trainer = L.Trainer() # ugly workaround
  nbn.recons_loss = None # for the torchscript
  nbn.eval()

  scripted = ScriptedNoiseBandNet(nbn).to('cpu')
  scripted.export_to_ts(config.output_path)
