import nn_tilde
import argparse
import torch
import lightning as L
import cached_conv as cc

from modules import NoiseBandNet

from typing import Optional

torch.enable_grad(False)
torch.set_printoptions(threshold=10000)


class ScriptedNoiseBandNet(nn_tilde.Module):
  def __init__(self,
               pretrained: NoiseBandNet):
    super().__init__()

    self.pretrained = pretrained
    self.hx = torch.zeros(pretrained.gru.num_layers, 1, pretrained.gru.hidden_size)

    # # Calculate the input ratio
    x_len = 2**14
    x = [torch.zeros(1, 1, x_len) for _ in range(self.pretrained.n_control_params)]
    y, _ = self.pretrained(x)
    in_ratio = y.shape[-1] / x_len
    print(f"in_ratio: {in_ratio}")

    self.register_method(
      "decode",
      in_channels = self.pretrained.n_control_params,
      in_ratio = self.pretrained.resampling_factor,
      out_channels = 1, # number of output audio channels
      out_ratio = 1,
      input_labels=[f'(signal) Control Parameter {i}' for i in range(1, self.pretrained.n_control_params+1)],
      output_labels=['(signal) Audio Output'],
      test_method=True,
    )

  @torch.jit.export
  def decode(self, control_params):
    # Make into the list for nn~
    if not isinstance(control_params, list):
      control_params = [c.unsqueeze(-2) for c in control_params.permute(1, 0, 2)]

    # If the batch size does not match the hidden state, reset it
    if self.hx is not None and self.hx.shape[1] != control_params[0].shape[0]:
      hx = None
    else:
      hx = self.hx

    audio, self.hx = self.pretrained(control_params, hx)
    return audio


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_checkpoint', type=str, help='Path to the model checkpoint')
  parser.add_argument('--output_path', type=str, help='Directory to save the autoencoded audio')
  parser.add_argument('--streaming', type=bool, default=True, help='Whether to use streaming mode')
  config = parser.parse_args()

  cc.use_cached_conv(config.streaming)

  nbn = NoiseBandNet.load_from_checkpoint(config.model_checkpoint, strict=False)
  nbn._trainer = L.Trainer() # ugly workaround
  nbn.loss = None # for the torchscript
  nbn.eval()

  scripted = ScriptedNoiseBandNet(nbn)
  scripted.export_to_ts(config.output_path)
