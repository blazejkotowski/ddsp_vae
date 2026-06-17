import nn_tilde
import argparse
import os
import math
import yaml
import torch
import torch.nn.functional as F
import lightning as L
import cached_conv as cc
from typing import Optional

import time

from ddsp.utils import find_checkpoint

from ddsp import DDSP
from ddsp.interfaces import ControlField, ControlSpace
import torch
from ddsp.prior import Prior, PriorDiscrete
from ddsp.prior.kv_infer import KVCachedPrior
from ddsp.latent_compressor import LatentCompressor

torch.enable_grad(False)
torch.set_printoptions(threshold=10000)

class ScriptedDDSP(nn_tilde.Module):
  def __init__(self,
               pretrained: DDSP,
               prior_model: torch.nn.Module = None,
               target_fs: float = 16000.0):
    super().__init__()

    self.pretrained = pretrained

    self.resample_ratio = target_fs / self.pretrained.fs

    numerator = self.pretrained.resampling_factor * target_fs
    denominator = int(self.pretrained.fs)

    if numerator % denominator != 0:
      gcd = math.gcd(self.pretrained.resampling_factor, denominator)
      compatible_step = denominator // gcd
      raise ValueError(
        "target_fs={} is incompatible with nn~ export. "
        "Choose a multiple of {} to keep decode buffers aligned.".format(target_fs, compatible_step)
      )

    self._nn_decode_ratio = int(numerator // denominator)

    if prior_model is None:
      prior_model = FakePrior()
    elif isinstance(prior_model, Prior):
      prior_model = PriorWrapper(prior_model, resample_ratio=self.resample_ratio)
    # else: assume it's already a wrapper module compatible with ScriptedDDSP.prior()

    self.prior_model = prior_model


    # # # Calculate the input ratio
    # x_len = 2**14
    # x = torch.zeros(1, 1, x_len) for _ in range(self.pretrained.n_control_params)
    # y, _ = self.pretrained(x)
    # in_ratio = y.shape[-1] / x_len
    # print(f"in_ratio: {in_ratio}")

    # self.register_buffer("prior_buffer", torch.randn(1, self.prior_model._max_len, self.prior_model._num_params))

    self.register_attribute("limit_components", 0.0)
    self.register_attribute("noise_amplitude_attenuation", 0.0)
    self.register_attribute("sines_amplitude_attenuation", 0.0)
    self.register_attribute("waveshaping", 0.0)

    # self.register_method(
    #   "forward",
    #   in_channels = 1,
    #   in_ratio = 1,
    #   out_channels = 1,
    #   out_ratio = 1,
    #   input_labels=['(signal) Audio Input'],
    #   output_labels=['(signal) Audio Output'],
    #   test_method=True,
    # )

    total_params = self.pretrained.num_params + self.pretrained.n_features

    n_channels = self.pretrained.n_channels

    self.register_method(
      "decode",
      in_channels = total_params,
      in_ratio = self._nn_decode_ratio,
      out_channels = n_channels, # number of output audio channels
      # out_ratio = 1/3,
      out_ratio = 1,
      input_labels=[f'(signal) Latent Dimension {i}' for i in range(1, total_params+1)],
      output_labels=[f'(signal) Audio Output {i}' for i in range(1, n_channels+1)],
      test_method=True,
    )

    if self.pretrained.latent_size > 0:
      self.register_method(
        "encode",
        in_channels = n_channels,
        in_ratio = 1,
        out_channels = self.pretrained.num_params,
        out_ratio = self.pretrained.resampling_factor,
        input_labels=[f'(signal) Audio Input {i}' for i in range(1, n_channels+1)],
        output_labels=[f'(signal) Latent Dimension {i}' for i in range(1, self.pretrained.num_params+1)],
        test_method=True
      )

    if not isinstance(self.prior_model, FakePrior):
      if isinstance(self.prior_model, PriorDiscreteWrapper):
        # New layout: [LFO control envelope | normalised 2-D territory map | temperature].
        cd = int(self.prior_model.cond_dim)
        use_terr = bool(self.prior_model.use_terr_map)
        use_cfg = bool(self.prior_model.use_cfg)
        in_ch = int(self.prior_model.prior_in_channels)
        labels = [f'(signal) LFO {i}' for i in range(1, cd + 1)]
        if use_terr:
          labels += ['(signal) Territory X', '(signal) Territory Y']
        labels += ['(signal) Temperature']
        if use_cfg:
          labels += ['(signal) CFG Strength']
        if bool(self.prior_model.use_feat_smooth):
          labels += ['(signal) Smoothing']
        if bool(self.prior_model.use_reseed):
          labels += ['(signal) Reseed Trigger']
        self.register_method(
          "prior",
          in_channels=in_ch,
          in_ratio=self._nn_decode_ratio,
          out_channels=total_params,
          out_ratio=self.pretrained.resampling_factor,
          input_labels=labels,
        )
      else:
        self.register_method(
          "prior",
          in_channels=total_params + 2, # latent transposition + temperature + prediction_strength
          in_ratio=self._nn_decode_ratio,
          out_channels=total_params,
          out_ratio=self.pretrained.resampling_factor,
          input_labels=[f'(signal) Transposition {i}' for i in range(1, total_params+1)] + ['(signal) Temperature', '(signal) Prediction strenght'],
        )

  @torch.jit.export
  def decode(self, params: torch.Tensor):
    # params = params.permute(0, 2, 1)
    # print('params', params.shape)
    # print(self.pretrained.latent_size)
    # print(self.pretrained.num_params)
    latents = params[:, self.pretrained.n_features:, :]
    features = params[:, :self.pretrained.n_features, :]
    # print("Params shape:", params.shape)

    latents = latents.permute(0, 2, 1)
    # latents = self.pretrained.params_to_latents(latents)
    # latents = self.pretrained.denormalize_latents(latents)
    features = features.permute(0, 2, 1)

    if self.pretrained.latent_size == 0:
      latents = torch.zeros(latents.size(0), latents.size(1), 1, device=latents.device)

    synth_params = self.pretrained.decoder(features, latents)
    audio = self.pretrained._synthesize(synth_params, waveshaping_factor=self.waveshaping[0], limit_components=self.limit_components[0])
    # print("Audio shape before interpolation:", audio.shape)

    if self.resample_ratio != 1:
      audio = F.interpolate(audio, scale_factor=self.resample_ratio, mode='linear')

    # print("Audio shape after interpolation:", audio.shape)

    return audio.float()


  @torch.jit.export
  def encode(self, audio: torch.Tensor):
    if self.pretrained.encoder is None:
      raise RuntimeError("encode() requested but the pretrained model has no encoder (latent_size=0)")
    # Encoder downmixes the [B, n_channels, T] input internally
    mu, scale = self.pretrained.encoder(audio)
    latents, _ = self.pretrained.encoder.reparametrize(mu, scale)
    latents = self.pretrained._smooth_latents(latents)
    latents = self.pretrained.normalize_latents(latents)
    latents = self.pretrained.latents_to_params(latents)
    latents = latents.permute(0, 2, 1).float()

    return latents

  # @torch.jit.export
  # def forward(self, audio: torch.Tensor):
  #   return self.pretrained(audio.squeeze(1)).float()

  @torch.jit.export
  def prior(self, x: torch.Tensor):
    return self.prior_model(x)

  @torch.jit.export
  def get_waveshaping(self) -> float:
    return self.waveshaping[0]

  @torch.jit.export
  def set_waveshaping(self, value: float):
    self.waveshaping = (value, )
    return 0


  @torch.jit.export
  def get_limit_components(self) -> float:
    return self.limit_components[0]

  @torch.jit.export
  def set_limit_components(self, value: float):
    self.limit_components = (value, )
    return 0

  @torch.jit.export
  def get_noise_amplitude_attenuation(self) -> float:
    return self.noise_amplitude_attenuation[0]

  @torch.jit.export
  def set_noise_amplitude_attenuation(self, value: float):
    self.noise_amplitude_attenuation = (value, )
    return 0

  @torch.jit.export
  def get_sines_amplitude_attenuation(self) -> float:
    return self.sines_amplitude_attenuation[0]

  @torch.jit.export
  def set_sines_amplitude_attenuation(self, value: float):
    self.sines_amplitude_attenuation = (value, )
    return 0


class FakePrior(torch.nn.Module):
  def forward(self, x: torch.Tensor):
    return torch.zeros_like(x)


class PriorWrapper(torch.nn.Module):
  def __init__(self, prior: Prior, resample_ratio: float = 1.0):
    super().__init__()

    self.prior = prior
    self.resample_ratio = resample_ratio

    self.max_len = self.prior._max_len
    self.init_primer_len = self.max_len // 4
    self.current_buffer_len = self.init_primer_len
    self.register_buffer("prior_buffer", torch.zeros(1, self.max_len, self.prior._num_controls))
    # self.prior_buffer = torch.randn(1, self.max_len, self.prior._num_params)


  def append_to_buffer(self, x: torch.Tensor):
    """
    Appends the input tensor to the prior buffer, if the buffer is full,
    it is reset to the initial primer length.

    Args:
      x, torch.Tensor[batch_size, seq_len, num_params]
    """
    x = x[:1, ...] # only first in batch
    seq_len = x.shape[1]

    # TODO: Is this correct?
    if self.current_buffer_len + seq_len > self.max_len:
      if seq_len >= self.init_primer_len:
        # If seq_len is greater than or equal to init_primer_len, just keep the last init_primer_len elements from x
        self.prior_buffer[:, :self.init_primer_len, :] = x[:, -self.init_primer_len:, :]
      else:
        # Shift the last init_primer_len - seq_len elements to the beginning
        self.prior_buffer[:, :self.init_primer_len - seq_len, :] = self.prior_buffer[:, -self.init_primer_len + seq_len:, :].clone()
        # Place x at the end of the primer region
        self.prior_buffer[:, self.init_primer_len - seq_len:self.init_primer_len, :] = x
      self.current_buffer_len = self.init_primer_len
    else:
      self.prior_buffer[:, self.current_buffer_len:self.current_buffer_len+seq_len, :] = x
      self.current_buffer_len += seq_len


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x, torch.Tensor[batch_size, num_params + 1, seq_len]
    """
    # self.append_to_buffer(x.permute(0, 2, 1))

    # Ignore the batch dimension
    transposition = x[:1, :-2, :]
    temperature = x[:1, -2, :]
    prediction_annealing = 1 - x[:1, -1, :]

    steps = x.shape[-1]

    output = torch.zeros(1, steps, self.prior._num_controls)
    local_buffer = self.prior_buffer.clone()
    current_len = self.current_buffer_len

    for i in range(steps):
      prime = local_buffer[:, :current_len, :]
      logits = self.prior(prime)
      latent = self.prior.sample(logits, temperature=temperature[0, i])[:, -1:, :]

      # transpose
      # print('latent', latent.shape, 'transpositions', transposition[:, :, i].shape)
      latent *= prediction_annealing[:, i]
      latent += transposition[:, :, i]

      local_buffer[:, current_len:current_len+1, :] = latent
      output[:, i, :] = latent[:, 0, :]

      current_len += 1

      if current_len >= self.max_len:
        local_buffer[:, :self.init_primer_len, :] = local_buffer[:, -self.init_primer_len:, :].clone()
        current_len = self.init_primer_len

    if x.size(0) > 1:
      output = output.repeat_interleave(x.size(0), dim=0)

    self.append_to_buffer(output)

    if self.resample_ratio != 1:
      output = F.interpolate(output.permute(0, 2, 1), scale_factor=self.resample_ratio, mode='linear').permute(0, 2, 1)

    return output.permute(0, 2, 1).float()


class LatentCompressorDecodeOnly(torch.nn.Module):
  def __init__(self, vq: torch.nn.Module, decoder: torch.nn.Module, compression_ratio: int):
    super().__init__()
    self.vq = vq
    self.decoder = decoder
    self.compression_ratio = int(compression_ratio)

  def decode_codes(self, indices: torch.Tensor, output_len: Optional[int] = None) -> torch.Tensor:
    z_q = self.vq.embed(indices)
    x_hat = self.decoder(z_q, None)

    if output_len is not None:
      T_out = int(x_hat.shape[1])
      if T_out > output_len:
        x_hat = x_hat[:, :output_len, :]
      elif T_out < output_len:
        x_hat = F.pad(x_hat, (0, 0, 0, output_len - T_out))

    return x_hat

  def forward(self, indices: torch.Tensor, output_len: Optional[int] = None) -> torch.Tensor:
    return self.decode_codes(indices, output_len=output_len)


class PriorDiscreteWrapper(torch.nn.Module):
  """Realtime nn~ wrapper for the (joint-codebook) discrete prior.

  Control layout in `forward(x)` (x: [B, C, steps], C = cond_dim + 2 + 1 + use_cfg):
    [0 : cond_dim)          -> LFO control envelope (the slow conditioning scaffold)
    [cond_dim : cond_dim+2) -> normalised 2-D "territory map" coordinate (blends zones)
    [cond_dim+2]            -> temperature (sampling randomness)
    [cond_dim+3]            -> CFG scale (territory contrast/strength; 1=off, >1 amplifies) [if available]
    [..]                    -> Smoothing (one-pole LPF on ALL control trajectories; 0=off..0.95 sluggish)
    [last]                  -> Reseed Trigger (rising edge >0.5 re-anchors the prior to a fresh phrase)
  No transposition, no prediction-strength. Joint (WS4) models sample the N codebooks
  autoregressively per frame (on-manifold); independent models fall back to per-codebook sampling.
  CFG runs a second (null-territory) KV cache in lockstep and guides each codebook's logits.
  """
  def __init__(self, prior: PriorDiscrete, compressor: torch.nn.Module, resample_ratio: float = 1.0,
               n_feature_channels: int = 2, terr_map_temp: float = 0.5, decode_lookahead: int = 2):
    super().__init__()

    # KV-cached incremental prior (weights mapped from the trained PriorDiscrete).
    self.kv = KVCachedPrior(prior)
    # Optional causal one-pole low-pass on the FEATURE output channels (loudness/centroid), to tame
    # fast generated-feature transients that the synth renders as clicks (percussive material).
    # Coefficient in [0,1): 0 = off; higher = smoother. Live-settable via set_feature_smoothing.
    self.n_feat_smooth = int(n_feature_channels)
    self.compressor = compressor
    self.resample_ratio = resample_ratio

    self.max_len = int(prior._max_len)
    self.init_primer_len = int(self.max_len // 4)
    self.num_codebooks = int(prior.num_codebooks)
    self.codebook_size = int(prior.codebook_size)
    self.start_id = int(getattr(prior, 'start_token_id', self.codebook_size))
    self.compression_ratio = int(getattr(self.compressor, 'compression_ratio', 32))

    self.is_joint = bool(getattr(prior, '_joint', False))
    self.cond_dim = int(getattr(prior, '_cond_dim', 0))
    self.use_cond = self.cond_dim > 0
    self.num_territories = int(getattr(prior, '_num_territories', 0))
    self.use_terr_map = self.num_territories > 0
    self.d_model = int(prior._d_model)
    # Classifier-free guidance is available when the model is joint AND was trained with a learned
    # NULL/uncond territory row (cfg_dropout > 0, so territory_weight has num_territories+1 rows).
    self.use_cfg = bool(self.is_joint and self.use_terr_map
                        and int(self.kv.territory_weight.shape[0]) > self.num_territories)
    # LFO-CFG (cond-axis guidance): depth_sample_last_cfg2 in kv_infer is ready; 3rd-cache wiring +
    # input channel are staged, not yet enabled (kept inert so the wrapper stays valid).
    self.use_lfo_cfg = False
    # nn~ inputs: [LFO(cond_dim) | TerritoryX,Y | Temperature | (CFG) | Smoothing | Reseed | (LFO CFG)].
    self.use_feat_smooth = int(n_feature_channels) > 0
    self.use_reseed = True  # beat-synced phrase re-anchor (rising-edge trigger)
    self.prior_in_channels = ((self.cond_dim if self.use_cond else 0)
                              + (2 if self.use_terr_map else 0) + 1 + (1 if self.use_cfg else 0)
                              + (1 if self.use_feat_smooth else 0) + (1 if self.use_reseed else 0)
                              + (1 if self.use_lfo_cfg else 0))
    self.terr_map_temp = float(terr_map_temp)  # blend sharpness of the 2-D territory map

    with torch.no_grad():
      _dummy = torch.zeros(1, 1, self.num_codebooks, dtype=torch.long)
      _num_controls = int(compressor.decode_codes(_dummy).shape[2])
    self.num_controls = _num_controls

    # 2-D territory map: PCA of the learned territory embeddings (first num_territories rows;
    # the CFG null row is excluded) -> a 2-D coordinate per zone the user can navigate/blend.
    if self.use_terr_map:
      W = self.kv.territory_weight[:self.num_territories].float()        # [T, D]
      Wc = W - W.mean(0, keepdim=True)
      try:
        _, _, V = torch.pca_lowrank(Wc, q=2)
        xy = Wc @ V[:, :2]
      except Exception:
        xy = Wc[:, :2]
      xy = xy / (xy.std(0, keepdim=True) + 1e-6)
      self.register_buffer("territory_xy", xy.contiguous())             # [T, 2]
      self.register_buffer("territory_table", W.contiguous())           # [T, D]
    else:
      self.register_buffer("territory_xy", torch.zeros(1, 2))
      self.register_buffer("territory_table", torch.zeros(1, self.d_model))

    # CFG: a second KV cache runs the UNCONDITIONED (null-territory) pass in lockstep, plus the
    # null territory embedding (the learned cfg-dropout row at index num_territories).
    self.kv_uncond = KVCachedPrior(prior) if self.use_cfg else self.kv
    if self.use_cfg:
      self.register_buffer("null_terr_vec", self.kv.territory_weight[self.num_territories].view(1, self.d_model).contiguous())
    else:
      self.register_buffer("null_terr_vec", torch.zeros(1, self.d_model))

    _init_buf = torch.zeros(1, self.max_len, self.num_codebooks, dtype=torch.long)
    _init_buf[:, 0, :] = self.start_id
    self.register_buffer("token_buffer", _init_buf)
    self.register_buffer("_current_len", torch.tensor(1, dtype=torch.long))
    # Pending time-context [1, D] predicting the token at index `_current_len` (cond + uncond).
    self.register_buffer("_pending_context", torch.zeros(1, self.d_model))
    self.register_buffer("_pending_context_uncond", torch.zeros(1, self.d_model))
    self.register_buffer("_ctrl_buf", torch.zeros(1, self.compression_ratio, _num_controls))
    self.register_buffer("_ctrl_pos", torch.tensor(self.compression_ratio, dtype=torch.long))

    self.decode_ctx = 8
    self.decode_lookahead = int(decode_lookahead)
    self.register_buffer("_emit_idx", torch.tensor(1, dtype=torch.long))

    # Smoothing low-pass: per-channel running state over ALL control channels (loudness, centroid,
    # latents), control rate, causal; strength from the input channel.
    self.register_buffer("_feat_ema", torch.zeros(1, _num_controls))
    # Reseed: last trigger value (for rising-edge detection across calls).
    self.register_buffer("_reseed_prev", torch.zeros(()))

    self.reset_state()

  def _territory_vec(self, xy: torch.Tensor) -> torch.Tensor:
    """Blend territory embeddings by softmax over -distance^2 to the 2-D map points. Returns [1, D]."""
    d2 = ((self.territory_xy - xy.view(1, 2)) ** 2).sum(-1)             # [T]
    w = torch.softmax(-d2 / self.terr_map_temp, dim=0)                  # [T]
    return (w.view(-1, 1) * self.territory_table).sum(0, keepdim=True)  # [1, D]

  def _sample_next(self, h_last: torch.Tensor, temp: float) -> torch.Tensor:
    """Sample one frame's N codebooks from time-context h_last [1, D]. Returns [1, N] long."""
    if self.is_joint:
      return self.kv.depth_sample_last(h_last, temp, 1.0)
    fc = F.linear(h_last, self.kv.fc_weight, self.kv.fc_bias).view(1, self.num_codebooks, self.codebook_size)
    t = temp if temp > 1e-4 else 1e-4
    probs = torch.softmax(fc / t, dim=-1)
    return torch.multinomial(probs.reshape(-1, self.codebook_size), 1).reshape(1, self.num_codebooks)

  @torch.jit.export
  def set_territory(self, t: int):
    # Optional manual default: snap the map to a single zone's 2-D point.
    pass

  @torch.jit.export
  def reset_state(self):
    self.token_buffer.zero_()
    self.token_buffer[:, 0, :] = self.start_id
    self.kv.reset()
    h = self.kv.decode_context(self.token_buffer[:, :1, :], 0, None, None)  # [1,1,D]
    self._pending_context.copy_(h[:, -1, :])
    if self.use_cfg:
      self.kv_uncond.reset()
      hu = self.kv_uncond.decode_context(self.token_buffer[:, :1, :], 0, None, None)
      self._pending_context_uncond.copy_(hu[:, -1, :])
    self._current_len.fill_(1)
    self._emit_idx.fill_(1)
    self._ctrl_buf.zero_()
    self._ctrl_pos.fill_(self.compression_ratio)
    self._feat_ema.zero_()
    self._reseed_prev.zero_()

  def _reanchor(self):
    """Beat-synced re-seed: drop the token history and re-anchor the prior context to the START
    token (a fresh phrase), WITHOUT interrupting frame emission (current control block plays out)
    or the smoothing state. Conditioning (LFO/territory) keeps coming from the live inputs."""
    self.token_buffer.zero_()
    self.token_buffer[:, 0, :] = self.start_id
    self.kv.reset()
    h = self.kv.decode_context(self.token_buffer[:, :1, :], 0, None, None)
    self._pending_context.copy_(h[:, -1, :])
    if self.use_cfg:
      self.kv_uncond.reset()
      hu = self.kv_uncond.decode_context(self.token_buffer[:, :1, :], 0, None, None)
      self._pending_context_uncond.copy_(hu[:, -1, :])
    self._current_len.fill_(1)
    self._emit_idx.fill_(1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    steps = int(x.shape[-1])
    if steps <= 0:
      return torch.zeros(1, self.num_controls, 0)

    cd = self.cond_dim
    # Parse the control layout: [LFO(cond_dim) | TerritoryX,Y | Temperature | (CFG)].
    if self.use_cond:
      cond_in = x[:1, :cd, :].permute(0, 2, 1).contiguous()  # [1, steps, cond_dim]
    else:
      cond_in = torch.zeros(1, steps, 1)
    base = cd if self.use_cond else 0
    if self.use_terr_map:
      terr_xy_in = x[:1, base:base + 2, :]   # [1, 2, steps]
      temp_in = x[:1, base + 2, :]           # [1, steps]
      cfg_in = x[:1, base + 3, :] if self.use_cfg else torch.ones(1, steps)
      smooth_idx = base + 3 + (1 if self.use_cfg else 0)
    else:
      terr_xy_in = torch.zeros(1, 2, steps)
      temp_in = x[:1, base, :]
      cfg_in = torch.ones(1, steps)
      smooth_idx = base + 1
    feat_smooth = float(torch.clamp(x[0, smooth_idx, -1], 0.0, 0.999)) if self.use_feat_smooth else 0.0

    # Reseed trigger: rising edge (low->high) re-anchors the prior to a fresh phrase. Detect across
    # this block AND the inter-call boundary so a one-sample bang/ramp is never missed.
    if self.use_reseed:
      reseed_idx = smooth_idx + (1 if self.use_feat_smooth else 0)
      last = float(self._reseed_prev.item())
      fired = False
      for t in range(steps):
        v = float(x[0, reseed_idx, t])
        if v > 0.5 and last <= 0.5:
          fired = True
        last = v
      self._reseed_prev.fill_(last)
      if fired:
        self._reanchor()

    current_len = int(self._current_len.item())
    emit_idx    = int(self._emit_idx.item())
    ctrl_buf    = self._ctrl_buf.clone()
    ctrl_pos    = int(self._ctrl_pos.item())
    pending     = self._pending_context            # [1, D], predicts token at index current_len
    pending_u   = self._pending_context_uncond     # [1, D], unconditioned (null-territory) pass

    out = torch.zeros(1, steps, self.num_controls)
    frames_written = 0
    CR = self.compression_ratio

    for _iter in range(steps + CR):
      if frames_written >= steps:
        break

      if ctrl_pos >= CR:
        while current_len < emit_idx + 1 + self.decode_lookahead:
          ti = int(min(steps - 1, frames_written))
          # Per-token conditioning sampled from the (slow) input controls at the current frame.
          cond_tok = cond_in[:, ti:ti + 1, :] if self.use_cond else None      # [1,1,cond_dim]
          tvec = self._territory_vec(terr_xy_in[0, :, ti]) if self.use_terr_map else None  # [1,D]
          temp = float(torch.clamp(temp_in[:, ti], min=1e-4).item())
          cfg = float(cfg_in[0, ti])

          if current_len >= self.max_len:
            shift = current_len - self.init_primer_len
            self.token_buffer[:, :self.init_primer_len, :] = self.token_buffer[:, shift:current_len, :].clone()
            current_len = self.init_primer_len
            emit_idx = emit_idx - shift
            self.kv.reset()
            reprimed = self.kv.decode_context(self.token_buffer[:, :current_len, :], 0, None, tvec)
            pending = reprimed[:, -1, :]
            if self.use_cfg:
              self.kv_uncond.reset()
              reprimed_u = self.kv_uncond.decode_context(self.token_buffer[:, :current_len, :], 0, None, self.null_terr_vec)
              pending_u = reprimed_u[:, -1, :]

          # Sample the next frame's codebooks (CFG-guided when enabled).
          if self.use_cfg:
            samples = self.kv.depth_sample_last_cfg(pending, pending_u, temp, 1.0, cfg)  # [1, N]
          else:
            samples = self._sample_next(pending, temp)                                   # [1, N]
          self.token_buffer[:, current_len:current_len + 1, :] = samples.unsqueeze(1)
          # Advance the cache(s) by feeding the just-sampled token with its conditioning.
          advanced = self.kv.decode_context(samples.unsqueeze(1), 0, cond_tok, tvec)
          pending = advanced[:, -1, :]
          if self.use_cfg:
            advanced_u = self.kv_uncond.decode_context(samples.unsqueeze(1), 0, cond_tok, self.null_terr_vec)
            pending_u = advanced_u[:, -1, :]
          current_len += 1

        lo = max(0, emit_idx - self.decode_ctx)
        hi = emit_idx + 1 + self.decode_lookahead
        decode_win = self.token_buffer[:, lo:hi, :].clamp(min=0, max=self.codebook_size - 1)
        decoded    = self.compressor.decode_codes(decode_win)
        off        = emit_idx - lo
        ctrl_buf   = decoded[:, off * CR:(off + 1) * CR, :].detach()
        ctrl_pos   = 0
        emit_idx  += 1

      take = min(steps - frames_written, CR - ctrl_pos)
      out[:, frames_written:frames_written + take, :] = ctrl_buf[:, ctrl_pos:ctrl_pos + take, :]
      frames_written += take
      ctrl_pos       += take

    out = out.permute(0, 2, 1)  # [1, D, steps]

    # Causal one-pole low-pass over ALL control trajectories (loudness/centroid/latents) at control
    # rate, state carried across calls. A global "smoothness" macro from the Smoothing input channel
    # (0=off .. ~0.95 very sluggish).
    a = feat_smooth
    if a > 0.0:
      ema = self._feat_ema  # [1, num_controls]
      for t in range(steps):
        ema = a * ema + (1.0 - a) * out[:, :, t]
        out[:, :, t] = ema
      self._feat_ema.copy_(ema)

    self._current_len.fill_(current_len)
    self._emit_idx.fill_(emit_idx)
    self._pending_context.copy_(pending)
    if self.use_cfg:
      self._pending_context_uncond.copy_(pending_u)
    self._ctrl_buf.copy_(ctrl_buf)
    self._ctrl_pos.fill_(ctrl_pos)

    if x.size(0) > 1:
      out = out.repeat_interleave(x.size(0), dim=0)
    if self.resample_ratio != 1:
      out = F.interpolate(out, scale_factor=self.resample_ratio, mode='linear')

    return out.float()


class ONNXDDSP(torch.nn.Module):
  def __init__(self,
               pretrained: DDSP):
    super().__init__()

    self.pretrained = pretrained

  def decode(self, latents: torch.Tensor):
    synth_params = self.pretrained.decoder(latents.permute(0, 2, 1))
    audio = self.pretrained._synthesize(synth_params)
    return audio

  def encode(self, audio: torch.Tensor):
    if self.pretrained.encoder is None:
      raise RuntimeError("encode() requested but the pretrained model has no encoder (latent_size=0)")
    mu, scale = self.pretrained.encoder(audio.squeeze(1))
    latents, _ = self.pretrained.encoder.reparametrize(mu, scale)
    return latents.permute(0, 2, 1)

  def forward(self, audio: torch.Tensor):
    return self.pretrained(audio.squeeze(1))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default=None,
                      help='Experiment config (name in configs/ or a path). Derives all checkpoint '
                           'paths, prior_kind, target_fs and output_path from experiment.name/training_dir.')
  parser.add_argument('--model_directory', type=str, default=None, help='DDSP synth training dir (derived from --config if omitted)')
  parser.add_argument('--prior_directory', type=str, default=None, help='Prior training dir (derived from --config if omitted)')
  parser.add_argument('--output_path', type=str, default=None, help='Output .ts path (derived from --config if omitted)')
  parser.add_argument('--streaming', type=bool, default=True, help='Whether to use streaming mode')
  parser.add_argument('--type', default='best', help='Type of model to export', choices=['best', 'last'])
  parser.add_argument('--target_fs', type=float, default=None, help='Target sampling rate (defaults to the model fs from --config)')
  parser.add_argument('--prior_kind', default=None, choices=['mulaw', 'discrete'], help='Derived from cfg.prior.discrete.enabled if omitted')
  parser.add_argument('--compressor_checkpoint', type=str, default=None, help='LatentCompressor checkpoint (derived from --config for prior_kind=discrete)')
  config = parser.parse_args()

  # Derive any unset options from the experiment config (explicit args win).
  if config.config is not None:
    cfg_path = config.config
    if not os.path.isfile(cfg_path):
      _name = config.config if config.config.endswith('.yaml') else config.config + '.yaml'
      cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', _name)
    with open(cfg_path) as f:
      _cfg = yaml.safe_load(f)
    _exp = _cfg.get('experiment', {}) or {}
    name = _exp.get('name'); tdir = _exp.get('training_dir', 'training')
    _disc = (_cfg.get('prior', {}) or {}).get('discrete', {}) or {}
    if config.prior_kind is None:
      config.prior_kind = 'discrete' if bool(_disc.get('enabled', False)) else 'mulaw'
    if config.model_directory is None:
      config.model_directory = os.path.join(tdir, 'synth', name)
    if config.prior_directory is None:
      sub = 'prior_discrete' if config.prior_kind == 'discrete' else 'prior'
      config.prior_directory = os.path.join(tdir, sub, name)
    if config.compressor_checkpoint is None and config.prior_kind == 'discrete':
      config.compressor_checkpoint = _disc.get('compressor_ckpt', None) or os.path.join(tdir, 'compressor', name, 'best.ckpt')
    if config.target_fs is None:
      config.target_fs = float((_cfg.get('audio', {}) or {}).get('fs', 44100))
    if config.output_path is None:
      config.output_path = os.path.join('models', f'{name}.ts')
    print(f"[config={config.config}] kind={config.prior_kind} model_dir={config.model_directory} "
          f"prior_dir={config.prior_directory} compressor={config.compressor_checkpoint} "
          f"target_fs={config.target_fs} out={config.output_path}")

  # Fallbacks / validation when running without --config.
  if config.prior_kind is None:
    config.prior_kind = 'mulaw'
  if config.target_fs is None:
    config.target_fs = 16000.0
  if config.model_directory is None:
    parser.error('--model_directory is required when --config is not given')
  if config.output_path is None:
    parser.error('--output_path is required when --config is not given')

  cc.use_cached_conv(config.streaming)

  checkpoint_path = find_checkpoint(config.model_directory, typ=config.type)
  print(f"exporting model from checkpoint: {checkpoint_path}")

  format = config.output_path.split('.')[-1]
  if format not in ['ts', 'onnx']:
    raise ValueError(f'Invalid format: {format}, supported formats are: ts, onnx')

  prior = None
  prior_discrete = None
  compressor = None
  if config.prior_directory is not None:
    prior_checkpoint_path = None
    if config.prior_kind == 'discrete':
      if config.type == 'best':
        # Prefer best_acc checkpoint for discrete prior.
        for root, _, files in os.walk(config.prior_directory):
          for file in files:
            if 'best_acc' in file and file.endswith('.ckpt'):
              p = os.path.join(root, file)
              if prior_checkpoint_path is None or os.path.getctime(p) > os.path.getctime(prior_checkpoint_path):
                prior_checkpoint_path = p
      if prior_checkpoint_path is None:
        prior_checkpoint_path = find_checkpoint(config.prior_directory, typ=config.type)
      print("exporting discrete prior model from checkpoint: ", prior_checkpoint_path)

      if config.compressor_checkpoint is None:
        raise RuntimeError("--compressor_checkpoint is required when --prior_kind=discrete")

      prior_discrete = PriorDiscrete.load_from_checkpoint(prior_checkpoint_path, strict=False).to('cpu')
      prior_discrete.eval()
      prior_discrete._trainer = L.Trainer()

      compressor_full = LatentCompressor.load_from_checkpoint(config.compressor_checkpoint, strict=False).to('cpu')
      compressor_full.eval()
      compressor_full._trainer = L.Trainer()

      if getattr(compressor_full, 'use_skip_connections', True):
        raise RuntimeError("LatentCompressor must have use_skip_connections=False for codes-only decoding")
      if getattr(compressor_full, 'vq', None) is None:
        raise RuntimeError("LatentCompressor must have vq_enabled=True for discrete-prior export")

      compressor = LatentCompressorDecodeOnly(
        vq=compressor_full.vq,
        decoder=compressor_full.decoder,
        compression_ratio=int(getattr(compressor_full, 'compression_ratio', 32)),
      ).to('cpu')
      compressor.eval()
    else:
      prior_checkpoint_path = find_checkpoint(config.prior_directory, typ=config.type)
      print("exporting prior model from checkpoint: ", prior_checkpoint_path)
      prior = Prior.load_from_checkpoint(prior_checkpoint_path, strict=False).to('cpu')
      prior.eval()
      if prior._normalization_dict is not None:
        for k, v in prior._normalization_dict.items():
          if torch.is_tensor(v):
            prior._normalization_dict[k] = v.to('cpu')

      prior._trainer = L.Trainer()

  # Reconstruct minimal ControlSpace from checkpoint hyperparameters
  ckpt = torch.load(checkpoint_path, map_location='cpu')
  hparams = ckpt.get('hyper_parameters', {})
  feature_dim = int(hparams.get('feature_dim', 0))
  latent_size = int(hparams.get('latent_size', 0))
  fields = []
  if feature_dim > 0:
    fields.append(ControlField(name='features', dim=feature_dim, source='feature', extractor=None))
  if latent_size > 0:
    fields.append(ControlField(name='latents', dim=latent_size, source='latent', extractor=None))
  if len(fields) == 0:
    raise RuntimeError("Checkpoint missing feature_dim/latent_size hparams; cannot reconstruct ControlSpace for export.")
  control_space = ControlSpace(tuple(fields))

  ddsp = DDSP.load_from_checkpoint(checkpoint_path, strict=False, streaming=True, device='cpu', control_space=control_space).to('cpu')
  ddsp.streaming(True)

  if config.prior_kind == 'discrete' and prior_discrete is not None:
    prior = PriorDiscreteWrapper(prior_discrete, compressor, resample_ratio=(config.target_fs / float(ddsp.fs)),
                                 n_feature_channels=feature_dim,
                                 terr_map_temp=float(_disc.get('terr_map_temp', 0.5)),
                                 decode_lookahead=int(_disc.get('decode_lookahead', 2)))

  if format == 'onnx':
    ddsp.eval()
    scripted = ONNXDDSP(ddsp).to('cpu')
    torch.onnx.dynamo_export(
      scripted,
      torch.zeros(1, ddsp.n_channels, 2**14),
    ).save(config.output_path)
  elif format == 'ts':
    # ugly workaround for the torchscript
    ddsp._trainer = L.Trainer()
    ddsp._recons_loss = None
    ddsp._mr_stft_loss = None
    ddsp._mr_mel_loss = None
    ddsp._mr_chroma_loss = None
    ddsp._m2l_loss = None
    ddsp._discriminator = None
    ddsp._sliced_wasserstein_loss = None

    ddsp.eval()


    scripted = ScriptedDDSP(ddsp, prior, config.target_fs).to('cpu')
    # Registering/scripting the nn~ methods can advance the discrete-prior wrapper's
    # token buffer with junk; clear it so the saved model starts from a clean START.
    if isinstance(prior, PriorDiscreteWrapper):
      prior.reset_state()
      print("discrete prior wrapper state after reset:",
            int(prior._current_len.item()), int(prior._emit_idx.item()))
    scripted.export_to_ts(config.output_path)

    print("Model exported to: ", config.output_path)
