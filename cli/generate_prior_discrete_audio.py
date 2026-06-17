"""
Offline generation: discrete prior -> latent compressor -> DDSP synth -> WAV.

Mirrors the inference path used by the nn~ export so you can audition a trained
codec prior outside Max/MSP. Supports mono and multichannel (e.g. stereo) models.
"""
import argparse
import math
import os
import sys
import tarfile
from typing import Optional, Tuple

# Ensure THIS repo's `ddsp` is imported (not a globally pip-installed one) when
# run as `python cli/generate_prior_discrete_audio.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa

from ddsp import DDSP
from ddsp.interfaces import ControlField, ControlSpace, build_control_space
from ddsp.latent_compressor import LatentCompressor
from ddsp.prior import PriorDiscrete
from ddsp.registry import FEATURE_EXTRACTORS


def _build_control_space_from_ddsp_ckpt(ddsp_ckpt_path: str) -> Tuple[ControlSpace, int, int]:
  ckpt = torch.load(ddsp_ckpt_path, map_location='cpu')
  h = ckpt.get('hyper_parameters', {})
  feature_dim = int(h.get('feature_dim', 0) or 0)
  latent_size = int(h.get('latent_size', 0) or 0)

  fields = []
  if feature_dim > 0:
    fields.append(ControlField(name='features', dim=feature_dim, source='feature', extractor=None))
  if latent_size > 0:
    fields.append(ControlField(name='latents', dim=latent_size, source='latent', extractor=None))
  return ControlSpace(tuple(fields)), feature_dim, latent_size


def _extract_features_audio_rate(x_audio: torch.Tensor, fs: int, control_space: ControlSpace) -> torch.Tensor:
  """Compute audio-rate feature matrix [T, D_feat] following ControlSpace spec.

  x_audio is mono [T].
  """
  feats = []
  for field in control_space.fields:
    if field.source != 'feature':
      continue
    if not field.extractor:
      raise RuntimeError(f"ControlField '{field.name}' missing extractor; pass a ControlSpace built from config")

    extractor = FEATURE_EXTRACTORS.create(field.extractor, **(field.params or {}))
    feat = extractor(x_audio, fs)
    if feat.ndim == 1:
      feat = feat.unsqueeze(-1)

    norm = dict(field.normalization) if field.normalization is not None else {}
    if 'mean' in norm and 'std' in norm:
      mean = torch.as_tensor(norm['mean'], device=feat.device, dtype=feat.dtype)
      std = torch.as_tensor(norm['std'], device=feat.device, dtype=feat.dtype)
      feat = (feat - mean) / (std + 1e-8)
    elif 'min' in norm and 'max' in norm:
      minv = torch.as_tensor(norm['min'], device=feat.device, dtype=feat.dtype)
      maxv = torch.as_tensor(norm['max'], device=feat.device, dtype=feat.dtype)
      feat = (feat - minv) / (maxv - minv + 1e-8)

    feats.append(feat)

  if not feats:
    return x_audio.new_zeros((x_audio.shape[0], 0))
  return torch.cat(feats, dim=-1)


def _load_audio_multichannel(wav_path: str, fs: int, n_channels: int, seconds: float, offset_s: float,
                             device: str) -> Tuple[torch.Tensor, torch.Tensor]:
  """Load audio coerced to n_channels. Returns (multi [n_ch, T], mono [T])."""
  x_np, _ = librosa.load(wav_path, sr=int(fs), mono=False, offset=float(offset_s), duration=float(seconds))
  x = torch.from_numpy(np.atleast_2d(x_np)).to(device=device, dtype=torch.float32)  # [c, T]

  need = int(round(float(seconds) * float(fs)))
  if x.shape[-1] < need:
    x = F.pad(x, (0, need - x.shape[-1]))

  c = x.shape[0]
  if c == n_channels:
    multi = x
  elif c == 1:
    multi = x.repeat(n_channels, 1)
  elif c > n_channels:
    multi = x[:n_channels]
  else:  # 1 < c < n_channels
    multi = torch.cat([x, x[-1:].repeat(n_channels - c, 1)], dim=0)

  mono = multi.mean(dim=0)  # [T]
  return multi, mono


@torch.no_grad()
def _prime_tokens_from_wav(
  *,
  ddsp: DDSP,
  compressor: LatentCompressor,
  control_space: ControlSpace,
  feature_dim: int,
  latent_size: int,
  wav_path: str,
  seconds: float,
  offset_s: float,
  device: str,
) -> torch.Tensor:
  n_channels = int(getattr(ddsp, 'n_channels', 1))
  x_multi, x_mono = _load_audio_multichannel(wav_path, int(ddsp.fs), n_channels, seconds, offset_s, device)

  # features: audio-rate (from mono) -> control-rate
  x_feat_audio = _extract_features_audio_rate(x_mono, int(ddsp.fs), control_space)  # [T_audio, D_feat]
  T_ctl = int(math.ceil(x_mono.shape[0] / float(ddsp.resampling_factor)))
  if x_feat_audio.shape[-1] != feature_dim:
    raise RuntimeError(f'feature_dim mismatch: expected {feature_dim}, got {x_feat_audio.shape[-1]}')

  x_feat_ctl = F.interpolate(
    x_feat_audio.to(device).T.unsqueeze(0),
    size=T_ctl,
    mode='linear',
    align_corners=False,
  ).squeeze(0).T  # [T_ctl, D_feat]

  if ddsp.encoder is None:
    raise RuntimeError('Cannot prime: DDSP model has no encoder')

  # Encoder downmixes the [1, n_channels, T] input internally.
  mu, scale = ddsp.encoder(x_multi.unsqueeze(0))
  z, _ = ddsp.encoder.reparametrize(mu, scale)
  z = ddsp._smooth_latents(z)

  T_min = min(int(x_feat_ctl.shape[0]), int(z.shape[1]))
  x_feat_ctl = x_feat_ctl[:T_min].unsqueeze(0)
  z = z[:, :T_min, :]

  if z.shape[-1] != latent_size:
    raise RuntimeError(f'latent_size mismatch: expected {latent_size}, got {z.shape[-1]}')

  controls = torch.cat([x_feat_ctl, z], dim=-1)[:, :, :(feature_dim + latent_size)]  # [1, T, D]
  tokens = compressor.encode_codes(controls)
  if tokens.dim() == 2:
    tokens = tokens.unsqueeze(-1)
  return tokens


@torch.no_grad()
def _make_lfo_primer_tokens(compressor, feature_dim, latent_size, control_rate, seconds, beat_hz, device):
  """Build beat-synced LFO trajectories on each control channel, encode to tokens.
  loudness = peaky kick at the beat; centroid = slow sine; latents = slower synced sines.
  This is synthetic, controllable priming (no audio in)."""
  T = int(seconds * control_rate)
  D = int(feature_dim + latent_size)
  t = torch.arange(T, device=device).float() / float(control_rate)
  ph = 2.0 * math.pi * float(beat_hz) * t
  chans = []
  # Features: loudness as a peaky per-beat pulse, centroid as a half-beat sine.
  if feature_dim >= 1:
    chans.append(0.08 + 0.55 * ((1 + torch.sin(ph - math.pi / 2)) / 2) ** 3)  # kick on the beat
  if feature_dim >= 2:
    chans.append(0.55 + 0.20 * torch.sin(0.5 * ph))                          # brightness sway
  for c in range(2, feature_dim):
    chans.append(0.5 + 0.2 * torch.sin((c) * ph))
  # Latents: slower, beat-synced sines at distinct rates/phases.
  lat_rates = [0.5, 0.25, 1.0, 0.75]
  for i in range(latent_size):
    r = lat_rates[i % len(lat_rates)]
    chans.append(0.0 + 0.5 * torch.sin(r * ph + i * math.pi / 3.0))
  ctrl = torch.stack(chans, dim=-1)[:, :D].unsqueeze(0)  # [1, T, D]
  tok = compressor.encode_codes(ctrl)
  if tok.dim() == 2:
    tok = tok.unsqueeze(-1)
  return tok  # [1, T_low, N]


def _make_lfo_cond_env(n_points, rate, feature_dim, latent_size, beat_hz, device):
  """Beat-synced LFO control envelope at token rate, [1, n_points, D]. Same channel
  shapes as the LFO primer, used as continuous conditioning (the user's control surface)."""
  D = int(feature_dim + latent_size)
  t = torch.arange(n_points, device=device).float() / float(rate)
  ph = 2.0 * math.pi * float(beat_hz) * t
  chans = []
  if feature_dim >= 1:
    chans.append(0.08 + 0.55 * ((1 + torch.sin(ph - math.pi / 2)) / 2) ** 3)
  if feature_dim >= 2:
    chans.append(0.55 + 0.20 * torch.sin(0.5 * ph))
  for c in range(2, feature_dim):
    chans.append(0.5 + 0.2 * torch.sin((c) * ph))
  lat_rates = [0.5, 0.25, 1.0, 0.75]
  for i in range(latent_size):
    chans.append(0.0 + 0.5 * torch.sin(lat_rates[i % len(lat_rates)] * ph + i * math.pi / 3.0))
  return torch.stack(chans, dim=-1)[:, :D].unsqueeze(0)  # [1, n_points, D]


@torch.no_grad()
def _sample_tokens(
  prior: PriorDiscrete,
  n_tokens: int,
  primer_len: int,
  temperature: float,
  sampling: str,
  device: str,
  primer_tokens: Optional[torch.Tensor] = None,
  territory: int = -1,
  top_p: float = 1.0,
  reset_every: int = 0,
  persist_primer: bool = False,
  cond_env: Optional[torch.Tensor] = None,
  territory_vec_env: Optional[torch.Tensor] = None,
  cfg_scale: float = 1.0,
  lfo_cfg: float = 1.0,
) -> torch.Tensor:
  codebook_size = int(prior.codebook_size)
  num_codebooks = int(prior.num_codebooks)
  max_len = int(getattr(prior, '_max_len', 256))
  start_id = int(getattr(prior, 'start_token_id', codebook_size))

  if n_tokens < 1:
    raise ValueError('n_tokens must be >= 1')

  sampling = str(sampling).lower()
  if sampling not in ('multinomial', 'argmax'):
    raise ValueError("sampling must be 'multinomial' or 'argmax'")

  # Position 0 is the learned START token (matches the nn~ wrapper cold start).
  # A real-audio primer (if given) follows START; generation fills the rest.
  buf = torch.full((1, n_tokens + 1, num_codebooks), start_id, dtype=torch.long, device=device)

  if primer_tokens is None:
    gen_start = 1
  else:
    if primer_tokens.dtype != torch.long:
      primer_tokens = primer_tokens.long()
    if primer_tokens.dim() != 3 or primer_tokens.shape[-1] != num_codebooks:
      raise ValueError('primer_tokens must have shape [1, T, num_codebooks]')
    L = min(int(primer_len), int(primer_tokens.shape[1]), int(n_tokens))
    buf[:, 1:1 + L, :] = primer_tokens[:, :L, :].to(device)
    gen_start = 1 + L

  tid = None
  if territory is not None and territory >= 0 and getattr(prior, '_num_territories', 0) > 0:
    tid = torch.tensor([int(territory)], dtype=torch.long, device=device)
  # Classifier-free guidance: amplify the conditioned distribution away from the unconditioned
  # (NULL-territory) one. Requires a prior trained with cfg_dropout (so the null row is learned).
  cfg_on = (float(cfg_scale) != 1.0 and tid is not None
            and getattr(prior, '_cfg_dropout', 0.0) > 0.0)
  null_tid = (torch.tensor([int(getattr(prior, '_cfg_null', 0))], dtype=torch.long, device=device)
              if cfg_on else None)

  start_col = torch.full((1, 1, num_codebooks), int(start_id), dtype=torch.long, device=device)
  anchor = gen_start  # index of the first token in the current (post-reset) phrase
  # Persistent primer: keep START+primer pinned at the front of every context window so the
  # prior is *continuously* conditioned on the (LFO) scaffold, not just seeded once.
  persist_head = (1 + primer_len) if (persist_primer and primer_tokens is not None) else 0

  for t in range(gen_start, n_tokens + 1):
    if persist_head > 0:
      head = buf[:, :persist_head, :]
      tail_lo = max(persist_head, t - (max_len - persist_head))
      ctx = torch.cat([head, buf[:, tail_lo:t, :]], dim=1)[:, -max_len:, :]
      logits = prior(ctx, territory_id=tid)
      next_logits = logits[:, -1, :, :]
      temp = max(1e-4, float(temperature))
      probs = torch.softmax(next_logits / temp, dim=-1)
      buf[:, t, :] = torch.multinomial(probs.reshape(-1, codebook_size), 1).reshape(1, num_codebooks)
      continue
    cond_slice = None
    if reset_every > 0:
      if t - anchor >= reset_every:
        anchor = t  # start a fresh phrase: drop drifted history, re-anchor to START
      # context = START + tokens since the anchor (in-distribution "phrase start"), capped to max_len
      ctx = torch.cat([start_col, buf[:, anchor:t, :]], dim=1)[:, -max_len:, :]
    else:
      lo = max(0, t - max_len)
      ctx = buf[:, lo:t, :]
      if cond_env is not None:
        cond_slice = cond_env[:, lo:t, :]
    tvec = None
    if territory_vec_env is not None:
      tvec = territory_vec_env[:, min(t - 1, territory_vec_env.shape[1] - 1), :]  # [1, D] blend at this step
    # Joint codebook head: decode the N codebooks autoregressively from the time-context (on-manifold).
    # LFO-CFG: guide toward the cond envelope (uncond = cond zeroed). Strengthens LFO grip so it
    # does not dissipate as the context fills (needs a cond_dropout-trained model). Takes precedence
    # over territory CFG when active.
    lfo_cfg_on = (float(lfo_cfg) != 1.0 and cond_slice is not None
                  and getattr(prior, '_cond_dropout', 0.0) > 0.0)
    if getattr(prior, 'is_joint', False):
      h = prior.time_context(ctx, territory_id=tid, cond=cond_slice, territory_vec=tvec)[:, -1, :]  # [1, D]
      hu = None; gscale = float(cfg_scale)
      if lfo_cfg_on:
        hu = prior.time_context(ctx, territory_id=tid, cond=torch.zeros_like(cond_slice), territory_vec=tvec)[:, -1, :]
        gscale = float(lfo_cfg)
      elif cfg_on:
        hu = prior.time_context(ctx, territory_id=null_tid, cond=cond_slice)[:, -1, :]
      buf[:, t, :] = prior.depth_decode_last(h, temperature=float(temperature), top_p=float(top_p),
                                             h_last_uncond=hu, cfg_scale=gscale)
      continue

    logits = prior(ctx, territory_id=tid, cond=cond_slice, territory_vec=tvec)  # [1, S, N, K]
    next_logits = logits[:, -1, :, :]  # [1, N, K]
    if lfo_cfg_on:
      ul = prior(ctx, territory_id=tid, cond=torch.zeros_like(cond_slice), territory_vec=tvec)[:, -1, :, :]
      next_logits = ul + float(lfo_cfg) * (next_logits - ul)
    elif cfg_on:
      ul = prior(ctx, territory_id=null_tid, cond=cond_slice)[:, -1, :, :]  # uncond [1, N, K]
      next_logits = ul + float(cfg_scale) * (next_logits - ul)

    if sampling == 'argmax':
      buf[:, t, :] = torch.argmax(next_logits, dim=-1)
      continue

    temp = max(1e-4, float(temperature))
    probs = torch.softmax(next_logits / temp, dim=-1)  # [1, N, K]
    if top_p < 1.0:
      # Nucleus filter per codebook: keep the smallest set of tokens whose cumulative
      # probability >= top_p, renormalise, sample from those (anti-drift on-manifold).
      sp, si = torch.sort(probs, dim=-1, descending=True)
      csum = sp.cumsum(dim=-1)
      keep = csum - sp <= top_p  # always keeps the top-1
      sp = sp * keep
      sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
      probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    samp = torch.multinomial(probs.reshape(-1, codebook_size), 1).reshape(1, num_codebooks)
    buf[:, t, :] = samp

  # Drop the START token; return the n_tokens real tokens.
  return buf[:, 1:, :]


def main():
  ap = argparse.ArgumentParser(description='Generate audio from a discrete prior -> latent compressor -> DDSP synth.')
  ap.add_argument('--prior_ckpt', type=str, required=True, help='Path to PriorDiscrete checkpoint (.ckpt).')
  ap.add_argument('--ddsp_ckpt', type=str, required=True, help='Path to DDSP checkpoint (.ckpt).')
  ap.add_argument('--ddsp_config', type=str, default='configs/experiment_hybrid.yaml',
                  help='DDSP YAML config used to build ControlSpace (required for priming feature extraction).')
  ap.add_argument('--compressor_ckpt', type=str, required=True, help='Path to LatentCompressor checkpoint (.ckpt).')

  ap.add_argument('--seconds', type=float, default=30.0, help='How many seconds to generate (approx).')
  ap.add_argument('--seed', type=int, default=0, help='Random seed for sampling.')
  ap.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (used for multinomial sampling).')
  ap.add_argument('--territory', type=int, default=-1, help='Territory/zone to condition on (-1 = unconditioned; requires a territory-trained prior).')
  ap.add_argument('--top_p', type=float, default=1.0, help='Nucleus sampling threshold per codebook (1.0 = off).')
  ap.add_argument('--reset_every', type=int, default=0, help='Re-anchor context to START every N tokens (0 = off) to prevent progressive drift.')
  ap.add_argument('--prime_lfo', action='store_true', help='Prime with synthetic beat-synced LFO control trajectories (no audio in).')
  ap.add_argument('--lfo_beat_hz', type=float, default=2.0, help='LFO beat rate in Hz (2.0 = 120 bpm).')
  ap.add_argument('--persist_primer', action='store_true', help='Keep the primer pinned at the front of every context (continuous scaffold).')
  ap.add_argument('--cond_lfo', action='store_true', help='Drive a cond-trained prior with a beat-synced LFO control envelope.')
  ap.add_argument('--cond_lfo_beat_hz', type=float, default=2.0, help='LFO beat rate for envelope conditioning.')
  ap.add_argument('--lfo_amount', type=float, default=1.0, help='Scale the LFO envelope (1=full, 0=freeform; needs a cond_dropout-trained model for clean 0).')
  ap.add_argument('--lfo_cfg', type=float, default=1.0, help='LFO classifier-free guidance (1=off, >1 strengthens LFO adherence so it does not dissipate; needs cond_dropout).')
  ap.add_argument('--territory_a', type=int, default=-1, help='Interpolation start territory (with --territory_b).')
  ap.add_argument('--territory_b', type=int, default=-1, help='Interpolation end territory; output morphs A->B over the clip.')
  ap.add_argument('--cfg_scale', type=float, default=1.0, help='Classifier-free guidance scale (1.0 = off; >1 amplifies territory faithfulness; needs a cfg_dropout-trained prior).')
  ap.add_argument('--sampling', type=str, default='multinomial', choices=['multinomial', 'argmax'],
                  help="Sampling strategy: 'multinomial' (stochastic) or 'argmax' (greedy).")
  ap.add_argument('--primer_frac', type=float, default=0.25, help='Primer length as a fraction of prior max_len.')

  ap.add_argument('--prime_wav', type=str, default='',
                  help='Optional: path to a WAV file used to prime the prior (real tokens for the primer window).')
  ap.add_argument('--prime_seconds', type=float, default=4.0, help='How many seconds of prime_wav to use.')
  ap.add_argument('--prime_offset_s', type=float, default=0.0, help='Offset (seconds) into prime_wav.')

  ap.add_argument('--target_fs', type=int, default=0, help='Output WAV sample rate (0 = keep DDSP fs).')
  ap.add_argument('--out_dir', type=str, default='outputs/generated_prior_discrete', help='Output directory.')
  ap.add_argument('--prefix', type=str, default='', help='Optional filename prefix. If empty, derived from settings.')
  ap.add_argument('--device', type=str, default='', help="'cuda' or 'cpu'. Default: auto.")

  ap.add_argument('--no_plots', action='store_true', help='Skip the diagnostic plots.')
  ap.add_argument('--make_archive', action='store_true', help='Create a .tgz archive of the artifacts.')
  args = ap.parse_args()

  device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(int(args.seed))
  np.random.seed(int(args.seed))

  os.makedirs(args.out_dir, exist_ok=True)

  # Build ControlSpace. For priming we need real feature extractors, so prefer config.
  if args.ddsp_config and os.path.exists(args.ddsp_config):
    with open(args.ddsp_config, 'r') as f:
      ddsp_cfg = yaml.safe_load(f)
    control_space = build_control_space(ddsp_cfg['model']['control_space'])
    _, feature_dim, latent_size = _build_control_space_from_ddsp_ckpt(args.ddsp_ckpt)
  else:
    control_space, feature_dim, latent_size = _build_control_space_from_ddsp_ckpt(args.ddsp_ckpt)

  ddsp = DDSP.load_from_checkpoint(
    args.ddsp_ckpt,
    strict=False,
    streaming=False,
    device=device,
    control_space=control_space,
  ).to(device)
  ddsp.eval()

  compressor = LatentCompressor.load_from_checkpoint(args.compressor_ckpt, strict=False).to(device)
  compressor.eval()

  prior = PriorDiscrete.load_from_checkpoint(args.prior_ckpt, strict=False).to(device)
  prior.eval()

  compression_ratio = int(getattr(compressor, 'compression_ratio', 32))
  max_len = int(getattr(prior, '_max_len', 256))
  n_channels = int(getattr(ddsp, 'n_channels', 1))

  control_rate = float(ddsp.fs) / float(ddsp.resampling_factor)
  tokens_per_sec = control_rate / float(compression_ratio)
  n_tokens = int(math.ceil(float(args.seconds) * tokens_per_sec))

  primer_len = max(1, int(round(float(args.primer_frac) * max_len)))
  primer_len = min(primer_len, max_len)

  if not args.prefix:
    args.prefix = f'prior_discrete_seed{args.seed}_{int(args.seconds)}s_{args.sampling}_t{args.temperature}'

  print('device:', device)
  print('ddsp.fs:', ddsp.fs, 'resampling_factor:', ddsp.resampling_factor, 'control_rate:', control_rate)
  print('n_channels:', n_channels, 'feature_dim:', feature_dim, 'latent_size:', latent_size)
  print('compression_ratio:', compression_ratio, 'tokens_per_sec:', round(tokens_per_sec, 3))
  print('n_tokens:', n_tokens, 'prior_max_len:', max_len, 'primer_len:', primer_len)

  primer_tokens = None
  if args.prime_lfo:
    print('priming from synthetic LFO trajectories, beat_hz:', args.lfo_beat_hz)
    primer_tokens = _make_lfo_primer_tokens(
      compressor=compressor, feature_dim=feature_dim, latent_size=latent_size,
      control_rate=control_rate, seconds=float(args.prime_seconds),
      beat_hz=float(args.lfo_beat_hz), device=device,
    )
    print('lfo prime_tokens_len:', int(primer_tokens.shape[1]))
    if primer_tokens.shape[1] < primer_len:
      primer_len = int(primer_tokens.shape[1])
  elif args.prime_wav:
    print('priming from wav:', args.prime_wav)
    primer_tokens = _prime_tokens_from_wav(
      ddsp=ddsp,
      compressor=compressor,
      control_space=control_space,
      feature_dim=feature_dim,
      latent_size=latent_size,
      wav_path=args.prime_wav,
      seconds=float(args.prime_seconds),
      offset_s=float(args.prime_offset_s),
      device=device,
    )
    print('prime_tokens_len:', int(primer_tokens.shape[1]))
    if primer_tokens.shape[1] < primer_len:
      print(f'WARNING: prime_tokens shorter than primer_len; shrinking primer_len to {primer_tokens.shape[1]}')
      primer_len = int(primer_tokens.shape[1])

  print('sampling tokens...')
  cond_env = None
  if args.cond_lfo and int(getattr(prior, '_cond_dim', 0)) > 0:
    cond_env = _make_lfo_cond_env(n_tokens + 1, tokens_per_sec, feature_dim, latent_size,
                                  float(args.cond_lfo_beat_hz), device)
    # LFO amount: scale the envelope. 1.0 = full LFO; 0.0 = zeros = in-distribution freeform
    # (for a cond_dropout-trained "switch" model). cond_proj(0)=bias is still applied (not None).
    cond_env = cond_env * float(args.lfo_amount)
    print('cond LFO envelope:', tuple(cond_env.shape), 'beat_hz', args.cond_lfo_beat_hz, 'amount', args.lfo_amount)

  # Territory interpolation: morph the territory embedding A->B linearly across the clip.
  territory_vec_env = None
  tbl = prior.territory_embedding_table() if hasattr(prior, 'territory_embedding_table') else None
  if args.territory_a >= 0 and args.territory_b >= 0 and tbl is not None:
    a = tbl[int(args.territory_a)]; b = tbl[int(args.territory_b)]  # [D]
    alpha = torch.linspace(0, 1, n_tokens + 1, device=device).unsqueeze(-1)  # [T,1]
    territory_vec_env = ((1 - alpha) * a + alpha * b).unsqueeze(0)  # [1, T, D]
    print(f'territory morph {args.territory_a}->{args.territory_b} over {n_tokens} tokens')

  tokens = _sample_tokens(
    prior,
    n_tokens=n_tokens,
    primer_len=primer_len,
    temperature=args.temperature,
    sampling=args.sampling,
    device=device,
    primer_tokens=primer_tokens,
    territory=int(args.territory),
    top_p=float(args.top_p),
    reset_every=int(args.reset_every),
    persist_primer=bool(args.persist_primer),
    cond_env=cond_env,
    territory_vec_env=territory_vec_env,
    cfg_scale=float(args.cfg_scale),
    lfo_cfg=float(args.lfo_cfg),
  )

  # Diagnostic: fraction of unique tokens (low -> collapsed / "loopy").
  uniq = int(torch.unique(tokens).numel())
  print(f'unique token ids used: {uniq} / {int(prior.codebook_size) * int(prior.num_codebooks)} (low => collapsed/repetitive)')

  print('decoding codes -> controls...')
  controls = compressor.decode_codes(tokens)  # [1, T, D]
  controls = controls[:, :, :(feature_dim + latent_size)]
  features = controls[:, :, :feature_dim]
  latents = controls[:, :, feature_dim:feature_dim + latent_size]

  print('DDSP decode + synth...')
  synth_params = ddsp.decoder(features, latents)
  audio = ddsp._synthesize(synth_params)  # [1, n_channels, T] (or [1, T])
  if audio.dim() == 2:
    audio = audio.unsqueeze(1)

  mx = audio.abs().max().clamp(min=1e-8)
  audio = (audio / mx).clamp(-1, 1)

  orig_fs = int(ddsp.fs)
  target_fs = int(args.target_fs) or orig_fs
  wav = audio[0].cpu()  # [n_channels, T]
  if target_fs != orig_fs:
    print(f'resampling {orig_fs} -> {target_fs} ...')
    wav = torchaudio.functional.resample(wav, orig_freq=orig_fs, new_freq=target_fs)

  wav_path = os.path.join(args.out_dir, f'{args.prefix}.wav')
  torchaudio.save(wav_path, wav, sample_rate=target_fs)
  print('wrote:', wav_path)

  npz_path = os.path.join(args.out_dir, f'{args.prefix}_arrays.npz')
  np.savez(
    npz_path,
    tokens=tokens.detach().cpu().numpy()[0],
    controls=controls.detach().float().cpu().numpy()[0],
    feature_dim=feature_dim,
    latent_size=latent_size,
    ddsp_fs=orig_fs,
    n_channels=n_channels,
    control_rate=control_rate,
    compression_ratio=compression_ratio,
    temperature=float(args.temperature),
    sampling=str(args.sampling),
    seconds=float(args.seconds),
  )
  print('wrote:', npz_path)

  artifacts = [wav_path, npz_path]
  if not args.no_plots:
    controls_np = controls.detach().float().cpu().numpy()[0]
    n = controls_np.shape[0]
    ds = max(1, n // 5000)
    t = (np.arange(n)[::ds] / control_rate)

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i in range(feature_dim):
      ax[0].plot(t, controls_np[::ds, i], label=f'feature[{i}]', linewidth=0.8)
    ax[0].set_title('Generated DDSP features (control-rate)')
    ax[0].legend(ncol=4, fontsize=8)
    ax[0].grid(True, alpha=0.3)
    for i in range(latent_size):
      ax[1].plot(t, controls_np[::ds, feature_dim + i], label=f'latent[{i}]', linewidth=0.8)
    ax[1].set_title('Generated DDSP latents (control-rate)')
    ax[1].legend(ncol=4, fontsize=8)
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xlabel('time (s)')
    controls_png = os.path.join(args.out_dir, f'{args.prefix}_controls.png')
    fig.tight_layout()
    fig.savefig(controls_png, dpi=150)
    plt.close(fig)
    print('wrote:', controls_png)

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.hist(tokens.detach().cpu().numpy().reshape(-1), bins=50)
    ax.set_title('Token histogram (all codebooks)')
    ax.set_xlabel('token id')
    ax.set_ylabel('count')
    ax.grid(True, alpha=0.3)
    hist_png = os.path.join(args.out_dir, f'{args.prefix}_token_hist.png')
    fig.tight_layout()
    fig.savefig(hist_png, dpi=150)
    plt.close(fig)
    print('wrote:', hist_png)
    artifacts += [controls_png, hist_png]

  if args.make_archive:
    archive_path = os.path.join(args.out_dir, f'{args.prefix}_artifacts.tgz')
    with tarfile.open(archive_path, 'w:gz') as tf:
      for p in artifacts:
        tf.add(p, arcname=os.path.basename(p))
    print('wrote:', archive_path)


if __name__ == '__main__':
  main()
