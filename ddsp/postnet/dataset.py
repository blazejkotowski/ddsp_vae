"""Paired (rough, real, control) cache for faithful post-net training.

The post-net is trained on the *frozen* synth's output: for each audio window we run the trained DDSP
model (encoder -> latents -> decoder -> synth) to get the "rough" reconstruction, keep the original
audio as the "real" target, and keep the control trajectory (features + latents at control rate) as
the conditioning the post-net sees at inference. This mirrors the control layout the exported
`decode()` feeds the post-net: `params = [features | latents]` (see cli/export.py::decode).

Dense overlapping windows are used for training; a contiguous tail of the timeline is held out
(non-overlapping) for validation, with a gap so train/val windows never overlap.
"""
import os
import json
import math
import hashlib

import torch

from ddsp.audio_feature_dataset import AudioFeatureDataset


def _cache_key(cfg, synth_ckpt: str) -> str:
  """Stable key over everything that changes the paired cache contents."""
  pn = cfg.get('postnet', {}) or {}
  data = (pn.get('data', {}) or {})
  try:
    mtime = os.path.getmtime(synth_ckpt) if synth_ckpt and os.path.exists(synth_ckpt) else 0
  except OSError:
    mtime = 0
  payload = {
    'dataset': str(cfg.data.dataset_path),
    'fs': int(cfg.audio.fs),
    'n_channels': int(getattr(cfg.audio, 'n_channels', 1)),
    'resampling_factor': int(cfg.model.resampling_factor),
    'chunk_s': float(data.get('chunk_s', 1.0)),
    'hop_s': float(data.get('hop_s', 0.25)),
    'val_fraction': float((pn.get('training', {}) or {}).get('val_fraction', 0.1)),
    'synth_ckpt': str(synth_ckpt),
    'synth_mtime': int(mtime),
  }
  blob = json.dumps(payload, sort_keys=True).encode()
  return hashlib.sha1(blob).hexdigest()[:12]


@torch.no_grad()
def _windows_from_model(model, audio, feats, starts, W, rf, device, batch=8):
  """Run the frozen synth over the given window starts; return (rough, real, cond) lists per channel."""
  feature_dim = int(model.feature_dim)
  roughs, reals, conds = [], [], []
  for i in range(0, len(starts), batch):
    span = starts[i:i + batch]
    aa, ff = [], []
    for s in span:
      aa.append(audio[:, s:s + W])
      if feature_dim > 0:
        fdim = feats[s:s + W]                                   # [W, feature_dim] (audio-rate)
        T_ctl = math.ceil(W / rf)
        fd = torch.nn.functional.interpolate(
          fdim.T.unsqueeze(0), size=T_ctl, mode="linear", align_corners=False).squeeze(0).T
        ff.append(fd)
    a = torch.stack(aa).to(device)                             # [B, C, W]
    if model.encoder is not None:
      mu, scale = model.encoder(a)
      z, _ = model.encoder.reparametrize(mu, scale)
      z = model._smooth_latents(z)                             # [B, T_z, latent]
    else:
      z = None

    if feature_dim > 0:
      f = torch.stack(ff).to(device)                           # [B, T_ctl, feature_dim]
      Tm = f.shape[1] if z is None else min(f.shape[1], z.shape[1])
      fc = f[:, :Tm]
    else:
      Tm = z.shape[1] if z is not None else math.ceil(W / rf)
      fc = torch.zeros(a.shape[0], Tm, 1, device=device)

    if z is not None:
      zc = z[:, :Tm]
      cond_parts = ([fc] if feature_dim > 0 else []) + [zc]
      cond = torch.cat(cond_parts, dim=-1).permute(0, 2, 1)    # [B, feat+lat, Tm]
    else:
      zc = torch.zeros(a.shape[0], Tm, max(int(model.latent_size), 1), device=device)
      cond = fc.permute(0, 2, 1)

    sp = model.decoder(fc, zc)
    r = model._synthesize(sp)                                  # [B, C, T]
    real = a[..., :r.shape[-1]].float()
    C = r.shape[1]
    for c in range(C):
      roughs.append(r[:, c:c + 1, :].cpu())
      reals.append(real[:, c:c + 1, :].cpu())
      conds.append(cond.cpu())
  return roughs, reals, conds


def build_or_load_postnet_cache(cfg, model, device: str):
  """Build (or load) the paired post-net training cache from the frozen DDSP model.

  Returns a dict with train tensors (rough/real/cond) and held-out eval tensors, plus `cond_dim`.
  Cached under `<dataset_dir>/postnet_cache_<name>_<key>/` and reused across runs.
  """
  pn = cfg.get('postnet', {}) or {}
  data_cfg = (pn.get('data', {}) or {})
  train_cfg = (pn.get('training', {}) or {})
  chunk_s = float(data_cfg.get('chunk_s', 1.0))
  hop_s = float(data_cfg.get('hop_s', 0.25))
  val_fraction = float(train_cfg.get('val_fraction', 0.1))

  fs = int(cfg.audio.fs)
  rf = int(cfg.model.resampling_factor)
  n_channels = int(getattr(cfg.audio, 'n_channels', 1))
  dataset_path = str(cfg.data.dataset_path)

  synth_ckpt = getattr(model, '_postnet_synth_ckpt', '')
  key = _cache_key(cfg, synth_ckpt)
  name = str(cfg.experiment.name)
  cache_dir = os.path.join(os.path.dirname(os.path.abspath(dataset_path)),
                           f"postnet_cache_{name}_{key}")
  tr_p = os.path.join(cache_dir, "train.pt")
  ev_p = os.path.join(cache_dir, "eval.pt")

  if os.path.exists(tr_p) and os.path.exists(ev_p):
    print(f"[postnet] using cached windows: {cache_dir}")
    tr = torch.load(tr_p)
    ev = torch.load(ev_p)
    return {
      'train_rough': tr['rough'], 'train_real': tr['real'], 'train_cond': tr['cond'],
      'eval_rough': ev['rough'], 'eval_real': ev['real'], 'eval_cond': ev['cond'],
      'cond_dim': int(tr['cond'].shape[1]),
    }

  os.makedirs(cache_dir, exist_ok=True)
  W = int(fs * chunk_s)
  hop = int(fs * hop_s)
  ds = AudioFeatureDataset(dataset_path=dataset_path, n_signal=W, sampling_rate=fs,
                           resampling_factor=rf, control_space=model.control_space,
                           transform_fn=None, n_channels=n_channels)
  audio, feats = ds._audio, ds._features
  N = int(audio.shape[-1])
  if N < 2 * W:
    raise RuntimeError(
      f"[postnet] dataset too short ({N} samples) for chunk_s={chunk_s}s ({W} samples). "
      f"Reduce postnet.data.chunk_s or use longer audio.")

  # Blocked split: reserve a contiguous tail for eval; dense train windows come from the head, with a
  # one-window gap so no train window overlaps an eval window (leakage-safe).
  split = int(N * (1.0 - val_fraction))
  train_starts = list(range(0, max(0, split - W - hop), hop))
  eval_starts = list(range(split, N - W, W))
  if not train_starts:
    train_starts = list(range(0, N - W, hop))
  if not eval_starts:
    eval_starts = [max(0, N - W)]

  print(f"[postnet] building paired cache: {len(train_starts)} train / {len(eval_starts)} eval "
        f"windows ({chunk_s}s, hop {hop_s}s) -> {cache_dir}")

  tr_r, tr_l, tr_c = _windows_from_model(model, audio, feats, train_starts, W, rf, device)
  ev_r, ev_l, ev_c = _windows_from_model(model, audio, feats, eval_starts, W, rf, device)

  tr = {'rough': torch.cat(tr_r, 0), 'real': torch.cat(tr_l, 0), 'cond': torch.cat(tr_c, 0)}
  ev = {'rough': torch.cat(ev_r, 0), 'real': torch.cat(ev_l, 0), 'cond': torch.cat(ev_c, 0)}
  torch.save(tr, tr_p)
  torch.save(ev, ev_p)
  with open(os.path.join(cache_dir, "meta.json"), "w") as fh:
    json.dump({'key': key, 'chunk_s': chunk_s, 'hop_s': hop_s, 'val_fraction': val_fraction,
               'cond_dim': int(tr['cond'].shape[1]), 'n_train': int(tr['rough'].shape[0]),
               'n_eval': int(ev['rough'].shape[0])}, fh, indent=2)
  print(f"[postnet] cache built: train={tuple(tr['rough'].shape)} cond={tuple(tr['cond'].shape)} "
        f"eval={tuple(ev['rough'].shape)}")

  return {
    'train_rough': tr['rough'], 'train_real': tr['real'], 'train_cond': tr['cond'],
    'eval_rough': ev['rough'], 'eval_real': ev['real'], 'eval_cond': ev['cond'],
    'cond_dim': int(tr['cond'].shape[1]),
  }
