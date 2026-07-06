"""Config-driven loaders for the evaluation notebooks.

The whole point of this module is that *the config name is enough*: given an
experiment config (e.g. ``topiary_stereo``) it locates and loads the matching
DDSP synth, latent compressor and discrete prior checkpoints, plus the dataset,
deriving every parameter (fs, channels, resampling, paths, control space) from
the YAML so nothing has to be typed by hand in the notebook.

Mirrors the inference path used by ``cli/generate_prior_discrete_audio.py`` and
``cli/train_prior.py`` so the notebooks audition exactly what gets exported.
"""
import os
import sys
import glob
import math
from typing import Optional, Tuple

# Make sure THIS repo's `ddsp` is importable when running from notebooks/.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import yaml
import torch
import torch.nn.functional as F

from ddsp import DDSP, AudioFeatureDataset
from ddsp.latent_compressor import LatentCompressor
from ddsp.prior import PriorDiscrete
from ddsp.interfaces import ControlField, ControlSpace, build_control_space
from ddsp.utils import find_checkpoint
from ddsp.checkpoint_compat import allow_full_checkpoints, force_weights_only_false

# torch>=2.6 defaults torch.load to weights_only=True, which rejects the OmegaConf objects in our
# checkpoints' hyper_parameters. These are trusted, locally produced checkpoints — load them fully.
allow_full_checkpoints()
force_weights_only_false()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(config_name: str) -> dict:
  """Load configs/<config_name>.yaml as a plain dict."""
  config_name = config_name[:-5] if config_name.endswith(".yaml") else config_name
  path = os.path.join(REPO_ROOT, "configs", f"{config_name}.yaml")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Config not found: {path}")
  with open(path) as f:
    cfg = yaml.safe_load(f)
  cfg["_config_name"] = config_name
  cfg["_config_path"] = path
  return cfg


def training_dir(cfg: dict) -> str:
  """Absolute path of the experiment's training output directory."""
  return os.path.join(REPO_ROOT, cfg["experiment"]["training_dir"])


# --------------------------------------------------------------------------- #
# Control spaces
# --------------------------------------------------------------------------- #
def _ddsp_dims_from_ckpt(ckpt_path: str) -> Tuple[int, int]:
  ckpt = torch.load(ckpt_path, map_location="cpu")
  h = ckpt.get("hyper_parameters", {})
  return int(h.get("feature_dim", 0) or 0), int(h.get("latent_size", 0) or 0)


def model_control_space(feature_dim: int, latent_size: int) -> ControlSpace:
  """Minimal control space (no extractors) for instantiating the DDSP model."""
  fields = []
  if feature_dim > 0:
    fields.append(ControlField(name="features", dim=feature_dim, source="feature", extractor=None))
  if latent_size > 0:
    fields.append(ControlField(name="latents", dim=latent_size, source="latent", extractor=None))
  return ControlSpace(tuple(fields))


def dataset_control_space(cfg: dict, latent_size: int) -> ControlSpace:
  """Full control space WITH feature extractors, as the dataset needs them.

  Built from the config's ``model.control_space``; latent fields are re-added
  (the dataset ignores them but the model expects matching dims).
  """
  all_fields = build_control_space(cfg["model"]["control_space"])
  fields = [f for f in all_fields.fields if f.source == "feature"]
  if latent_size > 0:
    fields.append(ControlField(name="latents", dim=latent_size, source="latent"))
  return ControlSpace(tuple(fields))


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #
def synth_ckpt(cfg: dict) -> str:
  d = os.path.join(training_dir(cfg), "synth", cfg["experiment"]["name"])
  ckpt = find_checkpoint(d, return_none=True, typ="best")
  if ckpt is None:
    ckpt = find_checkpoint(d, return_none=False, typ="last")
  return ckpt


def compressor_ckpt(cfg: dict) -> str:
  return os.path.join(training_dir(cfg), "compressor", cfg["experiment"]["name"], "best.ckpt")


def prior_ckpt(cfg: dict, prefer: str = "best_acc") -> str:
  """Find a discrete-prior checkpoint under training/prior_discrete/<name>/<run>/.

  Picks the most-recently-modified run directory, then prefers `prefer`
  (best_acc / best_loss / last).
  """
  base = os.path.join(training_dir(cfg), "prior_discrete", cfg["experiment"]["name"])
  runs = [r for r in glob.glob(os.path.join(base, "*")) if os.path.isdir(r)]
  if not runs:
    raise FileNotFoundError(f"No prior run directories under {base}")
  run = max(runs, key=os.path.getmtime)
  order = [prefer, "best_acc", "best_loss", "last"]
  for name in dict.fromkeys(order):  # dedupe, keep order
    p = os.path.join(run, f"{name}.ckpt")
    if os.path.exists(p):
      return p
  raise FileNotFoundError(f"No checkpoint ({order}) in {run}")


# --------------------------------------------------------------------------- #
# Model / dataset loaders
# --------------------------------------------------------------------------- #
def load_ddsp(cfg: dict, device: str = "cuda") -> DDSP:
  ckpt = synth_ckpt(cfg)
  print(f"[ddsp] {ckpt}")
  feature_dim, latent_size = _ddsp_dims_from_ckpt(ckpt)
  ddsp = DDSP.load_from_checkpoint(
    ckpt,
    strict=False,
    streaming=False,
    device=device,
    control_space=model_control_space(feature_dim, latent_size),
    weights_only=False
  ).to(device)
  ddsp.eval()
  return ddsp


def load_compressor(cfg: dict, device: str = "cuda") -> LatentCompressor:
  ckpt = compressor_ckpt(cfg)
  print(f"[compressor] {ckpt}")
  comp = LatentCompressor.load_from_checkpoint(ckpt, strict=False).to(device)
  comp.eval()
  return comp


def load_prior(cfg: dict, prefer: str = "best_acc", device: str = "cuda") -> PriorDiscrete:
  ckpt = prior_ckpt(cfg, prefer=prefer)
  print(f"[prior] {ckpt}")
  prior = PriorDiscrete.load_from_checkpoint(ckpt, strict=False).to(device)
  prior.eval()
  return prior


def load_dataset(cfg: dict, ddsp: DDSP, chunk_duration_s: Optional[float] = None,
                 device: str = "cuda") -> AudioFeatureDataset:
  audio = cfg["audio"]
  fs = int(audio["fs"])
  n_channels = int(audio.get("n_channels", 1))
  dur = float(chunk_duration_s if chunk_duration_s is not None else audio["chunk_duration_s"])
  n_signal = int(fs * dur)
  ds = AudioFeatureDataset(
    dataset_path=str(cfg["data"]["dataset_path"]),
    n_signal=n_signal,
    sampling_rate=fs,
    resampling_factor=int(ddsp.resampling_factor),
    control_space=dataset_control_space(cfg, int(ddsp.latent_size)),
    device=device,
    n_channels=n_channels,
  )
  return ds


# --------------------------------------------------------------------------- #
# Inference helpers (control-rate, mirror the export path)
# --------------------------------------------------------------------------- #
def _device(ddsp: DDSP) -> torch.device:
  return next(ddsp.parameters()).device


@torch.no_grad()
def autoencode(ddsp: DDSP, audio: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
  """Full DDSP autoencode of one chunk. audio [n_channels, T], features [T_ctl, D_feat]
  -> audio [1, n_channels, T_audio]."""
  dev = _device(ddsp)
  y = ddsp(audio.unsqueeze(0).to(dev), features.unsqueeze(0).to(dev))
  if y.dim() == 2:
    y = y.unsqueeze(1)
  return y


@torch.no_grad()
def encode_latents(ddsp: DDSP, audio: torch.Tensor) -> torch.Tensor:
  """audio [n_channels, T] or [B, n_channels, T] -> smoothed latents z [B, T_ctl, latent]."""
  audio = audio.to(_device(ddsp))
  if audio.dim() == 2:
    audio = audio.unsqueeze(0)
  mu, scale = ddsp.encoder(audio)
  z, _ = ddsp.encoder.reparametrize(mu, scale)
  return ddsp._smooth_latents(z)


@torch.no_grad()
def build_controls(ddsp: DDSP, audio: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
  """Assemble the [1, T_ctl, feature_dim+latent_size] control tensor the compressor sees.

  audio:    [n_channels, T]      (single chunk)
  features: [T_ctl, feature_dim] (control rate, from the dataset)
  """
  z = encode_latents(ddsp, audio)                  # [1, T_z, latent]
  feat = features.unsqueeze(0).to(z.device)         # [1, T_ctl, feature_dim]
  T = min(feat.shape[1], z.shape[1])
  controls = torch.cat([feat[:, :T], z[:, :T]], dim=-1)
  return controls[:, :, : (ddsp.feature_dim + ddsp.latent_size)]


@torch.no_grad()
def synth_from_controls(ddsp: DDSP, controls: torch.Tensor) -> torch.Tensor:
  """controls [1, T, feature_dim+latent_size] -> audio [1, n_channels, T_audio]."""
  fd, ls = ddsp.feature_dim, ddsp.latent_size
  features = controls[:, :, :fd] if fd > 0 else torch.zeros(
    controls.shape[0], controls.shape[1], 1, device=controls.device)
  latents = controls[:, :, fd:fd + ls]
  synth_params = ddsp.decoder(features, latents)
  audio = ddsp._synthesize(synth_params)
  if audio.dim() == 2:
    audio = audio.unsqueeze(1)
  return audio


@torch.no_grad()
def compressor_roundtrip(ddsp: DDSP, compressor: LatentCompressor,
                         controls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  """Quantize controls through the VQ compressor and resynthesize.

  Returns (audio [1, n_channels, T], codes [1, T_low, num_codebooks]).
  """
  codes = compressor.encode_codes(controls)
  if codes.dim() == 2:
    codes = codes.unsqueeze(-1)
  controls_hat = compressor.decode_codes(codes, output_len=controls.shape[1])
  controls_hat = controls_hat[:, :, : (ddsp.feature_dim + ddsp.latent_size)]
  return synth_from_controls(ddsp, controls_hat), codes


@torch.no_grad()
def sample_tokens(prior: PriorDiscrete, n_tokens: int, temperature: float = 1.0,
                  sampling: str = "multinomial", primer_tokens: Optional[torch.Tensor] = None,
                  primer_len: int = 0, device: str = "cuda") -> torch.Tensor:
  """Autoregressively sample `n_tokens` from the prior, cold-starting on START.

  Matches cli/generate_prior_discrete_audio.py / the nn~ wrapper: position 0 is
  the learned START token, an optional real-audio primer follows, generation
  fills the rest. Returns tokens [1, n_tokens, num_codebooks].
  """
  K = int(prior.codebook_size)
  N = int(prior.num_codebooks)
  max_len = int(getattr(prior, "_max_len", 256))
  start_id = int(getattr(prior, "start_token_id", K))
  sampling = sampling.lower()
  assert sampling in ("multinomial", "argmax")

  buf = torch.full((1, n_tokens + 1, N), start_id, dtype=torch.long, device=device)
  gen_start = 1
  if primer_tokens is not None:
    if primer_tokens.dtype != torch.long:
      primer_tokens = primer_tokens.long()
    L = min(int(primer_len), int(primer_tokens.shape[1]), int(n_tokens))
    buf[:, 1:1 + L, :] = primer_tokens[:, :L, :].to(device)
    gen_start = 1 + L

  for t in range(gen_start, n_tokens + 1):
    ctx = buf[:, max(0, t - max_len):t, :]
    next_logits = prior(ctx)[:, -1, :, :]  # [1, N, K]
    if sampling == "argmax":
      buf[:, t, :] = torch.argmax(next_logits, dim=-1)
    else:
      probs = torch.softmax(next_logits / max(1e-4, temperature), dim=-1)
      buf[:, t, :] = torch.multinomial(probs.reshape(-1, K), 1).reshape(1, N)

  return buf[:, 1:, :]


def tokens_per_second(ddsp: DDSP, compressor: LatentCompressor) -> float:
  control_rate = float(ddsp.fs) / float(ddsp.resampling_factor)
  return control_rate / float(getattr(compressor, "compression_ratio", 32))


# --------------------------------------------------------------------------- #
# TensorBoard stats
# --------------------------------------------------------------------------- #
def read_tb_scalars(log_root: str) -> dict:
  """Read the newest events file under `log_root` -> {tag: {last, step, min, max, n}}."""
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  events = glob.glob(os.path.join(log_root, "**", "events.out.tfevents.*"), recursive=True)
  if not events:
    raise FileNotFoundError(f"No tfevents under {log_root}")
  ea = EventAccumulator(max(events, key=os.path.getmtime), size_guidance={"scalars": 0})
  ea.Reload()
  out = {}
  for tag in ea.Tags()["scalars"]:
    vals = ea.Scalars(tag)
    ys = [v.value for v in vals]
    out[tag] = {"last": vals[-1].value, "step": vals[-1].step,
                "min": min(ys), "max": max(ys), "n": len(ys)}
  return out


def prior_stats(cfg: dict) -> dict:
  """Last/min/max of the discrete prior's logged scalars (val_acc, val_loss, ...)."""
  run = os.path.dirname(prior_ckpt(cfg))
  return read_tb_scalars(os.path.join(run, "logs"))


def compressor_stats(cfg: dict) -> dict:
  run = os.path.join(training_dir(cfg), "compressor", cfg["experiment"]["name"])
  return read_tb_scalars(run)


def synth_stats(cfg: dict) -> dict:
  """Last/min/max of the DDSP synth's logged scalars (val_loss, MRSTFT, kld, ...)."""
  run = os.path.join(training_dir(cfg), "synth", cfg["experiment"]["name"])
  return read_tb_scalars(run)
