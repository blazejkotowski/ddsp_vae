"""Faithful post-net LightningModule.

Wraps the deployable streaming transform (`cli.streaming_postnet.StreamingSpecTransform`) exactly as
the research lab's `StreamFX` did (`self.mod = StreamingSpecTransform(...)`), so training uses the same
math as deployment and the exported state-dict keys stay `mod.*` (what cli/export.py unwraps).

Recipe (the lab's proven default): MRSTFT + L1 reconstruction, bend augmentation (same spectral bend
applied to rough+target so the transform learns to preserve bends), a gain-slew penalty for smooth
per-bin gains, and EMA (via ddsp.postnet.ema.EMACallback).
"""
import math

import torch
import torch.nn.functional as F
import lightning as L

from cli.streaming_postnet import StreamingSpecTransform


def make_mrstft(fs: int):
  """The model's exact reconstruction MRSTFT (perceptual), matching the recon loss the synth is
  trained/evaluated with (see experiments/postnet/common.py::mrstft)."""
  import auraloss
  ffts = [2053, 1021, 509, 257, 129, 65, 33]
  return auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=ffts, hop_sizes=[n // 4 for n in ffts], win_lengths=ffts,
    perceptual_weighting=True, sample_rate=int(fs))


def bend_pair(x, y, roll, stretch, n_fft=2048, hop=512):
  """Apply the same STFT-bin remap bend (roll/stretch) to both tensors [B,1,T].

  Teaches the post-net that a spectral bend on the rough must survive to the output. Ported from
  experiments/postnet/lab.py::_bend_pair (limit-aug omitted; it was a CUDA-assert source there).
  """
  win = torch.hann_window(n_fft, device=x.device)

  def bend(a):
    X = torch.stft(a.squeeze(1), n_fft, hop, window=win, return_complex=True)
    mag, ph = X.abs(), torch.angle(X)
    Nb = mag.shape[1]
    idx = torch.arange(Nb, device=a.device, dtype=torch.float32)

    def gather(mm, srcm, wrap):
      s = srcm % Nb if wrap else srcm.clamp(0, Nb - 1)
      i0 = s.floor().long().clamp(0, Nb - 1)
      frac = (s - i0.float()).view(1, Nb, 1)
      i1 = ((i0 + 1) % Nb) if wrap else (i0 + 1).clamp(max=Nb - 1)
      i1 = i1.clamp(0, Nb - 1)
      return mm.index_select(1, i0) * (1 - frac) + mm.index_select(1, i1) * frac

    if stretch != 0.0:
      mag = gather(mag, idx / (2.0 ** stretch), False)
    if roll != 0.0:
      mag = gather(mag, idx - roll * Nb, True)
    out = torch.istft(torch.polar(mag, ph), n_fft, hop, window=win, length=a.shape[-1])
    return out.unsqueeze(1)

  return bend(x), bend(y)


class PostNet(L.LightningModule):
  """Streaming faithful post-net, trained on frozen-synth (rough, real, control) pairs."""

  def __init__(self, cond_dim: int = 4, ch: int = 128, layers: int = 10, ch2: int = 96,
               layers2: int = 8, max_gain_db: float = 15.0, phase_cap: float = 1.57, la_f: int = 3,
               lr: float = 3e-4, max_steps: int = 18000, bend_aug_p: float = 0.4,
               gain_slew_w: float = 0.5, fs: int = 44100):
    super().__init__()
    self.save_hyperparameters()
    self.mod = StreamingSpecTransform(
      None, cond_dim=cond_dim, ch=ch, layers=layers, ch2=ch2, layers2=layers2,
      max_gain_db=max_gain_db, phase_cap=phase_cap, la_f=la_f)
    # MRSTFT (auraloss, perceptual) carries its own params/buffers. Keep it OUT of this module's
    # parameter/state tree (object.__setattr__ bypasses nn.Module registration) so it never lands in
    # state_dict — otherwise EMA snapshots and the export unwrap (mod.*) would see spurious keys.
    object.__setattr__(self, "_mr", None)
    # Optional: the frozen synth, attached at train time so validation reports MRSTFT in the *exact*
    # same style as `cli.train` (same loss objects, argument order, channel-folding, length align).
    object.__setattr__(self, "_synth", None)

  def attach_synth_metrics(self, synth):
    """Report validation MRSTFT identically to the synth trainer by reusing its own loss machinery.

    The synth's perceptual MRSTFT is asymmetric, so the argument order matters: `cli.train` logs
    `val_loss = _reconstruction_loss(pred, target)` (monitored) and `val/<LossName>` via
    `_loss_component_values(pred, target)`. Reusing the synth's methods reproduces both exactly.
    """
    object.__setattr__(self, "_synth", synth)

  def _mrstft(self):
    if self._mr is None:
      object.__setattr__(self, "_mr", make_mrstft(int(self.hparams.fs)).to(self.device))
    return self._mr

  def forward(self, rough, cond):
    return self.mod.train_forward(rough, cond)

  def _recon(self, pred, target):
    L2 = min(pred.shape[-1], target.shape[-1])
    p, t = pred[..., :L2].float(), target[..., :L2].float()
    loss = self._mrstft()(p, t) + 0.1 * (p - t).abs().mean()
    g = getattr(self.mod, "_last_gain", None)
    if float(self.hparams.gain_slew_w) > 0.0 and g is not None and g.dim() >= 2 and g.shape[-1] > 1:
      loss = loss + float(self.hparams.gain_slew_w) * (g[..., 1:] - g[..., :-1]).abs().mean()
    return loss

  def training_step(self, batch, batch_idx):
    rough, real, cond = batch
    # cond-safe augmentation: polarity flip + occasional input noise
    sgn = (torch.randint(0, 2, (rough.shape[0], 1, 1), device=rough.device).float() * 2 - 1)
    rough, real = rough * sgn, real * sgn
    if float(torch.rand(1)) < 0.5:
      rough = rough + torch.randn_like(rough) * (0.003 * float(torch.rand(1)))
    bend_p = float(self.hparams.bend_aug_p)
    if bend_p > 0.0 and float(torch.rand(1)) < bend_p:
      roll = float(torch.rand(1)) * 0.4
      stretch = (float(torch.rand(1)) - 0.5) * 2.0
      rough, real = bend_pair(rough, real, roll, stretch)
    pred = self(rough, cond)
    loss = self._recon(pred, real)
    self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
    return loss

  def validation_step(self, batch, batch_idx):
    rough, real, cond = batch
    pred = self(rough, cond).float()
    real = real.float()
    bs = rough.shape[0]
    if self._synth is not None:
      # Report exactly what cli.train reports, using the synth's own methods (channel-folding + length
      # alignment included; pred is total_delay shorter than real):
      #   val_loss     = _reconstruction_loss(pred, real)   -> loss_fn(target, pred)  [synth-monitored]
      #   val/<Name>   = _loss_component_values(pred, real) -> loss_fn(pred, target)  [conventional]
      # The synth's perceptual MRSTFT is asymmetric, so these two differ. We log both for direct
      # comparability, but CHECKPOINT on the conventional pred-vs-target metric (`val_mrstft`) — the
      # reversed `val_loss` can make a genuine improvement look worse and would misguide selection.
      val_loss = self._synth._reconstruction_loss(pred, real)
      self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
      comps = self._synth._loss_component_values(pred, real)
      for name, v in comps.items():
        self.log(f'val/{name}', v, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)
      val_conv = sum(float(w) * comps[fn.__class__.__name__] for fn, w in self._synth._loss_items)
      self.log('val_mrstft', val_conv, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
      return val_conv
    # Fallback (no synth attached, e.g. unit tests): plain MRSTFT.
    L2 = min(pred.shape[-1], real.shape[-1])
    val = self._mrstft()(pred[..., :L2], real[..., :L2])
    self.log('val_mrstft', val, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
    return val

  def configure_optimizers(self):
    opt = torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(self.hparams.max_steps))
    return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}
