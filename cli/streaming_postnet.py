"""
Streaming (stateful) export module for the faithful STFT-domain post-net (SpecTransform).

Strategy: "cached-context recompute" — keep enough input history to recompute each causal conv
stack over [context + new] and emit only the new part, so chunked output matches one-shot output.
Each STFT stage uses hop-aligned overlap-add with carried input/output tails (a fixed (n_fft - hop)
sample delay per stage); a small look-ahead applies each frame's gain/phase to a frame slightly in
its past via magnitude/phase FIFOs, and the dry path is delayed to match for a phase-coherent blend.

Requires: nn~ buffer sizes that are multiples of the coarse-stage hop (256) — true for all
power-of-2 buffers.
"""
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F



class _MBlock(nn.Module):
  def __init__(self, ch: int, dilation: int):
    super().__init__()
    self.conv = nn.Conv1d(ch, ch, 3, dilation=dilation)
    self.pad = 2 * dilation

  def forward(self, h: torch.Tensor) -> torch.Tensor:
    v = self.conv(F.pad(h, (self.pad, 0)))
    return h + F.gelu(v)




class StreamingSpecTransform(nn.Module):
  """
  Stateful streaming version of the lab SpecTransform FX post-net (v8): a pure TRANSFORMATION of
  the rough — out = iSTFT(g*|R|, angle(R)+dphi) at two resolutions, params from [log-mag, controls].
  No generation path: zero rough in => silence out, bends propagate by construction.

  Stage 1: n_fft 1024 / hop 256, causal frame convs + la_f-frame lookahead (mag/ph FIFO).
  Stage 2: n_fft 256 / hop 64, causal, same-frame application.
  Total path delay = (1024-256) + la_f*256 + (256-64) samples (= 1728 for la_f=3, 36 ms @48k).
  Buffer length must be a multiple of 256 (true for power-of-2 nn~ buffers >= 256).
  """

  def __init__(self, state: dict, cond_dim: int = 4, n_fft: int = 1024, hop: int = 256,
               ch: int = 96, layers: int = 8, cycle: int = 4, max_gain_db: float = 18.0,
               phase_cap: float = 1.57, la_f: int = 3,
               n_fft2: int = 256, hop2: int = 64, ch2: int = 64, layers2: int = 6,
               max_channels: int = 2, cond_rf: int = 128):
    super().__init__()
    self.cond_dim = cond_dim
    self.max_gain_ln = float(max_gain_db) * math.log(10.0) / 20.0
    self.phase_cap = float(phase_cap)
    C = max_channels

    # ---- stage 1 ----
    self.n_fft, self.hop, self.la_f = n_fft, hop, la_f
    Fb = n_fft // 2 + 1
    self.Fb = Fb
    self.inp = nn.Conv1d(Fb + cond_dim, ch, 1)
    self.dil1: List[int] = [2 ** (i % cycle) for i in range(layers)]
    self.blocks = nn.ModuleList([_MBlock(ch, d) for d in self.dil1])
    self.out = nn.Conv1d(ch, 2 * Fb, 1)
    self.m_ctx = int(sum(2 * d for d in self.dil1))
    self.register_buffer("win", torch.hann_window(n_fft))
    self.ola1 = float((torch.hann_window(n_fft) ** 2).sum() / hop)
    self.d1 = n_fft - hop
    self.register_buffer("in_tail1", torch.zeros(C, self.d1))
    self.register_buffer("ola_tail1", torch.zeros(C, self.d1))
    self.register_buffer("fctx1", torch.zeros(C, Fb + cond_dim, self.m_ctx))
    self.register_buffer("mag_fifo", torch.zeros(C, Fb, max(la_f, 1)))
    self.register_buffer("ph_fifo", torch.zeros(C, Fb, max(la_f, 1)))

    # ---- stage 2 ----
    self.n_fft2, self.hop2 = n_fft2, hop2
    Fb2 = n_fft2 // 2 + 1
    self.Fb2 = Fb2
    self.inp2 = nn.Conv1d(Fb2 + cond_dim, ch2, 1)
    self.dil2: List[int] = [2 ** (i % cycle) for i in range(layers2)]
    self.blocks2 = nn.ModuleList([_MBlock(ch2, d) for d in self.dil2])
    self.out2 = nn.Conv1d(ch2, 2 * Fb2, 1)
    self.m_ctx2 = int(sum(2 * d for d in self.dil2))
    self.register_buffer("win2", torch.hann_window(n_fft2))
    self.ola2 = float((torch.hann_window(n_fft2) ** 2).sum() / hop2)
    self.d2 = n_fft2 - hop2
    self.register_buffer("in_tail2", torch.zeros(C, self.d2))
    self.register_buffer("ola_tail2", torch.zeros(C, self.d2))
    self.register_buffer("fctx2", torch.zeros(C, Fb2 + cond_dim, self.m_ctx2))

    self.total_delay = self.d1 + la_f * hop + self.d2
    self.register_buffer("dry_delay", torch.zeros(C, self.total_delay))
    # cond entering stage 2 lags by the stage-1 delay: exactly (d1 + la_f*hop)/cond_rf ctl frames
    self.cond_lag = (self.d1 + la_f * hop) // cond_rf
    self.register_buffer("cond_delay", torch.zeros(cond_dim, max(self.cond_lag, 1)))

    self._last_gain = torch.zeros(1)

    if state is None:
      # fresh init for TRAINING (zero heads = identity transform at start)
      nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
      nn.init.zeros_(self.out2.weight); nn.init.zeros_(self.out2.bias)
      return
    # ---- load trained weights (key remap: blocks.N.* -> blocks.N.conv.*) ----
    _state_bufs = ("in_tail", "ola_tail", "fctx", "mag_fifo", "ph_fifo", "dry_delay", "cond_delay")
    sd = {}
    for k, v in state.items():
      if any(k.startswith(b) for b in _state_bufs):
        continue                                   # streaming state, not weights (size = max_channels)
      for pre in ("blocks.", "blocks2."):
        if k.startswith(pre) and ".conv." not in k:
          n, rest = k[len(pre):].split(".", 1)
          k = f"{pre}{n}.conv.{rest}"
          break
      sd[k] = v
    missing, unexpected = self.load_state_dict(sd, strict=False)
    assert len(unexpected) == 0, f"unexpected keys: {unexpected}"
    assert len(sd) >= 20, f"too few weights: {len(sd)}"

  @torch.jit.export
  def reset(self):
    self.in_tail1.zero_(); self.ola_tail1.zero_(); self.fctx1.zero_()
    self.mag_fifo.zero_(); self.ph_fifo.zero_()
    self.in_tail2.zero_(); self.ola_tail2.zero_(); self.fctx2.zero_()
    self.dry_delay.zero_(); self.cond_delay.zero_()
    return 0

  def _net1(self, f: torch.Tensor) -> torch.Tensor:
    h = self.inp(f)
    for blk in self.blocks:
      h = blk(h)
    return self.out(h)

  def _net2(self, f: torch.Tensor) -> torch.Tensor:
    h = self.inp2(f)
    for blk in self.blocks2:
      h = blk(h)
    return self.out2(h)

  def _stage(self, x: torch.Tensor, cond_f: torch.Tensor, one: bool, use_state: bool) -> torch.Tensor:
    """One STFT transform stage, channel-batched. x [C,T]; cond_f [cond, F_t] or [C, cond, F_t]."""
    C, T = x.shape[0], x.shape[1]
    n_fft = self.n_fft if one else self.n_fft2
    hop = self.hop if one else self.hop2
    Fb = self.Fb if one else self.Fb2
    d = self.d1 if one else self.d2
    win = self.win if one else self.win2
    ola = self.ola1 if one else self.ola2
    ctx_frames = self.m_ctx if one else self.m_ctx2
    it = (self.in_tail1 if one else self.in_tail2)[:C] if use_state else torch.zeros(C, d, device=x.device, dtype=x.dtype)
    seg = torch.cat([it, x], dim=-1)
    if use_state:
      if one:
        self.in_tail1[:C] = seg[:, T:]
      else:
        self.in_tail2[:C] = seg[:, T:]
    frames = seg.unfold(-1, n_fft, hop)                                 # [C, F_t, n_fft]
    spec = torch.fft.rfft(frames * win, dim=-1)
    mag = spec.abs().transpose(1, 2)                                    # [C, Fb, F_t]
    ph = torch.angle(spec).transpose(1, 2)
    cf = cond_f.unsqueeze(0).expand(C, -1, -1) if cond_f.dim() == 2 else cond_f
    feats = torch.cat([torch.log1p(mag), cf], dim=1)
    fc = (self.fctx1 if one else self.fctx2)[:C] if use_state else torch.zeros(C, feats.shape[1], ctx_frames, device=x.device, dtype=x.dtype)
    fseq = torch.cat([fc, feats], dim=-1)
    if use_state:
      if one:
        self.fctx1[:C] = fseq[:, :, -ctx_frames:]
      else:
        self.fctx2[:C] = fseq[:, :, -ctx_frames:]
    o = (self._net1(fseq) if one else self._net2(fseq))[:, :, ctx_frames:]
    gain = torch.exp(torch.tanh(o[:, :Fb]) * self.max_gain_ln)
    if one:
      self._last_gain = gain
    dphi = torch.tanh(o[:, Fb:]) * self.phase_cap
    if one and self.la_f > 0:
      mfi = self.mag_fifo[:C] if use_state else torch.zeros(C, Fb, self.la_f, device=x.device, dtype=x.dtype)
      pfi = self.ph_fifo[:C] if use_state else torch.zeros(C, Fb, self.la_f, device=x.device, dtype=x.dtype)
      mag_seq = torch.cat([mfi, mag], dim=-1)
      ph_seq = torch.cat([pfi, ph], dim=-1)
      if use_state:
        self.mag_fifo[:C] = mag_seq[:, :, -self.la_f:]
        self.ph_fifo[:C] = ph_seq[:, :, -self.la_f:]
      mag_use = mag_seq[:, :, :mag.shape[-1]]
      ph_use = ph_seq[:, :, :ph.shape[-1]]
    else:
      mag_use, ph_use = mag, ph
    shaped = torch.polar(mag_use * gain, ph_use + dphi).transpose(1, 2)
    fr = torch.fft.irfft(shaped, n=n_fft, dim=-1) * win
    n_new = T // hop
    y = torch.zeros(C, T + d, device=x.device, dtype=x.dtype)
    for i in range(n_new):
      y[:, i * hop: i * hop + n_fft] += fr[:, i]
    y = y / ola
    out = y[:, :T].clone()
    if use_state:
      ot = self.ola_tail1 if one else self.ola_tail2
      out[:, :d] += ot[:C]
      if one:
        self.ola_tail1[:C] = y[:, T:]
      else:
        self.ola_tail2[:C] = y[:, T:]
    return out

  def forward(self, rough: torch.Tensor, cond: torch.Tensor, mix: float) -> torch.Tensor:
    """rough [1|B, C, T] (T multiple of 256); cond [same B, cond_dim, T_ctl]; mix in [0,1]."""
    B = rough.shape[0]
    if B > 1:
      items: List[torch.Tensor] = []
      for bi in range(B):
        items.append(self._fwd(rough[bi:bi + 1], cond[bi:bi + 1], mix, False))
      return torch.cat(items, dim=0)
    return self._fwd(rough, cond, mix, True)

  @torch.jit.ignore
  def train_forward(self, rough: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Lab-training path: rough [B,1,T], cond [B,cond,T_ctl] -> refined [B,1,T-total_delay].
    Identical math to cold-start streaming (stateless), so train == deploy exactly."""
    B, T = rough.shape[0], rough.shape[-1]
    x = rough[:, 0, :]                                                  # batch -> channel axis
    n1, n2 = T // self.hop, T // self.hop2
    cf1 = F.interpolate(cond, size=n1, mode="nearest")                  # [B, cond, n1]
    y1 = self._stage(x, cf1, True, False)
    lag = min(self.cond_lag, cond.shape[-1] - 1)
    cd2 = torch.cat([cond[:, :, :1].expand(-1, -1, lag), cond[:, :, : cond.shape[-1] - lag]], dim=-1) if lag > 0 else cond
    cf2 = F.interpolate(cd2, size=n2, mode="nearest")
    y2 = self._stage(y1, cf2, False, False)
    return y2[:, self.total_delay:].unsqueeze(1)

  def _fwd(self, rough: torch.Tensor, cond: torch.Tensor, mix: float, use_state: bool) -> torch.Tensor:
    C, T = rough.shape[1], rough.shape[2]
    x = rough[0]
    cd = cond[0]
    n1 = T // self.hop
    n2 = T // self.hop2
    cf1 = F.interpolate(cd.unsqueeze(0), size=n1, mode="nearest")[0]
    y1 = self._stage(x, cf1, True, use_state)
    # cond entering stage 2 lags by the stage-1 delay (cond_lag ctl frames)
    if self.cond_lag > 0:
      cdel = self.cond_delay if use_state else torch.zeros_like(self.cond_delay)
      Tc = cd.shape[-1]
      cseq = torch.cat([cdel, cd], dim=-1)
      cd2 = cseq[:, :Tc]
      if use_state:
        self.cond_delay.copy_(cseq[:, Tc:Tc + self.cond_lag])
    else:
      cd2 = cd
    cf2 = F.interpolate(cd2.unsqueeze(0), size=n2, mode="nearest")[0]
    y2 = self._stage(y1, cf2, False, use_state)
    dry_d = self.dry_delay[:C] if use_state else torch.zeros(C, self.total_delay, device=x.device, dtype=x.dtype)
    dseq = torch.cat([dry_d, x], dim=-1)
    dry = dseq[:, :T]
    if use_state:
      self.dry_delay[:C] = dseq[:, T:]
    return ((1.0 - mix) * dry + mix * y2).unsqueeze(0)
