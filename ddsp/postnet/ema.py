"""Exponential moving average of the post-net weights, as a Lightning callback.

Mirrors the research lab's "adopt EMA" behavior: a shadow copy of the weights is updated every step,
and swapped into the module for validation *and* checkpointing so the saved `best`/`last` checkpoints
hold the (usually better) EMA weights. The raw weights are restored when the next training batch
begins, so optimization still runs on the live parameters.
"""
import torch
import lightning as L


class EMACallback(L.Callback):
  def __init__(self, decay: float = 0.999):
    super().__init__()
    self.decay = float(decay)
    self.shadow = None
    self._backup = None

  def on_fit_start(self, trainer, pl_module):
    if self.decay <= 0.0:
      return
    self.shadow = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if self.shadow is None:
      return
    with torch.no_grad():
      for k, v in pl_module.state_dict().items():
        if v.dtype.is_floating_point:
          self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
        else:
          self.shadow[k].copy_(v)

  def _swap_in_ema(self, pl_module):
    if self.shadow is None or self._backup is not None:
      return
    self._backup = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
    pl_module.load_state_dict(self.shadow)

  def _restore(self, pl_module):
    if self._backup is None:
      return
    pl_module.load_state_dict(self._backup)
    self._backup = None

  # EMA weights are active from validation start (so val metric + checkpoint see them) until the next
  # training batch begins — this makes the swap correct regardless of callback ordering.
  def on_validation_start(self, trainer, pl_module):
    self._swap_in_ema(pl_module)

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    self._restore(pl_module)

  def on_train_end(self, trainer, pl_module):
    # Leave EMA weights loaded at the end so a final manual save captures them.
    self._swap_in_ema(pl_module)
