import lightning as L
from typing import Any

class BetaWarmupCallback(L.Callback):
  """
  A callback that warms up the β parameter of a β-VAE loss function.
  Arguments:
    - start_steps: int, the step to start the warmup
    - end_steps: int, the step to end the warmup
    - beta: float, the β parameter
  """
  def __init__(self, start_steps: int = 300, end_steps: int = 1300, beta: float = 1.0):
    super().__init__()
    self.start_steps = start_steps
    self.end_steps = end_steps
    self.beta = beta

  def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int) -> None:
    current_step = trainer.global_step
    if current_step < self.start_steps:
      beta = 0.0
    elif current_step >= self.end_steps:
      beta = self.beta
    else:
      beta = self.beta * (current_step - self.start_steps) / (self.end_steps - self.start_steps)

    pl_module._beta = beta

class CyclicalBetaWarmupCallback(L.Callback):
  """
  A callback that warms up the β parameter of a β-VAE loss function cyclically.
  Arguments:
    - start_epoch: int, the epoch to start the warmup
    - cycle_duration: int, the duration of a cycle in epochs
    - beta: float, the β parameter
  """
  def __init__(self, start_epoch: int = 100, cycle_duration: int = 50, beta: float = 1.0):
    super().__init__()
    self.start_epoch = start_epoch
    self.cycle_duration = cycle_duration
    self._beta = beta

  def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
    current_epoch = trainer.current_epoch
    if current_epoch < self.start_epoch:
      beta = 0.0
    else:
      cycle_progress = ((current_epoch - self.start_epoch) % self.cycle_duration) / self.cycle_duration
      beta = self.beta * cycle_progress

    pl_module._beta = beta
