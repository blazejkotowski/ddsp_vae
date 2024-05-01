import lightning as L

class BetaWarmupCallback(L.Callback):
  """
  A callback that warms up the β parameter of a β-VAE loss function.
  Arguments:
    - start_epoch: int, the epoch to start the warmup
    - end_epoch: int, the epoch to end the warmup
    - beta: float, the β parameter
  """
  def __init__(self, start_epoch: int = 100, end_epoch: int = 400, beta: float = 1.0):
    super().__init__()
    self.start_epoch = start_epoch
    self.end_epoch = end_epoch
    self.beta = beta

  def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
    current_epoch = trainer.current_epoch
    if current_epoch < self.start_epoch:
      beta = 0.0
    elif current_epoch >= self.end_epoch:
      beta = self.beta
    else:
      beta = self.beta * (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

    pl_module.beta = beta
