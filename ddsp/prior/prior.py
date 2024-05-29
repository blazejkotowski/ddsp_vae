import lightning as L
import torch.nn as nn
import torch

class Prior(L.LightningModule):
  """
  Simple GRU-based prior model predicting the mu and scale of the latents.

  Args:
    - latent_size: int, the size of the latent space
    - hidden_size: int, the size of the hidden state in the GRU
    - num_layers: int, the number of layers in the GRU
    - dropout: float, the dropout rate
  """

  def __init__(self,
               latent_size: int = 8,
               hidden_size: int = 512,
               num_layers: int = 8,
               dropout: float = 0.01,
               lr: float = 1e-3):
    super().__init__()
    self.save_hyperparameters()

    self.latent_size = latent_size

    self._lr = lr

    self._gru = nn.GRU(
      input_size=latent_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      dropout=dropout,
      batch_first = True
    )

    self._out = nn.Linear(hidden_size, latent_size)

    self._loss = nn.MSELoss()

  def forward(self, x):
    """
    Predicts the latents from the input.

    Args:
      - x: torch.Tensor[batch_size, seq_len, latent_size], the input sequence of latents
    Returns:
      - out: torch.Tensor[batch_size, seq_len, latent_size], the predicted sequence of latents
    """
    out, _ = self._gru(x)
    out = self._out(out)
    return out

  def training_step(self, batch, batch_idx):
    """
    Computes the loss for a batch of data.

    Args:
      - batch: torch.Tensor[batch_size, seq_len, latent_size], the input sequence of latents
      - batch_idx: int, the index of the batch
    Returns:
      - loss: torch.Tensor[1], the loss
    """
    train_loss = self._step(batch)
    self.log("train_loss", train_loss, prog_bar=True, logger=True)

    return train_loss

  def validation_step(self, batch, batch_idx):
    """
    Computes the loss for a batch of validation data.

    Args:
      - batch: torch.Tensor[batch_size, seq_len, latent_size], the input sequence of latents
      - batch_idx: int, the index of the batch
    Returns:
      - loss: torch.Tensor[1], the loss
    """
    val_loss = self._step(batch)
    self.log("val_loss", val_loss, prog_bar=True, logger=True)

    return val_loss

  def _step(self, batch):
    x, y = batch
    y_hat = self(x)[:, -1, :]

    loss = self._loss(y_hat, y)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self._lr)
