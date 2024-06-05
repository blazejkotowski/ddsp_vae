import lightning as L
import torch.nn as nn
import torch
import numpy as np

def _is_batch_size_one(x: torch.Tensor) -> bool:
  return x.shape[0] == 1

class Prior(L.LightningModule):
  """
  Simple GRU-based prior model predicting the mu and scale of the latents.

  Args:
    - latent_size: int, the size of the latent space
    - hidden_size: int, the size of the hidden state in the GRU
    - num_layers: int, the number of layers in the GRU
    - dropout: float, the dropout rate
    - lr: float, the learning rate
    - streaming: bool, whether to run the model in streaming mode
    - sequence_length: int, the length of the preceding latent code sequence for prediction
    - rnn_type: str, the type of the RNN to use ['gru', 'lstm']
  """

  def __init__(self,
               latent_size: int = 8,
               hidden_size: int = 512,
               num_layers: int = 8,
               dropout: float = 0.01,
               lr: float = 1e-3,
               streaming: bool = False,
               sequence_length: int = 10,
               rnn_type: str = 'gru'):
    super().__init__()
    self.save_hyperparameters()

    self.sequence_length = sequence_length
    self.latent_size = latent_size
    self._type = rnn_type

    self._streaming = streaming
    self._lr = lr

    # Build the network

    # GRU layer
    if self._type == 'gru':
      self._gru = nn.GRU(
        input_size=latent_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_first=True
      )

    elif self._type == 'lstm':
      self._lstm = nn.LSTM(
        input_size=latent_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_first=True
      )

    # Densely connected output layer
    self._out = nn.Linear(hidden_size, latent_size)

    # Try MAE (nn.L1Loss) or Huber (nn.HuberLoss) loss instead of MSE
    self._loss = nn.MSELoss()
    # self._loss = nn.L1Loss()

    # For keeping the GRU hidden state in streaming mode
    self.register_buffer('_hidden_state', torch.zeros(num_layers, 1, hidden_size), persistent=False)
    self.register_buffer('_cell_state', torch.zeros(num_layers, 1, hidden_size), persistent=False)

  def forward(self, x):
    """
    Predicts the latents from the input.

    Args:
      - x: torch.Tensor[batch_size, seq_len, latent_size], the input sequence of latents
    Returns:
      - out: torch.Tensor[batch_size, seq_len, latent_size], the predicted sequence of latents
    """
    x = self._rnn(x)[:, -1, :]

    # Densely connected layer
    out = self._out(x)

    return out

  def _rnn(self, x: torch.Tensor) -> torch.Tensor:
    """
    Run the RNN on the input.

    Args:
      - x: torch.Tensor[batch_size, seq_len, latent_size], the input sequence of latents
    Returns:
      - out: torch.Tensor[batch_size, seq_len, latent_size], the predicted sequence of latents
    """
    if self._type == 'gru':
      if self._streaming and _is_batch_size_one(x):
        out, hx = self._gru(x, self._hidden_state)
        self._hidden_state.copy_(hx)
      else:
        out, _ = self._gru(x)

    elif self._type == 'lstm':
      if self._streaming and _is_batch_size_one(x):
        out, (hx, cx) = self._lstm(x, (self._hidden_state, self._cell_state))
        self._hidden_state.copy_(hx)
        self._cell_state.copy_(cx)
      else:
        out, (_, _) = self._lstm(x)

    return out


  def training_step(self, sequence, batch_idx):
    """
    Computes the loss for a batch of data.

    Args:
      - batch: torch.Tensor[batch_size, seq_len*2, latent_size], the input sequence of latents
      - batch_idx: int, the index of the batch
    Returns:
      - loss: torch.Tensor[1], the loss
    """

    # Teacher forcing
    total_loss = self._teacher_forcing(sequence)
    self.log("train_loss", total_loss, prog_bar=True, logger=True)

    return total_loss


  def _teacher_forcing(self, sequence: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss using teacher forcing.
    Predict the next latent code from the preceding sequence with shifting window.
    Attach previously predicted latent code to the input sequence.

    Args:
      - sequence: torch.Tensor[batch_size, seq_len*2, latent_size], the input sequence of latents
    Returns:
      - total_loss: torch.Tensor[1], the loss
    """

    x = sequence[:, :self.sequence_length, :]
    total_loss = torch.zeros(1, requires_grad=True)
    for i in range(self.sequence_length):
      y = sequence[:, self.sequence_length+i, :]
      y_hat = self(x)
      total_loss = total_loss + self._loss(y_hat, y)
      x = torch.cat([x[:, 1:, :], y_hat.unsqueeze(1)], dim=1)

    return total_loss


  def validation_step(self, batch, batch_idx):
    """
    Computes the loss for a batch of validation data.

    Args:
      - batch: torch.Tensor[batch_size, seq_len*2, latent_size], the input sequence of latents
      - batch_idx: int, the index of the batch
    Returns:
      - loss: torch.Tensor[1], the loss
    """
    val_loss = self._teacher_forcing(batch)
    self.log("val_loss", val_loss, prog_bar=True, logger=True)

    return val_loss


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self._lr)
