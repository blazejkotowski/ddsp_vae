import lightning as L
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    self.register_buffer('_hidden_state', None, persistent=False)
    self.register_buffer('_cell_state', None, persistent=False)

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
    batch_size = x.size(0)
    if self._type == 'gru':
      if self._streaming:
        if self._hidden_state is None or batch_size != self._hidden_state.size(1):
          self._hidden_state = torch.zeros(self._gru.num_layers, batch_size, self._gru.hidden_size).to(x.device)

        out, hx = self._gru(x, self._hidden_state)
        self._hidden_state = hx.detach()
      else:
        out, _ = self._gru(x)

    elif self._type == 'lstm':
      if self._streaming:
        if self._hidden_state is None or batch_size != self._hidden_state.size(1):
          self._hidden_state = torch.zeros(self._lstm.num_layers, batch_size, self._lstm.hidden_size).to(x.device)
          self._cell_state = torch.zeros(self._lstm.num_layers, batch_size, self._lstm.hidden_size).to(x.device)

        out, (hx, cx) = self._lstm(x, (self._hidden_state, self._cell_state))
        self._hidden_state = hx.detach()
        self._cell_state = cx.detach()
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

    preserve_streaming = self._streaming
    self._streaming = True

    self.reset_state()

    losses = []

    x = sequence[:, :self.sequence_length, :]
    y = sequence[:, self.sequence_length, :]

    for i in range(self.sequence_length):
      y_hat = self(x)
      losses.append(self._loss(y_hat, y))
      x = y_hat.unsqueeze(1)
      y = sequence[:, self.sequence_length + i + 1, :]

    self._streaming = preserve_streaming
    return torch.mean(torch.stack(losses))


  def reset_state(self):
    self._hidden_state = None
    self._cell_state = None


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
    optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=1e-2)
    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": ReduceLROnPlateau(optimizer, patience=1),
        "monitor": "train_loss",
      }
    }
