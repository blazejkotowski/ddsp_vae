import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

import lightning as L

class Prior(L.LightningModule):
  def __init__(self, latent_size: int = 8, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dropout: float = 0.1, max_len: int = 256, lr: float = 1e-4):
    """
    Arguments:
      - latent_size: int, the size of the latent code
      - d_model: int, the model dimension
      - nhead: int, the number of heads in the multiheadattention models
      - num_layers: int, the number of sub-encoder-layers in the encoder
      - dropout: float, the dropout value
      - max_len: int, the maximum length of the sequence
    """
    super(Prior, self).__init__()

    self.save_hyperparameters()

    self._d_model = d_model
    self._lr = lr
    self._latent_size = latent_size
    self._max_len = max_len

    self._projection = nn.Linear(latent_size, d_model)
    encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    self._encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
    self._positional_encoding = LearnablePositionalEncoding(d_model = d_model, max_len = max_len, dropout=dropout)
    self._activation = nn.ReLU()
    self._dropout = nn.Dropout(dropout)

    # Entire sequence predicting one code
    self._out = nn.Linear(d_model * max_len, latent_size)

    # Mean squared error loss
    self._loss = nn.MSELoss()


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: torch.Tensor[batch_size, seq_len, n_latents], the preceding latent code sequence
    """
    # project to transformer space
    u = self._projection(x)

    # add positional encoding
    u = self._positional_encoding(u)

    # permute to comply with transformer shape
    u = u.permute(1, 0, 2) # => [seq_len, batch_size, d_model]

    # encode in causal mode
    causal_mask = torch.triu(torch.ones(u.size(0), u.size(0)), diagonal=1) * float('-inf').to(u.device)
    enc = self._encoder(u, mask=causal_mask) * math.sqrt(self._d_model)

    # permute back
    enc = enc.permute(1, 0, 2)

    # non-linearity and droput
    ac = self._dropout(self._activation(enc))

    # flatten
    stacked = ac.reshape(ac.size(0), -1)

    # predict next code
    return self._out(stacked)


  def training_step(self, batch, batch_idx):
    loss = self._step(batch)
    self.log('train_loss', loss, prog_bar=True)
    return loss


  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    self.log('val_loss', loss, prog_bar=True)
    return loss


  def _step(self, batch):
    """
    Arguments:
      - batch: torch.Tensor[batch_size, seq_len, latent_size], the batch of latent codes sequences
    """
    x = batch[:, :-1, :]
    y = batch[:, -1, :]

    y_hat = self(x)
    loss = self._loss(y_hat, y)

    return loss


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self._lr)

class LearnablePositionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
    """
    Arguments:
      - d_model: int, the model dimension
      - max_len: int, the maximum length of the sequence
      - dropout: float, the dropout rate
    """
    super(LearnablePositionalEncoding, self).__init__()

    self._positional_encoding = nn.Parameter(torch.empty(max_len, 1, d_model))
    nn.init.uniform_(self._positional_encoding, -0.02, 0.02)

    self._dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self._positional_encoding[:x.size(0), ...]
    return self._dropout(x)
