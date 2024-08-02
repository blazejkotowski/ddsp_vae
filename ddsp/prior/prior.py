import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

import lightning as L

from typing import Dict

class Prior(L.LightningModule):
  def __init__(self,
               latent_size: int = 8,
               d_model: int = 512,
               nhead: int = 8,
               num_layers: int = 6,
               dropout: float = 0.1,
               max_len: int = 256,
               lr: float = 1e-4):
    """
    Arguments:
      - latent_size: int, the size of the latent code
      - d_model: int, the model dimension
      - nhead: int, the number of heads in the multiheadattention models
      - num_layers: int, the number of sub-encoder-layers in the encoder
      - dropout: float, the dropout value
      - max_len: int, the maximum length of the sequence
      - lr: float, the learning rate
    """
    super(Prior, self).__init__()

    self.save_hyperparameters()

    self._d_model = d_model
    self._lr = lr
    self._latent_size = latent_size
    self._max_len = max_len

    self._projection = nn.Sequential(
      nn.LayerNorm(latent_size),
      nn.Linear(latent_size, d_model)
    )
    encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
    self._encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
    self._positional_encoding = LearnablePositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
    # self._positional_encoding = FixedPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
    self._activation = nn.ReLU()
    self._dropout = nn.Dropout(dropout)

    self._out = nn.Linear(d_model, latent_size)

    # # seq2seq
    # self._out = nn.Sequential(
    #   nn.LayerNorm(d_model * max_len),
    #   nn.Linear(d_model * max_len, latent_size * seq_out_len)
    # )

    # Mean squared error loss
    self._loss = nn.MSELoss()


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: torch.Tensor[batch_size, seq_len, latent_size], the preceding latent code sequence
    """
    # permute to comply with transformer shape
    x = x.permute(1, 0, 2) # => [seq_len, batch_size, latent_size]

    # project to transformer space
    u = self._projection(x) # => [seq_len, batch_size, d_model]

    # add positional encoding
    pos = self._positional_encoding(u)

    # Construct causal mask
    causal_mask = ~torch.triu(torch.ones(pos.size(0), pos.size(0)), diagonal=1).bool().to(pos.device)

    # Mask certain positions for better generalization (idea code from Behzad, cite if published)
    teacher_forcing_ratio = 0.75
    indices = torch.rand(causal_mask.size(0))
    indices = indices > teacher_forcing_ratio
    causal_mask[:, indices] = False

    # encode using transformer encoder
    enc = self._encoder(pos, mask=causal_mask) * math.sqrt(self._d_model)

    # non-linearity
    act = self._activation(enc)

    # permute back
    act = act.permute(1, 0, 2) # => [batch_size, seq_len, d_model]

    # dropout
    act = self._dropout(act) # => [batch_size, seq_len, d_model]

    # project back to latent space
    out = self._out(act) # => [batch_size, seq_len, latent_size]

    return out


  def training_step(self, batch, batch_idx):
    loss = self._step(batch)
    self.log('train_loss', loss, prog_bar=True)
    self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
    return loss


  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    self.log('val_loss', loss, prog_bar=True)
    return loss


  def configure_optimizers(self):
    # reduce on plateau combined with adam optimizer
    optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, threshold=1e-4)

    scheduler = {
      'scheduler': lr_scheduler,
      'monitor': 'val_loss',
      'interval': 'epoch'
    }

    return [optimizer], [scheduler]


  def _step(self, batch):
    """
    Arguments:
      - batch: torch.Tensor[batch_size, seq_len, latent_size], the batch of latent codes sequences
    """
    x = batch[:, :-1, :]
    y = batch[:, 1:, :]

    y_hat = self(x)

    loss = self._loss(y_hat, y)

    return loss

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


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
  r"""Inject some information about the relative or absolute position of the tokens
      in the sequence. The positional encodings have the same dimension as
      the embeddings, so that the two can be summed. Here, we use sine and cosine
      functions of different frequencies.
  .. math::
      \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
      \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
      \text{where pos is the word position and i is the embed idx)
  Args:
      d_model: the embed dim (required).
      dropout: the dropout value (default=0.1).
      max_len: the max. length of the incoming sequence (default=1024).
  """

  def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
    super(FixedPositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)  # positional encoding
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

  def forward(self, x):
    r"""Inputs of forward function
    Args:
        x: the sequence fed to the positional encoder model (required).
    Shape:
        x: [sequence length, batch size, embed dim]
        output: [sequence length, batch size, embed dim]
    """
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)
