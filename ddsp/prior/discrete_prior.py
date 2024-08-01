import math

import torch
from torch import nn
import torch.nn.functional as F

import lightning as L
import logging
from typing import Optional

from .blocks import QuantizedNormal

class DiscretePrior(L.LightningModule):
  def __init__(self,
               latent_size: int = 8,
               embedding_dim: int = 32,
               nhead: int = 8,
               num_layers: int = 6,
               dropout: float = 0.1,
               max_len: int = 1024,
               lr: float = 1e-4,
               resolution: int = 64):
    """
    Arguments:
    - latent_size: int, the size of the latent code
    - embedding_dim: int, the embedding dimensionality
    - nhead: int, the number of heads in the multiheadattention models
    - num_layers: int, the number of sub-encoder-layers in the encoder
    - dropout: float, the dropout value
    - max_len: int, the maximum length of the sequence
    - lr: float, the learning rate
    - resolution: int, the resolution of the quantized normal
    """

    super(DiscretePrior, self).__init__()

    self.save_hyperparameters()

    self._latent_size = latent_size
    self._embedding_dim = embedding_dim
    self._d_model = embedding_dim * latent_size
    self._nhead = nhead
    self._num_layers = num_layers
    self._dropout = dropout
    self._max_len = max_len
    self._lr = lr
    self._resolution = resolution

    self._quantized_normal = QuantizedNormal(resolution=self._resolution)

    self._embedding = nn.Embedding(self._resolution*self._latent_size, self._embedding_dim)

    self._positional_encoding = LearnablePositionalEncoding(
      embedding_dim=self._embedding_dim,
      max_len=self._max_len,
      latent_size=self._latent_size,
      dropout=self._dropout
    )

    encoder_layer = nn.TransformerEncoderLayer(d_model=self._d_model, nhead=self._nhead, dropout=self._dropout, batch_first=True)
    self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=self._num_layers)

    self._activation = nn.ReLU()
    self._dropout_layer = nn.Dropout(self._dropout)

    self._fc = nn.Linear(self._d_model*self._max_len, self._latent_size*self._resolution)


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, latent_size = x.shape

    assert latent_size == self._latent_size, f'Expected latent size {self._latent_size}, got {latent_size}'

    x = self._quantized_normal.encode(x) + self._resolution * torch.arange(self._latent_size).to(x.device).unsqueeze(0).unsqueeze(0)
    x = self._embedding(x) * math.sqrt(self._embedding_dim)
    # x = self._positional_encoding(x)

    # so far each latent variable was embedded separately
    # now we stack them together, preparing for the transformer
    x = x.view(batch_size, seq_len, self._d_model)

    x = self._encoder(x)
    # x = self._activation(x)
    x = self._dropout_layer(x)

    out = self._fc (x.view(batch_size, seq_len*self._d_model))
    logits = out.view(batch_size, self._latent_size, self._resolution)
    
    return logits
  
  def sample(self, logits: torch.Tensor) -> torch.Tensor:
    samples = torch.argmax(logits, -1)
    return self._quantized_normal.decode(samples)
  

  def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross entropy loss
    
    Arguments:
      - logits: torch.Tensor[batch_size, latent_size, resolution], the logits
      - target: torch.Tensor[batch_size, latent_size], the target
    Returns
      - loss: torch.Tensor, the loss
    """
    return F.cross_entropy(logits.permute(0, 2, 1), target).mean()
  

  def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
    """
    Predicting one step ahead
    
    Arguments:
      - batch: torch.Tensor[batch_size, seq_len+1, latent_size], the batch of sequences
    Returns
      - loss: torch.Tensor, the loss
    """

    x, y = batch[:, :self._max_len, ...], batch[:, self._max_len, ...]

    target = self._quantized_normal.encode(y)
    logits = self(x)
    
    loss = self._loss(logits, target)

    self.log('train_loss', loss, prog_bar=True)
    self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

    return loss
  

  def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
    """
    Predicting one step ahead
    
    Arguments:
      - batch: torch.Tensor[batch_size, seq_len+1, latent_size], the batch of sequences
    Returns
      - loss: torch.Tensor, the loss
    """

    x, y = batch[:, :self._max_len, ...], batch[:, self._max_len, ...]

    target = self._quantized_normal.encode(y)
    logits = self(x)

    loss = self._loss(logits, target)

    self.log('val_loss', loss, prog_bar=True)

    return loss
  

  def configure_optimizers(self):
    # reduce on plateau combined with adam optimizer
    optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True, threshold=1e-6)
    
    return {
      'optimizer': optimizer,
      'lr_scheduler': scheduler,
      'monitor': 'val_loss'
    }



class LearnablePositionalEncoding(nn.Module):
  def __init__(self, embedding_dim: int, max_len: int, latent_size: int, dropout: float):
    """
    Arguments:
      - embedding_dim: int, the embedding dimensionality
      - max_len: int, the maximum length of the sequence
      - latent_size: int, the size of the latent code
      - dropout: float, the dropout rate
    """
    super(LearnablePositionalEncoding, self).__init__()

    self._positional_encoding = nn.Parameter(torch.empty(max_len, latent_size, embedding_dim))
    nn.init.uniform_(self._positional_encoding, -0.02, 0.02)

    self._dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self._positional_encoding[:, :x.size(0), ...].unsqueeze(0) # add batch dimension for broadcasting
    return self._dropout(x)


class MultivariateEmbedding(nn.Module):

    def __init__(self,
                 num_tokens: int,
                 num_features: int,
                 num_quantizers: int,
                 from_pretrained: Optional[str] = None) -> None:
        super().__init__()
        self.from_pretrained = from_pretrained

        self.embedder = nn.Embedding(num_quantizers * num_tokens, num_features)
        self.proj = None

        self.num_quantizers = num_quantizers
        self.num_tokens = num_tokens

    def forward(self, x: torch.Tensor,
                sum_over_quantizers: bool) -> torch.Tensor:
        if sum_over_quantizers:
            x = x + torch.arange(x.shape[-1]).type_as(x) * self.num_tokens

        x = self.embedder(x.long())

        if sum_over_quantizers:
            x = x.sum(-2)

        return x