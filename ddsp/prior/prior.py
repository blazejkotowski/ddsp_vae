import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchaudio.transforms import MuLawEncoding, MuLawDecoding
from torch.nn.functional import cross_entropy, softmax, mse_loss

import math

import lightning as L

from typing import Dict
from typing import Optional

class Prior(L.LightningModule):
  def __init__(self,
               latent_size: int = 8,
               embedding_dim: int = 32,
               quantization_channels: int = 32,
               nhead: int = 8,
               num_layers: int = 6,
               dropout: float = 0.1,
               max_len: int = 256,
               lr: float = 1e-2,
               normalization_dict: Optional[Dict[str, float]] = None,
               device='cuda'):
    """
    Arguments:
      - latent_size: int, the size of the latent code
      - embedding_dim: int, the embedding dimensionality
      - quantization_channels: int, the number of quantization channels
      - nhead: int, the number of heads in the multiheadattention models
      - num_layers: int, the number of sub-encoder-layers in the encoder
      - dropout: float, the dropout value
      - max_len: int, the maximum length of the sequence
      - lr: float, the learning rate
      - normalization_dict: Dict[str, float], the normalization containing mean and variance
      - device: str, the torch device to use
    """
    super(Prior, self).__init__()

    self.save_hyperparameters()

    self._device = device

    self.eval_mode = False

    self._normalization_dict = normalization_dict

    self._d_model = embedding_dim * latent_size
    self._embedding_dim = embedding_dim
    self._lr = lr
    self._latent_size = latent_size
    self._max_len = max_len

    self._quantization_channels = quantization_channels
    self._quantizer = MuLawEncoding(self._quantization_channels)
    self._dequantizer = MuLawDecoding(self._quantization_channels)

    self._embedding = nn.Embedding(self._quantization_channels, self._embedding_dim)

    encoder_layer = TransformerEncoderLayer(d_model=self._d_model, nhead=nhead, dropout=dropout)
    self._encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
    # self._positional_encoding = LearnablePositionalEncoding(embedding_dim=self._embedding_dim, max_len=max_len, dropout=dropout, device=self._device)
    self._positional_encoding = FixedPositionalEncoding(embedding_dim=self._embedding_dim, max_len=max_len, dropout=dropout, device=self._device)

    self._activation = nn.ReLU()
    self._dropout = nn.Dropout(dropout)

    self._fc = nn.Linear(self._d_model, latent_size * self._quantization_channels)

    # Cross-entropy loss
    self._loss = nn.CrossEntropyLoss(reduce=False)


  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the latent codes.

    Arguments:
      - x: torch.Tensor[batch_size, seq_len, latent_size], the sequences of latent codes
    Returns:
      - x: torch.Tensor[batch_size, seq_len, latent_size], the normalized latent codes
    """
    min_x, max_x = self._normalization_dict['min'], self._normalization_dict['max']
    return -1 + 2 * (x - min_x) / (max_x - min_x)
    # return (x - self._normalization_dict['mean']) / self._normalization_dict['var']


  def denormalize(self, x: torch.Tensor) -> torch.Tensor:
    """
    Denormalize the latent codes.

    Arguments:
      - x: torch.Tensor[batch_size, seq_len, latent_size], the sequences of normalized latent codes
    Returns:
      - x: torch.Tensor[batch_size, seq_len, latent_size], the denormalized latent codes
    """
    min_x, max_x = self._normalization_dict['min'], self._normalization_dict['max']
    return ((x + 1) / 2) * (max_x - min_x) + min_x
    # return x * self._normalizatisleon_dict['var'] + self._normalization_dict['mean']


  def sample(self, logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """
    Sample from the logits.

    Arguments:
      - logits: torch.Tensor[batch_size, seq_len, latent_size, quantization_channels], the logits
      - temperature: float, the temperature (how much should softmax smooth the distribution)
    Returns:
      - x: torch.Tensor[batch_size, seq_len, latent_size], the sampled latent codes
    """
    temperature += 1e-4
    # probs = softmax(logits / temperature)
    # x = torch.distributions.Categorical(probs=probs).sample()
    x = torch.distributions.Categorical(logits=logits/temperature).sample()
    # x = torch.argmax(probs, dim=-1)
    return self.denormalize(self._dequantizer(x))


  def generate(self, prime: torch.Tensor, seq_len: int, temperature: float = 0.0) -> torch.Tensor:
    """
    Arguments:
      - prime: torch.Tensor[prime_len, latent_size], the preceding latent code sequence
      - seq_len: int, the length of the generated sequence
      - temperature: float, the temperature for sampling
    Returns
      - torch.Tensor[seq_len, latent_size], the generated latent code sequence
    """
    prime_len = prime.size(0)
    # self.eval()
    output_seq = torch.full((seq_len, self._latent_size), fill_value=0, device=self.device, dtype=torch.float32)

    output_seq[:prime.shape[0], :] = prime.clone()
    x = prime.clone().unsqueeze(0)

    for i in range(prime_len, seq_len):
      with torch.no_grad():
        logits = self(x)
        y_hat = self.sample(logits, temperature=temperature)
      x = torch.cat((x, y_hat[:, -1:, :]), dim=1)
      output_seq[i, :] = y_hat.squeeze()[-1, :]

      if x.size(1) > self._max_len:
        x = x[:, -prime_len:, :]

    return output_seq


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: torch.Tensor[batch_size, seq_len, latent_size], the preceding latent code sequence
    """
    # permute to comply with transformer shape
    x = x.permute(1, 0, 2) # => [seq_len, batch_size, latent_size]
    seq_len, batch_size, latent_size = x.shape

    x = self.normalize(x)
    x = self._quantizer(x)

    # flattent latent codes
    # embed the latent codes
    indices = x.long() # + self._quantization_channels * torch.arange(self._latent_size, device=self._device).reshape(1, 1, -1)
    embed = self._embedding(indices) # * math.sqrt(self._embedding_dim) # => [seq_len, batch_size, embedding_dim]

    # add positional encoding
    pos = self._positional_encoding(embed)

    # so far each latent variable was embedded separately
    # now we stack them together, preparing for the transformer
    pos = pos.view(seq_len, batch_size, self._d_model) # => [seq_len, batch_size, d_model]

    # Construct causal mask
    causal_mask = torch.triu(torch.ones(pos.size(0), pos.size(0), device=self._device), diagonal=1).bool().to(pos.device)

    # if self.training:
    #   # Mask certain positions for better generalization (idea code from Behzad, cite if published)
    #   teacher_forcing_ratio = 0.7
    #   indices = torch.rand(causal_mask.size(0))
    #   indices = indices > teacher_forcing_ratio
    #   causal_mask[:, indices] = True

    #   # # Every position should be to attend at least itself (unmask the diagonal)
    #   causal_mask.fill_diagonal_(False)

    # encode using transformer encoder
    enc = self._encoder(pos, mask=causal_mask) * math.sqrt(self._d_model)

    # permute back
    enc = enc.permute(1, 0, 2) # => [batch_size, seq_len, d_model]

    # non-linearity
    enc = self._activation(enc)

    # dropout
    # enc = self._dropout(enc) # => [batch_size, seq_len, d_model]

    # project back to latent space
    fc = self._fc(enc) # [batch_size, seq_len, latent_size * quantization_channels]

    logits = fc.view(batch_size, seq_len, latent_size, self._quantization_channels) # => [batch_size, seq_len, latent_size, quantization_channels]

    return logits


  def training_step(self, batch, batch_idx):
    loss = self._step(batch)
    self.log('loss', loss['loss'], prog_bar=True)
    self.log('mse', loss['mse'], prog_bar=True)
    self.log('acc', loss['acc'], prog_bar=True)
    self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
    return loss


  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    self.log('val_loss', loss['loss'], prog_bar=True)
    self.log('val_mse', loss['mse'], prog_bar=True)
    self.log('val_acc', loss['acc'], prog_bar=True)
    return loss


  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
    # optimizer = torch.optim.SGD(self.parameters(), lr=self._lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=False, threshold=1e-4)

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
    y_discrete = self._quantizer(self.normalize(y))

    logits = self(x) # [batch_size, seq_len, latent_size, quantization_channels]

    batch_size = batch.size(0)

    # Calcualate class accuracy
    y_hat_discrete = torch.argmax(logits, dim=-1) # [batch_size, seq_len, latent_size]
    acc = (y_hat_discrete == y_discrete).float().sum() / y_hat_discrete.numel()

    # Calcualte mean square error on real values
    y_hat = self.denormalize(self._dequantizer(y_hat_discrete)) # [batch_size, seq_len, latent_size]
    mse = mse_loss(y_hat, y, reduction='none').nanmean()

    # Calculate cross-entropy loss
    ce_loss = cross_entropy(logits.permute(0, 3, 1, 2).view(batch_size, self._quantization_channels, -1), y_discrete.view(batch_size, -1), reduce=False).nanmean()

    loss = {
      'mse': mse,
      'acc': acc,
      'loss': ce_loss
    }

    return loss

class LearnablePositionalEncoding(nn.Module):
  def __init__(self, embedding_dim: int, max_len: int = 256, dropout: float = 0.1, device='cuda'):
    """
    Arguments:
      - embedding_dim: int, embedding diemnsionality
      - max_len: int, the maximum length of the sequence
      - dropout: float, the dropout rate
    """
    super(LearnablePositionalEncoding, self).__init__()

    self._device = device
    self._positional_encoding = nn.Parameter(torch.empty(max_len, 1, embedding_dim, device=self._device))
    nn.init.uniform_(self._positional_encoding, -0.02, 0.02)

    self._dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self._positional_encoding[:x.size(1), ...]
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
      embedding_dim: the embed dim (required).
      dropout: the dropout value (default=0.1).
      max_len: the max. length of the incoming sequence (default=1024).
  """

  def __init__(self, embedding_dim, dropout=0.1, max_len=1024, scale_factor=1.0, device='cuda'):
    super(FixedPositionalEncoding, self).__init__()
    self._device = device
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, embedding_dim, device=self._device)  # positional encoding
    position = torch.arange(0, max_len, dtype=torch.float, device=self._device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=self._device).float() * (-math.log(10000.0) / embedding_dim))
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
    x = x + self.pe[:x.size(1), :]
    return self.dropout(x)
