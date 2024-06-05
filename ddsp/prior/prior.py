import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchaudio
from torch.optim.lr_scheduler import ReduceLROnPlateau

def _is_batch_size_one(x: torch.Tensor) -> bool:
  return x.shape[0] == 1

class Prior(L.LightningModule):
  """
  Simple GRU-based prior model predicting the mu and scale of the latents.

  Args:
    - latent_size: int, the number of latents being input and predicted
    - hidden_size: int, the size of the hidden state in the GRU
    - num_layers: int, the number of layers in the GRU
    - dropout: float, the dropout rate
    - lr: float, the learning rate
    - streaming: bool, whether to run the model in streaming mode
    - sequence_length: int, the length of the preceding latent code sequence for prediction
    - rnn_type: str, the type of the RNN to use ['gru', 'lstm']
    - embedding_dim: int, the dimension of the embedding layer
    - x_min: float, the minimum value of the input (for normalization step)
    - x_max: float, the maximum value of the input (for normalization step)
  """

  def __init__(self,
               latent_size: int = 8,
               hidden_size: int = 512,
               num_layers: int = 8,
               dropout: float = 0.01,
               lr: float = 1e-3,
               streaming: bool = False,
               sequence_length: int = 100,
               rnn_type: str = 'gru',
               embedding_dim: int = 16,
               x_min: float = -1,
               x_max: float = 1):
    super().__init__()
    self.save_hyperparameters()

    self.sequence_length = sequence_length
    self.latent_size = latent_size

    self._type = rnn_type
    self._streaming = streaming
    self._lr = lr
    self._embedding_dim = embedding_dim

    # These are for normalization
    self._x_min = x_min
    self._x_max = x_max

    self._quantization_channels = 16
    self._quantize = torchaudio.transforms.MuLawEncoding(self._quantization_channels)
    self._dequantize = torchaudio.transforms.MuLawDecoding(self._quantization_channels)

    # Build the network

    ## Embedding layer
    self._embedding = nn.Embedding(self._quantization_channels, self._embedding_dim)

    ## GRU layer
    if self._type == 'gru':
      self._gru = nn.GRU(
        input_size=self._embedding_dim * self.latent_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_first=True
      )

    elif self._type == 'lstm':
      self._lstm = nn.LSTM(
        input_size=self._embedding_dim * self.latent_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_first=True
      )

    ## For keeping the GRU hidden state in streaming mode
    self.register_buffer('_hidden_state', torch.zeros(num_layers, 1, hidden_size), persistent=False)
    self.register_buffer('_cell_state', torch.zeros(num_layers, 1, hidden_size), persistent=False)

    ## Densely connected output layer mapping to logits (one class per quantization bin)
    self._fc = nn.Linear(hidden_size, self.latent_size * self._quantization_channels)


  def forward(self, input):
    """
    Predicts the latents from the input.

    Args:
      - input: torch.Tensor[batch_size, seq_len, latent_size], the input sequence of latents
    Returns:
      - logits: torch.Tensor[batch_size, latent_size, quantization_channels], the predicted logits
    """
    batch_size = input.shape[0]
    seq_len = input.shape[1]
    # Quantize the input into bins
    x = self._quantize(self._normalize(input)).long()

    # Create embeddings
    embeds = self._embedding(x)
    flattened = embeds.view(batch_size, seq_len, -1)

    # Pass through RNN
    rnn_out = self._rnn(flattened)[:, -1, :] # take the last one

    # Densely connected layer
    fc_out = self._fc(rnn_out)

    # Reshape to [batch_size, latent_size, quantization_channels]
    logits = fc_out.view(batch_size, self.latent_size, self._quantization_channels)

    return logits


  def loss(self, x, y):
    """
    Computes the cross-entropy loss for a batch of data.
    The x and y gets normalised and quantised into 256 bins.
    Next they are encoded using one-hot encoding and the loss is calculated using the cross-entropy loss.

    Args:
      - x: torch.Tensor[batch_size, seq_len, latent_size], the input sequence of latents
      - y: torch.Tensor[batch_size, latent_size], the target latents
    Returns:
      - loss: torch.Tensor[1], the loss
    """
    logits = self(x)
    target = self._quantize(self._normalize(y)).long()

    # Original
    # breakpoint()
    # loss = F.cross_entropy(logits.view(-1, self._quantization_channels), target.view(-1))
    loss = F.cross_entropy(logits.permute(0, 2, 1), target)
    return loss

    # Alternative
    # target_one_hot = F.one_hot(target, self._quantization_channels)
    # logits = torch.log_softmax(logits, dim=-1)
    # cross_entropy = -torch.sum(target_one_hot * logits, dim=-1)
    # return cross_entropy.mean()


    # sample = self.sample(logits)


  def sample(self, logits, temperature=1.0):
    """
    Samples from the logits.

    Args:
      - logits: torch.Tensor[batch_size, latent_size], the logits
    Returns:
      - sample: torch.Tensor[batch_size, latent_size], the sampled latents
    """
    batch_size = logits.shape[0]

    logits /= temperature
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs.view(-1, self._quantization_channels), 1)
    sampled = sampled.view(batch_size, self.latent_size)

    return self._denormalize(self._dequantize(sampled))


  def training_step(self, batch, batch_idx):
    """
    Computes the loss for a batch of data.

    Args:
      - batch: torch.Tensor[batch_size, seq_len, latent_size], the input sequence of latents
      - batch_idx: int, the index of the batch
    Returns:
      - loss: torch.Tensor[1], the loss
    """
    x, y = batch
    train_loss = self.loss(x, y)
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
    x, y = batch
    val_loss = self.loss(x, y)
    self.log("val_loss", val_loss, prog_bar=True, logger=True)
    self.log("accuracy" , self._accuracy(y, self(x)), prog_bar=True, logger=True)

    return val_loss


  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=1e-2)
    # optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, weight_decay=1e-2, lr=self._lr)
    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": ReduceLROnPlateau(optimizer, patience=1),
        "monitor": "train_loss",
      }
    }


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

  def _accuracy(self, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the MSE for a batch of data.

    Args:
      - y: torch.Tensor[batch_size, seq_len, latent_size], the target latents
      - logits: torch.Tensor[batch_size, seq_len, latent_size], the predicted logits
    Returns:
      - mse: torch.Tensor[1], the mean squared error
    """
    with torch.no_grad():
      y_hat = self.sample(logits)
      return F.mse_loss(y_hat, y)


  def _normalize(self, x: torch.Tensor) -> torch.Tensor:
    """Normalize the input tensor to [-1, 1]"""
    return 2*((x-self._x_min) / (self._x_max-self._x_min)) - 1


  def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
    """Denormalize the input tensor from [-1, 1] to its original scale"""
    return 0.5*(x+1) * (self._x_max-self._x_min) + self._x_min
