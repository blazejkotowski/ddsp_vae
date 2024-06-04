import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchaudio

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
               sequence_length: int = 10,
               rnn_type: str = 'gru',
               x_min: float = -1,
               x_max: float = 1):
    super().__init__()
    self.save_hyperparameters()

    self.sequence_length = sequence_length
    self.latent_size = latent_size

    self._type = rnn_type
    self._streaming = streaming
    self._lr = lr

    # These are for normalization
    self._x_min = x_min
    self._x_max = x_max

    self._quantization_channels = 256
    self._quantize = torchaudio.transforms.MuLawEncoding(self._quantization_channels)
    self._dequantize = torchaudio.transforms.MuLawDecoding(self._quantization_channels)

    # Build the network
    ## GRU layer
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

    ## For keeping the GRU hidden state in streaming mode
    self.register_buffer('_hidden_state', torch.zeros(num_layers, 1, hidden_size), persistent=False)
    self.register_buffer('_cell_state', torch.zeros(num_layers, 1, hidden_size), persistent=False)

    ## Densely connected output layer mapping to logits (one class per quantization bin)
    self._fc = nn.Linear(hidden_size, latent_size * self._quantization_channels)


  def forward(self, input):
    """
    Predicts the latents from the input.

    Args:
      - input: torch.Tensor[batch_size, seq_len, latent_size], the input sequence of latents
    Returns:
      - logits: torch.Tensor[batch_size, latent_size, quantization_channels], the predicted logits
    """
    batch_size = input.shape[0]
    # Normalize the input
    x = self._quantize(self._normalize(input)).float()
    rnn_out = self._rnn(x)[:, -1, :] # take the last one

    # Densely connected layer
    logits = self._fc(rnn_out)

    # Reshape to [batch_size, latent_size, quantization_channels]
    logits = logits.view(batch_size, self.latent_size, self._quantization_channels)

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

    loss = F.cross_entropy(logits.view(-1, self._quantization_channels), target.view(-1))
    return loss


  def sample(self, logits, temperature=1.0):
    """
    Samples from the logits using the Gumbel-Softmax trick.

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

    return self._dequantize(self._denormalize(logits))


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

    return val_loss


  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self._lr)


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


  def _normalize(self, x: torch.Tensor) -> torch.Tensor:
    """Normalize the input tensor to [-1, 1]"""
    return 2*((x-self._x_min) / (self._x_max-self._x_min)) - 1


  def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
    """Denormalize the input tensor from [-1, 1] to its original scale"""
    return 0.5*(x+1) * (self._x_max-self._x_min) + self._x_min
