import torch
import math

from typing import Callable

def multiply_and_sum_tensors(ten_a: torch.Tensor, ten_b: torch.Tensor, n_chunks: int = 4) -> torch.Tensor:
  """
  Optimization of torch.sum(a * b, dim=1, keepdim=True)
  Multiply and sum two tensors, optimized for saving memory at cost of speed.
  Takes two tensors, splits them into N chunks, multiplies and sums them chunk by chunk.
  Arguments:
    - a: torch.Tensor[batch_size, k, l], the first tensor
    - b: torch.Tensor[?, k, l] the second tensor
    - n_chunks: int, the number of chunks to split the tensors into
  Returns:
    - c: torch.Tensor[batch_size, 1, l], the result of the multiplication and sum
  """
  batch_size, k, l = ten_a.shape
  if ten_b.dim() == 2:
     ten_b = ten_b.unsqueeze(0)  # Add batch dimension if not present

  c = torch.zeros(batch_size, 1, l, device=ten_a.device)

  k_chunk_size = math.ceil(k / n_chunks)

  for i in range(n_chunks):
    a_chunk = ten_a[:, i*k_chunk_size:(i+1)*k_chunk_size, :]
    b_chunk = ten_b[:, i*k_chunk_size:(i+1)*k_chunk_size, :]
    c += torch.sum(a_chunk * b_chunk, dim=1, keepdim=True)

  return c

def apply_method_in_place(tensor: torch.Tensor, method: Callable, n_chunks: int = 4) -> torch.Tensor:
  """
  Optimization of array method optimized for saving memory at cost of speed.

  Arguments:
    - tensor: torch.Tensor, the tensor to apply the method to
    - method: Callable, the method to apply to the tensor
    - n_chunks: int, the number of chunks to split the tensor into
  """
  size = math.ceil(tensor.shape[-1] / n_chunks)
  for i in range(n_chunks):
    tensor[..., i*size:(i+1)*size] = method(tensor[..., i*size:(i+1)*size])

  return tensor
