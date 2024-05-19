import torch

def multiply_and_sum_tensors(a: torch.Tensor, b: torch.Tensor, n_chunks: int = 4) -> torch.Tensor:
    """
    Optimization of torch.sum(a * b, dim=1, keepdim=True)
    Multiply and sum two tensors, optimized for saving memory at cost of speed.
    Takes two tensors, splits them into N chunks, multiplies and sums them chunk by chunk.
    Arguments:
      - a: torch.Tensor[batch_size, k, l], the first tensor
      - b: torch.Tensor[batch_size, ?, l] the second tensor
      - n_chunks: int, the number of chunks to split the tensors into
    Returns:
      - c: torch.Tensor[batch_size, 1, l], the result of the multiplication and sum
    """
    c = torch.zeros(a.shape[0], 1, a.shape[2], device=a.device)
    for a_chunk, b_chunk in zip(a.chunk(n_chunks, dim=1), b.chunk(n_chunks, dim=-2)):
      c += torch.sum(a_chunk * b_chunk, dim=1, keepdim=True)
    return c


def cumsum(tensor: torch.Tensor, n_chunks: int = 4) -> torch.Tensor:
    """
    Optimization of torch.cumsum(tensor, dim=-1)
    Compute the cumulative sum of a tensor, optimized for saving memory at cost of speed.
    Takes a tensor, splits it into N chunks, computes the cumulative sum chunk by chunk.
    Arguments:
      - tensor: torch.Tensor, the input tensor
    - n_chunks: int, the number of chunks to split the tensor into
    Returns:
      - result: torch.Tensor, the cumulative sum of the input tensor
    """
    # Split the omegas tensor into chunks along the last dimension
    chunks = torch.chunk(tensor, n_chunks, dim=-1)

    # Initialize the cumulative sum tensor
    result = torch.Tensor()
    last_sum = 0

    # Iterate over each chunk and compute the cumulative sum
    for chunk in chunks:
      chunk_cumsum = torch.cumsum(chunk, dim=-1) + last_sum
      result = torch.cat([result, chunk_cumsum], dim=-1)
      last_sum = result[..., -1].unsqueeze(-1)

    return result
