from torch.utils.data import Dataset as TorchDataset
import os

class Dataset(TorchDataset):
  """
  Dataset class for the NoiseBandNet model.
  """
  def __init__(self, path: str, n_signal: int):
    """
    Args:
      - path: str, path to the dataset
      - n_signal: int, length of audio chank in samples
    """
    os.walk()


  def __len__(self):
    pass

  def __getitem__(self, idx):
    pass
