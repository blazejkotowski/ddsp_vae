
import os

def find_checkpoint(model_directory: str, return_none: bool = False, typ: str = 'best') -> str:
  """
  Finds the last checkpoint recursively looking in the model directory

  Args:
    model_directory: str, the path to the model directory
    return_none: bool, whether to return None if no checkpoint is found
    typ: str, the type of checkpoint to find, either 'best' or 'last'
  Returns:
    path: str, the path to the checkpoint
  """
  if typ not in ['best', 'last']:
    raise ValueError(f"Invalid type: {typ}, supported types are: best, last")

  checkpoints = []
  for root, _, files in os.walk(model_directory):
    for file in files:
      if typ in file and file.endswith('.ckpt'):
        checkpoints.append(os.path.join(root, file))

  if not checkpoints:
    if return_none:
      return None
    else:
      raise ValueError(f"No checkpoints found in {model_directory}")

  return max(checkpoints, key=os.path.getctime)
