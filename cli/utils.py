
import os

def find_checkpoint(model_directory: str) -> str:
  """Finds the last checkpoint recursively looking in the model directory"""
  checkpoints = []
  for root, _, files in os.walk(model_directory):
    for file in files:
      if file.endswith('.ckpt'):
        checkpoints.append(os.path.join(root, file))

  if not checkpoints:
    raise ValueError(f"No checkpoints found in {model_directory}")

  return max(checkpoints, key=os.path.getctime)
