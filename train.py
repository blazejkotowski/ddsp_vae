import lightning as L
import torch
torch.set_default_dtype(torch.float32)
import argparse

from torch.utils.data import DataLoader
from lib.dataset_tool import AudioDataset

from modules import NoiseBandNet

DATASET_PATH = '/Users/bl/code/noisebandnet/datasets/freesound-walking/processed'
SAMPLING_RATE = 44100
AUDIO_CHUNK_DURATION = 1.5
N_SIGNAL = int(SAMPLING_RATE * AUDIO_CHUNK_DURATION)
BATCH_SIZE = 16
TORCH_DEVICE = 'cpu'

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_path', help='Directory of the training sound/sounds')
  parser.add_argument('--device', help='Device to use', default='cuda', choices=['cuda', 'cpu'])
  parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
  parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
  parser.add_argument('--n_band', type=int, default=2048, help='Number of bands of the filter bank')
  parser.add_argument('--fs', type=int, default=44100, help='Sampling rate of the audio')
  parser.add_argument('--audio_chunk_duration', type=float, default=1.5, help='Duration of the audio chunks in seconds')
  parser.add_argument('--resampling_factor', type=int, default=32, help='Resampling factor for the control signal and noise bands')
  parser.add_argument('--output_path', help='Directory to save the model')
  parser.add_argument('--mixed_precision', type=bool, default=False, help='Use mixed precision')
  config = parser.parse_args()

  n_signal = int(config.audio_chunk_duration * config.fs)

  dataset = AudioDataset(
    dataset_path=config.dataset_path,
    audio_size_samples=n_signal,
    min_batch_size=config.batch_size,
    sampling_rate=config.fs,
    device=config.device
  )ls

  train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

  nbn = NoiseBandNet(learning_rate=1e-3)

  precision = 16 if config.mixed_precision else 32
  trainer = L.Trainer(max_epochs=10000, accelerator=config.device, precision=precision, log_every_n_steps=4)
  trainer.fit(model=nbn, train_dataloaders=train_loader)
