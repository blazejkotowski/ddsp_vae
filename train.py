import lightning as L
import torch
import argparse
torch.set_default_device('cpu')

from torch.utils.data import DataLoader
from lib.dataset_tool import AudioDataset

from modules import NoiseBandNet

DATASET_PATH = '/Users/bl/code/noisebandnet/datasets/freesound-walking/processed'
SAMPLING_RATE = 44100
AUDIO_CHUNK_DURATION = 1.5
N_SIGNAL = int(SAMPLING_RATE * AUDIO_CHUNK_DURATION)
BATCH_SIZE = 16
TORCH_DEVICE = 'cpu'

config = argparse.ArgumentParser()
config.add_argument('--dataset_path', help='Directory of the training sound/sounds')
config.add_argument('--device', help='Device to use', default='cuda', choices=['cuda', 'cpu'])
config.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
config.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
config.add_argument('--n_band', type=int, default=2048, help='Number of bands of the filter bank')
config.add_argument('--fs', type=int, default=44100, help='Sampling rate of the audio')
config.add_argument('--audio_chunk_duration', type=float, default=1.5, help='Duration of the audio chunks in seconds')
config.add_argument('--resampling_factor', type=int, default=32, help='Resampling factor for the control signal and noise bands')
config.add_argument('--output_path', help='Directory to save the model')
config.parse_args()

n_signal = config.audio_chunk_duration * config.fs


dataset = AudioDataset(
  dataset_path=DATASET_PATH,
  audio_size_samples=N_SIGNAL,
  min_batch_size=BATCH_SIZE,
  sampling_rate=SAMPLING_RATE,
  device=TORCH_DEVICE
)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

nbn = NoiseBandNet(learning_rate=1e-3)

trainer = L.Trainer(limit_train_batches=100, max_epochs=100, gradient_clip_val=0.5, precision=32, accelerator=TORCH_DEVICE)
trainer.fit(model=nbn, train_dataloaders=train_loader)
