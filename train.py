import lightning as L
from torch.utils.data import DataLoader
from lib.dataset_tool import AudioDataset

from modules import NoiseBandNet

DATASET_PATH = '/Users/bl/code/noisebandnet/datasets/freesound-walking/processed'
SAMPLING_RATE = 44100
AUDIO_CHUNK_DURATION = 1.5
N_SIGNAL = int(SAMPLING_RATE * AUDIO_CHUNK_DURATION)
BATCH_SIZE = 16
TORCH_DEVICE = 'mps'

dataset = AudioDataset(
  dataset_path=DATASET_PATH,
  audio_size_samples=N_SIGNAL,
  min_batch_size=BATCH_SIZE,
  sampling_rate=SAMPLING_RATE,
  device=TORCH_DEVICE
)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

nbn = NoiseBandNet(learning_rate=1e-6)

trainer = L.Trainer(limit_train_batches=100, max_epochs=100, gradient_clip_val=0.5, precision=32)
trainer.fit(model=nbn, train_dataloaders=train_loader)
