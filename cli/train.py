import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import torch
torch.set_default_dtype(torch.float32)

import argparse
import os

from torch.utils.data import DataLoader, random_split
from ddsp import DDSP, AudioDataset
from ddsp.callbacks import BetaWarmupCallback, CyclicalBetaWarmupCallback
from ddsp.utils import find_checkpoint

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_path', help='Directory of the training sound/sounds')
  parser.add_argument('--device', help='Device to use', default='cuda', choices=['cuda', 'cpu'])
  parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
  parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
  parser.add_argument('--n_band', type=int, default=512, help='Number of bands of the filter bank')
  parser.add_argument('--n_sines', type=int, default=500, help='Number of sines to synthesise')
  parser.add_argument('--fs', type=int, default=44100, help='Sampling rate of the audio')
  parser.add_argument('--encoder_ratios', type=int, nargs='+', default=[8, 4, 2], help='Capacity ratios for the encoder')
  parser.add_argument('--decoder_ratios', type=int, nargs='+', default=[2, 4, 8], help='Capacity ratios for the decoder')
  parser.add_argument('--capacity', type=int, default=64, help='Capacity of the model')
  parser.add_argument('--latent_size', type=int, default=16, help='Dimensionality of the latent space')
  parser.add_argument('--audio_chunk_duration', type=float, default=1.5, help='Duration of the audio chunks in seconds')
  parser.add_argument('--resampling_factor', type=int, default=32, help='Resampling factor for the control signal and noise bands')
  parser.add_argument('--mixed_precision', type=bool, default=True, help='Use mixed precision')
  parser.add_argument('--training_dir', type=str, default='training', help='Directory to save the training logs')
  parser.add_argument('--model_name', type=str, default='ddsp', help='Name of the model')
  parser.add_argument('--max_epochs', type=int, default=10000, help='Maximum number of epochs')
  parser.add_argument('--control_params', type=str, nargs='+', default=['loudness', 'centroid'], help='Control parameters to use, possible: aloudness, centroid, flatness')
  parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for the beta-VAE loss')
  parser.add_argument('--warmup_start', type=int, default=300, help='Step to start the beta warmup')
  parser.add_argument('--warmup_end', type=int, default=1300, help='Step to end the beta warmup')
  parser.add_argument('--kld_weight', type=float, default=0.0001, help='Weight for the KLD loss')
  parser.add_argument('--early_stopping', type=bool, default=False, help='Use early stopping')
  parser.add_argument('--force_restart', type=bool, default=False, help='Force restart the training. Ignore the existing checkpoint.')
  # parser.add_argument('--warmup_cycle', type=int, default=50, help='Number of epochs for a full beta cycle')
  config = parser.parse_args()

  n_signal = int(config.audio_chunk_duration * config.fs)

  # Create training directory
  training_path = os.path.join(config.training_dir, config.model_name)
  os.makedirs(training_path, exist_ok=True)

  # Load Dataset
  dataset = AudioDataset(
    dataset_path=config.dataset_path,
    n_signal=n_signal,
    sampling_rate=config.fs,
  )

  # Split into training and validation
  train_set, val_set = random_split(dataset, [0.9, 0.1])
  train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

  # Core model
  ddsp = DDSP(
    n_filters=config.n_band,
    n_sines=config.n_sines,
    latent_size=config.latent_size,
    fs=config.fs,
    encoder_ratios=config.encoder_ratios,
    decoder_ratios=config.decoder_ratios,
    capacity=config.capacity,
    resampling_factor=config.resampling_factor,
    learning_rate=config.lr,
    kld_weight=config.kld_weight,
  )

  # Tensorboard
  tb_logger = TensorBoardLogger(config.training_dir, name=config.model_name)

  training_callbacks = []

  # Warming up beta parameter
  beta_warmup = BetaWarmupCallback(
    beta=config.beta,
    start_steps=config.warmup_start,
    end_steps=config.warmup_end
  )
  training_callbacks.append(beta_warmup)

  # Early stopping
  if config.early_stopping:
    training_callbacks += [EarlyStopping(monitor='train_loss', patience=10, mode='min')]

  # Define the checkpoint callback
  checkpoint_callback = ModelCheckpoint(
      filename='best',
      monitor='train_loss',
      mode='min',
  )
  training_callbacks.append(checkpoint_callback)

  # Configure the trainer
  precision = 16 if config.mixed_precision else 32
  trainer = L.Trainer(
    callbacks=training_callbacks,
    max_epochs=config.max_epochs,
    accelerator=config.device,
    precision=precision,
    log_every_n_steps=4,
    logger=tb_logger,
  )

  # Try to find previously trained checkpoint
  ckpt_path = find_checkpoint(training_path, return_none=True) if not config.force_restart else None
  if ckpt_path is not None:
    print(f"Resuming from checkpoint: {ckpt_path}")

  # Start training
  trainer.fit(model=ddsp,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    ckpt_path=ckpt_path
  )
