import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import argparse
import os

from torch.utils.data import DataLoader, random_split

from ddsp.prior import Prior, PriorDataset
from ddsp.prior.prior_dataset import DummyMultivariateSequenceDataset
from ddsp.utils import find_checkpoint

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
  parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
  parser.add_argument('--sequence_length', type=int, default=5, help='Number of the preceding latent codes')
  parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
  parser.add_argument('--n_layers', type=int, default=6, help='Number of layers in the transformer')
  parser.add_argument('--fs', type=int, default=44100, help='Sampling rate of the audio')
  parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
  parser.add_argument('--training_dir', type=str, default='training/prior', help='Directory to save the training logs')
  parser.add_argument('--force_restart', type=bool, default=False, help='Force restart the training. Ignore the existing checkpoint.')
  parser.add_argument('--device', type=str, default='cuda', help='Device to use', choices=['cuda', 'cpu', 'mps'])
  parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
  parser.add_argument('--model_path', type=str, help='Path to the encoding model', required=True)
  parser.add_argument('--dataset_path', help='Directory of the training sound/sounds', required=True)
  parser.add_argument('--early_stopping', type=bool, default=True, help='Enable early stopping')
  config = parser.parse_args()


  # Create training directory
  training_path = os.path.join(config.training_dir, config.model_name)
  os.makedirs(training_path, exist_ok=True)

  # Load Dataset
  dataset = PriorDataset(
    audio_dataset_path=config.dataset_path,
    encoding_model_path=config.model_path,
    sequence_length=config.sequence_length,
    sampling_rate=config.fs,
    device=config.device
  )
  # For denormalizing
  normalization_dict = dataset.normalization_dict

  # Dummy dataset
  # dataset = DummyMultivariateSequenceDataset(num_features=16, seq_length=config.sequence_length, n_examples=10000)
  # normalization_dict = {'mean': 0.0, 'var': 1.0}


  # Split into training and validation
  train_set, val_set = random_split(dataset, [0.9, 0.1])
  train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

  latent_size = dataset[0][0].shape[-1]

  # Core model
  prior = Prior(
    latent_size=latent_size,
    dropout=config.dropout,
    lr=config.lr,
    d_model=config.d_model,
    num_layers=config.n_layers,
    normalization_dict=normalization_dict,
    max_len=config.sequence_length,
  )

  # Setup the logger
  logger = TensorBoardLogger(training_path, name=config.model_name)


  # Early stopping callback
  callbacks = []
  if config.early_stopping:
    callbacks.append(EarlyStopping(monitor='val_loss', patience=10))

  # Checkpoint callback
  checkpoint_callback = ModelCheckpoint(
    dirpath=training_path,
    filename='best',
    monitor='val_loss',
    mode='min'
  )
  callbacks.append(checkpoint_callback)

  # Configure the trainer
  trainer = L.Trainer(
    callbacks=callbacks,
    accelerator=config.device,
    log_every_n_steps=4,
    logger=logger,
  )

  # Try to find previous checkpoint
  ckpt_path = find_checkpoint(training_path, return_none=True) if not config.force_restart else None
  if ckpt_path is not None:
    print(f'Resuming training from checkpoint {ckpt_path}')

  # Start training
  trainer.fit(
    model=prior,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    ckpt_path=ckpt_path,
  )


