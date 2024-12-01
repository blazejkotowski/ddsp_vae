import os

import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import EarlyStopping

from random import randint
import lightning as L
import torch
torch.set_default_device('cuda')
torch.set_float32_matmul_precision('medium')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ddsp.prior import Prior, PriorDataset

model_name = 'drums'
training_path_root = '/home/btadeusz/code/ddsp_vae/training/seven-manifolds'
models_path_root = '/home/btadeusz/code/ddsp_vae/models/seven-manifolds'
# dataset_path = '/mnt/mariadata/datasets/seven_manifolds/liget'
dataset_path = '/mnt/mariadata/datasets/seven_manifolds/drums'
fs = 44100

# Training dir
training_path = os.path.join(training_path_root, model_name)
synth_output_path = os.path.join(models_path_root, f'{model_name}.ts')

# Fixed prior config
prior_epochs = 100
prior_model_name = f'{model_name}-prior'
lr = 1e-3

seq_idx = [randint(0, 100) for _ in range(4)]

# Training dir
prior_training_path = os.path.join(training_path_root, prior_model_name)

def generate_audio(prior, prior_dataset, idx, num_steps=15000):
  # Prepare the models
  synth = torch.jit.load(synth_output_path)
  prior = prior.train().to('cuda')

  sequence = prior_dataset[idx].unsqueeze(0).to('cuda')
  sequence_length = sequence.shape[1]-1

  for _ in range(num_steps):
    with torch.no_grad():
      next_code = prior(sequence[:, -sequence_length:, :])

    sequence = sequence = torch.cat((sequence, next_code[:, -1:, :]), dim=1)

  mu, logvar = sequence.chunk(2, dim=-1)

  with torch.no_grad():
    latents, _ = synth.pretrained.encoder.reparametrize(mu, logvar)
    audio = synth.decode(latents.permute(0, 2, 1).to('cpu'))

  audio = audio.cpu().numpy().squeeze()
  audio = audio / audio.max()

  return audio


def train_model():
  # Wandb config
  wandb.init()
  wandb_logger = WandbLogger()
  config = wandb.config

  # Load dataset
  prior_dataset = PriorDataset(
    audio_dataset_path=dataset_path,
    encoding_model_path=synth_output_path,
    sequence_length=config['sequence_length']+1,
    sampling_rate=fs,
    device='cuda',
    stride_factor=0.25,
    dropout=config['dropout']
  )

  batch_size = config['batch_size']
  # batch_size = 64

  generator = torch.Generator(device='cuda').manual_seed(42)
  prior_train_set, prior_val_set = random_split(prior_dataset, [0.9, 0.1], generator=generator)

  prior_train_loader = DataLoader(prior_train_set, batch_size=batch_size, shuffle=True, generator=generator)
  prior_val_loader = DataLoader(prior_val_set, batch_size=batch_size, shuffle=False, generator=generator)

  latent_size = prior_dataset[0].shape[-1]


  prior = Prior(
    latent_size = latent_size,
    d_model = config['d_model'],
    num_layers = 6,
    nhead = config['nhead'],
    max_len = config['sequence_length'],
    lr = 1e-3
  )

  early_stopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', min_delta=1e-6)

  prior_trainer = L.Trainer(
    callbacks=[early_stopping],
    accelerator='cuda',
    log_every_n_steps=4,
    logger=wandb_logger,
    max_epochs=config['epochs'],
    check_val_every_n_epoch=1,
  )

  prior_trainer.fit(
    model=prior,
    train_dataloaders=prior_train_loader,
    val_dataloaders=prior_val_loader
  )

  wandb_logger.log_audio('outcome_audio', [generate_audio(prior, prior_dataset, idx) for idx in seq_idx], sample_rate=[fs for _ in seq_idx])


if __name__ == '__main__':

  sweep_config = {
    'method': 'random',
    'name': 'ddsp-prior-sweep-5',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
      'sequence_length': {'values': [64, 128]},
      'batch_size': {'values': [16, 64, 128]},
      'd_model': {'values': [32, 64]},
      # 'num_layers': {'values': [2, 3, 6]},
      'nhead': {'values': [2, 4]},
      'epochs': {'values': [25, 50, 75, 100]},
      'dropout': {'values': [0, 0.1, 0.2, 0.5]}
      # 'dataset_stride': {'values': [0.25, 0.5, 1]}
    }
  }

  sweep_id=wandb.sweep(sweep_config, project="ddsp-prior")
  wandb.agent(sweep_id=sweep_id, function=train_model, count=96)
