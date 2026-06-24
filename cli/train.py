import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import torch
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

import os
import shutil

from torch.utils.data import DataLoader, Subset, random_split
from ddsp import DDSP, AudioFeatureDataset
from ddsp.synths import BendableNoiseBandSynth
from ddsp.callbacks import BetaWarmupEpochCallback
from ddsp.interfaces import build_control_space
from ddsp.augmentations import build_audio_augmentation_pipeline
from ddsp.registry import SYNTHS

from ddsp.prior import Prior, PriorDataset
from ddsp.utils import find_checkpoint

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
  # Set default device/tensor type only during CLI execution
  if torch.cuda.is_available():
    try:
      torch.set_default_device('cuda')
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
    except Exception:
      pass
  # Shared audio params
  fs = int(cfg.audio.fs)
  resampling_factor = int(cfg.model.resampling_factor)
  chunk_duration_s = float(cfg.audio.chunk_duration_s)
  n_channels = int(getattr(cfg.audio, 'n_channels', 1))
  n_signal = int(fs * chunk_duration_s)

  # Paths
  dataset_path = cfg.data.dataset_path
  model_name = cfg.experiment.name
  training_dir = cfg.experiment.training_dir
  synth_training_path = os.path.join(training_dir, 'synth', model_name)
  os.makedirs(synth_training_path, exist_ok=True)

  # Control space
  control_space = build_control_space(cfg.model.control_space)

  # Optional audio augmentations
  transform_fn = build_audio_augmentation_pipeline(
    getattr(cfg.data, 'augmentations', None),
    n_signal=n_signal,
    sampling_rate=fs,
  )

  # Dataset
  synth_dataset = AudioFeatureDataset(
    dataset_path=dataset_path,
    n_signal=n_signal,
    sampling_rate=fs,
    resampling_factor=resampling_factor,
    control_space=control_space,
    transform_fn=transform_fn,
    n_channels=n_channels,
  )

  # Dataloaders
  batch_size = int(cfg.trainer.batch_size)
  generator = torch.Generator(device='cuda')
  synth_total_len = len(synth_dataset)
  synth_val_len = int(0.2 * synth_total_len)
  synth_indices = torch.randperm(synth_total_len, generator=generator)
  synth_val_indices = synth_indices[:synth_val_len]
  train_indices = synth_indices
  train_loader = DataLoader(Subset(synth_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
  val_loader = DataLoader(Subset(synth_dataset, synth_val_indices), batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)

  # Synths from config
  synth_configs = []
  for s in cfg.model.synths:
    synth_configs.append({"class": s.type, "params": dict(s.params)})

  # Capture the active Hydra config name for logging/hparams
  try:
    active_config_name = HydraConfig.get().job.config_name
  except Exception:
    active_config_name = None

  # Core model
  ddsp = DDSP(
    control_space=control_space,
    synth_configs=synth_configs,
    fs=fs,
    n_channels=n_channels,
    resampling_factor=resampling_factor,
    latent_smoothing_kernel=int(cfg.model.latent_smoothing_kernel),
    decoder_gru_layers=int(cfg.model.decoder_gru_layers),
    decoder_temporal_stride=int(getattr(cfg.model, 'decoder_temporal_stride', 1)),
    learning_rate=float(cfg.model.learning_rate),
    perceptual_loss_weight=float(getattr(cfg.model, 'perceptual_loss_weight', 0.0)),
    plateau_patience=int(cfg.model.plateau_patience),
    capacity=int(cfg.model.capacity),
    losses=[dict(l) for l in getattr(cfg, 'losses', [])],
    adversarial_loss=bool(cfg.adversarial.enabled),
    adv_g_start_epoch=int(cfg.adversarial.schedule.g_start_epoch),
    adv_d_start_epoch=int(cfg.adversarial.schedule.d_start_epoch),
    adv_gen_weight=float(cfg.adversarial.weights.gen),
    adv_disc_weight=float(cfg.adversarial.weights.disc),
    adv_fm_weight=float(cfg.adversarial.weights.fm),
    adv_ndf=int(getattr(cfg.adversarial, 'ndf', 4)),
    adv_disc_update_every=int(getattr(cfg.adversarial, 'disc_update_every', 5)),
    ddsp_min_lr=float(getattr(cfg.model, 'min_lr', 0.0)),
    config_name=active_config_name,
  )

  # Tensorboard
  tb_logger = TensorBoardLogger(synth_training_path, name=model_name)

  # Callbacks
  callbacks = []
  if 'beta' in cfg.model:
    callbacks.append(BetaWarmupEpochCallback(
      beta=float(cfg.model.beta.target),
      start_epoch=int(cfg.model.beta.start_epoch),
      end_epoch=int(cfg.model.beta.end_epoch),
    ))

  callbacks.append(ModelCheckpoint(dirpath=synth_training_path, filename='best', monitor='val_loss', mode='min', save_top_k=1, save_last=True))

  # Trainer
  synth_trainer = L.Trainer(
    callbacks=callbacks,
    max_epochs=int(cfg.trainer.max_epochs),
    log_every_n_steps=int(cfg.trainer.log_every_n_steps),
    logger=tb_logger,
    enable_progress_bar=(os.environ.get('SYNTH_NO_PBAR', '0') != '1'),
  )

  # Resume checkpoint
  synth_ckpt_path = find_checkpoint(synth_training_path, return_none=True, typ='last') if not bool(cfg.trainer.force_restart) else None
  if cfg.trainer.force_restart:
    print("Force restart set to True")
  elif synth_ckpt_path is not None:
    print(f"Resuming from checkpoint: {synth_ckpt_path}")

  # Train
  synth_trainer.fit(model=ddsp, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=synth_ckpt_path)
  print(f"Synthesizer training completed. Your checkpoints are in {synth_training_path}")


if __name__ == '__main__':
  main()
