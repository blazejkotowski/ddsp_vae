import os

import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig

# torch>=2.6 checkpoint-load compatibility (Colab). See ddsp/checkpoint_compat.py.
from ddsp.checkpoint_compat import weights_only_false_kwargs, allow_full_checkpoints, force_weights_only_false
allow_full_checkpoints()
force_weights_only_false()

from ddsp.interfaces import build_control_space
from ddsp.utils import find_checkpoint
from ddsp.prior.lmdb_cache import infer_ddsp_checkpoint_path
from ddsp.postnet import PostNet, build_or_load_postnet_cache, EMACallback

# Reuse the exact frozen-synth loader the prior trainer uses (same load kwargs).
from cli.train_prior import _load_ddsp


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
  """Hydra-driven training for the faithful post-net (streaming spectral transform).

  Reads the same config as cli.train / cli.train_prior. Requires a trained synth under
  training/synth/<name>/. Writes to training/postnet/<name>/ and is auto-consumed by cli.export.
  """
  L.seed_everything(cfg.get('seed', 42))

  pn_cfg = cfg.get('postnet', None)
  if pn_cfg is None or not bool(pn_cfg.get('enabled', False)):
    print("cfg.postnet.enabled is False (or absent); skipping post-net training.")
    return

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Frozen synth
  control_space = build_control_space(cfg.model.control_space)
  synth_configs = []
  for s in cfg.model.synths:
    synth_configs.append({"class": s.type, "params": dict(s.params)})
  model = _load_ddsp(cfg, control_space, synth_configs, device)
  try:
    model._postnet_synth_ckpt = infer_ddsp_checkpoint_path(cfg)
  except Exception:
    model._postnet_synth_ckpt = ''

  # Paired (rough, real, control) cache from the frozen synth
  cache = build_or_load_postnet_cache(cfg, model, device)
  cond_dim = int(cache['cond_dim'])

  train_ds = TensorDataset(cache['train_rough'], cache['train_real'], cache['train_cond'])
  eval_ds = TensorDataset(cache['eval_rough'], cache['eval_real'], cache['eval_cond'])

  m = pn_cfg.get('model', {}) or {}
  t = pn_cfg.get('training', {}) or {}
  batch_size = int(t.get('batch_size', 16))
  max_steps = int(os.environ.get('POSTNET_MAX_STEPS', str(int(t.get('max_steps', 18000)))))

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
  val_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

  postnet = PostNet(
    cond_dim=cond_dim,
    ch=int(m.get('ch', 128)),
    layers=int(m.get('layers', 10)),
    ch2=int(m.get('ch2', 96)),
    layers2=int(m.get('layers2', 8)),
    max_gain_db=float(m.get('max_gain_db', 15.0)),
    phase_cap=float(m.get('phase_cap', 1.57)),
    la_f=int(m.get('la_f', 3)),
    lr=float(t.get('lr', 3e-4)),
    max_steps=max_steps,
    bend_aug_p=float(t.get('bend_aug_p', 0.4)),
    gain_slew_w=float(t.get('gain_slew_w', 0.5)),
    fs=int(cfg.audio.fs),
  )
  # Report validation MRSTFT in the same style as cli.train (reuses the synth's own loss machinery).
  postnet.attach_synth_metrics(model)
  print(f"[postnet] cond_dim={cond_dim} params={sum(p.numel() for p in postnet.parameters())/1e3:.1f}K "
        f"max_steps={max_steps}")

  out_dir = os.path.join(cfg.experiment.training_dir, 'postnet', cfg.experiment.name)
  os.makedirs(out_dir, exist_ok=True)
  # Monitor the CONVENTIONAL pred-vs-target MRSTFT (`val_mrstft`), not the synth's reversed-arg
  # `val_loss` — the latter is asymmetric under perceptual weighting and would misguide selection.
  # Both are still logged (val_loss for direct comparability with cli.train).
  ckpt_cb = ModelCheckpoint(dirpath=out_dir, filename='best', monitor='val_mrstft', mode='min',
                            save_top_k=1, save_last=True, enable_version_counter=False)
  callbacks = [ckpt_cb]
  ema_decay = float(t.get('ema', 0.0))
  if ema_decay > 0.0:
    callbacks.append(EMACallback(ema_decay))
  logger = TensorBoardLogger(save_dir=out_dir, name='logs')

  # Validate a few times over the run (val_mrstft drives checkpointing; val_loss also logged for
  # direct comparability with cli.train's reported metric).
  val_interval = max(1, max_steps // 10)
  trainer = L.Trainer(
    max_steps=max_steps,
    accelerator='gpu' if device == 'cuda' else 'cpu',
    devices=1,
    gradient_clip_val=5.0,
    val_check_interval=val_interval,
    check_val_every_n_epoch=None,
    log_every_n_steps=10,
    callbacks=callbacks,
    logger=logger,
    default_root_dir=out_dir,
    enable_progress_bar=(os.environ.get('POSTNET_NO_PBAR', '0') != '1'),
  )

  force_restart = bool(t.get('force_restart', False))
  ckpt_path = None
  if not force_restart:
    ckpt_path = find_checkpoint(out_dir, return_none=True, typ='last')
    if ckpt_path is not None:
      print(f"[postnet] resuming from checkpoint: {ckpt_path}")
  else:
    print("[postnet] force restart; starting from scratch.")

  trainer.fit(postnet, train_loader, val_loader, ckpt_path=ckpt_path,
              **weights_only_false_kwargs(trainer.fit))
  print(f"[postnet] training complete. Checkpoints in {out_dir}")
  if ckpt_cb.best_model_score is not None:
    print(f"[postnet] best val_mrstft={float(ckpt_cb.best_model_score):.4f} "
          f"(conventional pred-vs-target; val_loss is also logged in cli.train's reversed-arg style)")


if __name__ == '__main__':
  main()
