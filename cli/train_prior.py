import os
import math
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig

from ddsp import DDSP
from ddsp.audio_feature_dataset import AudioFeatureDataset
from ddsp.prior.prior import Prior
from ddsp.prior.prior_discrete import PriorDiscrete
from ddsp.prior.prior_dataset import PriorDataset
from ddsp.prior.dataset import PriorSequenceDataset, PriorTokenSequenceDataset
from ddsp.latent_compressor import LatentCompressor
from ddsp.prior.lmdb_cache import (
  build_or_load_prior_cache_from_cfg,
  build_or_load_prior_tokens_cache_from_cfg,
  infer_ddsp_checkpoint_path,
)
from ddsp.interfaces import build_control_space
from ddsp.utils import find_checkpoint


def _load_ddsp(cfg: DictConfig, control_space, synth_configs, device: str) -> DDSP:
  """Load the trained DDSP model (same kwargs as the prior cache loader)."""
  ckpt_path = infer_ddsp_checkpoint_path(cfg)
  print(f"Loading DDSP checkpoint: {ckpt_path}")
  ddsp = DDSP.load_from_checkpoint(
    ckpt_path,
    strict=False,
    streaming=False,
    device=device,
    control_space=control_space,
    synth_configs=synth_configs,
    fs=int(cfg.audio.fs),
    resampling_factor=int(cfg.model.resampling_factor),
    latent_smoothing_kernel=int(cfg.model.latent_smoothing_kernel),
    decoder_gru_layers=int(cfg.model.decoder_gru_layers),
    learning_rate=float(cfg.model.learning_rate),
    plateau_patience=int(cfg.model.plateau_patience),
    capacity=int(cfg.model.capacity),
    losses=[dict(l) for l in getattr(cfg, 'losses', [])],
    adversarial_loss=bool(cfg.adversarial.enabled),
    adv_g_start_epoch=int(cfg.adversarial.schedule.g_start_epoch),
    adv_d_start_epoch=int(cfg.adversarial.schedule.d_start_epoch),
    adv_gen_weight=float(cfg.adversarial.weights.gen),
    adv_disc_weight=float(cfg.adversarial.weights.disc),
    adv_fm_weight=float(cfg.adversarial.weights.fm),
  ).to(device)
  ddsp.eval()
  return ddsp


def _train_compressor(cfg: DictConfig, control_space, synth_configs, device: str) -> str:
  """Train the LatentCompressor over DDSP control sequences; return best checkpoint path."""
  comp_cfg = cfg.compressor
  ddsp_model = _load_ddsp(cfg, control_space, synth_configs, device)

  control_rate = int(ddsp_model.fs) / int(ddsp_model.resampling_factor)
  input_dim = int(ddsp_model.feature_dim + ddsp_model.latent_size)

  # Training window length (in control frames) is derived from the control rate and
  # chunk duration; an explicit compressor.sequence_length overrides if ever needed.
  seq_len_override = comp_cfg.get('sequence_length', None)
  if seq_len_override is not None:
    sequence_length = int(seq_len_override)
  else:
    sequence_length = int(round(control_rate * float(cfg.audio.chunk_duration_s)))
  print(
    f"[compressor] control rate: {control_rate:.1f} Hz, input_dim: {input_dim}, "
    f"sequence_length: {sequence_length} frames (~{cfg.audio.chunk_duration_s}s)"
  )

  n_signal = int(cfg.audio.fs * cfg.audio.chunk_duration_s)
  audio_dataset = AudioFeatureDataset(
    dataset_path=str(cfg.data.dataset_path),
    n_signal=n_signal,
    sampling_rate=int(cfg.audio.fs),
    resampling_factor=int(ddsp_model.resampling_factor),
    control_space=ddsp_model.control_space,
    device=device,
  )
  prior_dataset = PriorDataset(
    audio_dataset=audio_dataset,
    sequence_length=sequence_length,
    encoding_model=ddsp_model,
    sampling_rate=int(cfg.audio.fs),
    stride_factor=float(comp_cfg.stride_factor),
    device=device,
  )
  print(f"[compressor] PriorDataset size: {len(prior_dataset)} sequences")

  total_len = len(prior_dataset)
  val_len = max(1, int(total_len * 0.1))
  train_set, val_set = random_split(
    prior_dataset, [total_len - val_len, val_len],
    generator=torch.Generator().manual_seed(int(cfg.get('seed', 42))),
  )

  sample = prior_dataset[0]
  sample_is_cuda = isinstance(sample, torch.Tensor) and sample.is_cuda
  num_workers = int(comp_cfg.get('num_workers', 0))
  if sample_is_cuda and num_workers != 0:
    print("[compressor] PriorDataset returns CUDA tensors; forcing num_workers=0")
    num_workers = 0

  train_loader = DataLoader(train_set, batch_size=int(comp_cfg.batch_size), shuffle=True,
                            num_workers=num_workers, pin_memory=False, persistent_workers=False)
  val_loader = DataLoader(val_set, batch_size=int(comp_cfg.batch_size), shuffle=False,
                          num_workers=num_workers, pin_memory=False, persistent_workers=False)

  vq_cfg = comp_cfg.get('vq', {}) or {}
  max_channels = comp_cfg.get('max_channels', None)
  if max_channels is not None:
    max_channels = int(max_channels)

  compressor = LatentCompressor(
    input_dim=input_dim,
    hidden_dim=int(comp_cfg.hidden_dim),
    compressed_dim=int(comp_cfg.compressed_dim),
    strides=list(comp_cfg.get('strides', [8, 4])),
    num_residual_layers=int(comp_cfg.num_residual_layers),
    kernel_size=int(comp_cfg.get('kernel_size', 7)),
    max_channels=max_channels,
    learning_rate=float(comp_cfg.learning_rate),
    use_skip_connections=bool(comp_cfg.get('use_skip_connections', True)),
    vq_enabled=bool(vq_cfg.get('enabled', False)),
    vq_codebook_size=int(vq_cfg.get('codebook_size', 1024)),
    vq_beta=float(vq_cfg.get('beta', 0.25)),
    vq_loss_weight=float(vq_cfg.get('loss_weight', 1.0)),
    vq_num_codebooks=int(vq_cfg.get('num_codebooks', 1)),
  )
  print(f"[compressor] params: {compressor.get_num_params() / 1e6:.2f}M  compression: {compressor.compression_ratio}x")

  out_dir = os.path.join(cfg.experiment.training_dir, 'compressor', cfg.experiment.name)
  os.makedirs(out_dir, exist_ok=True)
  ckpt_cb = ModelCheckpoint(dirpath=out_dir, filename='best', monitor='val_loss',
                            mode='min', save_top_k=1, save_weights_only=True,
                            enable_version_counter=False)
  early = EarlyStopping(monitor='val_loss', patience=80, mode='min')

  trainer = L.Trainer(
    max_epochs=int(comp_cfg.max_epochs),
    accelerator='gpu' if device == 'cuda' else 'cpu',
    devices=1,
    callbacks=[ckpt_cb, early],
    log_every_n_steps=10,
    default_root_dir=out_dir,
  )
  print("[compressor] training...")
  trainer.fit(compressor, train_loader, val_loader)

  best_path = ckpt_cb.best_model_path or os.path.join(out_dir, 'best.ckpt')
  if ckpt_cb.best_model_score is not None:
    print(f"[compressor] done. best val_loss={float(ckpt_cb.best_model_score):.6f} -> {best_path}")
  return best_path


def _ensure_compressor(cfg: DictConfig, control_space, synth_configs, device: str) -> str:
  """Return a usable compressor checkpoint, training one first if none exists."""
  explicit = getattr(cfg.prior.discrete, 'compressor_ckpt', None)
  default_ckpt = os.path.join(cfg.experiment.training_dir, 'compressor', cfg.experiment.name, 'best.ckpt')
  target = str(explicit) if explicit else default_ckpt

  force = bool(cfg.compressor.get('force_restart', False)) if 'compressor' in cfg else False
  if os.path.exists(target) and not force:
    print(f"[compressor] using existing checkpoint: {target}")
    return target

  print("[compressor] no checkpoint found; training the compressor first.")
  return _train_compressor(cfg, control_space, synth_configs, device)


def _train_discrete(cfg: DictConfig, control_space, synth_configs, in_memory: bool):
  """Train the codec prior: compressor (if needed) -> token cache -> PriorDiscrete.

  Self-contained so the continuous-Prior path above stays unchanged.
  """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  compressor_ckpt = _ensure_compressor(cfg, control_space, synth_configs, device)

  lmdb_path, stats = build_or_load_prior_tokens_cache_from_cfg(
    cfg,
    control_space=control_space,
    synth_configs=synth_configs,
    device=device,
    compressor_ckpt=compressor_ckpt,
  )
  print(f"Prior tokens cache: {lmdb_path} (rebuilt={stats.get('rebuilt', False)})")
  ds = PriorTokenSequenceDataset(path=lmdb_path, in_memory=in_memory)

  # Train/val split. Highly-overlapping windows can leak across a naive random
  # split (val windows nearly identical to train), so default to a blocked split.
  val_fraction = float(cfg.prior.training.get('val_fraction', 0.05))
  n_total = len(ds)

  meta = getattr(ds, '_meta', None) or {}
  if meta:
    print(
      f"Dataset meta: num_sequences={n_total} seq_len={ds.seq_len} "
      f"stride={meta.get('stride')} stride_factor={meta.get('stride_factor')}"
    )

  if n_total < 2:
    train_ds, val_ds = ds, ds
  else:
    n_val = max(1, int(n_total * val_fraction))
    if n_val >= n_total:
      n_val = 1

    def _gap_from_meta(seq_len: int, meta: dict, split_gap) -> int:
      stride = meta.get('stride', None)
      if split_gap is None:
        if stride is None:
          return 0
        return int(math.ceil(float(seq_len) / float(stride)))
      return int(split_gap)

    def _blocked_random_split_indices(n_total: int, n_val: int, gap: int, seed: int):
      g = torch.Generator().manual_seed(int(seed))
      perm = torch.randperm(n_total, generator=g).tolist()
      blocked = [False] * n_total
      val_idx = []
      for idx in perm:
        if blocked[idx]:
          continue
        val_idx.append(int(idx))
        lo = max(0, int(idx) - gap)
        hi = min(n_total - 1, int(idx) + gap)
        for j in range(lo, hi + 1):
          blocked[j] = True
        if len(val_idx) >= n_val:
          break
      if len(val_idx) < n_val and gap > 0:
        return _blocked_random_split_indices(n_total, n_val, 0, seed)
      val_set = set(val_idx)
      train_idx = [i for i, is_blocked in enumerate(blocked) if (not is_blocked) and (i not in val_set)]
      return train_idx, val_idx

    split_strategy = str(cfg.prior.training.get('split_strategy', 'blocked_random'))
    split_gap = cfg.prior.training.get('split_gap', None)
    gap = _gap_from_meta(int(ds.seq_len), meta, split_gap)

    if split_strategy == 'contiguous':
      n_train = n_total - n_val - gap
      if n_train < 1:
        gap = 0
        n_train = n_total - n_val
      train_ds = torch.utils.data.Subset(ds, list(range(0, n_train)))
      val_ds = torch.utils.data.Subset(ds, list(range(n_total - n_val, n_total)))
      print(f"Split: contiguous (n_train={n_train}, n_val={n_val}, gap={gap})")
    elif split_strategy == 'blocked_random':
      seed = int(cfg.prior.training.get('split_seed', cfg.get('seed', 42)))
      train_idx, val_idx = _blocked_random_split_indices(n_total, n_val, gap, seed)
      train_ds = torch.utils.data.Subset(ds, train_idx)
      val_ds = torch.utils.data.Subset(ds, val_idx)
      print(f"Split: blocked_random (n_train={len(train_idx)}, n_val={len(val_idx)}, gap={gap}, seed={seed})")
    else:
      n_train = n_total - n_val
      seed = int(cfg.prior.training.get('split_seed', cfg.get('seed', 42)))
      g = torch.Generator().manual_seed(seed)
      train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)
      print(f"Split: random (n_train={n_train}, n_val={n_val}, seed={seed})")

  num_workers = cfg.prior.training.get('num_workers', 4)
  pin_memory = torch.cuda.is_available()
  prefetch_factor = cfg.prior.training.get('prefetch_factor', 2)

  train_dl = DataLoader(
    train_ds,
    batch_size=cfg.prior.training.get('batch_size', 16),
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=num_workers > 0,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
  )
  val_dl = DataLoader(
    val_ds,
    batch_size=cfg.prior.training.get('batch_size', 16),
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=num_workers > 0,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
  )

  # Output/checkpoints
  model_name = cfg.experiment.name
  training_dir = cfg.experiment.training_dir
  run_tag = getattr(cfg.prior.discrete, 'run_tag', None)
  if run_tag is None:
    run_tag = cfg.prior.training.get('run_tag', None)
  if run_tag is None:
    base = f"e{int(cfg.prior.model.embedding_dim)}_h{int(cfg.prior.model.nhead)}_l{int(cfg.prior.model.num_layers)}_len{int(ds.seq_len)}"
    run_tag = f"discrete_{base}_k{int(ds.codebook_size)}_n{int(ds.num_codebooks)}"

  output_dir = os.path.join(training_dir, 'prior_discrete', model_name, str(run_tag))
  os.makedirs(output_dir, exist_ok=True)

  ckpt_acc = ModelCheckpoint(
    dirpath=output_dir,
    filename='best_acc',
    save_last=True,
    save_top_k=1,
    monitor='val_acc',
    mode='max',
    save_on_train_epoch_end=True,
    enable_version_counter=False,
  )
  ckpt_loss = ModelCheckpoint(
    dirpath=output_dir,
    filename='best_loss',
    save_last=False,
    save_top_k=1,
    monitor='val_loss',
    mode='min',
    save_on_train_epoch_end=True,
    enable_version_counter=False,
  )
  callbacks = [ckpt_acc, ckpt_loss]
  # Optional periodic checkpoints (for coherence-vs-training-step analysis).
  _ckpt_every = int(os.environ.get('PRIOR_CKPT_EVERY', '0'))
  if _ckpt_every > 0:
    callbacks.append(ModelCheckpoint(
      dirpath=os.path.join(output_dir, 'steps'),
      filename='step{step}',
      every_n_train_steps=_ckpt_every,
      save_top_k=-1,
      save_on_train_epoch_end=False,
      enable_version_counter=False,
    ))
  logger = TensorBoardLogger(save_dir=output_dir, name='logs')

  model = PriorDiscrete(
    num_codebooks=ds.num_codebooks,
    codebook_size=ds.codebook_size,
    embedding_dim=int(cfg.prior.model.embedding_dim),
    nhead=int(cfg.prior.model.nhead),
    num_layers=int(cfg.prior.model.num_layers),
    dim_feedforward=int(getattr(cfg.prior.model, 'dim_feedforward', 2048)),
    dropout=float(cfg.prior.model.dropout),
    max_len=int(ds.seq_len),
    lr=float(cfg.prior.training.lr),
    num_territories=int(getattr(ds, 'num_territories', 0)),
    cond_dim=int(getattr(ds, 'cond_dim', 0)),
    cfg_dropout=float(getattr(getattr(cfg.prior, 'discrete', {}), 'cfg_dropout', 0.0)),
    cond_dropout=float(getattr(getattr(cfg.prior, 'discrete', {}), 'cond_dropout', 0.0)),
    joint_codebooks=bool(getattr(getattr(cfg.prior, 'discrete', {}), 'joint_codebooks', False)),
    style_dim=int(getattr(getattr(cfg.prior, 'discrete', {}), 'style_dim', 0)),
    style_dropout=float(getattr(getattr(cfg.prior, 'discrete', {}), 'style_dropout', 0.0)),
    context_dropout=float(getattr(getattr(cfg.prior, 'discrete', {}), 'context_dropout', 0.0)),
    style_aux_weight=float(getattr(getattr(cfg.prior, 'discrete', {}), 'style_aux_weight', 0.0)),
    ss_prob=float(getattr(getattr(cfg.prior, 'discrete', {}), 'ss_prob', 0.0)),
    ss_anneal_steps=int(getattr(getattr(cfg.prior, 'discrete', {}), 'ss_anneal_steps', 0)),
    ss_temperature=float(getattr(getattr(cfg.prior, 'discrete', {}), 'ss_temperature', 1.0)),
    ss_iters=int(getattr(getattr(cfg.prior, 'discrete', {}), 'ss_iters', 1)),
    device=device,
  )
  if int(getattr(ds, 'num_territories', 0)) > 0:
    print(f"Territory conditioning enabled: {ds.num_territories} territories")
  if int(getattr(getattr(cfg.prior, 'discrete', {}), 'style_dim', 0)) > 0:
    print(f"Style encoder enabled: style_dim={cfg.prior.discrete.style_dim} "
          f"style_dropout={getattr(cfg.prior.discrete, 'style_dropout', 0.0)}")

  force_restart = cfg.prior.training.get('force_restart', False)
  ckpt_path = None
  if not force_restart:
    ckpt_path = find_checkpoint(output_dir, return_none=True, typ='last')
    if ckpt_path is not None:
      print(f"Resuming from checkpoint: {ckpt_path}")
  else:
    print("Force restart requested; starting from scratch.")

  # Step cap from config (prior.training.max_steps; -1 = uncapped); PRIOR_MAX_STEPS env overrides.
  _max_steps = int(os.environ.get('PRIOR_MAX_STEPS', str(int(cfg.prior.training.get('max_steps', -1)))))
  trainer = L.Trainer(
    max_epochs=cfg.prior.training.get('max_epochs', 10),
    max_steps=_max_steps,
    accelerator='gpu' if device == 'cuda' else 'cpu',
    devices=1,
    log_every_n_steps=10,
    # Clip global grad norm: the step-based cosine LR (unlike the old plateau scheduler) does NOT
    # back off on a loss spike, so a single high-variance batch can blow up a gradient and collapse
    # the model to ~random with no recovery (seen on VCTK speech at step ~19k). Clipping prevents it.
    gradient_clip_val=float(cfg.prior.training.get('gradient_clip', 1.0)),
    callbacks=callbacks,
    logger=logger,
    default_root_dir=output_dir,
    enable_progress_bar=(os.environ.get('PRIOR_NO_PBAR', '0') != '1'),
  )

  trainer.fit(model, train_dl, val_dl, ckpt_path=ckpt_path)

  best_val_acc = ckpt_acc.best_model_score
  best_val_loss = ckpt_loss.best_model_score
  print(f"Best val_acc: {float(best_val_acc):.6f}" if best_val_acc is not None else "Best val_acc: <none>")
  print(f"Best val_loss: {float(best_val_loss):.6f}" if best_val_loss is not None else "Best val_loss: <none>")
  print(f"tokens_per_step: {int(ds.num_codebooks)}")
  print(f"codebook_size: {int(ds.codebook_size)}")
  if hasattr(ds, '_meta') and ds._meta is not None and 'steps_per_second' in ds._meta:
    print(f"steps_per_second: {float(ds._meta['steps_per_second']):.6f}")


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig):
  """Hydra-driven training for Prior.

  Expected cfg fields:
    cfg.prior.model: {embedding_dim, quantization_channels, nhead, num_layers, dropout, lr}
    cfg.prior.dataset: {hdf5_path?, seq_len, stride_factor}
    cfg.prior.training: {batch_size, num_workers, max_epochs}
  """
  L.seed_everything(cfg.get('seed', 42))

  if not bool(cfg.prior.get('enabled', True)):
    print("cfg.prior.enabled is False; skipping prior training.")
    return

  # Dataset (auto-cache controls/latents for prior)
  in_memory = cfg.prior.dataset.get('in_memory', True)

  # Infer control_space and synth configs from experiment config (same as DDSP training)
  control_space = build_control_space(cfg.model.control_space)
  synth_configs = []
  for s in cfg.model.synths:
    synth_configs.append({"class": s.type, "params": dict(s.params)})

  discrete_enabled = bool(getattr(getattr(cfg.prior, 'discrete', {}), 'enabled', False))
  if discrete_enabled:
    _train_discrete(cfg, control_space, synth_configs, in_memory)
    return

  lmdb_path, stats = build_or_load_prior_cache_from_cfg(
    cfg,
    control_space=control_space,
    synth_configs=synth_configs,
    device='cuda' if torch.cuda.is_available() else 'cpu',
  )
  print(f"Prior cache: {lmdb_path} (rebuilt={stats.get('rebuilt', False)})")
  ds = PriorSequenceDataset(path=lmdb_path, in_memory=in_memory)
  num_workers = cfg.prior.training.get('num_workers', 4)
  pin_memory = torch.cuda.is_available()
  prefetch_factor = cfg.prior.training.get('prefetch_factor', 2)
  dl = DataLoader(
    ds,
    batch_size=cfg.prior.training.get('batch_size', 16),
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=num_workers > 0,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
  )

  # Output/checkpoints
  model_name = cfg.experiment.name
  training_dir = cfg.experiment.training_dir
  output_dir = os.path.join(training_dir, 'prior', model_name)
  os.makedirs(output_dir, exist_ok=True)

  checkpoint_cb = ModelCheckpoint(
    dirpath=output_dir,
    filename='best',
    save_last=True,
    save_top_k=1,
    monitor='loss',
    mode='min',
    save_on_train_epoch_end=True,
    enable_version_counter=False,
  )
  logger = TensorBoardLogger(save_dir=output_dir, name='logs')

  # Model
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = Prior(
    num_controls=ds.num_controls,
    embedding_dim=cfg.prior.model.embedding_dim,
    quantization_channels=cfg.prior.model.quantization_channels,
    nhead=cfg.prior.model.nhead,
    num_layers=cfg.prior.model.num_layers,
    dropout=cfg.prior.model.dropout,
    max_len=cfg.prior.model.max_len,
    lr=cfg.prior.training.lr,
    normalization_dict=cfg.prior.dataset.get('normalization', None),
    device=device
  )

  # Trainer
  force_restart = cfg.prior.training.get('force_restart', False)
  ckpt_path = None
  if not force_restart:
    ckpt_path = find_checkpoint(output_dir, return_none=True, typ='last')
    if ckpt_path is not None:
      print(f"Resuming from checkpoint: {ckpt_path}")
  elif force_restart:
    print("Force restart requested; starting from scratch.")

  trainer = L.Trainer(
    max_epochs=cfg.prior.training.get('max_epochs', 10),
    accelerator='gpu' if device == 'cuda' else 'cpu',
    devices=1,
    log_every_n_steps=10,
    callbacks=[checkpoint_cb],
    logger=logger,
    default_root_dir=output_dir,
  )

  trainer.fit(model, dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
  main()


