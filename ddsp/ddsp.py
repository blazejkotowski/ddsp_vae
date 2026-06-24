import lightning as L
import torch
import torch.nn.functional as F

import numpy as np
import auraloss

from ddsp.blocks import VariationalEncoder, Decoder
from ddsp.discriminator import Discriminator
from ddsp.synths import BaseSynth, SineSynth, SubbandSineSynth, NoiseBandSynth, BendableNoiseBandSynth
from sklearn.decomposition import PCA

from typing import List, Tuple, Dict, Any

from ddsp.losses import MultiScaleSlicedWassersteinLoss
from ddsp.interfaces import ControlSpace


class DDSP(L.LightningModule):
  """
  A neural network that learns how to resynthesise signal, predicting amplitudes of
  precalculated, loopable noise bands.

  Arguments:
    - control_space: ControlSpace (required), explicit control definition
    - synth_configs: List[Dict], differentiable synthesiser configs (BaseSynth.from_config)
    - fs : int, the sampling rate of the input signal
    - resampling_factor: int, internal up/down sampling factor for controls/synths
    - encoder_ratios: List[int], capacity ratios for encoder layers
    - decoder_ratios: List[int], capacity ratios for decoder layers
    - latent_smoothing_kernel: int, smoothing kernel for latents
    - n_melbands: int, mel bands for audio encoder
    - decoder_gru_layers: int, GRU layers in decoder
    - capacity: int, base capacity multiplier
    - learning_rate: float, optimizer LR
    - losses: List[Dict], configurable reconstruction loss entries (e.g., MRSTFT)
    - streaming: bool, streaming mode
    - perceptual_loss_weight: float, weight for perceptual terms
    - plateau_patience: int, LR scheduler patience
    - adversarial_loss: bool, enable adversarial training
    - device: str, device
  """

  # Fixed (non-configurable) scaling on the KLD term. The β schedule modulates
  # this further; recovered from the pre-refactor default (commit bb76ec9).
  _KLD_WEIGHT = 0.0001
  def __init__(self,
               control_space: ControlSpace,
               synth_configs: List[Dict[Any, Any]] = [],
               fs: int = 44100,
               n_channels: int = 1,
               encoder_ratios: List[int] = [8, 4, 2],
               latent_smoothing_kernel: int = 1,
               n_melbands: int = 128,
               decoder_ratios: List[int] = [2, 4, 8],
               decoder_gru_layers: int = 1,
               decoder_temporal_stride: int = 1,
               capacity: int = 64,
               resampling_factor: int = 32,
               learning_rate: float = 1e-3,
               losses: List[Dict[Any, Any]] = [],
               perceptual_loss_weight: float = 0.0,
               streaming: bool = False,
               plateau_patience: int = 20,
               adversarial_loss: bool = True,
               # adversarial schedule and weights
               adv_g_start_epoch: int = 100,
               adv_d_start_epoch: int = 150,
               adv_gen_weight: float = 1.0,
               adv_disc_weight: float = 1.0,
               adv_fm_weight: float = 1.0,
               adv_ndf: int = 4,
               adv_disc_update_every: int = 5,
               ddsp_min_lr: float = 0.0,
               config_name: str | None = None,
               device: str = 'cuda'):
    super().__init__()
    # Turn of autoamtic optimization
    self.automatic_optimization = False

    # Save hyperparameters in the checkpoints
    # Avoid serializing frozen dataclasses like ControlSpace in hparams
    self.save_hyperparameters(ignore=['control_space'])
    self._synth_configs = synth_configs
    self._device = device
    self._plateau_patience = plateau_patience
    self.fs = fs
    self.n_channels = n_channels
    # Control space is the single source of dims
    self.control_space = control_space
    self.latent_size = self.control_space.latent_dim
    self.feature_dim = self.control_space.feature_dim
    # Store dims for convenience in hparams (safe to mutate)
    try:
      self.hparams.feature_dim = int(self.feature_dim)
      self.hparams.latent_size = int(self.latent_size)
      if config_name is not None:
        self.hparams.config_name = str(config_name)
    except Exception:
      pass
    # For PCA utilities; keep num_params aligned to latent_size by default
    self.num_params = self.latent_size
    self.resampling_factor = resampling_factor
    self._latent_smoothing_kernel = latent_smoothing_kernel
    self._adversarial_loss = adversarial_loss
    # Adversarial config
    self._adv_g_start_epoch = int(adv_g_start_epoch)
    self._adv_d_start_epoch = int(adv_d_start_epoch)
    self._adv_gen_weight = float(adv_gen_weight)
    self._adv_disc_weight = float(adv_disc_weight)
    self._adv_fm_weight = float(adv_fm_weight)
    self._adv_disc_update_every = max(1, int(adv_disc_update_every))
    self._ddsp_min_lr = float(ddsp_min_lr)

    # self.synths = synths
    # self.synths = torch.nn.ModuleList([builder() for builder in self._synth_builders])
    print(f"Building synthesizers..., resampling_factor: {resampling_factor}")
    # Build the synthesizers from config
    synths: List[BaseSynth] = []
    for cfg in self._synth_configs:
      cfg['params']['resampling_factor'] = resampling_factor
      cfg['params']['fs'] = fs
      synth = BaseSynth.from_config(cfg).to(self._device)
      synths.append(synth)
    self.synths = torch.nn.ModuleList(synths)

    # Latent space analyzis buffers

    self.register_buffer('_latent_pca', torch.eye(self.latent_size))
    self.register_buffer('_latent_mean', torch.zeros(self.latent_size))
    self.register_buffer('_latent_std', torch.ones(self.latent_size))
    self.register_buffer('_latent_quantiles', torch.zeros(2, self.latent_size))

    self._total_synth_params = sum([s.n_params for s in self.synths])
    # Sanity checks on dims
    assert isinstance(self.feature_dim, int) and self.feature_dim >= 0, "feature_dim must be >= 0"
    assert isinstance(self.latent_size, int) and self.latent_size >= 0, "latent_dim must be >= 0"
    assert self._total_synth_params > 0, "Total synth param size must be > 0"
    # self.register_buffer('_total_synth_params', torch.tensor(total_params, dtype=torch.long))
    # self._total_synth_params = sum([s.n_params for s in self.synths])
    # total_synth_params = 1000

    # self.synths = [NoiseBandSynth(n_filters=1000, fs=fs, resampling_factor=resampling_factor)]

    # ELBO regularization params
    self._beta = 0

    # Define the neural network
    ## Encoder to extract latents from the input audio signal
    # Encoder maps audio -> latents if latent space is requested
    self.encoder = None
    if self.latent_size > 0:
      self.encoder = VariationalEncoder(
        layer_sizes=(np.array(encoder_ratios)*capacity).tolist(),
        sample_rate=fs,
        latent_size=self.latent_size,
        streaming=streaming,
        n_melbands=n_melbands,
        resampling_factor=self.resampling_factor,
      )

    ## Decoder to predict the amplitudes of the noise bands
    self.decoder = Decoder(
      n_params=self._total_synth_params,
      latent_size=max(self.latent_size, 1),
      n_features=max(self.feature_dim, 1),
      n_channels=n_channels,
      layer_sizes=(np.array(decoder_ratios)*capacity).tolist(),
      gru_layers=decoder_gru_layers,
      temporal_stride=decoder_temporal_stride,
      streaming=streaming,
    )

    # (hyperparameters already saved above)

    # Flexible loss list; built from config in CLI via self.hparams.losses
    # Supports built-in 'MRSTFT' and registry-based losses from ddsp.losses
    from ddsp.registry import LOSSES
    self._loss_items = []  # list of tuples (loss_fn, weight)
    losses_cfg = getattr(self.hparams, 'losses', []) or []
    if len(losses_cfg) == 0:
      # Default to MRSTFT only if nothing specified
      self._loss_items.append((self._construct_mrstft_loss(), 1.0))
    else:
      for entry in losses_cfg:
        typ = entry.get('type')
        weight = float(entry.get('weight', 1.0))
        params = dict(entry.get('params', {}))
        if typ == 'MRSTFT':
          # Allow overriding fft_sizes via params
          loss = self._construct_mrstft_loss(**params)
        else:
          # Instantiate from registry
          loss = LOSSES.create(typ, **params)
        self._loss_items.append((loss, weight))

    self._disc_num_D = 3 # number of discriminators at different scales
    self._disc_ndf = int(adv_ndf) # number of filters in the first layer of the discriminator
    self._disc_n_layers = 3 # number of layers in each discriminator
    self._disc_downsample_factor = 4 # downsampling factor between discriminators
    self._disc_feature_weight = 1.0 # weight for the feature matching loss
    self._discriminator = Discriminator(
      self._disc_num_D,
      self._disc_ndf,
      self._disc_n_layers,
      self._disc_downsample_factor
    )

    # Learning rate
    self._learning_rate = learning_rate

    # Validation inputs and outputs
    self._last_validation_in = None
    self._last_validation_out = None
    self._validation_index = 1


  @property
  def resampling_factor(self) -> int:
    return self._resampling_factor

  @resampling_factor.setter
  def resampling_factor(self, value: int) -> None:
    """
    Sets the resampling factor for the model, encoder, and decoder.

    Args:
      value: int, the new resampling factor
    """
    self._resampling_factor = value
    if hasattr(self, 'encoder') and self.encoder is not None:
      self.encoder.resampling_factor = value

  def streaming(self, streaming: bool) -> None:
    """Set streaming mode"""
    self._streaming = streaming
    for synth in self.synths:
      synth.streaming = streaming

    self.decoder.streaming = streaming

  @property
  def n_features(self) -> int:
    """Backward-compat alias for exporter script."""
    return int(self.feature_dim)


  def analyze_latent_space(self, z: torch.Tensor) -> None:
    """
    Saves the PCA decomposition and the mean of the latent space

    Args:
      z: torch.Tensor[n, latent_size]
    """
    # Extract and store the mean
    latent_mean = z.mean(0)
    self._latent_mean.copy_(latent_mean)

    # Extract and store the std
    latent_std = z.std(0)
    self._latent_std.copy_(latent_std)

    # Center the latents
    z -= latent_mean

    # Fit and strore the PCA
    latent_pca = PCA(n_components=self.latent_size)
    latent_pca.fit(z.cpu().numpy())
    self._latent_pca.copy_(torch.from_numpy(latent_pca.components_))

    # Compute 5th and 95th percentile for each component
    quantiles = torch.quantile(z, torch.tensor([0.01, 0.99]), dim=0)
    self._latent_quantiles.copy_(quantiles)


  def normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the given latents using the saved mean and quantils

    Args:
      z: torch.Tensor[batch_size, n, latent_size]
    Returns:
      z: torch.Tensor[batch_size, n, latent_size], normalized latents
    """
    if self._latent_quantiles is None or self._latent_mean is None:
      return z

    # Center the latents
    z_centered = z - self._latent_mean

    # Normalize the latents using the quantiles to range [-1, 1]
    quantiles = self._latent_quantiles
    z_normalized = 2*(z_centered - quantiles[0]) / (quantiles[1] - quantiles[0])-1

    # Ensure the latents are in the range [-1, 1]
    z_normalized = torch.clamp(z_normalized, -1.0, 1.0)

    return z_normalized


  def denormalize_latents(self, z: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes the given latents using the saved mean and quantils

    Args:
      z: torch.Tensor[batch_size, n, latent_size]
    Returns:
      z: torch.Tensor[batch_size, n, latent_size], denormalized latents
    """
    if self._latent_quantiles is None or self._latent_mean is None:
      return z

    # Ensure the latents are in the range [-1, 1]
    z_clamped = torch.clamp(z, -1.0, 1.0)

    # Denormalize the latents using the quantiles
    quantiles = self._latent_quantiles

    # From [-1, 1] to [quantiles[0], quantiles[1]]
    z_denormalized = (z_clamped + 1) / 2 * (quantiles[1] - quantiles[0]) + quantiles[0]

    # Add the mean back
    z_denormalized += self._latent_mean

    return z_denormalized



  def latents_to_params(self, z: torch.Tensor) -> torch.Tensor:
    """
    Converts the given latents to params using the saved PCA and mean

    Args:
      - z: torch.Tensor[batch_size, n, latent_size]
    Returns:
      - params: torch.Tensor[batch_size, n, num_params]
    """
    # Do not do anything if no projection is necessary
    if z.shape[-1] == self.num_params:
      return z

    z_centered = z - self._latent_mean
    pca = self._latent_pca[:self.num_params, :]
    params = torch.matmul(pca, z_centered.permute(0, 2, 1))
    return params.permute(0, 2, 1)


  def params_to_latents(self, params: torch.Tensor) -> torch.Tensor:
    """
    Converts given params to the latent space using the saved PCA and mean.
    It fills the missing information with gaussian noise with the right mean.

    Args:
      - params: torch.Tensor[batch_size, n, num_params]
    Returns:
      - latents: torch.Tensor[batch_size, n, latent_size]
    """
    # Do not do anything if no projection is necessary
    if params.shape[-1] == self.latent_size:
      return params

    # params: [batch_size, n, num_params]
    batch_size, n, num_params = params.shape
    latent_size = self.latent_size

    # Get PCA and mean
    pca = self._latent_pca[:num_params, :]  # [num_params, latent_size]
    latent_mean = self._latent_mean  # [latent_size]

    # Invert PCA: x = pca @ z.T  =>  z = pca_pinv @ x.T
    pca_pinv = torch.linalg.pinv(pca)  # [latent_size, num_params]
    params_t = params.permute(0, 2, 1)  # [batch_size, num_params, n]
    z_centered = torch.matmul(pca_pinv, params_t)  # [batch_size, latent_size, n]
    z_centered = z_centered.permute(0, 2, 1)  # [batch_size, n, latent_size]

    # Fill missing latent dims with noise if needed
    if num_params < latent_size:
      noise = torch.randn(batch_size, n, latent_size - num_params, device=params.device)
      # Scale the noise to have the same std as the latent space
      noise *= self._latent_std[num_params:]  # [batch_size, n, latent_size - num_params]
      z_centered[..., num_params:] = noise

    # Add mean back
    latents = z_centered + latent_mean
    return latents



  def forward(self, audio: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the network.
    Args:
      - audio: torch.Tensor[batch_size, n_signal], the input audio signal
    Returns:
      - signal: torch.Tensor, the synthesized signal
    """
    z = None
    if self.encoder is not None:
      mu, scale = self.encoder(audio)
      z, _ = self.encoder.reparametrize(mu, scale)
      z = self._smooth_latents(z)

    # features already at control rate [B, T_ctl, D_feat]
    # Predict the parameters of the synthesiser and synthesize
    # Features must be control-rate sequences [B, T_ctl, D_feat]
    assert features.dim() == 3, "features must be [B, T_ctl, D_feat]"
    assert features.shape[-1] == self.feature_dim, "features last dim must equal feature_dim from ControlSpace"
    # Prepare dummy features for decoder pathway when feature_dim==0 (JIT-stable interface)
    features_for_decoder = features if self.feature_dim > 0 else torch.zeros(features.shape[0], features.shape[1], 1, device=features.device, dtype=features.dtype)
    if z is None:
      # Provide zero latents when latent space is disabled (keeps decoder interface stable for JIT)
      z = torch.zeros(features.shape[0], features.shape[1], max(self.latent_size, 1), device=features.device, dtype=features.dtype)

    # Align time dimensions: encoder downsampling may differ from dataset's T_ctl by ±1
    T_feat = features_for_decoder.shape[1]
    T_z = z.shape[1]
    if T_feat != T_z:
      T_min = min(T_feat, T_z)
      features_for_decoder = features_for_decoder[:, :T_min, :]
      z = z[:, :T_min, :]

    synth_params = self.decoder(features_for_decoder, z)
    # Decoder outputs [B, n_channels, n_params, T_ctl]; synthesize to audio-rate
    assert synth_params.shape[2] == self._total_synth_params, "decoder param size mismatch"
    signal = self._synthesize(synth_params)
    return signal


  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
    """
    Compute the loss for a batch of data

    Args:
      batch:
        Tuple[
            torch.Tensor[batch_size, n_signal],
            torch.Tensor[params_number, batch_size, n_signal]
          ], audio, control_params
      batch_idx: int, index of the batch (unused)
    Returns:
      loss: torch.Tensor[batch_size, 1], tensor of loss
    """
    x_audio, x_features = batch
    opt_ddsp, opt_disc = self.optimizers()
    sched_ddsp, sched_disc = self.lr_schedulers()

    # Autoencode once
    y_audio, kld_loss = self._autoencode(x_audio, x_features)

    # Match lengths if needed
    if y_audio.shape[-1] != x_audio.shape[-1]:
      min_length = min(x_audio.shape[-1], y_audio.shape[-1])
      x_audio = x_audio[..., :min_length]
      y_audio = y_audio[..., :min_length]

    # Defaults (so we can log even before adversarial starts)
    device = x_audio.device
    adv_d = torch.tensor(0.0, device=device) # discriminator loss
    adv_g = torch.tensor(0.0, device=device) # generator adversarial loss

    # -------------------------
    # 1) Discriminator update
    # -------------------------
    if self._adversarial_loss and self.current_epoch >= self._adv_d_start_epoch:
      self.toggle_optimizer(opt_disc)

      # D(real) and D(fake.detach()) with grads ONLY for D
      pred_real = self._discriminate(x_audio.float())
      pred_fake_det = self._discriminate(y_audio.float().detach())

      self.log("D(real)", sum([out[-1].mean() for out in pred_real]).item(), prog_bar=True, logger=True)
      self.log("D(fake)", sum([out[-1].mean() for out in pred_fake_det]).item(), prog_bar=True, logger=True)

      # Hinge loss for D
      ld = 0.0
      for out in pred_fake_det:
        ld = ld + F.relu(1 + out[-1]).mean() # for a good disceriminator, D(fake) should be negative, so the loss part here should be small
      for out in pred_real:
        ld = ld + F.relu(1 - out[-1]).mean() # for a good disceriminator, D(real) should be positive, so the loss part here should be small
      adv_d = ld * self._adv_disc_weight

      if batch_idx % self._adv_disc_update_every == 0:
        opt_disc.zero_grad(set_to_none=True)
        self.manual_backward(adv_d)
        self.clip_gradients(opt_disc, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_disc.step()
        sched_disc.step(adv_d)
        self.untoggle_optimizer(opt_disc)
      else:
        self.untoggle_optimizer(opt_disc)

    # -------------------------
    # 2) Generator/DDSP update
    #    (freeze D BEFORE using it)
    # -------------------------
    # Reconstruction loss (perceptual/MRSTFT etc.)
    recons_loss = self._reconstruction_loss(y_audio.float(), x_audio.float())
    # Log individual loss components for visibility
    self._log_loss_components(y_audio.float(), x_audio.float(), split='train')
    ddsp_loss = recons_loss + self._KLD_WEIGHT * self._beta * kld_loss  # adv term added below if enabled

    if self._adversarial_loss and self.current_epoch >= self._adv_g_start_epoch:
      self.toggle_optimizer(opt_ddsp)                 # freezes D params for this block

      pred_real = self._discriminate(x_audio.float())
      # D(fake) WITHOUT detach, but D is frozen so grads flow to G only
      pred_fake = self._discriminate(y_audio.float())

      # Generator hinge term: -E[D(fake)]
      for out in pred_fake:
        adv_g = adv_g + (-out[-1].mean())

      # Feature matching
      loss_feat = torch.tensor(0.0, device=device)
      feat_weights = 4.0 / (self._disc_n_layers + 1)
      D_weights = 1.0 / self._disc_num_D
      wt = D_weights * feat_weights

      # Reuse pred_real stats from D step, but detach them
      for i in range(self._disc_num_D):
        for j in range(len(pred_fake[i]) - 1):
          loss_feat = loss_feat + wt * F.l1_loss(pred_fake[i][j], pred_real[i][j].detach())

      # Weighting
      adv_g = self._adv_gen_weight * adv_g + (self._adv_fm_weight * loss_feat)

      ddsp_loss = ddsp_loss + adv_g

      opt_ddsp.zero_grad(set_to_none=True)
      self.manual_backward(ddsp_loss)
      self.clip_gradients(opt_ddsp, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
      opt_ddsp.step()
      sched_ddsp.step(ddsp_loss)
      self.untoggle_optimizer(opt_ddsp)
    else:
      # No adversarial yet: do a plain DDSP step
      self.toggle_optimizer(opt_ddsp)
      opt_ddsp.zero_grad(set_to_none=True)
      self.manual_backward(ddsp_loss)
      opt_ddsp.step()
      sched_ddsp.step(ddsp_loss)
      self.untoggle_optimizer(opt_ddsp)

    # Logging
    self.log('lr_adv_d', self.trainer.optimizers[1].param_groups[0]['lr'], prog_bar=True)
    self.log("adv_d", adv_d, prog_bar=True, logger=True)
    self.log('lr_ddsp', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
    self.log("adv_g", adv_g, prog_bar=True, logger=True)
    self.log("kld", kld_loss, prog_bar=True, logger=True)
    self.log("beta", self._beta, prog_bar=True, logger=True)
    self.log("recons_loss", recons_loss, prog_bar=True, logger=True)
    self.log("ddsp_loss", ddsp_loss, prog_bar=True, logger=True)
    # Also expose a canonical 'loss' for dashboards/alerting
    self.log("loss", ddsp_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)


  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
    """Compute the loss for validation data"""
    x_audio, x_features = batch

    with torch.no_grad():
      y_audio, _ = self._autoencode(x_audio, x_features)

    val_loss = self._reconstruction_loss(y_audio.float(), x_audio.float())
    # Log validation loss components (epoch-level)
    self._log_loss_components(y_audio.float(), x_audio.float(), split='val')

    self.log("val_loss", val_loss, prog_bar=True, logger=True)

    # if self._last_validation_in is None:
    #   self._last_validation_in = x_audio
    #   self._last_validation_out = y_audio.squeeze(1)

    return y_audio


  def configure_optimizers(self):
    """Configure the optimizer for the model"""
    # return torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    disc_param_ids = {id(p) for p in self._discriminator.parameters()}
    ddsp_params = [p for p in self.parameters() if id(p) not in disc_param_ids]
    disc_params = list(self._discriminator.parameters())

    opt_ddsp = torch.optim.Adam(ddsp_params, lr=self._learning_rate)
    # min_lr floors the generator LR so an adversarial phase that starts AFTER recons convergence
    # still has a usable LR to fine-tune with (without a floor it decays to ~1e-8 and G can't move).
    sched_ddsp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ddsp, mode='min', factor=0.1, patience=self._plateau_patience, threshold=1e-3, min_lr=self._ddsp_min_lr)

    opt_disc = torch.optim.Adam(disc_params, lr=self._learning_rate)
    sched_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_disc, mode='min', factor=0.1, patience=int(self._plateau_patience*10), threshold=1e-3)

    optimizers = [opt_ddsp, opt_disc]
    schedulers = [
      {'scheduler': sched_ddsp, 'monitor': 'recons_loss', 'interval': 'epoch'},
      {'scheduler': sched_disc, 'monitor': 'adv_d', 'interval': 'epoch'}
    ]

    return optimizers, schedulers


  def _smooth_latents(self, z: torch.Tensor, mode: str = 'causal') -> torch.Tensor:
    """
    smooth the latent envelopes using a low-pass filter

    Args:
      - z: torch.Tensor[batch_size, n_latent, latent_size], the latent envelopes
    Returns:
      - z: torch.Tensor[batch_size, n_latent, latent_size], the smoothed latent envelopes
    """
    if self._latent_smoothing_kernel == 1:
      return z

    smoothing_kernel = self._latent_smoothing_kernel
    # Ensure the latent smoothing kernel is odd
    if smoothing_kernel % 2 == 0:
      smoothing_kernel += 1

    if mode=='causal':
      B, T, C = z.shape
      x = z.transpose(1, 2)  # [B, C, T] for conv/pool
      w = torch.ones(C, 1, smoothing_kernel, device=z.device, dtype=z.dtype) / smoothing_kernel
      y = F.conv1d(F.pad(x, (smoothing_kernel-1, 0)), w, groups=C)  # [B, C, T]
      return y.transpose(1, 2)

    else:
      # smooth the latent envelopes, individually
      # Apply a simple moving average (low-pass filter) along the time axis for each latent dimension
      padding = smoothing_kernel // 2
      z_smooth = torch.nn.functional.avg_pool1d(
        z.transpose(1, 2), kernel_size=smoothing_kernel, stride=1, padding=padding
      ).transpose(1, 2)
      return z_smooth


  def _autoencode(self, x_audio: torch.Tensor, x_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Autoencode the audio signal
    Args:
      - x_audio: torch.Tensor[batch_size, n_signal], the input audio signal
    Returns:
      - y_audio: torch.Tensor[batch_size, n_signal], the autoencoded audio signal
      - kld: torch.Tensor[], the KL divergence of the latent distribution
    """
    # Encode the audio signal if latent space is used
    kld = torch.tensor(0.0, device=x_audio.device)
    if self.encoder is not None:
      mu, scale = self.encoder(x_audio)
      z, kld = self.encoder.reparametrize(mu, scale)
      z = self._smooth_latents(z)
    else:
      # supply zeros latents for decoder interface (keeps JIT-friendly shapes)
      z = torch.zeros(x_features.shape[0], x_features.shape[1], max(self.latent_size, 1), device=x_features.device, dtype=x_features.dtype)

    # Features are already at control rate; verify shape
    assert x_features.dim() == 3 and x_features.shape[-1] == self.feature_dim, "x_features must be [B, T_ctl, feature_dim]"

    # Align time dimensions: the encoder's scale_factor downsampling may
    # produce a slightly different T than the dataset's ceil-based T_ctl.
    T_feat = x_features.shape[1]
    T_z = z.shape[1]
    if T_feat != T_z:
      T_min = min(T_feat, T_z)
      x_features = x_features[:, :T_min, :]
      z = z[:, :T_min, :]

    # When feature_dim==0, create a dummy single-channel zero feature for decoder path
    x_features_for_decoder = x_features if self.feature_dim > 0 else torch.zeros(x_features.shape[0], x_features.shape[1], 1, device=x_features.device, dtype=x_features.dtype)
    synth_params = self.decoder(x_features_for_decoder, z)

    # Synthesize the output signal
    y_audio = self._synthesize(synth_params)

    return y_audio, kld


  # def on_validation_epoch_end(self):
  #   """At the end of the validation epoch, log the validation audio"""
  #   if self._last_validation_out is not None:
  #     device = self._last_validation_out.device
  #   else:
  #     device = self.device

  #   # audio = torch.FloatTensor(0).to(device) # Concatenated audio
  #   # silence = torch.zeros(1, int(self.fs/2)).to(device) # 0.5s silence
  #   # for input, output in zip(self._last_validation_in, self._last_validation_out):
  #   #   audio = torch.cat((audio, input.unsqueeze(0), silence, output.unsqueeze(0), silence.repeat(1, 3)), dim=-1)

  #   # audio = audio.clip_(-1, 1) # Clip the audio to stay in range
  #   # self.logger.experiment.add_audio("audio_validation", audio, self._validation_index, self.fs)

  #   self._last_validation_in = None
  #   self._last_validation_out = None
  #   self._validation_index += 1


  def _discriminate(self, audio: torch.Tensor):
    """
    Run the (mono) discriminator on each channel independently by folding the channel axis
    into the batch. Downstream hinge/feature-matching reductions then average over batch*channels,
    i.e. a per-channel discriminator. Accepts [B, N, T] or legacy [B, T]/[B, 1, T].
    """
    if audio.dim() == 2:
      audio = audio.unsqueeze(1)
    B, N, T = audio.shape
    return self._discriminator(audio.reshape(B * N, 1, T))


  def _synthesize(self, params: torch.Tensor, waveshaping_factor: float=0.0, limit_components: float=0.0) -> torch.Tensor:
    """
    Synthesizes a signal from the predicted amplitudes and the baked noise bands.
    Args:
      - params: torch.Tensor[batch_size, params, sig_length], the predicted synth parameters
      - waveshaping: float, the amount of waveshaping to apply to the signal
      - sines_number_attenuation: float, attenuation factor for the number of sines
      - noise_amp_attenuation: float, attenuation factor for the noise amplitude
      - sines_amp_attenuation: float, attenuation factor for the sines amplitude
    Returns:
      - signal: torch.Tensor[batch_size, sig_length], the synthesized signal
    """
    # Fold the channel axis into the batch so each synth processes [B*N, n_params, T]
    # transparently. Legacy/JIT callers may pass [B, n_params, T]; treat it as a single channel.
    if params.dim() == 3:
      params = params.unsqueeze(1)
    B, N = params.shape[0], params.shape[1]
    params = params.reshape(B * N, params.shape[2], params.shape[3])

    params_idx = 0
    audio = []

    for synth in self.synths:
      synth_params = params[:, params_idx:params_idx+synth.n_params, :]
      params_idx += synth.n_params
      # Explicit per-synth routing without introspection
      name = synth.jit_name

      if name in ("NoiseBandSynth", "SubbandSineSynth", "BendableNoiseBandSynth", "ComplexSineSynth"):
        audio.append(synth(synth_params, limit_components=limit_components, waveshaping_factor=waveshaping_factor))
      elif name in ("SineSynth",):
        audio.append(synth(synth_params, limit_components=limit_components))
      else:
        # Default call with parameters only
        audio.append(synth(synth_params))

    # Sum across synths -> [B*N, 1, T], then restore the channel axis -> [B, N, T]
    mixed = torch.sum(torch.hstack(audio), dim=1, keepdim=True)
    return mixed.reshape(B, N, mixed.shape[-1])


  def _reconstruction_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes reconstruction as weighted sum of configured losses."""
    # Ensure [B,1,T] and aligned lengths
    if y.dim() == 2:
      y = y.unsqueeze(1)
    if x.dim() == 2:
      x = x.unsqueeze(1)
    if x.shape[-1] != y.shape[-1]:
      min_length = min(x.shape[-1], y.shape[-1])
      x = x[..., :min_length]
      y = y[..., :min_length]
    # Fold channels into the batch so each loss sees mono [B*N, 1, T] (per-channel averaged).
    x = x.reshape(x.shape[0] * x.shape[1], 1, x.shape[-1])
    y = y.reshape(y.shape[0] * y.shape[1], 1, y.shape[-1])

    total = 0.0
    # Each loss function may expect shapes [B,1,T] or [B,T]; try both
    for loss_fn, w in self._loss_items:
      try:
        val = loss_fn(y, x)
      except Exception:
        val = loss_fn(y.squeeze(1), x.squeeze(1))
      total = total + (w * val)
    return total

  def _loss_component_values(self, y: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute individual reconstruction loss component values without weighting.
    Returns a dict mapping component name -> value.
    """
    # Ensure [B,1,T] and aligned lengths
    if y.dim() == 2:
      y = y.unsqueeze(1)
    if x.dim() == 2:
      x = x.unsqueeze(1)
    if x.shape[-1] != y.shape[-1]:
      min_length = min(x.shape[-1], y.shape[-1])
      x = x[..., :min_length]
      y = y[..., :min_length]
    # Fold channels into the batch so each loss sees mono [B*N, 1, T] (per-channel averaged).
    x = x.reshape(x.shape[0] * x.shape[1], 1, x.shape[-1])
    y = y.reshape(y.shape[0] * y.shape[1], 1, y.shape[-1])

    comps: Dict[str, torch.Tensor] = {}
    for loss_fn, _w in self._loss_items:
      name = loss_fn.__class__.__name__
      val = loss_fn(y, x)
      comps[name] = val
    return comps

  def _log_loss_components(self, y: torch.Tensor, x: torch.Tensor, split: str = 'train') -> None:
    """Log individual reconstruction loss components for better visibility.
    split: 'train' logs on_step+on_epoch; 'val' logs on_epoch only.
    """
    comps = self._loss_component_values(y, x)
    for name, val in comps.items():
      if split == 'train':
        self.log(f"{split}/{name}", val, prog_bar=False, logger=True, on_step=True, on_epoch=True)
      else:
        self.log(f"{split}/{name}", val, prog_bar=False, logger=True, on_step=False, on_epoch=True)


  @torch.jit.ignore
  def _construct_mel_loss(self):
    """Construct the loss function for the model: a multi-resolution STFT loss"""
    fft_sizes = np.array([32768, 16384, 8192, 4096, 2048])
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[32768, 16384, 8192, 4096, 2048],
                                                hop_sizes=fft_sizes//4,
                                                win_lengths=fft_sizes,
                                                scale='mel',
                                                n_bins=256,
                                                sample_rate=self.fs,
                                                perceptual_weighting=True).to(self._device)

  @torch.jit.ignore
  def _construct_chroma_loss(self):
    fft_sizes = np.array([16384, 8192, 4096])
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[16384, 8192, 4096],
                                                hop_sizes=fft_sizes//4,
                                                win_lengths=fft_sizes,
                                                scale='chroma',
                                                n_bins=128,
                                                sample_rate=self.fs,
                                                perceptual_weighting=True).to(self._device)

  @torch.jit.ignore
  def _construct_mrstft_loss(self, fft_sizes: np.ndarray | None = None):
    """Construct the loss function for the model: a multi-resolution STFT loss"""
    fft_sizes = np.array(fft_sizes) if fft_sizes is not None else np.array([2053, 1021, 509, 257, 129, 65, 33])
    # Modification from log to log1p according to
    # Schwär, S., & Müller, M. (2023). Multi-Scale Spectral Loss Revisited.
    # IEEE Signal Processing Letters, 30, 1712–1716. https://doi.org/10.1109/LSP.2023.3333205
    return auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[2053, 1021, 509, 257, 129, 65, 33],
                                                hop_sizes=fft_sizes//4,
                                                win_lengths=fft_sizes,
                                                # window='flattop_window',
                                                # mag_distance='L2',
                                                perceptual_weighting=True,
                                                sample_rate=self.fs,
                                                ).to(self._device)

