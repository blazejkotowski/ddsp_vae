# Faithful Post-Net: Architecture

## Overview

The **post-net** is a small neural enhancer applied *after* the DDSP synth. The noise-band synth
renders a "rough" reconstruction from the control trajectory; the post-net sharpens transients and
corrects the spectral envelope to recover detail the parameter-free synth cannot produce on its own.

It is a **transform of the rough audio**, not a generator. For a rough signal `R` with STFT
`|R|·e^{j∠R}`, the output is

$$y = \text{iSTFT}\big(\, g \cdot |R| \;,\; \angle R + \Delta\phi \,\big)$$

where a per-frequency-bin **gain** `g` and **phase nudge** `Δφ` are predicted from the rough's own
log-magnitude spectrogram together with the 4-channel control signal. Both heads reshape the rough's
spectrum rather than adding new signal, which gives the model two structural properties:

- **Zero-rough → silence.** `g·0 = 0`, so muting the synth mutes the output regardless of the
  control or the network's prediction.
- **Bends propagate.** A spectral bend applied to the rough changes `|R|` and `∠R`; the post-net
  reshapes that bent spectrum, so the bend survives to the output.

## Network structure

Two-resolution STFT chain. Each stage analyzes the signal, predicts gain + phase corrections per
bin, and resynthesizes; the second stage refines the first at finer time resolution.

| Stage | `n_fft` / `hop` | Body | Heads |
|---|---|---|---|
| 1 (coarse) | 1024 / 256 | 10 dilated causal conv blocks, 128 ch | gain (513 bins) + phase (513 bins) |
| 2 (fine) | 256 / 64 | 8 dilated causal conv blocks, 96 ch | gain (129 bins) + phase (129 bins) |

Per frame, each stage's conv stack receives `[ log(1+|R|) , control ]` — the rough's log-magnitude
concatenated with the 4-channel control (loudness, centroid, latent 0, latent 1), nearest-upsampled
to the STFT frame rate. Conv bodies are dilated over time (frequency bins are the channel axis).

Both heads are **bounded**:

- gain: `g = exp( tanh(·) · ln_bound )`, `ln_bound` = **±15 dB**.
- phase: `Δφ = tanh(·) · φ_cap`, `φ_cap = π/2`.

A tighter gain bound makes the output lean harder on the rough (stronger bend transmission) at a
small cost in reconstruction accuracy.

## Streaming and latency

The model runs in `nn~` in fixed audio buffers and streams **click-free**:

- **Causal convolutions with cached context** — at each buffer the module keeps enough input history
  to recompute the conv stack over `[context + new]` and emit only the new part, so chunked output
  is identical to one-shot output.
- **STFT overlap-add with carried tails** — each stage carries its input and overlap-add tails
  across buffers (a fixed `n_fft − hop` sample delay per stage).
- **Look-ahead** — the gain/phase from a frame are applied to a frame 3 frames (768 samples) in its
  past via magnitude/phase FIFOs, recovering most of a non-causal model's quality. The dry path is
  delayed to match.

**Total latency** ≈ 768 (stage 1) + 768 (look-ahead) + 192 (stage 2) ≈ **1728 samples (~36 ms)** at
48 kHz. A buffer-size-invariant, EMA-smoothed loudness cap keeps the output level anchored to the
rough.

## Runtime control — `postnet_mix`

The `decode` stage exposes the nn~ attribute **`postnet_mix`** (`0…1`), set live with
`set postnet_mix <value>`:

```
audio = (1 − mix) · rough_synth  +  mix · postnet(rough_synth, control)
```

- `0` → raw synth (post-net bypassed; also skips its compute).
- `1` → fully enhanced (default).
- intermediate → phase-coherent wet/dry blend (the dry path is delay-aligned to the wet path).

Spectral bends (`spectral_roll` / `spectral_stretch` / `limit_components` …) pass *through* the
post-net at `postnet_mix 1`, so bend + post-net combine musically. (The separate `post_net_enabled`
attribute is an unrelated, unused enhancer — leave it at `0`.)

## Training

The post-net is a first-class training stage (`cli.train_postnet`) that reads the same config as
`cli.train` / `cli.train_prior`. It trains after the synth, on the synth's frozen output:

```zsh
python -m cli.train_postnet -cn <name> \
  data.dataset_path=/abs/path ++experiment.name=my_run ++postnet.enabled=true
```

Under the hood:

1. Train the DDSP synth (`cli.train`).
2. `cli.train_postnet` loads the frozen synth and builds a paired `(rough, real, control)` cache of
   dense overlapping windows (`ddsp/postnet/dataset.py :: build_or_load_postnet_cache`) — rough from
   the frozen synth, real the target audio, control = `[features | latents]` at control rate (the same
   layout `decode()` feeds the post-net). Cached next to the dataset and reused across runs.
3. It trains `PostNet` (`ddsp/postnet/postnet.py`), which wraps the exact streaming module
   (`StreamingSpecTransform`), with an MRSTFT + L1 loss, **bend augmentation** (the same bend applied
   to both rough and target, teaching the transform to preserve bends), a gain-slew penalty for smooth
   gains, and EMA (`ddsp/postnet/ema.py`). Training uses the exact streaming computation, so training
   and deployment match. Checkpoints land in `training/postnet/<name>/`. Recipe knobs live in the
   `postnet` block of `configs/template.yaml`.

   **Validation reporting** reuses the synth's own loss machinery (`PostNet.attach_synth_metrics`), so
   the post-net logs MRSTFT in the *same style* as `cli.train`: `val_loss` (the synth's monitored
   metric) and `val/<LossName>` (its per-component metric) are directly comparable to the synth run.
   Note the synth's perceptual MRSTFT is **asymmetric** — its `val_loss` calls `loss_fn(target, pred)`
   while `val/MultiResolutionSTFTLoss` calls `loss_fn(pred, target)` — so the two differ, and an
   improvement can *raise* the reversed `val_loss` while *lowering* the conventional component. The
   post-net therefore **checkpoints on the conventional `val_mrstft`** (pred-vs-target), not the
   reversed `val_loss`.
4. `cli.export --config <name>` **auto-resolves** the post-net from `training/postnet/<name>/` and
   wires the streaming module into `decode()` behind `postnet_mix` (`--no_postnet` to skip;
   `--postnet <ckpt>` to override).

Expected reconstruction at the standard 4-channel / 375 Hz control budget is ~0.62–0.66 MRSTFT,
down from ~0.85 for the raw synth.

A research playground with alternative architectures (queue-driven) remains in `experiments/postnet/`,
but `cli.train_postnet` is the canonical, supported path.

## Files

| File | Role |
|---|---|
| `cli/streaming_postnet.py` | `StreamingSpecTransform` — deployed streaming module (cached-context convs, OLA tails, look-ahead FIFOs, loudness cap) |
| `cli/train_postnet.py` | Hydra training CLI (the canonical path) |
| `ddsp/postnet/postnet.py` | `PostNet` LightningModule (wraps `StreamingSpecTransform`) + loss/bend-aug |
| `ddsp/postnet/dataset.py` | paired `(rough, real, control)` cache builder from the frozen synth |
| `ddsp/postnet/ema.py` | `EMACallback` — EMA of the weights, swapped in for validation/checkpointing |
| `cli/export.py` | auto-resolves + wires the streaming module into `decode()` behind `postnet_mix` |
| `experiments/postnet/` | research playground (alternative archs, queue-driven) — not the supported path |
