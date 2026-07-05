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

The post-net is trained after the synth, on the synth's frozen output:

1. Train the DDSP synth (`cli.train`).
2. Build a paired `(rough, real, control)` cache of dense overlapping windows
   (`experiments/postnet/common.py :: build_cache_cond_dense`) — rough from the frozen synth, real
   the target audio, control the trajectory the synth was driven by.
3. Train the post-net (`experiments/postnet/lab.py`, `arch: streamfx`) with an MRSTFT + L1 loss,
   **bend augmentation** (the same bend applied to both rough and target, teaching the transform to
   preserve bends), a gain-slew penalty for smooth gains, and EMA averaging. The lab trains the
   exact streaming computation, so training and deployment match.
4. Export with the prior (`cli.export --postnet <ckpt>`), which wires the streaming module into
   `decode()` behind `postnet_mix`.

Expected reconstruction at the standard 4-channel / 375 Hz control budget is ~0.62–0.66 MRSTFT,
down from ~0.85 for the raw synth.

## Files

| File | Role |
|---|---|
| `cli/streaming_postnet.py` | `StreamingSpecTransform` — deployed streaming module (cached-context convs, OLA tails, look-ahead FIFOs, loudness cap) |
| `experiments/postnet/lab.py` | `SpecTransform` / `StreamFX` arch + training loop |
| `experiments/postnet/common.py` | paired `(rough, real, control)` cache builders |
| `cli/export.py` | wires the streaming module into `decode()` behind `postnet_mix` |
