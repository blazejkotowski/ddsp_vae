# Discrete Codec Prior — Architecture, Control, Config & nn~

A realtime generative system: a DDSP‑VAE synthesiser is driven by a **control trajectory** that is
**compressed to discrete tokens** by a grouped‑VQ codec, and an **autoregressive Transformer prior**
generates those tokens. The prior is **conditioned** (slow "LFO" envelopes, selectable territories,
classifier‑free guidance) and runs in realtime via a KV cache, exported to `nn~` for Max/MSP.

---

## 1. Architecture

```
                         ┌──────────────────────── TRAIN-TIME (analysis) ───────────────────────┐
  audio ──▶ DDSP.encoder ──▶ latents z ─┐
        └─▶ feature extract ─▶ loudness │   control [loudness, centroid, z0, z1]   (control rate)
                              centroid ─┘            │  [T, 4] @ 375 Hz
                                                     ▼
                                       LatentCompressor.encode  (grouped VQ, ratio 16)
                                                     │
                                                     ▼
                                          tokens  [S, N=4]   @ 23.4 Hz   (4 codebooks × 256)
                                                     │
                         ┌───────────────────── PriorDiscrete (causal Transformer) ──────────────┐
                         │  per-codebook embed + START + positional enc  ──▶  + conditioning:     │
                         │     • LFO/cond envelope (cond_proj, additive)                          │
                         │     • territory embedding (additive)            ──▶ Transformer ──▶ h   │
                         │  joint depth head: P(codebook i | codebooks <i, h)                      │
                         └───────────────────────────────────────────────────────────────────────┘
                                                     │  next-token logits
                         └──────────────────────── INFER-TIME (synthesis) ──────────────────────┐
  tokens ──▶ LatentCompressor.decode_codes ──▶ control [T,4] ──▶ DDSP.decoder ──▶ synth params
                                                                          │
                                                                          ▼
                                                       BendableNoiseBandSynth ──▶ audio
```

**Stages**

| Component | Role | Key params (morelli/harmsworth) |
|---|---|---|
| `DDSP` (`ddsp/ddsp.py`) | VAE synth: encoder (audio→latents), decoder (control→synth params), `BendableNoiseBandSynth` | `resampling_factor=128` → control rate **375 Hz**; `decoder_temporal_stride=4` (linear upsampling) |
| `LatentCompressor` (`ddsp/latent_compressor.py`) | grouped‑VQ codec over the control sequence; **codes‑only decode** (`use_skip_connections=false`) | `compression_ratio=16` → token rate **23.4 Hz**; `strides=[4,2,2]`, `num_codebooks=4`, `codebook_size=256` |
| `PriorDiscrete` (`ddsp/prior/prior_discrete.py`) | causal Transformer over `[S, N=4]` tokens + conditioning + joint codebook head | `d_model = embedding_dim(64) × num_codebooks(4) = 256`, `nhead=8`, `num_layers=4`, `max_len=512` (~22 s) |
| `KVCachedPrior` (`ddsp/prior/kv_infer.py`) | incremental, KV‑cached equivalent of `PriorDiscrete.forward` (batch=1, scriptable) | weights mapped from the trained prior; parity ≈ 1e‑5 |
| `PriorDiscreteWrapper` (`cli/export.py`) | realtime nn~ wrapper: token generation loop, streaming compressor decode, control I/O | KV cache(s), `decode_lookahead`, smoothing + reseed |

**Control vector** = `[loudness, centroid, latent0, latent1]` (`num_controls = feature_dim + latent_size = 4`).

---

## 2. Conditioning mechanisms

All conditioning is **additive at the Transformer input** (kept separate from the codebook embeddings).

- **LFO / cond envelope** — the control low‑passed (avg‑pool, `cond_smooth_frames=64` ≈ 170 ms) and
  sampled at token rate; projected by `cond_proj` and added at every position. It is a slow,
  per‑channel **scaffold** the prior learns to fill in around. At inference you draw these 4 curves.
  - **`cond_dropout`**: zero the envelope for a fraction of training windows ⇒ the model learns
    **freeform** (`cond=0`) *and* LFO‑driven generation. At inference an **LFO‑amount** `α` scales the
    envelope (`α·LFO`): `1` = full follow, `0` = freeform, in‑between = blend. One model, switchable.
- **Territory** — a per‑window label (`terr_by_track` = source track, or k‑means clusters) → an
  additive `territory_embedding`. Exposed as a **2‑D map**: PCA of the embeddings places each zone at
  an (x,y); the wrapper blends zones by softmax over −distance² (`terr_map_temp`).
- **CFG (classifier‑free guidance)** — `cfg_dropout` trains a **null** territory row (index
  `num_territories`). At inference, a second null pass runs in lockstep and logits are guided
  `uncond + scale·(cond − uncond)` ⇒ a **territory‑contrast** knob (1 = off, >1 amplifies).
- **Joint codebook head (WS4)** — instead of sampling the 4 codebooks independently (off‑manifold),
  a small **depth head** predicts codebook `i` from the time‑context `h` and the already‑decoded
  codebooks `<i` (parallel prefix‑sum in training, N sequential steps at inference). Keeps frames
  on‑manifold.

---

## 3. Realtime control surface (nn~ `prior` method)

Inputs are signal‑rate channels (a "knob" is just a signal). Full model = **10 inputs → 4 control out**:

| # | label | meaning |
|---|---|---|
| 1–4 | `LFO 1..4` | control‑envelope scaffold (loudness, centroid, latent0, latent1). `0` everywhere = freeform |
| 5–6 | `Territory X / Y` | normalised 2‑D map position; blends zones (≈ ±2 range, `(0,0)` = neutral) |
| 7 | `Temperature` | sampling randomness (low = locked, high = varied) |
| 8 | `CFG Strength` | territory contrast (1 = off, ~3 = strong; needs `cfg_dropout`) |
| 9 | `Smoothing` | causal one‑pole LPF on **all** control trajectories (0 = off … ~0.95 = sluggish glide) |
| 10 | `Reseed Trigger` | rising edge >0.5 re‑anchors the prior to a fresh phrase (beat‑sync restart) |

Channels are added only when the model supports them (`cond_dim>0`, `num_territories>0`, CFG‑capable).
Click fix is baked in (decoder linear upsampling), independent of these knobs.

---

## 4. Config (`prior` section)

Everything is config‑driven; nothing about conditioning is hardcoded. Canonical block:

```yaml
prior:
  enabled: true
  discrete:
    enabled: true
    compressor_ckpt: null         # auto-train/locate if null

    # ── LFO / envelope conditioning ──
    cond_envelope: true           # add the slow control-envelope scaffold
    cond_smooth_frames: 64        # envelope low-pass window (control frames)
    cond_augment: false           # light cond augmentation (kept off; it weakens the LFO)
    cond_film: false              # FiLM conditioning instead of additive (kept off)
    cond_dropout: 0.2             # >0 ⇒ switchable LFO/freeform (zero the cond this often)

    # ── territories ──
    num_territories: 6            # 0 = no territory conditioning
    terr_by_track: true           # territory = source track (else k-means)
    terr_rich: true               # rich descriptor for clustering (when not by_track)

    # ── guidance & sampling head ──
    cfg_dropout: 0.15             # >0 ⇒ CFG-capable (trains a null territory)
    joint_codebooks: true         # depth head (on-manifold sampling; kills codebook-independence clicks)

    # ── nn~ wrapper (export-time, optional) ──
    terr_map_temp: 0.5            # 2-D territory-map blend sharpness
    decode_lookahead: 2           # streaming-decode lookahead (tokens of latency)

  model:                          # Transformer
    embedding_dim: 64             # d_model = embedding_dim × num_codebooks
    nhead: 8
    num_layers: 4
    dim_feedforward: 1024
    dropout: 0.0
    max_len: 512                  # ~22 s context (KV cache makes this realtime)

  dataset:
    stride_factor: 0.01
    respect_boundaries: true      # windows never cross track boundaries
    in_memory: true

  training:
    batch_size: 64
    max_epochs: 10000             # cap steps with PRIOR_MAX_STEPS env var
    lr: 5e-4
    val_fraction: 0.0
```

The codec lives under the sibling `compressor:` section (`compression_ratio` derived from `strides`,
`num_codebooks`, `codebook_size`, `hidden_dim`, `compressed_dim`, `use_skip_connections: false`).

---

## 5. Train & export

```bash
# 1. synth (DDSP)            → training/synth/<name>
python cli/train.py        --config-name <name>
# 2. compressor + prior      → training/compressor|prior_discrete/<name>   (compressor auto-trains)
python cli/train_prior.py  --config-name <name>            # PRIOR_MAX_STEPS=8000 to cap
# 3. export to nn~           → models/<name>.ts
python cli/export.py       --config <name> --type best
```

The synth is dataset‑specific (it changes the latent space); the compressor and prior sit on top.
Reusing an existing synth/compressor across prior variants is done with symlinks under `training/`.

### nn~ integration
`cli/export.py` wraps the trained models in `ScriptedDDSP` (an `nn_tilde.Module`) exposing:

- **`encode`** — audio → control (analysis).
- **`decode`** — control → audio (the synth; streaming).
- **`prior`** — the 10‑channel control surface above → control, at the synth's resampling ratio.

In Max, drive `prior` with your control signals and feed its output to `decode` (or patch a `prior →
decode` chain). State (KV cache, token buffer, smoothing, reseed) persists across audio blocks; cold
start is the learned START token.
