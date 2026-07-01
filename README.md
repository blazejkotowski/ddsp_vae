# Performative Latents for Adaptive Unsupervised DDSP (PLAUD)

PLAUD is a modular PyTorch framework for Differentiable Digital Signal Processing (DDSP). It
targets small, personal datasets and emphasizes playability, modularity, and real-time use. The
system is configurable end-to-end — datasets, feature/control spaces, synthesis blocks, losses,
training regimes (including optional adversarial), and a Transformer **prior** for generative
control — and exports to `nn~` for Max/MSP and PureData.

**The flagship instrument** is a realtime generative synth: a DDSP-VAE synthesiser is driven by a
**control trajectory**, that trajectory is compressed to **discrete tokens** by a grouped-VQ codec,
and an **autoregressive Transformer prior** generates those tokens. The prior is steered live by a
learned **style** XY-pad (morphing between the styles in your dataset), with optional slow **LFO**
envelopes, a **territory** XY-pad, and **classifier-free guidance**. It runs in realtime via a KV
cache. → See **[DISCRETE_PRIOR.md](DISCRETE_PRIOR.md)** for the full architecture, control surface,
and tuning guide.

**Core idea**: a single, explicit `ControlSpace` defines the controls (features and/or latents),
their dimensions, and the control rate. The dataset, model, and exports are all built to respect
this schema, keeping components composable and consistent.

## Highlights

- **Explicit ControlSpace**: authoritative source for control dimensions and rate.
- **Modular synth routing**: choose synth blocks and their parameters via config.
- **Multichannel output**: set `audio.n_channels` to synthesize/export N audio channels.
- **Configurable losses**: flexible list (MRSTFT by default) with per-component logging.
- **Optional adversarial regime**: gate discriminator/generator contributions by epoch.
- **Two priors**: a **discrete codec prior** (the style instrument, default) and a simpler
  **continuous mu-law prior** — switched by `prior.discrete.enabled`.
- **Realtime `nn~` export**: TorchScript (or ONNX) models with a streaming, KV-cached prior.

## Install

```zsh
pip install -r requirements.txt
pip install -e .
```

## Concepts

- **`ControlSpace`**: declares control fields (name, dim, source, extractor, normalization) and the
  rate. The DDSP model and dataset derive `feature_dim` / `latent_dim` strictly from this.
- **Registries**: features, synth blocks, losses, and augmentations are registered and instantiated
  by name from configs.
- **Codec (`LatentCompressor`)**: a grouped-VQ autoencoder that compresses the control sequence to
  discrete tokens (`num_codebooks` indices per token) at a chosen token rate.
- **Prior**: a causal Transformer that generates token sequences, conditioned on the style code
  (and optionally LFO / territory), consumed in realtime through a KV cache.

## Quickstart — the 3-stage workflow

All three stages read the **same** config file. Configs live in `configs/`; start from the annotated
**[`configs/template.yaml`](configs/template.yaml)**. Stages 1–2 are [Hydra](https://hydra.cc) apps
(`-cn <name>` selects the config; `++a.b=v` overrides an existing key, `+a.b=v` adds a new one).

### 1) Train the DDSP synth

```zsh
python -m cli.train -cn <name> \
  data.dataset_path=/absolute/path/to/dataset \
  ++experiment.name=my_run
```

Trains the DDSP-VAE synth (analysis encoder + control decoder + synth blocks). Artifacts land under
`training/synth/<name>/`. Optional adversarial fine-tuning is epoch-gated via the `adversarial`
section.

### 2) Train the codec + prior

```zsh
python -m cli.train_prior -cn <name> \
  data.dataset_path=/absolute/path/to/dataset \
  ++experiment.name=my_run
```

When `prior.discrete.enabled=true` (the default in the template), this first trains the **codec**
(`LatentCompressor`) under `training/compressor/<name>/`, then trains the **prior** under
`training/prior_discrete/<name>/`. On first run it builds an **LMDB token cache** next to the dataset
(`prior_cache_<key>.lmdb`) and reuses it afterwards.

Useful env overrides:
- `PRIOR_MAX_STEPS=<n>` — overrides `prior.training.max_steps` (the real stop; ~50k is the sweet spot).
- `PRIOR_CKPT_EVERY=<n>` — checkpoint every N steps.
- `PRIOR_NO_PBAR=1` — disable the progress bar (for logs).

```zsh
PRIOR_MAX_STEPS=50000 python -m cli.train_prior -cn <name> \
  data.dataset_path=/absolute/path/to/dataset ++experiment.name=my_run
```

To force rebuilding the cache, delete the `prior_cache_*.lmdb` directory (or change cache-relevant
settings like `prior.model.max_len`, `prior.dataset.stride_factor`, or the codec strides).

### 3) Export for `nn~`

```zsh
# TorchScript (recommended). --config derives model/prior/compressor dirs + output path from <name>.
python -m cli.export --config <name>

# Choose the checkpoint and target rate:
python -m cli.export --config <name> --type last --target_fs 48000
```

`--config <name>` resolves everything from the config: the synth dir, the prior dir, the compressor
checkpoint, the prior kind (from `prior.discrete.enabled`), and the output path
(`models/<name>.ts`). For a discrete prior the exporter also **auto-builds the style XY-pad map** from
the per-track style centroids. Flags: `--type {best,last}`, `--target_fs`, `--streaming`,
`--prior_checkpoint`, `--compressor_checkpoint`. (The lower-level `--model_directory` /
`--prior_directory` / `--output_path` flags are available for manual exports.)

For an ONNX export, pass an `--output_path` ending in `.onnx`.

## Realtime control surface (in `nn~`)

The exported `prior` method exposes signal-rate controls (each "knob" is a signal). A discrete
style model has **10 inputs → 4 control outputs**:

| Input | Name | Meaning |
|---|---|---|
| 1–4 | `LFO 1..4` | optional slow control-envelope scaffold (loudness, centroid, latent0/1); `0` = off/freeform |
| 5–6 | `Style X / Y` | 2-D style pad (`0…1` each axis) — blends the per-track style centroids |
| 7 | `Temperature` | sampling randomness (~0.5 locked … 1.0 varied; ~0.6 default) |
| 8 | `Style CFG` | how hard to commit to the pad's style (start ~1.5) |

If a model is exported with **territory** conditioning instead of style, channels 5–6 become
`Territory X/Y` and channel 8 is `CFG Strength`; the layout is otherwise identical. Start around
**CFG 1.5 / Temperature 0.6**; see DISCRETE_PRIOR.md §3 for the full tuning guide.

## Continuous vs discrete prior

`prior.discrete.enabled` selects the prior:

- **`true` (default, `configs/template.yaml`)** — the discrete codec + style prior described above.
  Uses the `compressor` section.
- **`false` (`configs/experiment*.yaml`)** — a simpler **continuous mu-law** Transformer prior over
  the control trajectory (no codec). Uses the mu-law `prior.model` keys (`quantization_channels`,
  `embedding_dim`, `nhead`, `num_layers`, `dim_feedforward`, `dropout`, `max_len`).

## Configuration

- Configs are Hydra YAML files in `configs/`. **[`configs/template.yaml`](configs/template.yaml)** is
  the annotated canonical template: everything uncommented is a minimal working discrete-prior recipe,
  and every optional block is commented with an explanation.
- Top-level sections: `experiment`, `audio`, `data`, `model` (ControlSpace + synths + decoder),
  `losses`, `adversarial`, `trainer`, `prior` (+ `prior.discrete`), and `compressor`.
- Override existing fields with `++path.to.field=value`; add new ones with `+path.to.field=value`.

### ControlSpace

`model.control_space` is a list of fields; each has `name`, `dim`, `source` (`feature` or `latent`),
and — for features — an `extractor` (registry name) plus `params`. `feature_dim` and `latent_dim` are
the sums of the respective fields; their total is the control vector the prior generates.

## Multichannel

`audio.n_channels` (default `1`) sets how many audio channels the model synthesizes.

- **Decoder per-channel heads**: one shared latent/feature stream drives `n_channels` independent sets
  of synth parameters. The encoder and feature extractors consume a **mono downmix** of the input.
- **Data**: preprocess with channels preserved (`utils/dataset_converter.py` keeps the source layout by
  default; `--channels N` forces one). At load time, files with fewer channels are zero-padded and
  files with more are cropped to the first `n_channels`.
- **Adversarial**: the discriminator runs per channel and is averaged.
- **Export**: `decode` exposes `n_channels` audio outlets; `encode` accepts `n_channels` inputs.
- **Mono default**: at `n_channels=1` the model is a standard single-channel model. Changing
  `n_channels` rebuilds the dataset cache (it is part of the cache key).

## Testing

```zsh
python -m pytest -q
```

## Troubleshooting

- **“No control windows produced”** when building the prior cache: your dataset chunks are too short
  for `prior.model.max_len`; reduce `max_len`, increase `audio.chunk_duration_s`, or use longer files.
- **Hydra struct-mode errors** when adding keys: prefix new-key overrides with `+`.
- **CPU-only runs**: set the Trainer `accelerator='cpu'` and ensure CUDA defaults are disabled.
- **Groove “comes and goes”**: the codec token rate is too coarse for the beat. Use finer strides
  (e.g. `[4,2,2]` ≈ 43 ms/token) rather than a coarse `[5,5,3]`; see DISCRETE_PRIOR.md §2.

## Max/MSP and PureData

PLAUD exports are compatible with the `nn~` external for Max/MSP and PureData. Install it from the
[nn~ repository](https://github.com/acids-ircam/nn_tilde) and load the exported `.ts` model.
```
