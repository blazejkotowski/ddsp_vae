"""Render the style/territory XY pad of a discrete-prior nn~ model as a 2-D terrain.

The exported instrument steers generation with a 2-D pad: per-style centroids sit at
`PriorDiscreteWrapper.style_xy` in [0,1]^2 and any pad point blends them by
`softmax(-dist^2 / _style_temp)` (see `cli/export.py:_style_vec`). This module turns that pad into an
informative terrain: for each style we GENERATE a short segment at a FIXED CFG/temperature/seed, RENDER
it to audio, and measure per-style AUDIO features (temporal: rhythmicity / onset density / percussiveness;
timbre: loudness / brightness / noisiness). Those per-style scalars are blended across the pad with the
exact same softmax the live pad uses, giving one disentangled [0,1] grid per feature plus an `elevation`
grid (blend dominance: 1 at each style peak, dipping into the crevices between styles).

Output is a JSON file (disentangled grids, style positions, generation metadata) that a Max JS object
renders, plus a PNG preview (one panel per feature + a composite). Everything is computed from the same
wrapper buffers the shipped `.ts` uses, so the terrain matches the deployed pad exactly.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

# Temporal features first (the prior chiefly models temporality), then timbre for contrast.
FEATURE_ORDER: List[str] = [
    "rhythmicity", "onset_density", "percussiveness",  # temporal (onset/amplitude envelope)
    "loudness", "brightness", "noisiness",             # timbre (spectral)
]
DEFAULT_RGB = {"r": "percussiveness", "g": "rhythmicity", "b": "brightness"}
_HOP = 512  # librosa onset_strength default hop_length
# All features are measured at this rate (the rendered audio is resampled to it). Keeping it fixed and
# modest makes the optional audio cache small and makes cache-recompute identical to a live run.
FEATURE_SR = 16000


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _audio_features(y: np.ndarray, sr: int, seconds: float) -> Dict[str, float]:
    """Per-style scalar audio features from a mono waveform `y` at rate `sr`."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    feats: Dict[str, float] = {}
    # Timbre (spectral).
    feats["loudness"] = float(np.sqrt(np.mean(y ** 2)))
    feats["brightness"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats["noisiness"] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    # Temporal (onset / amplitude envelope).
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
    # Percussiveness = fraction of energy that is PERCUSSIVE (broadband transients) vs harmonic (tonal /
    # sustained), via harmonic-percussive source separation. Independent of loudness and of how many
    # onsets there are -> "plucky/drummy" (~1) vs "pad/drone" (~0). (mean onset-strength was just tracking
    # loudness.)
    y_h, y_p = librosa.effects.hpss(y)
    e_p = float(np.sqrt(np.mean(y_p ** 2))); e_h = float(np.sqrt(np.mean(y_h ** 2)))
    feats["percussiveness"] = e_p / (e_p + e_h + 1e-9)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=_HOP)
    feats["onset_density"] = float(len(onsets) / max(seconds, 1e-6))  # events / second
    # Rhythmicity = presence of a clear PERIODIC pulse. Autocorrelate the MEAN-SUBTRACTED onset envelope
    # -- subtracting the DC is essential: the raw onset envelope is all-positive, so a sustained/noisy
    # texture autocorrelates to ~1 at every small lag and would score as highly "rhythmic" (a drone would
    # read as metronomic). After removing the mean, take the PROMINENCE of the strongest LOCAL peak in the
    # tempo band (40..240 BPM): a metronomic beat has a sharp prominent peak at its period; a drone/texture
    # has a monotonic decay with no local peak -> ~0.
    from scipy.signal import find_peaks
    oc = oenv - oenv.mean()
    ac = librosa.autocorrelate(oc)
    ac = ac / (ac[0] + 1e-9)
    fps = sr / float(_HOP)  # onset-envelope frames per second
    lo = max(1, int(round(60.0 * fps / 240.0)))
    hi = min(ac.size - 1, int(round(60.0 * fps / 40.0)))
    rhythmicity = 0.0
    if hi > lo + 2:
        band = ac[lo:hi + 1]
        peaks, props = find_peaks(band, prominence=0.0)
        if len(peaks):
            rhythmicity = float(np.clip(props["prominences"].max(), 0.0, 1.0))
    feats["rhythmicity"] = rhythmicity
    return feats


def _render_audio(tokens: torch.Tensor, compressor, ddsp, feature_dim: int, latent_size: int) -> np.ndarray:
    """tokens [1, T, N] -> mono waveform (numpy), mirroring generate_prior_discrete_audio's render path."""
    with torch.no_grad():
        controls = compressor.decode_codes(tokens)[:, :, : feature_dim + latent_size]
        features = controls[:, :, :feature_dim]
        latents = controls[:, :, feature_dim:feature_dim + latent_size]
        if latent_size == 0:  # decoder still expects a latent stream; feed zeros (matches export.decode)
            latents = torch.zeros(controls.size(0), controls.size(1), 1, device=controls.device)
        synth_params = ddsp.decoder(features, latents)
        audio = ddsp._synthesize(synth_params)  # [1, C, T] or [1, T]
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    y = audio[0].mean(0).detach().cpu().float().numpy()  # mono downmix
    mx = float(np.abs(y).max())
    return y / mx if mx > 1e-8 else y


def _render_feature_audio(tokens, compressor, ddsp, feature_dim, latent_size, seconds):
    """Render tokens -> mono audio resampled to FEATURE_SR and length-normalised (for features + cache)."""
    import librosa
    y = _render_audio(tokens, compressor, ddsp, feature_dim, latent_size)
    fs = int(ddsp.fs)
    if fs != FEATURE_SR:
        y = librosa.resample(y, orig_sr=fs, target_sr=FEATURE_SR)
    n = int(round(float(seconds) * FEATURE_SR))
    if y.shape[0] >= n:
        y = y[:n]
    else:
        y = np.pad(y, (0, n - y.shape[0]))
    return y.astype(np.float32)


def pad_feature_table(prior, compressor, ddsp, node_table: torch.Tensor, feature_dim: int, latent_size: int,
                      *, use_style: bool, cfg: float, temperature: float, seed: int, seconds: float,
                      device: str) -> Dict[str, np.ndarray]:
    """Per-style audio features [T] for every named feature.

    For each style/territory `t`: generate `seconds` of tokens at a FIXED cfg/temperature/seed (LFO off,
    freeform), render to audio, measure `_audio_features`. `node_table` is the style-centroid table
    (`[T, style_dim]`) for style models; for territory models it only supplies `T` (generation is driven
    by the territory index).
    """
    from cli.generate_prior_discrete_audio import _sample_tokens

    fs = int(ddsp.fs)
    control_rate = fs / float(ddsp.resampling_factor)
    comp_ratio = int(getattr(compressor, "compression_ratio", 32))
    n_tokens = max(8, int(round(float(seconds) * control_rate / comp_ratio)))
    n_styles = int(node_table.shape[0])

    per_style: List[Dict[str, float]] = []
    for t in range(n_styles):
        torch.manual_seed(int(seed) + t)  # deterministic per style, independent of order
        if use_style:
            style_vec = node_table[t:t + 1].to(device)  # [1, style_dim]
            tokens = _sample_tokens(prior, n_tokens, 0, float(temperature), "multinomial", device,
                                    style_vec=style_vec, style_cfg=float(cfg))
        else:
            tokens = _sample_tokens(prior, n_tokens, 0, float(temperature), "multinomial", device,
                                    territory=t, cfg_scale=float(cfg))
        y = _render_feature_audio(tokens, compressor, ddsp, feature_dim, latent_size, seconds)
        per_style.append(_audio_features(y, FEATURE_SR, float(seconds)))
        print(f"  style {t + 1}/{n_styles}: " + ", ".join(f"{k}={per_style[-1][k]:.3f}" for k in FEATURE_ORDER))

    return {name: np.array([ps[name] for ps in per_style], dtype=np.float64) for name in FEATURE_ORDER}


def _elevation_grid(pad_xy, temp: float, resolution: int) -> np.ndarray:
    """Elevation = proximity to the NEAREST style: exp(-min_dist^2 / temp), min-max normalised.

    Exactly 1 at every style (peaks), dipping into crevices along the equidistant ridges between styles
    and decaying to 0 in the empty regions far from any style. (This peaks AT the styles; a softmax
    "dominance" would instead plateau over whichever style is nearest, cresting in empty outskirts.)"""
    xy = np.asarray(pad_xy, dtype=np.float64).reshape(-1, 2)
    axis = np.linspace(0.0, 1.0, int(resolution))
    gx, gy = np.meshgrid(axis, axis)                          # row index -> y, col index -> x
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)        # [G, 2]
    d2 = ((grid[:, None, :] - xy[None, :, :]) ** 2).sum(-1)   # [G, T]
    prox = np.exp(-d2.min(1) / max(float(temp), 1e-6))
    pmin, pmax = float(prox.min()), float(prox.max())
    elev = (prox - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(prox)
    return elev.reshape(resolution, resolution)


def _upsample_norm(coarse: np.ndarray, resolution: int) -> np.ndarray:
    """Bicubic-upsample a coarse [N,N] feature grid to [R,R] and min-max normalise to [0,1]."""
    import torch.nn.functional as F
    t = torch.from_numpy(coarse.astype(np.float32)).view(1, 1, *coarse.shape)
    up = F.interpolate(t, size=(int(resolution), int(resolution)), mode="bicubic",
                       align_corners=True).view(resolution, resolution).numpy()
    mn, mx = float(up.min()), float(up.max())
    return np.clip((up - mn) / (mx - mn) if mx > mn else np.zeros_like(up), 0.0, 1.0)


def compute_pad_terrain(pad_xy, node_features: Dict[str, np.ndarray], temp: float, resolution: int) -> dict:
    """Blend per-style features across the [0,1]^2 pad -> one disentangled [0,1] grid per feature + elevation.

    Uses the SAME softmax(-dist^2/temp) blend as the live pad, so the terrain matches the deployed model.
    """
    xy = np.asarray(pad_xy, dtype=np.float64).reshape(-1, 2)  # [T, 2]
    axis = np.linspace(0.0, 1.0, int(resolution))
    gx, gy = np.meshgrid(axis, axis)                          # row index -> y (row0 == y0)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)         # [G, 2]
    d2 = ((grid[:, None, :] - xy[None, :, :]) ** 2).sum(-1)   # [G, T]
    w = _softmax(-d2 / max(float(temp), 1e-6), axis=1)        # [G, T]

    features_grid: Dict[str, np.ndarray] = {}
    for name, vals in node_features.items():
        v = np.asarray(vals, dtype=np.float64)
        vmin, vmax = float(v.min()), float(v.max())
        vn = (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)  # normalise across styles
        features_grid[name] = np.clip((w @ vn).reshape(resolution, resolution), 0.0, 1.0)

    return {
        "resolution": int(resolution),
        "features": features_grid,
        "elevation": _elevation_grid(xy, temp, resolution),
        "pad_xy": xy,
    }


def grid_feature_sample(prior, compressor, ddsp, wrapper, feature_dim: int, latent_size: int,
                        *, sample_grid: int, use_style: bool, cfg: float, temperature: float, seed: int,
                        seconds: float, device: str, avg_seeds: int = 1,
                        cache_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Sample audio features on an N x N grid of ACTUAL interpolated pad points (N = sample_grid).

    At each coarse cell we form the exact style/territory vector the live pad would use there
    (`wrapper._style_vec`/`_territory_vec`), generate at the fixed cfg/temperature, render to audio, and
    measure features -- so the between-style terrain reflects real generation (incl. nonlinear crevices),
    not a linear blend of the endpoints. With `avg_seeds > 1` each cell is generated that many times (with
    different seeds) and the features are averaged, denoising the map (the style's central tendency rather
    than one noisy rollout). Returns raw coarse grids `{name: [N, N]}`.
    """
    from cli.generate_prior_discrete_audio import _sample_tokens

    fs = int(ddsp.fs)
    control_rate = fs / float(ddsp.resampling_factor)
    comp_ratio = int(getattr(compressor, "compression_ratio", 32))
    n_tokens = max(8, int(round(float(seconds) * control_rate / comp_ratio)))
    N = int(sample_grid)
    S = max(1, int(avg_seeds))
    axis = np.linspace(0.0, 1.0, N)
    coarse = {name: np.zeros((N, N), dtype=np.float64) for name in FEATURE_ORDER}
    total = N * N
    # Optional audio cache: [N, N, S, T] float16 at FEATURE_SR, so features can be re-derived later
    # (tweak a definition) without re-generating. Written next to `cache_path` (a .npy) + a .json sidecar.
    T = int(round(float(seconds) * FEATURE_SR))
    audio_cache = np.zeros((N, N, S, T), dtype=np.float16) if cache_path else None

    def _gen_audio(xy_pad):
        if use_style:
            svec = wrapper._style_vec(xy_pad).to(device)  # [1, style_dim] exact pad blend at (x, y)
            tokens = _sample_tokens(prior, n_tokens, 0, float(temperature), "multinomial", device,
                                    style_vec=svec, style_cfg=float(cfg))
        else:
            tvec = wrapper._territory_vec(xy_pad).to(device).view(1, 1, -1).expand(1, n_tokens, -1).contiguous()
            tokens = _sample_tokens(prior, n_tokens, 0, float(temperature), "multinomial", device,
                                    territory_vec_env=tvec)
        return _render_feature_audio(tokens, compressor, ddsp, feature_dim, latent_size, seconds)

    k = 0
    for i in range(N):        # row -> y
        for j in range(N):    # col -> x
            xy = torch.tensor([axis[j], axis[i]], dtype=torch.float32, device=device)
            acc = {name: 0.0 for name in FEATURE_ORDER}
            for s in range(S):
                torch.manual_seed(int(seed) + k * S + s)  # distinct seed per (cell, repeat)
                y = _gen_audio(xy)
                if audio_cache is not None:
                    audio_cache[i, j, s] = y.astype(np.float16)
                f = _audio_features(y, FEATURE_SR, float(seconds))
                for name in FEATURE_ORDER:
                    acc[name] += f[name]
            for name in FEATURE_ORDER:
                coarse[name][i, j] = acc[name] / S
            k += 1
            if k % max(1, total // 50) == 0 or k == total:
                print(f"  grid cell {k}/{total} ({S} seed(s) each)  {time.strftime('%H:%M:%S')}", flush=True)

    if cache_path is not None:
        np.save(cache_path, audio_cache)  # [N, N, S, T] float16 @ FEATURE_SR (recompute reads this)
        print(f"wrote audio cache: {cache_path}  ({audio_cache.nbytes/1e9:.2f} GB, {N}x{N}x{S} clips @ {FEATURE_SR} Hz)")
    return coarse


def features_from_cache(audio: np.ndarray) -> Dict[str, np.ndarray]:
    """Recompute per-cell features from a cached audio array [N, N, S, T] (@ FEATURE_SR), averaging over
    the S seeds. Lets a feature-definition change re-render the terrain WITHOUT any generation."""
    N, _, S, T = audio.shape
    seconds = T / float(FEATURE_SR)
    coarse = {name: np.zeros((N, N), dtype=np.float64) for name in FEATURE_ORDER}
    for i in range(N):
        for j in range(N):
            acc = {name: 0.0 for name in FEATURE_ORDER}
            for s in range(S):
                f = _audio_features(audio[i, j, s].astype(np.float32), FEATURE_SR, seconds)
                for name in FEATURE_ORDER:
                    acc[name] += f[name]
            for name in FEATURE_ORDER:
                coarse[name][i, j] = acc[name] / S
    return coarse


def _round_grid(a: np.ndarray, nd: int = 4) -> list:
    return np.round(a.astype(np.float64), nd).tolist()


def write_terrain_json(terrain: dict, path: str, *, axis_labels=("Style X", "Style Y"),
                       feature_names: Optional[List[str]] = None, generation: Optional[dict] = None) -> None:
    names = feature_names or list(terrain["features"].keys())
    xy = terrain["pad_xy"]
    data = {
        "resolution": terrain["resolution"],
        "orientation": "row0_is_y0_bottom",
        "axes": {"x": axis_labels[0], "y": axis_labels[1]},
        "generation": generation or {},
        "feature_names": names,
        "default_rgb": {k: v for k, v in DEFAULT_RGB.items() if v in names},
        "styles": [{"index": i, "x": float(xy[i, 0]), "y": float(xy[i, 1])} for i in range(xy.shape[0])],
        "features": {n: _round_grid(terrain["features"][n]) for n in names},
        "elevation": _round_grid(terrain["elevation"]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"wrote terrain JSON: {path}  ({terrain['resolution']}x{terrain['resolution']}, {len(names)} features)")


def save_terrain_png(terrain: dict, path: str, *, feature_names: Optional[List[str]] = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - PNG is a convenience only
        print(f"skipping PNG preview (matplotlib unavailable: {e})")
        return

    names = feature_names or list(terrain["features"].keys())
    xy = terrain["pad_xy"]
    elev = terrain["elevation"]
    panels = names + ["composite"]
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    def _mark(ax):
        ax.scatter(xy[:, 0], xy[:, 1], s=28, facecolors="none", edgecolors="white", linewidths=1.2)
        for i in range(xy.shape[0]):
            ax.text(xy[i, 0], xy[i, 1], str(i), color="white", fontsize=7, ha="center", va="center")

    for k, panel in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        if panel == "composite":
            rgb = np.zeros((*elev.shape, 3))
            for ci, key in enumerate(("r", "g", "b")):
                fname = DEFAULT_RGB.get(key)
                if fname in terrain["features"]:
                    rgb[..., ci] = terrain["features"][fname]
            rgb = rgb * (0.25 + 0.75 * elev[..., None])  # dim crevices, keep a floor
            ax.imshow(np.clip(rgb, 0, 1), origin="lower", extent=(0, 1, 0, 1))
            ax.contour(np.linspace(0, 1, elev.shape[1]), np.linspace(0, 1, elev.shape[0]), elev,
                       levels=6, colors="white", linewidths=0.3, alpha=0.5)
            ax.set_title("composite (RGB x elevation)")
        else:
            ax.imshow(terrain["features"][panel], origin="lower", extent=(0, 1, 0, 1), cmap="magma", vmin=0, vmax=1)
            ax.set_title(panel)
        _mark(ax)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(panels), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote terrain PNG: {path}")


def render_terrain(prior, compressor, ddsp, wrapper, feature_dim: int, latent_size: int, *,
                   cfg: float = 1.5, temperature: float = 0.6, seed: int = 0, seconds: float = 8.0,
                   resolution: int = 128, sample_grid: int = 24, avg_seeds: int = 1,
                   features: Optional[List[str]] = None,
                   json_path: Optional[str] = None, png_path: Optional[str] = None,
                   cache_path: Optional[str] = None, device: str = "cpu") -> Optional[dict]:
    """End-to-end: pick the pad (style else territory) from `wrapper`, measure features, rasterize, write
    JSON/PNG. Returns the terrain dict, or None if the model has no XY pad.

    `sample_grid > 0` measures features on an N x N grid of ACTUAL interpolated pad points (real between-
    style landscape), upsampled to `resolution`. `sample_grid == 0` falls back to the fast node-only mode
    (features measured at the T styles, then linearly blended)."""
    use_style = bool(getattr(wrapper, "use_style", False))
    use_terr = bool(getattr(wrapper, "use_terr_map", False))
    if use_style:
        pad_xy = wrapper.style_xy.detach().cpu().numpy()
        temp = float(wrapper._style_temp)
        node_table = wrapper.style_table.detach().cpu()
        axis_labels = ("Style X", "Style Y")
    elif use_terr:
        pad_xy = wrapper.territory_xy.detach().cpu().numpy()
        temp = float(wrapper.terr_map_temp)
        node_table = wrapper.territory_table.detach().cpu()  # unused for generation (territory index drives it)
        axis_labels = ("Territory X", "Territory Y")
    else:
        print("no style/territory XY pad on this model; nothing to render.")
        return None

    names = [f for f in (features or FEATURE_ORDER) if f in FEATURE_ORDER]
    mode = f"grid {sample_grid}x{sample_grid}" if int(sample_grid) > 0 else "node-blend"
    print(f"rendering terrain: {pad_xy.shape[0]} nodes, {resolution}x{resolution} ({mode}), "
          f"cfg={cfg} temp={temperature} seed={seed} seconds={seconds} device={device}")
    if int(sample_grid) > 0:
        coarse = grid_feature_sample(prior, compressor, ddsp, wrapper, feature_dim, latent_size,
                                     sample_grid=int(sample_grid), use_style=use_style, cfg=cfg,
                                     temperature=temperature, seed=seed, seconds=seconds, device=device,
                                     avg_seeds=int(avg_seeds), cache_path=cache_path)
        terrain = {
            "resolution": int(resolution),
            "features": {n: _upsample_norm(coarse[n], resolution) for n in names},
            "elevation": _elevation_grid(pad_xy, temp, resolution),
            "pad_xy": np.asarray(pad_xy, dtype=np.float64).reshape(-1, 2),
        }
        if cache_path is not None:  # sidecar so recompute_terrain can rebuild WITHOUT generation
            import json as _json
            with open(os.path.splitext(cache_path)[0] + "_meta.json", "w") as _f:
                _json.dump({"resolution": int(resolution), "temp": float(temp),
                            "pad_xy": np.asarray(pad_xy, dtype=np.float64).reshape(-1, 2).tolist(),
                            "axis_labels": list(axis_labels), "feature_sr": FEATURE_SR,
                            "cfg": float(cfg), "temperature": float(temperature), "seed": int(seed),
                            "seconds": float(seconds), "avg_seeds": int(avg_seeds),
                            "pad": "style" if use_style else "territory"}, _f)
    else:
        F = pad_feature_table(prior, compressor, ddsp, node_table, feature_dim, latent_size,
                              use_style=use_style, cfg=cfg, temperature=temperature, seed=seed,
                              seconds=seconds, device=device)
        terrain = compute_pad_terrain(pad_xy, {n: F[n] for n in names}, temp, resolution)
    generation = {"cfg": float(cfg), "temperature": float(temperature), "seed": int(seed),
                  "seconds": float(seconds), "pad": "style" if use_style else "territory",
                  "sample_grid": int(sample_grid), "avg_seeds": int(avg_seeds)}
    if json_path:
        write_terrain_json(terrain, json_path, axis_labels=axis_labels, feature_names=names, generation=generation)
    if png_path:
        save_terrain_png(terrain, png_path, feature_names=names)
    return terrain
