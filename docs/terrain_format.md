# Style-pad terrain — format & workflow

A **terrain** is a 2-D map of the discrete-prior **style XY pad** (or territory pad): for a grid of
points across the pad it generates audio at that interpolated style, measures audio features, and
rasterizes each feature over the `[0,1]²` pad. It is exported as a JSON (for a Max/PureData JS
renderer) plus PNG previews, so a performer can *see* the sonic landscape they navigate — where the
rhythm is, where it's percussive, where it's bright — with the style embeddings sitting on peaks.

Produced by `scripts/render_style_terrain.py` (or the `--emit_terrain` hook of `cli/export.py`), and
implemented in `cli/terrain.py`.

---

## Files

| File | For | Notes |
|---|---|---|
| `<name>.json` | the Max/JS renderer | the terrain data (feature grids, style positions, elevation) |
| `<name>.png` | eyeballing | multi-panel preview (one panel per feature + a composite) |
| `<name>_audio.npy` (+ `_meta.json`) | re-deriving features offline | only with `--cache_audio`; ~1.2 GB; **not for Max** |

---

## JSON schema

```jsonc
{
  "resolution": 128,                       // every grid is 128×128
  "orientation": "row0_is_y0_bottom",      // grid[0] is the BOTTOM row (y=0)
  "axes": { "x": "Style X", "y": "Style Y" },
  "generation": { "cfg": 3.0, "temperature": 0.4, "seconds": 30.0,
                  "sample_grid": 16, "avg_seeds": 5, "seed": 0, "pad": "style" },  // provenance only
  "feature_names": ["rhythmicity","onset_density","percussiveness","loudness","brightness","noisiness"],
  "default_rgb": { "r": "percussiveness", "g": "rhythmicity", "b": "brightness" }, // a suggested mapping
  "styles": [ { "index": 0, "x": 0.40, "y": 0.39 }, ... ],   // style markers, x/y in [0,1]
  "features": {                            // one disentangled grid per feature, values in [0,1]
    "rhythmicity":   [[...128...], ... 128 rows ...],
    "onset_density": [[...]],
    ...
  },
  "elevation": [[...128...], ...]          // [0,1]; = 1 at each style, dips into the crevices between
}
```

### Coordinate convention (matches a standard XY pad)

- Grids are indexed **`grid[row][col]`**, where **`row` = Y and `col` = X**.
- `orientation: "row0_is_y0_bottom"` → **row 0 is the bottom** (y=0), col 0 is the left (x=0).
  So `grid[0][0]` is the **bottom-left** corner = pad `(x=0, y=0)`.

Look up the value under the pad position `(px, py)` (both in `[0,1]`):

```js
const col = Math.round(px * (resolution - 1));
const row = Math.round(py * (resolution - 1));
const val = features["rhythmicity"][row][col];   // 0..1
```

### Values are normalized per-map

Every feature grid (and `elevation`) is min-max normalized to `[0,1]` **within that render**: `0` =
lowest, `1` = highest *on this pad*. They are **relative**, not absolute physical units — good for
"where on the pad is X strongest", not for comparing magnitudes across different models/renders.

### Rendering

- **Single feature** — map each cell's value through a colormap (dark→bright). Clearest to read.
- **RGB (three features)** — `r = features[fr][row][col]`, `g`, `b`. `default_rgb` is only a suggested
  starting triple; map any three.
- **elevation** — optional: draw contour lines from it, and/or dim the between-style areas so the
  styles glow as peaks. Draw the `styles` as circles + index labels at their `(x, y)`.

### Features

Temporal (from the onset / amplitude envelope): `rhythmicity` (strength of a periodic pulse),
`onset_density` (events/sec), `percussiveness` (percussive vs sustained energy, via HPSS). Timbre
(spectral): `loudness` (RMS), `brightness` (spectral centroid), `noisiness` (spectral flatness). All
measured from audio generated at a **fixed** cfg / temperature / seed (recorded under `generation`) so
the map is deterministic and representative of what the instrument produces at that pad point.

---

## Workflow

```zsh
# 1) Render a terrain. ALWAYS pass --from_ts so the pad geometry matches the model you load in Max
#    (see the gotcha below). ~10s per grid cell on GPU → a 16×16 grid ≈ 45 min.
python -m scripts.render_style_terrain --config <name> --from_ts models/<name>.ts \
  --cfg 3 --temperature 0.4 --seconds 30 --sample_grid 16 --avg_seeds 5 --cache_audio \
  --out terrain_previews/<name>

# 2) Re-render the PNG previews (single-feature heatmaps + RGB composites) from the JSON — instant.
python -m scripts.preview_terrain --json terrain_previews/<name>.json --smooth 1.0

# 3) Changed a feature definition? Recompute the whole terrain from the cached audio — no GPU.
python -m scripts.recompute_terrain --cache terrain_previews/<name>_audio.npy
```

Key flags of `render_style_terrain.py`:

| Flag | Meaning |
|---|---|
| `--from_ts <model.ts>` | read the pad geometry (`style_xy` / `style_table`) from the exported model (see gotcha) |
| `--sample_grid N` | measure features on an N×N grid of real interpolated pad points (0 = fast node-blend) |
| `--avg_seeds N` | generate each cell N times and average — denoises the map |
| `--seconds` / `--cfg` / `--temperature` / `--seed` | generation settings (recorded in `generation`) |
| `--resolution` | display grid size (the coarse `sample_grid` is upsampled to this) |
| `--cache_audio` | also save the rendered audio so features can be recomputed offline |

`cli/export.py` can also emit a terrain right after a `.ts` export via `--emit_terrain` (default on;
`--terrain_grid` / `--terrain_avg_seeds` control it; `--no_emit_terrain` to skip). This is the
guaranteed-matching path — the terrain shares the exact geometry of the model it just exported.

---

## Max renderer (`max/terrain.js`)

`max/terrain.js` is a `jsui` script that draws a terrain JSON and **resizes with the object** — just
drag the jsui box. The `.json` (and `max/terrain.js`) must be on the Max search path or given as an
absolute path.

Messages (into the single inlet):

| Message | Effect |
|---|---|
| `read <file>` | load a terrain. Accepts a `.json` path, OR the model's `.ts` path — it derives `<base>_terrain.json` (falling back to `<base>.json`), so you can feed the same path you gave `nn~`. A bare `*.json`/`*.ts` symbol works too. |
| `<int>` | select a menu entry by index — wire a **`live.menu`** outlet here |
| `feature <name\|index>` | show one feature (magma colormap) |
| `rgb <a> <b> <c>` | show three features as R/G/B (names or indices; `none`/`-1` = channel off) |
| `cfg <0..20>` | **contrast**: `0` = flat/uniform terrain, higher = more prominent, `20` = maximal |
| `markers <0\|1>` | hide/show the style markers |
| `clear` (or `reset`) | unload the terrain → blank black display |
| `dumpmenu` | (re)populate the connected `live.menu` (also fires automatically when a terrain loads) |
| `bang` | redraw |

The `jsui` has **2 inlets**: inlet 0 (left) for the messages above + the `live.menu` selection index;
inlet 1 (right) for the `cfg` contrast value (a bare `0..20` number). `cfg` is a **saved attribute** — it
is stored with the patch and restored (and applied) on load, so a terrain that loads at startup is drawn
at your saved contrast rather than the default. Its **outlet** populates a
`live.menu` (via the `_parameter_range` attribute, sent automatically when a terrain loads) — connect the
outlet to the `live.menu` inlet, and the `live.menu` outlet back to inlet 0 to select the view. Set the
`live.menu`'s **Parameter Visibility** so it isn't stored, since its items are populated at runtime.

Patch wiring: send `dumpmenu` once to fill a `umenu` (its items become the 6 single features + a few
RGB presets, in menu order); connect the `umenu` outlet to the `jsui` inlet so selecting an item picks
the view. Feed `cfg $1` from a `[flonum]`/slider (0–20). The display uses the same coordinate convention
as the pad: **(0,0) is bottom-left**. The `cfg` contrast is a *display* control only — it reshapes the
feature values around their mean (it is unrelated to the model's generation-time CFG baked into the map).

## ⚠️ Gotcha: the terrain must match the model you load

The style pad layout (`style_xy`, a t-SNE of the per-style centroids) is **baked into each export** and
is **not reproducible** across exports — an older `.ts` was built with an unseeded window sample, so a
freshly regenerated terrain invents a *different* layout and every feature lands on the wrong pad
location (symptom: "the map doesn't match what I hear").

**Always render the terrain against the `.ts` you actually load in Max**, via
`--from_ts models/<name>.ts` (standalone) or by using the `cli/export.py --emit_terrain` hook (same
export). This reads `style_xy` / `style_table` straight from the model, so the terrain is guaranteed to
correspond to your pad. (`cli/export.py` now seeds the centroid sampling, so *new* exports are
reproducible — but pre-existing `.ts` files predate that.)
