// terrain.js — jsui renderer for a style-pad terrain JSON (see docs/terrain_format.md).
//
// Displays the pad terrain (a feature heatmap, or an RGB composite of 3 features) and adapts to the
// jsui box size (just resize the object). Style markers are drawn on top. The "CFG" value controls how
// strongly the terrain varies: cfg=0 -> flat/uniform, higher -> more contrast, up to cfg=20 = maximal.
//
// Inlets:  0 (left) = control messages / live.menu selection ;  1 (right) = cfg value (0..20).
// Outlet:  0 = populates a live.menu (sends its "_parameter_range"; auto-fires when a terrain loads,
//              and on demand via the "dumpmenu" message).
//
// Messages into inlet 0:
//   read <file>               load a terrain (a .json path, or the model's .ts -> <base>_terrain.json)
//   <int>                     select a menu entry by index (wire the live.menu's outlet here)
//   feature <name|index>      show a single feature (magma colormap)
//   rgb <a> <b> <c>           show 3 features as R/G/B (names or indices; "none"/-1 = channel off)
//   markers <0|1>             hide / show the style markers
//   clear  (or reset)         unload the terrain -> blank black display
//   dumpmenu                  (re)populate the connected live.menu
//   bang                      redraw
// Into inlet 1:
//   <0..20>                   contrast / prominence: 0 = flat, higher = more, 20 = maximal
//   (or the message  cfg <0..20>  into inlet 0). cfg is SAVED with the patch and restored + applied on
//   load, so a terrain that loads at startup is drawn at your saved contrast, not the default.
//
// Patch: connect [jsui] outlet -> [live.menu] inlet (auto-populates on load), and [live.menu] outlet ->
// [jsui] inlet 0. Feed a [live.dial]/[number] 0..20 into [jsui] inlet 1 for contrast.

autowatch = 1;
inlets = 2;
outlets = 1;
setinletassist(0, "messages: read / feature / rgb / dumpmenu / live.menu index");
setinletassist(1, "cfg 0..20 (display contrast)");
setoutletassist(0, "to live.menu: _parameter_range (auto on load, or send dumpmenu)");
// Persist the contrast with the patch and RESTORE (+ apply) it on load, so a terrain that loads at
// startup is drawn at the saved cfg rather than the hard-coded default.
declareattribute("cfg", "getcfg", "setcfg", 1);

mgraphics.init();
mgraphics.relative_coords = 0;   // pixel coordinates
mgraphics.autofill = 0;

var gTerrain = null;   // parsed JSON
var gNames = [];       // feature_names
var gMean = {};        // per-feature mean (for the contrast centre)
var gMenu = [];        // live.menu entries: single features + RGB presets
var gMode = "single";  // "single" | "rgb"
var gFeatureIdx = 0;   // active single-feature index
var gRgb = [-1, -1, -1];
var gCfg = 6.0;        // contrast strength 0..20
var gMarkers = 1;

// ---------------------------------------------------------------- messages

function read(f) { loadTerrain(f); }

function clear() {                       // reset to the empty (black) state
    gTerrain = null; gNames = []; gMean = {}; gMenu = [];
    gMode = "single"; gFeatureIdx = 0; gRgb = [-1, -1, -1];
    mgraphics.redraw();
}
function reset() { clear(); }            // alias

function anything() {
    var sel = messagename;
    if (nameIndex(sel) >= 0) { setSingle(nameIndex(sel)); return; }     // bare feature name (symbol)
    if (/\.(json|ts)$/i.test(sel)) { loadTerrain(sel); return; }        // bare filename (.json or .ts)
    post("terrain: unknown message '" + sel + "'\n");
}

function msg_int(i) {
    if (inlet === 1) { setcfg(i); return; }          // right inlet = cfg
    if (i >= 0 && i < gMenu.length) applyMenu(i);     // left inlet = live.menu index
}
function msg_float(x) {
    if (inlet === 1) setcfg(x);                       // right inlet = cfg (float)
}

function feature(a) { setSingle(resolveFeature(a)); }

function rgb() {
    var args = arrayfromargs(arguments);
    gMode = "rgb";
    for (var k = 0; k < 3; k++) gRgb[k] = (k < args.length) ? resolveFeatureLoose(args[k]) : -1;
    mgraphics.redraw();
}

// cfg is a saved attribute (see declareattribute above). setcfg is its setter — also called by Max when
// restoring the saved value on load, and by the right inlet / a "cfg <v>" message at runtime.
function getcfg() { return gCfg; }
function setcfg(v) { gCfg = Math.max(0.0, Math.min(20.0, Number(v))); mgraphics.redraw(); }

function markers(v) { gMarkers = v ? 1 : 0; mgraphics.redraw(); }

function bang() { mgraphics.redraw(); }

function dumpmenu() {
    // Populate a live.menu: its items are the enum range, set in one attribute message.
    if (gMenu.length === 0) return;
    var payload = [0, "_parameter_range"];
    for (var i = 0; i < gMenu.length; i++) payload.push(gMenu[i].label);
    outlet.apply(this, payload);
}

// ---------------------------------------------------------------- loading

// Accept a .json path directly, OR a .ts (model) path -> derive its terrain JSON. The exporter writes
// "<base>_terrain.json" (--emit_terrain); also try a plain "<base>.json". Loads the first that opens.
function terrainCandidates(fname) {
    var out = [];
    if (/\.json$/i.test(fname)) {
        out.push(fname);
    } else {
        var base = fname.replace(/\.[^./\\:]*$/, "");   // strip the extension (.ts etc.)
        out.push(base + "_terrain.json");
        out.push(base + ".json");
    }
    // Fallback: also try the bare filename so Max's SEARCH PATH is used when an absolute path (e.g. a
    // "Macintosh HD:/..." style path) fails to open directly. Add the folder to Options > File Preferences,
    // or keep the .json beside the patch, and this resolves it.
    for (var i = out.length - 1; i >= 0; i--) {
        var bn = out[i].split(/[\/:\\]/).pop();
        if (bn && out.indexOf(bn) === -1) out.push(bn);
    }
    return out;
}

function loadTerrain(fname) {
    var candidates = terrainCandidates(fname), f = null, used = "";
    for (var c = 0; c < candidates.length; c++) {
        // NB: no 3rd arg — it is a macOS file-TYPE filter (e.g. 'TEXT'), not an encoding; passing one
        // makes isopen false for normal .json files that lack that legacy type code.
        var trial = new File(candidates[c], "read");
        if (trial.isopen) { f = trial; used = candidates[c]; break; }
    }
    if (!f) { post("terrain: no JSON found for '" + fname + "' (tried " + candidates.join(", ") + ")\n"); return; }
    fname = used;
    // Read the whole file by looping readline() until it returns nothing. NB: Max's readline() caps at
    // 32767 chars per call, and f.eof/f.position are unreliable here (eof reports 0), so we must NOT
    // gate the loop on position<eof — just keep reading fixed chunks until empty. Concatenating the
    // pieces (readline drops newlines, which are insignificant JSON whitespace) reconstructs the file.
    var text = "", piece, guard = 0;
    while (guard < 1000000) {
        piece = f.readline(32767);
        if (piece === null || piece === undefined || piece.length === 0) break;
        text += piece;
        guard++;
    }
    f.close();
    if (!text.length) { post("terrain: read 0 bytes from '" + fname + "'\n"); return; }
    try { gTerrain = JSON.parse(text); }
    catch (e) {
        post("terrain: JSON parse error (" + e + "); read " + text.length + " chars\n");
        gTerrain = null; return;
    }

    gNames = gTerrain.feature_names || [];
    if (gNames.length === 0) for (var n in gTerrain.features) gNames.push(n);
    computeMeans();
    buildMenu();
    gMode = "single"; gFeatureIdx = 0;
    if (gTerrain.default_rgb) {
        gRgb = [nameIndex(gTerrain.default_rgb.r), nameIndex(gTerrain.default_rgb.g),
                nameIndex(gTerrain.default_rgb.b)];
    }
    post("terrain loaded: " + fname + "  " + gTerrain.resolution + "x" + gTerrain.resolution +
         ", " + gNames.length + " features\n");
    dumpmenu();      // auto-populate a connected live.menu with the feature/RGB entries
    mgraphics.redraw();
}

function computeMeans() {
    gMean = {};
    for (var n in gTerrain.features) {
        var g = gTerrain.features[n], s = 0, c = 0;
        for (var r = 0; r < g.length; r++) { var row = g[r]; for (var k = 0; k < row.length; k++) { s += row[k]; c++; } }
        gMean[n] = c ? s / c : 0.5;
    }
}

function buildMenu() {
    gMenu = [];
    for (var i = 0; i < gNames.length; i++) gMenu.push({ mode: "single", f: i, label: gNames[i] });
    addRgbPreset("percussiveness", "rhythmicity", "brightness", "RGB_default");
    addRgbPreset("rhythmicity", "onset_density", "percussiveness", "RGB_temporal");
    addRgbPreset("loudness", "brightness", "noisiness", "RGB_timbre");
}

function addRgbPreset(a, b, c, label) {
    var r = nameIndex(a), g = nameIndex(b), bl = nameIndex(c);
    if (r >= 0 && g >= 0 && bl >= 0) gMenu.push({ mode: "rgb", rgb: [r, g, bl], label: label });
}

// ---------------------------------------------------------------- selection helpers

function nameIndex(name) {
    for (var i = 0; i < gNames.length; i++) if (gNames[i] === name) return i;
    return -1;
}
function resolveFeature(a) {
    if (typeof a === "number") return (a >= 0 && a < gNames.length) ? a : -1;
    return nameIndex(String(a));
}
function resolveFeatureLoose(a) {
    if (typeof a === "number") return (a >= 0 && a < gNames.length) ? a : -1;
    var s = String(a);
    if (s === "none" || s === "-" || s === ".") return -1;
    return nameIndex(s);
}
function setSingle(idx) {
    if (idx < 0) return;
    gMode = "single"; gFeatureIdx = idx; mgraphics.redraw();
}
function applyMenu(i) {
    var m = gMenu[i]; if (!m) return;
    if (m.mode === "single") { gMode = "single"; gFeatureIdx = m.f; }
    else { gMode = "rgb"; gRgb = m.rgb.slice(); }
    mgraphics.redraw();
}

// ---------------------------------------------------------------- colour

// Contrast shaping around the feature mean. cfg=0 -> 0.5 everywhere (flat); higher cfg -> steeper
// sigmoid -> more contrast; cfg=20 approaches a hard split (maximal visibility).
function shape(v, m) {
    if (gCfg <= 1e-4) return 0.5;
    return 1.0 / (1.0 + Math.exp(-gCfg * (v - m)));
}

// magma-ish colormap (t in [0,1] -> [r,g,b]).
var MAGMA = [
    [0.00, 0.00, 0.01], [0.16, 0.045, 0.28], [0.35, 0.06, 0.43], [0.55, 0.09, 0.43],
    [0.74, 0.19, 0.37], [0.90, 0.32, 0.24], [0.98, 0.52, 0.18], [0.99, 0.75, 0.28], [0.99, 0.99, 0.75]
];
function magma(t) {
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    var x = t * (MAGMA.length - 1), i = Math.floor(x), f = x - i;
    if (i >= MAGMA.length - 1) return MAGMA[MAGMA.length - 1];
    var a = MAGMA[i], b = MAGMA[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// Bilinear sample of a feature grid at pad coords (u,v) in [0,1]. grid[row][col], row=Y (row 0 = y=0).
function sampleBilinear(grid, u, v, res) {
    var fx = u * (res - 1), fy = v * (res - 1);
    var x0 = Math.floor(fx), y0 = Math.floor(fy);
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    var x1 = x0 + 1 < res ? x0 + 1 : res - 1;
    var y1 = y0 + 1 < res ? y0 + 1 : res - 1;
    var tx = fx - x0, ty = fy - y0;
    var a = grid[y0][x0], b = grid[y0][x1], c = grid[y1][x0], d = grid[y1][x1];
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

// Colour at pad coords (u,v), bilinearly interpolated (smooth) then contrast-shaped.
function colorAtUV(u, v) {
    var res = gTerrain.resolution;
    if (gMode === "single") {
        var n = gNames[gFeatureIdx];
        return magma(shape(sampleBilinear(gTerrain.features[n], u, v, res), gMean[n]));
    }
    var out = [0, 0, 0];
    for (var k = 0; k < 3; k++) {
        var idx = gRgb[k];
        if (idx >= 0) { var nm = gNames[idx]; out[k] = shape(sampleBilinear(gTerrain.features[nm], u, v, res), gMean[nm]); }
    }
    return out;
}

// ---------------------------------------------------------------- paint

function paint() {
    var w = mgraphics.size[0], h = mgraphics.size[1];
    if (!gTerrain) { return; }   // nothing loaded -> paint nothing -> the jsui is transparent

    mgraphics.set_source_rgba(0, 0, 0, 1);   // black base under the heatmap
    mgraphics.rectangle(0, 0, w, h); mgraphics.fill();

    // Draw ~2px cells (independent per axis) with BILINEAR-interpolated colour -> smooth gradients.
    // Capped so a large box stays responsive.
    var cols = Math.max(24, Math.min(320, Math.round(w / 2)));
    var rows = Math.max(24, Math.min(320, Math.round(h / 2)));
    var cw = w / cols, ch = h / rows;
    for (var i = 0; i < rows; i++) {                 // i = display row from TOP
        var v = 1.0 - (i + 0.5) / rows;              // pad y (0 = bottom, 1 = top)
        var sy = i * ch;
        for (var j = 0; j < cols; j++) {
            var u = (j + 0.5) / cols;                // pad x (0 = left, 1 = right)
            var c = colorAtUV(u, v);
            mgraphics.set_source_rgba(c[0], c[1], c[2], 1);
            mgraphics.rectangle(j * cw, sy, cw + 0.6, ch + 0.6);   // +0.6 avoids seams
            mgraphics.fill();
        }
    }

    if (gMarkers && gTerrain.styles) drawMarkers(w, h);
}

function drawMarkers(w, h) {
    var st = gTerrain.styles, rad = Math.max(5, Math.min(w, h) * 0.018);
    mgraphics.select_font_face("Arial"); mgraphics.set_font_size(Math.max(8, rad));
    mgraphics.set_line_width(1.4);
    for (var i = 0; i < st.length; i++) {
        var sx = st[i].x * w, sy = h - st[i].y * h;     // flip Y
        mgraphics.set_source_rgba(0, 0, 0, 0.55);
        mgraphics.ellipse(sx - rad, sy - rad, rad * 2, rad * 2); mgraphics.fill();
        mgraphics.set_source_rgba(1, 1, 1, 0.95);
        mgraphics.ellipse(sx - rad, sy - rad, rad * 2, rad * 2); mgraphics.stroke();
        var lbl = String(st[i].index);
        mgraphics.move_to(sx - rad * 0.5 * lbl.length, sy + rad * 0.5);
        mgraphics.show_text(lbl);
    }
}

// initial blank paint
mgraphics.redraw();
