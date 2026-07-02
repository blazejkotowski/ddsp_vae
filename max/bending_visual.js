mgraphics.init();
mgraphics.relative_coords = 0;
mgraphics.autofill = 0;

autowatch = 1;
inlets = 7;
outlets = 0;

// The vertical axis of the orb == the spectrum (bottom = low band, top = high band).
// limit_mode chooses WHICH bands survive; roll/stretch/warp move them along that axis.
setinletassist(0, "limit amount 0..1 (0=all partials, 1=one)");
setinletassist(1, "waveshape 0..1 (0=noise, 0.5=sine, 1=square)");
setinletassist(2, "rotation speed 0..100");
setinletassist(3, "limit mode 0..5 (0 loudest,1 density,2 lower,3 higher,4 peaks,5 stochastic)");
setinletassist(4, "spectral roll 0..1 (shift up, wraps)");
setinletassist(5, "spectral stretch -1..1 (+ = toward highs)");
setinletassist(6, "spectral warp -1..1 (+ = toward lows/darker)");

var density   = 0.0;   // == limit_components (amount)
var waveshape = 0.0;
var speed     = 1.0;
var limit_mode = 0;    // 0 loudest,1 density,2 lower,3 higher,4 peaks,5 stochastic
var roll      = 0.0;   // spectral_roll   0..1
var stretch   = 0.0;   // spectral_stretch -1..1
var warp      = 0.0;   // spectral_warp   -1..1
var rot_y     = 0.4;
var rot_x     = 0.3;
var max_p     = 32;
var task      = null;
var time      = 0.0;

// per-frame surviving-band arrays (filled in paint)
var spoke_dirs    = [];
var spoke_amp     = [];
var spoke_bandidx = [];

// ghost frame storage for motion trail
var ghost_dirs  = [];
var ghost_rot_y = 0.0;
var ghost_rot_x = 0.0;
var ghost_time  = 0.0;
var prev_rot_y  = 0.4;
var frame_count = 0;

function msg_float(v) {
    if (inlet == 0) { density   = Math.max(0.0, Math.min(1.0, v)); mgraphics.redraw(); }
    if (inlet == 1) { waveshape = Math.max(0.0, Math.min(1.0, v)); mgraphics.redraw(); }
    if (inlet == 2) { set_speed(v); }
    if (inlet == 3) { limit_mode = Math.max(0, Math.min(5, Math.round(v))); mgraphics.redraw(); }
    if (inlet == 4) { roll    = Math.max(0.0,  Math.min(1.0, v)); mgraphics.redraw(); }
    if (inlet == 5) { stretch = Math.max(-1.0, Math.min(1.0, v)); mgraphics.redraw(); }
    if (inlet == 6) { warp    = Math.max(-1.0, Math.min(1.0, v)); mgraphics.redraw(); }
}
function msg_int(v) { msg_float(v); }
function bang()     { mgraphics.redraw(); }

function speed_to_rate(s) {
    if (s <= 0) return 0;
    var norm = Math.log(s + 1.0) / Math.log(101.0);
    return norm * 0.12;
}

function set_speed(v) {
    speed = Math.max(0.0, Math.min(100.0, v));
    if (task) { task.cancel(); task = null; }
    if (speed > 0.0) {
        task = new Task(tick, this);
        task.interval = 32;
        task.repeat();
    } else {
        mgraphics.redraw();
    }
}

function tick() {
    var rate   = speed_to_rate(speed);
    prev_rot_y = rot_y;
    rot_y     += rate;
    time      += rate * 0.4;
    frame_count++;
    // update ghost every 4 frames
    if (frame_count % 4 == 0) {
        ghost_rot_y = prev_rot_y;
        ghost_rot_x = rot_x;
        ghost_time  = time - rate * 1.6;
    }
    if (rot_y > Math.PI * 2.0) rot_y -= Math.PI * 2.0;
    mgraphics.redraw();
}

// ── colour from waveshape ─────────────────────────────────
// noise=0: cold blue, sine=0.5: white, square=1: warm amber

function ws_color(ws, alpha_mul) {
    var r, g, b;
    if (ws < 0.5) {
        var t = ws * 2.0;
        r = 1.0;
        g = 1.0 - t * 0.45;
        b = 0.9 - t * 0.9;
    } else {
        var t = (ws - 0.5) * 2.0;
        r = 1.0;
        g = 0.55 - t * 0.45;
        b = 0.0;
    }
    return [r, g, b, alpha_mul];
}
// ── 3d helpers ────────────────────────────────────────────

function rot_Y(x, y, z, a) {
    return {
        x:  x * Math.cos(a) + z * Math.sin(a),
        y:  y,
        z: -x * Math.sin(a) + z * Math.cos(a)
    };
}

function rot_X(x, y, z, a) {
    return {
        x: x,
        y: y * Math.cos(a) - z * Math.sin(a),
        z: y * Math.sin(a) + z * Math.cos(a)
    };
}

function project(p, cx, cy, R) {
    var fov  = 2.8;
    var dist = fov;
    var z    = p.z + dist;
    if (z < 0.01) z = 0.01;
    var scale = dist / z;
    return {
        x:     cx + p.x * R * scale,
        y:     cy + p.y * R * scale,
        scale: scale,
        z:     p.z
    };
}

function transform_with(x, y, z, ry, rx) {
    var p = rot_Y(x, y, z, ry);
    p     = rot_X(p.x, p.y, p.z, rx);
    return p;
}

function transform(x, y, z) {
    return transform_with(x, y, z, rot_y, rot_x);
}

function tanh_safe(x) {
    if (x > 10)  return  1.0;
    if (x < -10) return -1.0;
    var e2 = Math.exp(2.0 * x);
    return (e2 - 1.0) / (e2 + 1.0);
}

function spoke_disp(t, ws, spoke_idx) {
    var w = 0.22;
    if (ws < 0.5) {
        var tt    = ws * 2.0;
        var noise = Math.sin(t * 11.3  + spoke_idx * 3.7)  * 0.45
                  + Math.sin(t * 23.1  + spoke_idx * 7.1)  * 0.25
                  + Math.sin(t * 41.7  + spoke_idx * 1.9)  * 0.15
                  + Math.sin(t * 67.3  + spoke_idx * 5.3)  * 0.10
                  + Math.sin(t * 103.9 + spoke_idx * 11.7) * 0.05;
        var sine  = Math.sin(t * Math.PI * 3.0);
        return (noise * (1.0 - tt) + sine * tt) * w;
    } else {
        var tt       = (ws - 0.5) * 2.0;
        var sine_raw = Math.sin(t * Math.PI * 3.0);
        var sq       = tanh_safe(sine_raw * (1.0 + tt * 8.0));
        return (sine_raw * (1.0 - tt) + sq * tt) * w;
    }
}

// spoke length driven by its band amplitude (spoke_amp[s]) + a gentle idle wobble.
function spoke_length(s, t) {
    var amp   = (s < spoke_amp.length) ? spoke_amp[s] : 1.0;
    var phase = ((s < spoke_bandidx.length) ? spoke_bandidx[s] : s) * 2.399;
    var base  = 0.55 + 0.5 * amp;
    var range = 0.10;
    return base + Math.sin(t + phase) * range
                + Math.sin(t * 1.618 + phase * 0.7) * range * 0.4;
}

// ── band model (spectrum along the vertical axis) ─────────────
// A fixed spectral envelope over max_p bands (band 0 = low, max_p-1 = high) with three
// formant-like bumps, so the limit modes read clearly (loudest keeps the bumps, peaks keeps
// their crests, lower/higher keep the ends).
function band_amp(i, N) {
    var p = (N > 1) ? i / (N - 1) : 0.0;
    return 0.30
         + 0.55 * Math.exp(-Math.pow((p - 0.18) / 0.10, 2))
         + 0.45 * Math.exp(-Math.pow((p - 0.50) / 0.12, 2))
         + 0.30 * Math.exp(-Math.pow((p - 0.80) / 0.09, 2))
         + 0.12 * (1.0 - p);
}

// spectral bends: map a band's neutral position p in [0,1] to its bent position.
// Directions mirror the synth: stretch>0 -> toward highs, warp>0 -> toward lows, roll -> up + wrap.
function bend_pos(p) {
    var scale = Math.pow(2.0, stretch);
    p = Math.max(0.0, Math.min(1.0, p * scale));   // stretch: clamp (no wrap)
    var gamma = Math.pow(2.0, warp);
    p = Math.pow(p, gamma);                         // warp: energy-conserving skew
    if (roll != 0.0) {                              // roll: circular wrap only when active,
        p = (p + roll) % 1.0;                       // so a stretched band at the top pole
        if (p < 0.0) p += 1.0;                      // (p=1) is not collapsed to the bottom.
    }
    return p;
}

// direction on the unit sphere: bent band position -> latitude (y), stable golden longitude.
function band_dir(idx, N) {
    var p    = bend_pos((N > 1) ? idx / (N - 1) : 0.0);
    var y    = 2.0 * p - 1.0;
    var r_xz = Math.sqrt(Math.max(0.0, 1.0 - y * y));
    var lon  = Math.PI * (3.0 - Math.sqrt(5.0)) * idx;
    return { x: r_xz * Math.cos(lon), y: y, z: r_xz * Math.sin(lon) };
}

// pick which of the N bands survive, given the limit amount + mode.
function select_bands(N, amount, mode) {
    var k = Math.max(1, Math.round((1.0 - amount) * N));
    var keep = [];
    if (k >= N) { for (var i = 0; i < N; i++) keep.push(i); return keep; }

    if (mode == 1) {                 // density: evenly spaced
        for (var j = 0; j < k; j++) keep.push(Math.round(j * (N - 1) / (k > 1 ? k - 1 : 1)));
    } else if (mode == 2) {          // lower
        for (var i = 0; i < k; i++) keep.push(i);
    } else if (mode == 3) {          // higher
        for (var i = N - k; i < N; i++) keep.push(i);
    } else if (mode == 4) {          // peaks: k most prominent local maxima
        var peaks = [];
        for (var i = 0; i < N; i++) {
            var a = band_amp(i, N);
            var l = (i > 0)     ? band_amp(i - 1, N) : -1.0;
            var r = (i < N - 1) ? band_amp(i + 1, N) : -1.0;
            if (a >= l && a >= r) peaks.push({ i: i, a: a });
        }
        peaks.sort(function(x, y) { return y.a - x.a; });
        var kk = Math.min(k, peaks.length);
        for (var j = 0; j < kk; j++) keep.push(peaks[j].i);
    } else if (mode == 5) {          // stochastic: random subset (flickers each frame)
        var pool = [];
        for (var i = 0; i < N; i++) pool.push(i);
        for (var j = 0; j < k && pool.length > 0; j++) {
            var r = Math.floor(Math.random() * pool.length);
            keep.push(pool[r]); pool.splice(r, 1);
        }
    } else {                         // 0 loudest: global top-k by amplitude
        var arr = [];
        for (var i = 0; i < N; i++) arr.push({ i: i, a: band_amp(i, N) });
        arr.sort(function(x, y) { return y.a - x.a; });
        for (var j = 0; j < k; j++) keep.push(arr[j].i);
    }
    keep.sort(function(a, b) { return a - b; });
    return keep;
}

// fill the module-level spoke arrays for this frame from (amount, mode, bends).
function build_bands() {
    var N = max_p;
    var keep = select_bands(N, density, limit_mode);
    spoke_dirs = []; spoke_amp = []; spoke_bandidx = [];
    for (var s = 0; s < keep.length; s++) {
        var bi = keep[s];
        spoke_dirs.push(band_dir(bi, N));
        spoke_amp.push(band_amp(bi, N));
        spoke_bandidx.push(bi);
    }
}

// draw a glowing spoke with taper + bloom layers
function draw_spoke_glow(g, dir, perp, r_inner, r_outer, i, cx, cy, R,
                          depth_t, col, ry, rx, t_time) {
    var steps = 32;
    var lw_base = 0.4 + depth_t * 0.8;

    // bloom layers — wide soft outer glow, then sharper inner lines
    var layers = [
        { lw_mul: 5.0, alpha_mul: 0.04 },
        { lw_mul: 3.0, alpha_mul: 0.07 },
        { lw_mul: 1.8, alpha_mul: 0.12 },
        { lw_mul: 1.0, alpha_mul: 1.00 },
    ];

    for (var li = 0; li < layers.length; li++) {
        var layer = layers[li];
        var lw    = lw_base * layer.lw_mul;
        var alpha = (0.2 + depth_t * 0.75) * layer.alpha_mul;

        // depth haze: back = bluer, front = warmer
        var haze_t = depth_t;
        var c      = col.slice();
        c[0] = c[0] * (0.5 + haze_t * 0.5);
        c[1] = c[1] * (0.6 + haze_t * 0.4);
        c[2] = c[2] * (0.8 + haze_t * 0.2) + (1.0 - depth_t) * 0.3;

        g.set_source_rgba(c[0], c[1], c[2], alpha);
        g.set_line_width(lw);

        var d0   = spoke_disp(0.0, waveshape, i);
        var p3_0 = transform_with(
            dir.x * r_inner + perp.x * d0,
            dir.y * r_inner + perp.y * d0,
            dir.z * r_inner + perp.z * d0,
            ry, rx
        );
        var s0 = project(p3_0, cx, cy, R);
        g.move_to(s0.x, s0.y);

        for (var s = 1; s <= steps; s++) {
            var t      = s / steps;
            var r      = r_inner + t * (r_outer - r_inner);
            // taper: line width modulated by position — thick at base, thin at tip
            var taper  = 1.0 - t * 0.7;
            var d      = spoke_disp(t, waveshape, i);
            var p3     = transform_with(
                dir.x * r + perp.x * d,
                dir.y * r + perp.y * d,
                dir.z * r + perp.z * d,
                ry, rx
            );
            var sp = project(p3, cx, cy, R);
            g.line_to(sp.x, sp.y);
        }
        g.stroke();
    }
}

// ── paint ─────────────────────────────────────────────────

function paint() {
    var g  = mgraphics;
    var W  = box.rect[2] - box.rect[0];
    var H  = box.rect[3] - box.rect[1];
    var cx = W * 0.5;
    var cy = H * 0.5;
    var R  = Math.min(W, H) * 0.5 - 8;

    // build the surviving bands (fills global spoke_dirs / spoke_amp / spoke_bandidx)
    build_bands();
    var n_parts  = spoke_dirs.length;
    var col      = ws_color(waveshape, 1.0);
    var r_inner  = 0.12;

    // tip positions current frame
    var tip_2d = [];
    for (var i = 0; i < n_parts; i++) {
        var d   = spoke_dirs[i];
        var len = spoke_length(i, time);
        var p3  = transform(d.x * len, d.y * len, d.z * len);
        var p2  = project(p3, cx, cy, R);
        tip_2d.push(p2);
    }

    // back-to-front order
    var order = [];
    for (var i = 0; i < n_parts; i++) order.push(i);
    order.sort(function(a, b) { return tip_2d[a].z - tip_2d[b].z; });

    // ── ghost trail ───────────────────────────────────────
    if (speed > 0.5) {
        var ghost_alpha = Math.min(0.18, speed_to_rate(speed) * 2.0);
        var ghost_tip_2d = [];
        for (var i = 0; i < n_parts; i++) {
            var d   = spoke_dirs[i];
            var len = spoke_length(i, ghost_time);
            var p3  = transform_with(d.x*len, d.y*len, d.z*len, ghost_rot_y, ghost_rot_x);
            var p2  = project(p3, cx, cy, R);
            ghost_tip_2d.push(p2);
        }

        for (var oi = 0; oi < order.length; oi++) {
            var i   = order[oi];
            var dir = spoke_dirs[i];
            var len = spoke_length(i, ghost_time);

            var dot_up = Math.abs(dir.y);
            var ref    = dot_up > 0.9 ? {x:1,y:0,z:0} : {x:0,y:1,z:0};
            var perp   = {
                x: dir.y*ref.z - dir.z*ref.y,
                y: dir.z*ref.x - dir.x*ref.z,
                z: dir.x*ref.y - dir.y*ref.x
            };
            var plen = Math.sqrt(perp.x*perp.x + perp.y*perp.y + perp.z*perp.z);
            perp.x /= plen; perp.y /= plen; perp.z /= plen;

            var c = col.slice();
            g.set_source_rgba(c[0]*0.6, c[1]*0.6, c[2], ghost_alpha);
            g.set_line_width(0.6);

            var d0   = spoke_disp(0.0, waveshape, i);
            var p3_0 = transform_with(
                dir.x * r_inner + perp.x * d0,
                dir.y * r_inner + perp.y * d0,
                dir.z * r_inner + perp.z * d0,
                ghost_rot_y, ghost_rot_x
            );
            var s0 = project(p3_0, cx, cy, R);
            g.move_to(s0.x, s0.y);

            for (var s = 1; s <= 24; s++) {
                var t  = s / 24;
                var r  = r_inner + t * (len - r_inner);
                var d  = spoke_disp(t, waveshape, i);
                var p3 = transform_with(
                    dir.x * r + perp.x * d,
                    dir.y * r + perp.y * d,
                    dir.z * r + perp.z * d,
                    ghost_rot_y, ghost_rot_x
                );
                var sp = project(p3, cx, cy, R);
                g.line_to(sp.x, sp.y);
            }
            g.stroke();
        }
    }

    // ── concentric inner rings ────────────────────────────
    var n_rings = 3;
    for (var ri = 0; ri < n_rings; ri++) {
        var ring_r   = (ri + 1) / (n_rings + 1) * R;
        var ring_a   = 0.03 + ri * 0.02;
        var ring_pulse = 1.0 + Math.sin(time * 1.3 + ri * 1.1) * 0.04;
        g.set_source_rgba(col[0], col[1], col[2], ring_a);
        g.set_line_width(0.4);
        g.arc(cx, cy, ring_r * ring_pulse, 0, Math.PI * 2.0);
        g.stroke();
    }

    // ── rim connections ───────────────────────────────────
    var drawn_edges = {};
    for (var oi = 0; oi < order.length; oi++) {
        var i  = order[oi];
        var p2 = tip_2d[i];

        var dists = [];
        for (var j = 0; j < n_parts; j++) {
            if (j == i) continue;
            var da  = spoke_dirs[i];
            var db  = spoke_dirs[j];
            var dot = da.x*db.x + da.y*db.y + da.z*db.z;
            dists.push({j: j, dot: dot});
        }
        dists.sort(function(a, b) { return b.dot - a.dot; });

        var n_connect = Math.min(3, dists.length);
        for (var k = 0; k < n_connect; k++) {
            var j   = dists[k].j;
            var key = Math.min(i,j) + "_" + Math.max(i,j);
            if (drawn_edges[key]) continue;
            drawn_edges[key] = true;

            var q2    = tip_2d[j];
            var avg_z = (p2.z + q2.z) * 0.5;
            var depth_t = (avg_z + 1.0) * 0.5;
            var ea    = 0.03 + depth_t * 0.08;

            // rim glow
            g.set_source_rgba(col[0], col[1], col[2], ea * 0.4);
            g.set_line_width(2.5);
            g.move_to(p2.x, p2.y);
            g.line_to(q2.x, q2.y);
            g.stroke();

            g.set_source_rgba(col[0], col[1], col[2], ea);
            g.set_line_width(0.4);
            g.move_to(p2.x, p2.y);
            g.line_to(q2.x, q2.y);
            g.stroke();
        }
    }

    // ── spokes with bloom + taper ─────────────────────────
    for (var oi = 0; oi < order.length; oi++) {
        var i     = order[oi];
        var dir   = spoke_dirs[i];
        var p2tip = tip_2d[i];
        var len   = spoke_length(i, time);

        var depth_t = (p2tip.z + 1.0) * 0.5;

        var dot_up = Math.abs(dir.y);
        var ref    = dot_up > 0.9 ? {x:1,y:0,z:0} : {x:0,y:1,z:0};
        var perp   = {
            x: dir.y*ref.z - dir.z*ref.y,
            y: dir.z*ref.x - dir.x*ref.z,
            z: dir.x*ref.y - dir.y*ref.x
        };
        var plen = Math.sqrt(perp.x*perp.x + perp.y*perp.y + perp.z*perp.z);
        perp.x /= plen; perp.y /= plen; perp.z /= plen;

        draw_spoke_glow(g, dir, perp, r_inner, len, i, cx, cy, R,
                        depth_t, col, rot_y, rot_x, time);

        // tip dot with glow
        var dot_r   = 1.2 + depth_t * 1.8;
        var tip_col = ws_color(waveshape, 1.0);

        // outer glow
        g.set_source_rgba(tip_col[0], tip_col[1], tip_col[2], 0.08 * depth_t);
        g.arc(p2tip.x, p2tip.y, dot_r * 3.5, 0, Math.PI * 2.0);
        g.fill();

        // mid glow
        g.set_source_rgba(tip_col[0], tip_col[1], tip_col[2], 0.18 * depth_t);
        g.arc(p2tip.x, p2tip.y, dot_r * 2.0, 0, Math.PI * 2.0);
        g.fill();

        // core dot
        g.set_source_rgba(tip_col[0], tip_col[1], tip_col[2], 0.7 + depth_t * 0.3);
        g.arc(p2tip.x, p2tip.y, dot_r, 0, Math.PI * 2.0);
        g.fill();
    }

    // ── centre orb ───────────────────────────────────────
    var ctr = project(transform(0, 0, 0), cx, cy, R);

    // outer glow rings
    g.set_source_rgba(col[0], col[1], col[2], 0.04);
    g.arc(ctr.x, ctr.y, 12, 0, Math.PI * 2.0);
    g.fill();

    g.set_source_rgba(col[0], col[1], col[2], 0.10);
    g.arc(ctr.x, ctr.y, 6, 0, Math.PI * 2.0);
    g.fill();

    g.set_source_rgba(col[0], col[1], col[2], 0.25);
    g.arc(ctr.x, ctr.y, 3.5, 0, Math.PI * 2.0);
    g.fill();

	g.set_source_rgba(col[0], col[1], col[2], 0.9);
	g.arc(ctr.x, ctr.y, 1.8, 0, Math.PI * 2.0);
	g.fill();
}

set_speed(1.0);
