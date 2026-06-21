"""Groove / rhythmic-structure persistence metric — captures "groove comes and goes".

On the rendered audio's onset-strength envelope, slide a window and per window measure PULSE CLARITY
(max autocorrelation in the beat lag range). A sustained groove => high pulse clarity in MOST windows
and a STABLE dominant period. "Coming in and out" => high variance / many low-clarity windows.

Reports per signal:
  clarity_mean   — average pulse clarity across windows (overall grooviness)
  clarity_lo     — fraction of windows BELOW a clarity threshold (the "groove drops out" fraction)
  period_stab    — 1 - normalized std of the dominant beat period across windows (tempo stability)
"""
import sys
sys.path.insert(0, '/home/btadeusz/code/ddsp_vae')
import numpy as np
import librosa


def groove_stats(mono, sr=48000, win_s=4.0, hop_s=1.0, bpm_lo=60, bpm_hi=180, clarity_thresh=0.25):
    onset = librosa.onset.onset_strength(y=mono.astype(np.float32), sr=sr, hop_length=512)
    fps = sr / 512.0
    lag_lo = int(fps * 60.0 / bpm_hi)   # smallest beat period (fastest tempo)
    lag_hi = int(fps * 60.0 / bpm_lo)   # largest beat period (slowest tempo)
    w = int(win_s * fps); h = int(hop_s * fps)
    clar, per = [], []
    for s in range(0, max(1, len(onset) - w), h):
        seg = onset[s:s + w]
        seg = seg - seg.mean()
        denom = (seg * seg).sum() + 1e-9
        ac = np.array([(seg[:-l] * seg[l:]).sum() / denom for l in range(lag_lo, lag_hi)])
        if len(ac) == 0:
            continue
        k = int(np.argmax(ac))
        clar.append(float(ac[k])); per.append(lag_lo + k)
    clar = np.array(clar); per = np.array(per, dtype=float)
    if len(clar) == 0:
        return dict(clarity_mean=0.0, clarity_lo=1.0, period_stab=0.0)
    return dict(
        clarity_mean=float(clar.mean()),
        clarity_lo=float((clar < clarity_thresh).mean()),
        period_stab=float(max(0.0, 1.0 - per.std() / (per.mean() + 1e-9))),
    )
