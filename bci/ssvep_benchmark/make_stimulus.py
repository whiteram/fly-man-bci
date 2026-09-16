"""SSVEP-Benchmark stimulus generator (Wang et al. 2016 paradigm).

40 targets on a 5x8 grid flicker sinusoidally at 8-15.8 Hz (0.2 Hz
steps) with joint frequency-phase coding (phase offset 0.5 s x column
index, as in the benchmark's 0.5 s phase gap).  Deviations from the
human benchmark, documented:
  - sinusoidal (not square-wave) luminance modulation -- cleaner
    spectrum for a detection pilot;
  - filled slots instead of characters (fly has no文字);
  - 10.5 s per trial (the pipeline's protocol length) instead of 6 s,
    and 60 fps frame rate (the benchmark's monitor refresh);
  - both fly eyes sample the WHOLE grid (per-eye full-frame mapping),
    so both optic lobes are driven by all 40 frequencies at once --
    "attention" is not modelled; the question is whether every flicker
    frequency is detectable in the simulated EEG.

Usage:
  python bci/ssvep_benchmark/make_stimulus.py --all       # 40 stimuli
  python bci/ssvep_benchmark/make_stimulus.py --freq 12.0 # single
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 60.0
DUR_S = 10.5
H, W = 72, 96
ROWS, COLS = 5, 8
F0, DF = 8.0, 0.2
PHASE_STEP = 0.5           # s of phase lag per column (benchmark-style)
BASE = 128                 # mean gray
AMP = 60                   # flicker amplitude (gray levels)


def freqs():
    return [round(F0 + i * DF, 1) for i in range(ROWS * COLS)]


def slot_rect(i):
    r, c = divmod(i, COLS)
    y0 = int(r * H / ROWS + 0.12 * H / ROWS)
    y1 = int((r + 1) * H / ROWS - 0.12 * H / ROWS)
    x0 = int(c * W / COLS + 0.12 * W / COLS)
    x1 = int((c + 1) * W / COLS - 0.12 * W / COLS)
    return y0, y1, x0, x1


def make(freq, catalog=True, single=False):
    n = int(DUR_S * FPS)
    t = np.arange(n) / FPS
    img = np.full((n, H, W), BASE, np.float64)
    bg = 18 * np.sin(np.arange(H)[:, None] / 9.0
                     * np.cos(np.arange(W)[None, :] / 11.0))
    img += bg[None]
    targets = freqs() if not single else [freq]
    for f_tgt in targets:
        i = freqs().index(f_tgt)
        y0, y1, x0, x1 = slot_rect(i)
        phase = 2 * np.pi * f_tgt * PHASE_STEP * (i % COLS)
        img[:, y0:y1, x0:x1] = BASE + AMP * np.sin(
            2 * np.pi * f_tgt * t + phase)[:, None, None]
    frames = np.clip(img, 0, 255).astype(np.uint8)
    out = ROOT / "viz" / "data" / "stimuli" / "ssvep"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"f{freq:.1f}" if not single else f"single_f{freq:.1f}"
    np.save(out / f"{stem}.npy", frames)
    if catalog:
        cat_path = ROOT / "viz" / "data" / "visual_inputs.json"
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        sid = f"ssvep1_f{freq:.1f}" if single else f"ssvep_f{freq:.1f}"
        entry = {"id": sid, "kind": "video",
                 "label": f"SSVEP {freq} Hz" + (" single-target" if single
                                                else " all-40"),
                 "source": f"data/stimuli/ssvep/{stem}.npy",
                 "fps": FPS, "blur_px": 0.0, "lo": 0.05, "hi": 3.0,
                 "linearize_srgb": False}
        keep = [v for v in cat["inputs"]
                if not v["id"].startswith(("ssvep_", "ssvep1_"))]
        cat["inputs"] = keep + [entry]
        cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--freq", type=float)
    a = ap.parse_args()
    if a.all:
        for f in freqs():
            make(f)
            print(f"ssvep f{f:.1f}")
    elif a.freq:
        make(a.freq)
        print(f"ssvep f{a.freq:.1f}")
    else:
        ap.error("--all or --freq required")
