"""Drifting-grating stimulus generator (motion-direction paradigm).

Four directions (0/90/180/270 deg), full-field sinusoidal gratings
drifting at 4 Hz temporal frequency, spatial period 24 px, 3.0 s @
60 fps.  T4/T5 (the fly's direction cells) sit behind the bci video
input path via the standard cascade.

Usage:
  python bci/direction/make_stimulus.py --all
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 60.0
DUR_S = 3.0
H, W = 72, 96
BASE = 128
AMP = 60
LAMBDA = 24.0                     # spatial period (px)
TF = 4.0                          # temporal drift frequency (Hz)
DIRS = [0, 90, 180, 270]          # motion-axis degrees


def make(deg, catalog=True):
    n = int(DUR_S * FPS)
    t = np.arange(n) / FPS
    th = np.deg2rad(deg)
    yy, xx = np.mgrid[0:H, 0:W]
    spatial = 2 * np.pi * (xx * np.cos(th) + yy * np.sin(th)) / LAMBDA
    phase = spatial[..., None] - 2 * np.pi * TF * t     # (H, W, frames)
    img = BASE + AMP * np.sin(phase)
    frames = np.clip(img, 0, 255).astype(np.uint8)
    out = ROOT / "viz" / "data" / "stimuli" / "direction"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"d{deg:03d}.npy", frames)
    if catalog:
        cat_path = ROOT / "viz" / "data" / "visual_inputs.json"
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        entry = {"id": f"dir_{deg:03d}", "kind": "video",
                 "label": f"drifting grating {deg} deg (4 Hz)",
                 "source": f"data/stimuli/direction/d{deg:03d}.npy",
                 "fps": FPS, "blur_px": 0.0, "lo": 0.05, "hi": 3.0,
                 "linearize_srgb": False}
        keep = [v for v in cat["inputs"] if not v["id"].startswith("dir_")]
        cat["inputs"] = keep + [entry]
        cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        for d in DIRS:
            make(d)
        print(f"direction: {len(DIRS)} stimuli written")
    else:
        ap.error("--all required")
