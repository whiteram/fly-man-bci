"""Looming stimulus generator (threat/escape paradigm).

Three classes, 3.0 s @ 60 fps, full-field (both eyes sample the whole
frame, bci-branch convention):
  loom    dark disk EXPANDS (2 -> 60 px over 300-2300 ms, then holds)
          -- the classical approaching-object (l/v collision-avoidance)
          stimulus that drives LC4/LPLC -> DN giant-fiber escape
  recede  dark disk SHRINKS (60 -> 2 px, same window) -- receding
          control
  static  constant mid-size disk (30 px) -- luminance-matched control

Usage:
  python bci/looming/make_stimulus.py --all
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
DISK = 20                      # dark object
R0, R1 = 2.0, 60.0             # start/end radius (px)
EXP0, EXP1 = 0.3, 2.3          # expansion window (s)
CLASSES = ["loom", "recede", "static"]


def frames_for(cls):
    n = int(DUR_S * FPS)
    t = np.arange(n) / FPS
    r = np.full(n, R0)
    if cls == "loom":
        r = R0 + (R1 - R0) * np.clip((t - EXP0) / (EXP1 - EXP0), 0, 1)
    elif cls == "recede":
        r = R1 - (R1 - R0) * np.clip((t - EXP0) / (EXP1 - EXP0), 0, 1)
    elif cls == "static":
        r = np.full(n, 30.0)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
    img = np.full((n, H, W), BASE, np.float64)
    for i in range(n):
        img[i][d <= r[i]] = DISK
    return np.clip(img, 0, 255).astype(np.uint8)


def make(cls, catalog=True):
    frames = frames_for(cls)
    out = ROOT / "viz" / "data" / "stimuli" / "looming"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{cls}.npy", frames)
    if catalog:
        cat_path = ROOT / "viz" / "data" / "visual_inputs.json"
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        entry = {"id": f"loom_{cls}", "kind": "video",
                 "label": {"loom": "looming (expanding dark disk)",
                           "recede": "receding (shrinking disk)",
                           "static": "static disk"}[cls],
                 "source": f"data/stimuli/looming/{cls}.npy",
                 "fps": FPS, "blur_px": 0.0, "lo": 0.05, "hi": 3.0,
                 "linearize_srgb": False}
        keep = [v for v in cat["inputs"] if not v["id"].startswith("loom_")]
        cat["inputs"] = keep + [entry]
        cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        for c in CLASSES:
            make(c)
        print(f"looming: {len(CLASSES)} stimuli written")
    else:
        ap.error("--all required")
