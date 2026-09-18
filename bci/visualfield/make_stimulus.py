"""bci/visualfield stimulus generator: monocular flicker halves.

Two videos (3.6 s @ 60 fps): the LEFT half of the frame flickers at
10 Hz (square wave) while the right half stays at BASE, and the
mirror image.  Combined with the new eye_map "half_split" mode in
export_data (each lobe samples its own half of the frame), vf_left
drives the RIGHT lobe only and vf_right the LEFT lobe only -- a
monocular/hemifield paradigm; the class = stimulated eye.

Usage:
  python bci/visualfield/make_stimulus.py    # both videos + catalog
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 60.0
DUR_S = 3.6
H, W = 72, 96
BASE = 128
AMP = 60.0
FFREQ = 10.0                     # flicker Hz


def make(side, catalog=True):
    n = int(DUR_S * FPS)
    t = np.arange(n) / FPS
    flick = (np.sign(np.sin(2 * np.pi * FFREQ * t)) + 1) / 2
    img = np.full((n, H, W), BASE, np.float64)
    if side == "left":
        img += AMP * flick[:, None, None] * \
            (np.arange(W)[None, None, :] < W // 2)
    else:
        img += AMP * flick[:, None, None] * \
            (np.arange(W)[None, None, :] >= W // 2)
    frames = np.clip(img, 0, 255).astype(np.uint8)
    out = ROOT / "viz" / "data" / "stimuli" / "visualfield"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{side}.npy", frames)
    if catalog:
        cat_path = ROOT / "viz" / "data" / "visual_inputs.json"
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        entry = {"id": f"vf_{side}", "kind": "video",
                 "label": f"{side}-half 10 Hz flicker "
                          f"(eye_map half_split)",
                 "source": f"data/stimuli/visualfield/{side}.npy",
                 "fps": FPS, "blur_px": 0.0, "lo": 0.05, "hi": 3.0,
                 "linearize_srgb": False, "eye_map": "half_split"}
        keep = [v for v in cat["inputs"] if not v["id"].startswith("vf_")]
        cat["inputs"] = keep + [entry]
        cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return frames


if __name__ == "__main__":
    for s in ("left", "right"):
        make(s)
    print("visualfield: 2 stimuli written")
