"""P300 oddball stimulus generator (peripheral-adaptation version).

A full-field flash train (SOA 800 ms, flash 100 ms): frequent DIM
standards (gray 148) and rare BRIGHT targets (gray 228) at fixed
positions.  The phototransduction cascade adapts to the repeated
standard, so the rare target evokes a distinctly larger transient --
a deviance response.  DOCUMENTED honest framing: this is sensory-
adaptation deviance detection, not the cognitive parietal P300 (the
model has no attention system; human P300 requires only silent
counting, no motor response -- the missing piece here is cognition,
not action).

Protocol: 12 s trial = 15 events at k*800+200 ms; targets at events
4 and 11 (13% rarity).

Usage:
  python bci/p300/make_stimulus.py    # write video + catalog entry
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 60.0
DUR_S = 12.0
H, W = 72, 96
BASE = 128
STD = 148                    # standard flash level
TGT = 228                    # target flash level
SOA = 800.0                  # ms between event onsets
FLASH = 100.0                # ms flash width
EVENTS = 15
TARGETS = (4, 11)            # target event indices


def make():
    n = int(DUR_S * FPS)
    img = np.full((n, H, W), BASE, np.float64)
    for k in range(EVENTS):
        a = (k * SOA + 100) / 1000.0
        b = (k * SOA + 100 + FLASH) / 1000.0
        i0, i1 = int(round(a * FPS)), int(round(b * FPS))
        img[i0:i1] = TGT if k in TARGETS else STD
    frames = np.clip(img, 0, 255).astype(np.uint8)
    out = ROOT / "viz" / "data" / "stimuli" / "p300"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train.npy", frames)
    cat_path = ROOT / "viz" / "data" / "visual_inputs.json"
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    entry = {"id": "p300_train", "kind": "video",
             "label": "P300 oddball train (SOA 800 ms, targets @4,11)",
             "source": "data/stimuli/p300/train.npy",
             "fps": FPS, "blur_px": 0.0, "lo": 0.05, "hi": 3.0,
             "linearize_srgb": False}
    keep = [v for v in cat["inputs"] if not v["id"].startswith("p300_")]
    cat["inputs"] = keep + [entry]
    cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return frames


if __name__ == "__main__":
    make()
    print("p300 train written")
