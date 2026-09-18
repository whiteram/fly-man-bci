"""c-VEP stimulus generator (BETA paradigm, Liu et al. 2021 analog).

40 targets flicker with cyclically shifted m-sequences (code-VEP:
the target identity is in the code SHAPE, not the frequency).  Each
stimulus here = the FULL FIELD driven by one target's code -- the
single-target analog of the benchmark's 40-slot display (same
documented deviation as the SSVEP branch: both fly eyes sample the
whole frame, "attention" is not modelled; the question is whether the
code shape is decodable from simulated EEG).

Codes:
  base = 127-bit m-sequence (LFSR, primitive polynomial x^7+x+1,
         seed all-ones; balance check 64 ones / 63 zeros asserted)
  target k code = circular shift of base by 3k bits (127/40 ~ 3.2 --
         near-maximal separation, standard c-VEP spacing)
  luminance = BASE + AMP * (code bit mapped to -1/+1), sampled at the
         60 fps monitor refresh; the phototransduction cascade provides
         the band-limiting (BETA band-limits the code instead --
         documented deviation)

Video timeline per trial (3.0 s @ 60 fps): 30 frames lead-in at BASE,
127 frames code period, 23 frames tail at BASE; decode window = the
code period (500-2617 ms).

Usage:
  python bci/cvep/make_code.py --all        # 40 stimuli + catalog
  python bci/cvep/make_code.py --target 7   # single
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 60.0
N_BITS = 127
SHIFT_STEP = 3
N_TARGETS = 40
H, W = 72, 96
BASE = 128
AMP = 60
LEAD_FR, TAIL_FR = 30, 23


def msequence(n_bits=7):
    """127-bit m-sequence: first degree-7 primitive-polynomial LFSR
    (Fibonacci form) that passes the balance + full-cycle checks."""
    candidates = ([7, 1], [7, 3], [7, 3, 2, 1], [7, 4, 3, 2])
    for taps in candidates:
        reg = [1] * n_bits
        out = []
        for _ in range((1 << n_bits) - 1):
            fb = 0
            for t in taps:
                fb ^= reg[t - 1]
            out.append(reg[0])
            reg = reg[1:] + [fb]
        if sum(out) == 64 and reg == [1] * n_bits:
            return np.array(out, dtype=np.float64)
    raise AssertionError("no primitive polynomial produced an m-sequence")


def codes():
    base = msequence()
    return [np.roll(base, SHIFT_STEP * k) for k in range(N_TARGETS)]


def make(target, catalog=True):
    code = codes()[target]
    n = LEAD_FR + N_BITS + TAIL_FR
    img = np.full((n, H, W), BASE, np.float64)
    val = BASE + AMP * (2.0 * code - 1.0)    # bit 1 -> +AMP, 0 -> -AMP
    img[LEAD_FR:LEAD_FR + N_BITS] = val[:, None, None]
    frames = np.clip(img, 0, 255).astype(np.uint8)
    out = ROOT / "viz" / "data" / "stimuli" / "cvep"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"t{target:02d}.npy", frames)
    if catalog:
        cat_path = ROOT / "viz" / "data" / "visual_inputs.json"
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        entry = {"id": f"cvep_t{target:02d}", "kind": "video",
                 "label": f"c-VEP target {target} "
                          f"(m-seq shift {SHIFT_STEP * target})",
                 "source": f"data/stimuli/cvep/t{target:02d}.npy",
                 "fps": FPS, "blur_px": 0.0, "lo": 0.05, "hi": 3.0,
                 "linearize_srgb": False}
        keep = [v for v in cat["inputs"]
                if not v["id"].startswith("cvep_")]
        cat["inputs"] = keep + [entry]
        cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--target", type=int)
    a = ap.parse_args()
    if a.all:
        for k in range(N_TARGETS):
            make(k)
        print(f"cvep: {N_TARGETS} stimuli written")
    elif a.target is not None:
        make(a.target)
        print(f"cvep t{a.target:02d} written")
    else:
        ap.error("--all or --target required")
