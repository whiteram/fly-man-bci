"""Head-model parameter sensitivity (TECHNICAL.md §12 item B7).

The scalp amplitude is linear in the kernel, so for a FIXED synaptic
current state the amplitude under a parameter variant scales as the
kernel coefficient norm. This script builds the 4-sphere kernel for a
representative occipital source ensemble under literature-spread
variants (conductivities +-20%, layer thickness +-1 mm) and reports the
per-electrode |coef| ratio vs baseline. No simulation needed.

Run:  python scripts/head_sensitivity.py   (~1 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffbm.forward import FourSpherePairField

OUT = ROOT / "scripts" / "outputs"

BASE = dict(r1=7.8e4, r2=8.0e4, r3=8.5e4, r4=9.2e4,
            sigma1=0.33, sigma2=1.79, sigma3=0.013, sigma4=0.33)

VARIANTS = {
    "baseline": {},
    "skull_sigma_-20%": {"sigma3": 0.013 * 0.8},
    "skull_sigma_+20%": {"sigma3": 0.013 * 1.2},
    "csf_sigma_-20%": {"sigma2": 1.79 * 0.8},
    "csf_sigma_+20%": {"sigma2": 1.79 * 1.2},
    "brain_sigma_+20%": {"sigma1": 0.33 * 1.2, "sigma4": 0.33 * 1.2},
    "skull_thinner_1mm": {"r3": 8.4e4},
    "skull_thicker_1mm": {"r3": 8.6e4},
    "csf_thinner_0.5mm": {"r2": 7.95e4},
    "csf_thicker_0.5mm": {"r2": 8.05e4},
}

# representative occipital ensemble: sources in a blob centred 0.3*r1
# from the centre toward the pole (the shifted network's regime)
rng = np.random.default_rng(0)
POLE = np.array([0.0, 0.0, -1.0])
n_pairs = 600
pre = POLE[None, :] * 0.30 * BASE["r1"] + rng.normal(
    scale=0.15 * BASE["r1"], size=(n_pairs, 3))
post = pre + rng.normal(scale=800.0, size=(n_pairs, 3))
post *= np.minimum(1.0, 0.72 * BASE["r1"]
                   / np.maximum(np.linalg.norm(post, axis=1), 1e-9))[:, None]
ELEC_DIRS = np.array([[0.0, 0.0, -1.0],      # occipital pole (hot)
                      [0.7, 0.0, -0.7],      # 45 deg off pole
                      [1.0, 0.0, 0.0]])      # equator


def build(params, dirs):
    kw = {**BASE, **params}
    elecs = 0.985 * kw["r4"] * dirs
    return FourSpherePairField(pre, post, elecs, center=np.zeros(3),
                               **kw).coef


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = {}
    base_coef = build({}, ELEC_DIRS)
    base_norm = np.linalg.norm(base_coef, axis=1)
    for name, params in VARIANTS.items():
        coef = build(params, ELEC_DIRS)
        ratio = np.linalg.norm(coef, axis=1) / base_norm
        table[name] = [round(float(x), 3) for x in ratio]
        print(f"{name:22s} pole {ratio[0]:6.3f}  45deg {ratio[1]:6.3f}  "
              f"equator {ratio[2]:6.3f}")
    lo = min(min(v) for k, v in table.items() if k != "baseline")
    hi = max(max(v) for k, v in table.items() if k != "baseline")
    print(f"\namplitude envelope across variants: x{lo:.2f} .. x{hi:.2f}")
    (OUT / "head_sensitivity.json").write_text(json.dumps(
        {"electrodes": ["pole", "45deg", "equator"], "ratios": table,
         "envelope": [round(lo, 3), round(hi, 3)]}, indent=1))
    print(f"written to {OUT / 'head_sensitivity.json'}")


if __name__ == "__main__":
    main()
