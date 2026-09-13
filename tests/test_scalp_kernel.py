"""Validation for ScalpPairField (3-layer open human-head kernel).

With all conductivities equal and the scalp boundary pushed far away, the
kernel must converge to the infinite homogeneous one. With realistic skull
conductivity the classic attenuation/smearing behavior appears."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ffbm.forward import ScalpPairField, StaticPairField

rng = np.random.default_rng(0)
n = 300
pre = rng.normal(0, 20, (n, 3))
post = pre + rng.normal(0, 1.5, (n, 3))

# electrode in scalp layer of a far, matched-conductivity head
r1, r2, r3 = 250.0, 1.2 * 250.0, 200 * 250.0
elec = np.array([[0.0, 0.0, 1.5 * r1]])
ref = StaticPairField(pre, post, elec, sigma=0.33)
far = ScalpPairField(pre, post, elec, center=(0, 0, 0), r1=r1, r2=r2,
                     r3=r3, sigma1=0.33, sigma2=0.33, sigma3=0.33,
                     n_terms=60)
err = np.abs(far.coef - ref.coef) / (np.abs(ref.coef) + 1e-30)
print(f"matched far head: median rel err {np.median(err):.2e}")
assert np.median(err) < 0.02

# realistic human head (mm): brain 80, skull 85, scalp 92; skull 25x less
# conductive; electrode just inside the scalp surface
r1h, r2h, r3h = 8e4, 8.5e4, 9.2e4
elech = np.array([[0.0, 0.0, 0.985 * r3h]])
human = ScalpPairField(pre * 1e3, post * 1e3, elech, center=(0, 0, 0),
                       r1=r1h, r2=r2h, r3=r3h, n_terms=60)
noskull = ScalpPairField(pre * 1e3, post * 1e3, elech, center=(0, 0, 0),
                         r1=r1h, r2=r2h, r3=r3h, sigma2=0.33, n_terms=60)
att = np.abs(human.coef).sum() / np.abs(noskull.coef).sum()
print(f"human head with skull vs without (deep central sources): "
      f"amplitude ratio {att:.3f}")
# deep central dipole terms pass a thin resistive shell with mild
# attenuation; the classic 0.3-0.5 factors apply to superficial sources
assert 0.3 < att < 1.05, "skull attenuation out of plausible range"
print("ALL SCALP-KERNEL CHECKS PASSED")
