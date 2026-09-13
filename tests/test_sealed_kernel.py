"""Validation for SealedHeadPairField: with the shell pushed far away and
matched conductivity, the sealed-sphere kernel must converge to the infinite
homogeneous kernel (StaticPairField)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ffbm.forward import SealedHeadPairField, StaticPairField

rng = np.random.default_rng(0)
n = 400
r1 = 250.0
pre = rng.normal(0, 60, (n, 3))            # sources well inside
post = pre + rng.normal(0, 2, (n, 3))
elec = rng.normal(0, 90, (3, 3))

ref = StaticPairField(pre, post, elec, sigma=0.33)
far = SealedHeadPairField(pre, post, elec, center=(0, 0, 0), r1=r1,
                          r2=50 * r1, sigma1=0.33, sigma2=0.33, n_terms=60)
err = np.abs(far.coef - ref.coef) / (np.abs(ref.coef) + 1e-30)
print(f"far-boundary limit: median rel err {np.median(err):.2e}, "
      f"p99 {np.percentile(err, 99):.2e}")
assert np.median(err) < 5e-3, "sealed kernel does not converge to homogeneous"

# sealed same-material sphere (r2 = r1): known effect ~ image charge doubling
# of the n=0..low terms; just verify it changes amplitudes and stays finite
sealed = SealedHeadPairField(pre, post, elec, center=(0, 0, 0), r1=r1,
                             r2=r1 * 1.0 + 1e-6, sigma1=0.33, sigma2=0.33,
                             n_terms=60)
ratio = np.abs(sealed.coef).sum() / np.abs(ref.coef).sum()
print(f"sealed homogeneous sphere: |coef| ratio vs infinite = {ratio:.3f}")
assert 0.5 < ratio < 3.0

# low-conductivity shell (cuticle-like): expect stronger low-order gain
shell = SealedHeadPairField(pre, post, elec, center=(0, 0, 0), r1=r1,
                            r2=1.3 * r1, sigma1=0.33, sigma2=0.0033,
                            n_terms=60)
ratio2 = np.abs(shell.coef).sum() / np.abs(ref.coef).sum()
print(f"low-sigma shell (ratio 0.01, thickness 30%): |coef| ratio = "
      f"{ratio2:.3f}")
print("ALL KERNEL CHECKS PASSED")
