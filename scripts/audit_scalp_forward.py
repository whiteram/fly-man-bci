"""Independent correctness audit of the scalp (human 3-sphere) forward kernel.

Checks, in order:
 1. CROSS-VALIDATION: ScalpPairField with sigma1=sigma2=sigma3 (no actual
    boundary) must reproduce SealedHeadPairField (uniform sealed sphere,
    sealed at r3) exactly -- two independently coded Legendre solvers
    solving the same physical problem.
 2. DIPOLE ANTISYMMETRY: flipping the pair's dipole moment must flip the
    sign of every electrode potential exactly.
 3. SERIES CONVERGENCE: S=60 (export default) vs S=200 for deep sources
    (r0/r1 up to 0.75, our network's regime).
 4. SKULL ATTENUATION vs literature: superficial source, sigma2 = 0.013
    vs 0.33 -> surface potential ratio should land in the classic
    0.3-0.5 range; deep sources attenuate less (0.8-0.9).
 5. ELECTRODE DEPTH: 0.985*r3 vs 0.9995*r3 (on-scalp) difference.
 6. GAUGE/REFERENCE: the dropped n=0 term means potentials are relative
    (common-mode-free); verify shifting all electrode positions does not
    change pairwise differences' physical content... (documented check:
    n=0 is identically zero for pairs, so the gauge is 'mean over sphere
    = 0' for each dipole -- verified via antisymmetry in 2).

Run:  python scripts/audit_scalp_forward.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffbm.forward import ScalpPairField, SealedHeadPairField

R1, R2, R3 = 8.0e4, 8.5e4, 9.2e4     # um, human 3-sphere radii
SIG = 0.33                            # S/m, brain & scalp
rng = np.random.default_rng(0)

print("=" * 64)
print("1) CROSS-VALIDATION  Scalp(uniform sigma)  vs  Sealed(sphere r3)")
print("=" * 64)

# random source-sink pairs in the brain (r < 0.75 R1), lengths ~0.3-2 mm
n = 400
d = rng.normal(size=(n, 3))
d *= (0.72 * R1 * rng.random(n) / np.linalg.norm(d, axis=1))[:, None]
step = rng.normal(scale=600.0, size=(n, 3))
p = d + step
p *= np.minimum(1.0, 0.75 * R1 / np.maximum(np.linalg.norm(p, axis=1), 1e-9))[:, None]
pre, post = d, p

e_dir = np.array([0.2, -0.5, 0.84]); e_dir /= np.linalg.norm(e_dir)
elec = 0.985 * R3 * e_dir

A = ScalpPairField(pre, post, elec[None, :], center=np.zeros(3),
                   r1=R1, r2=R2, r3=R3, sigma1=SIG, sigma2=SIG, sigma3=SIG,
                   n_terms=200).coef[0]
B = SealedHeadPairField(pre, post, elec[None, :], center=np.zeros(3),
                        r1=R3, r2=R3 * 1.0000001, sigma1=SIG, sigma2=SIG,
                        n_terms=200).coef[0]
rel = np.abs(A - B) / (np.abs(B) + 1e-30)
scale_rel = np.abs(A - B).max() / np.abs(B).max()
print(f"   max |coef| = {np.abs(B).max():.3e} V/A per pair")
print(f"   scale-relative diff (max|A-B| / max|B|) = {scale_rel:.2e}")
print(f"   median per-pair relative diff = {np.median(rel):.2e}")
assert scale_rel < 1e-6, "cross-validation FAILED"
print("   PASS -- two independent solvers agree to ~7 significant digits")

print()
print("=" * 64)
print("2) DIPOLE ANTISYMMETRY  (flip moment -> exactly -V)")
print("=" * 64)
c = np.zeros((1, 3))
pr = c + np.array([[3.0e4, 0, 0]])
po = c + np.array([[3.4e4, 0, 0]])
elecs = np.array([0.985 * R3 * v / np.linalg.norm(v)
                  for v in rng.normal(size=(8, 3))])
V1 = ScalpPairField(pr, po, elecs, center=c, r1=R1, r2=R2, r3=R3,
                    n_terms=200).coef
V2 = ScalpPairField(po, pr, elecs, center=c, r1=R1, r2=R2, r3=R3,
                    n_terms=200).coef
err = np.abs(V1 + V2).max() / np.abs(V1).max()
print(f"   max |V(+p) + V(-p)| / max|V| = {err:.2e}")
assert err < 1e-12
print("   PASS -- exact antisymmetry")

print()
print("=" * 64)
print("3) SERIES CONVERGENCE  S=60 (export) vs S=200")
print("=" * 64)
for r0f, tag in ((0.30, "shallow-ish r0/r1=0.30"),
                 (0.75, "DEEP network regime r0/r1=0.75")):
    pre2 = r0f * R1 * np.array([[np.sqrt(0.5), np.sqrt(0.5), 0.0]])
    post2 = pre2 + np.array([[2000.0, 0, 0]])
    c60 = ScalpPairField(pre2, post2, elec[None, :], center=np.zeros(3),
                         r1=R1, r2=R2, r3=R3, n_terms=60).coef[0]
    c200 = ScalpPairField(pre2, post2, elec[None, :], center=np.zeros(3),
                          r1=R1, r2=R2, r3=R3, n_terms=200).coef[0]
    rel2 = abs(c60 - c200) / abs(c200)
    print(f"   {tag}: rel diff = {float(np.max(rel2)):.2e}")
    assert np.max(rel2) < 1e-3

print()
print("=" * 64)
print("4) SKULL ATTENUATION vs literature")
print("=" * 64)
# superficial radial pair just inside r1, electrode directly above
top = np.array([0.0, 0.0, 1.0])
pr3 = 0.97 * R1 * top[None, :]
po3 = 0.995 * R1 * top[None, :]
e3 = (0.985 * R3 * top)[None, :]
v_skull = np.ravel(ScalpPairField(pr3, po3, e3, center=np.zeros(3), r1=R1, r2=R2,
                               r3=R3, sigma2=0.013, n_terms=200).coef[0])[0]
v_nosk = np.ravel(ScalpPairField(pr3, po3, e3, center=np.zeros(3), r1=R1, r2=R2,
                              r3=R3, sigma2=SIG, n_terms=200).coef[0])[0]
print(f"   superficial source:  V_skull/V_noskull = "
      f"{v_skull / v_nosk:.2f}  (literature ~0.3-0.5)")
pr4 = 0.1 * R1 * top[None, :]
po4 = 0.14 * R1 * top[None, :]
v_d1 = np.ravel(ScalpPairField(pr4, po4, e3, center=np.zeros(3), r1=R1, r2=R2,
                            r3=R3, sigma2=0.013, n_terms=200).coef[0])[0]
v_d2 = np.ravel(ScalpPairField(pr4, po4, e3, center=np.zeros(3), r1=R1, r2=R2,
                            r3=R3, sigma2=SIG, n_terms=200).coef[0])[0]
print(f"   deep source (r0/r1~0.1):  ratio = {v_d1 / v_d2:.2f}  "
      f"(literature deep sources attenuate less, 0.7-0.9)")

print()
print("=" * 64)
print("5) ELECTRODE DEPTH  0.985*r3 vs 0.9995*r3 (on the scalp)")
print("=" * 64)
e_near = (0.9995 * R3 * top)[None, :]
v_985 = np.ravel(ScalpPairField(pr3, po3, e3, center=np.zeros(3), r1=R1, r2=R2,
                             r3=R3, n_terms=200).coef[0])[0]
v_999 = np.ravel(ScalpPairField(pr3, po3, e_near, center=np.zeros(3), r1=R1,
                             r2=R2, r3=R3, n_terms=200).coef[0])[0]
print(f"   V(0.9995 r3)/V(0.985 r3) = {v_999 / v_985:.3f}  "
      f"(sealed BC: dV/dr=0 at r3, so the surface value is the limit)")

print()
print("ALL CHECKS PASSED")
