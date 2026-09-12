"""Synthetic sanity checks for exp002 pieces (run before the big data pass)."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "run", Path(__file__).parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

# --- Phototransduction sanity (experiment timing: 3 s flash) ---
photo = run.Phototransduction(dt=0.5)
trace = np.array([photo.step(350.0) for _ in range(7000)])  # 3.5 s flash
peak = trace[:400].max()          # early max (before adaptation bites)
steady = trace[-1000:].mean()     # last 500 ms of a 3 s flash
print(f"photo: peak={peak:.1f} pA, late={steady:.1f} pA, "
      f"sag_ratio={steady / peak:.3f} (expect ~0.72)")
assert 0.65 < steady / peak < 0.80, "adaptation sag off"

i50 = int(np.argmax(trace >= 0.5 * 350.0))
print(f"photo: 50%-rise time = {i50 * 0.5:.1f} ms (expect ~15-20 ms)")
assert 10 < i50 * 0.5 < 25, "phototransduction delay off"

photo2 = run.Phototransduction(dt=0.5)
for _ in range(3000):
    photo2.step(350.0)               # 1.5 s light
rec = [photo2.step(0.0) for _ in range(100)]
print(f"photo: after 50 ms dark, inc={rec[-1]:.2f} pA "
      f"(cascaded tail, analytic ~10 pA)")
assert abs(rec[-1]) < 15.0

# --- match_edge_pairs sanity ---
P = {1: np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])}
Q = {2: np.array([[0.1, 0.2, 0.0],      # ~0.22 from t-bar 0
                  [5.2, 0.1, 0.0],      # ~0.22 from t-bar 1
                  [9.9, 0.0, 0.3],      # ~0.32 from t-bar 2
                  [0.0, 4.0, 0.0],      # 4.0 from t-bar 0
                  [5.0, 5.0, 0.0],      # 5.0 from t-bar 1
                  [10.0, 6.0, 0.0]])}   # 6.0 from t-bar 2
pp, qq, pe, dd = run.match_edge_pairs(
    P, Q, np.array([1]), np.array([2]), np.array([4.0]))
print(f"pairs: n={len(dd)}, edge_ids={pe}, dists={np.sort(np.round(dd, 3))}")
assert len(dd) == 4
s = np.sort(dd)
assert np.all(s[:3] < 0.35), "3 close PSDs must be selected"
assert np.isclose(s[3], 4.0, atol=0.1), "4th must be the closest far PSD"
for p, q in zip(pp, qq):
    dbar = np.linalg.norm(P[1] - q, axis=1)
    assert np.isclose(np.linalg.norm(p - q), dbar.min()), "not nearest t-bar"

pp2, qq2, pe2, dd2 = run.match_edge_pairs(
    P, Q, np.array([1]), np.array([2]), np.array([10.0]))
print(f"oversubscribed edge: n_pairs={len(dd2)} (6 PSDs, w=10)")
assert len(dd2) == 6

print("ALL SANITY CHECKS PASSED")
