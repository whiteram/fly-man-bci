"""Geometry check for per-eye video sampling (no circuit assembly needed).

Question: does mapping each eye's R1-R6 cells in ITS OWN tangent-plane
basis (u_eye = that lobe's R-mean minus T45-mean, a = x orthogonalized,
b = u x a) give both eyes the same image orientation, so that "each eye
samples one complete copy of the frame" is well defined?

Checks, from the raw tables only (seconds):
  1. bases: print u/a/b per eye, angle between the two eye axes;
  2. face-on: each eye's own-basis footprint is compact (similar aspect
     in x/y), unlike the right eye projected onto the LEFT basis
     (edge-on sliver) -- the degenerate case the old union mapping had;
  3. same orientation: mirroring the right-lobe cells across the
     midsagittal (x = midline) plane and projecting them in the LEFT
     basis must reproduce the LEFT eye's footprint (percentile boxes
     overlap) -- head symmetry then guarantees both eyes sample an
     equally-oriented complete copy;
  4. preview simulation: per-eye percentile mapping onto a 96-wide grid
     -> pooled occupancy must span the full width (what the page's
     fly-eye view will show).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffbm import data as fdata  # noqa: E402

nodes = fdata.load_visual_nodes()
sites = fdata.load_neuron_sites()
pre_pos = fdata.site_positions(sites, "PreSyn")
types = nodes.set_index("bodyId")["type"]

r_ids = np.array(sorted(b for b in nodes.loc[
    nodes["type"] == "R1-R6", "bodyId"] if b in pre_pos), dtype=np.int64)
t45_ids = np.array(sorted(b for b in nodes.loc[
    nodes["type"].str.startswith(("T4", "T5")), "bodyId"]
    if b in pre_pos), dtype=np.int64)
r_pos = np.array([pre_pos[b] for b in r_ids])
t45_pos = np.array([pre_pos[b] for b in t45_ids])
print(f"R1-R6 cells with PreSyn sites: {len(r_ids):,}; "
      f"T4/T5: {len(t45_ids):,}")

# midsagittal split: 2-means on the pooled x (eyes are well separated)
x_all = np.concatenate([r_pos[:, 0], t45_pos[:, 0]])
lo, hi = x_all.min(), x_all.max()
mid = (lo + hi) / 2.0
for _ in range(50):
    mid_new = 0.5 * (x_all[x_all < mid].mean() + x_all[x_all >= mid].mean())
    if abs(mid_new - mid) < 1e-9:
        break
    mid = mid_new
side_r = r_pos[:, 0] < mid          # True = LEFT lobe (circuit convention)
side_t45 = t45_pos[:, 0] < mid
print(f"midline x={mid:.1f} um | left R {side_r.sum():,} / "
      f"right R {(~side_r).sum():,}")


def eye_axis(mask_r, mask_t):
    u = r_pos[mask_r].mean(axis=0) - t45_pos[mask_t].mean(axis=0)
    return u / np.linalg.norm(u)


def basis(u):
    a = np.array([1.0, 0.0, 0.0]) - u * u[0]
    a /= np.linalg.norm(a)
    return a, np.cross(u, a)


u_l = eye_axis(side_r, side_t45)      # == circuit["u_eye_left"]
u_r = eye_axis(~side_r, ~side_t45)
a_l, b_l = basis(u_l)
a_r, b_r = basis(u_r)
ang = np.degrees(np.arccos(np.clip(u_l @ u_r, -1, 1)))
print("\n[1] bases")
print(f"  u_eye_left = {np.round(u_l, 3)}   u_eye_right = {np.round(u_r, 3)}")
print(f"  angle between eye axes: {ang:.1f} deg "
      f"(mirror-symmetric pair -> ~180 - 2*tilt)")
print(f"  a_l={np.round(a_l, 3)} a_r={np.round(a_r, 3)} "
      f"(both x>0: same world horizontal)")
print(f"  b_l={np.round(b_l, 3)} b_r={np.round(b_r, 3)} "
      f"(z-signs same: same world vertical: "
      f"{np.sign(b_l[2]) == np.sign(b_r[2])})")


def box(x, y):
    xlo, xhi = np.percentile(x, [1, 99])
    ylo, yhi = np.percentile(y, [1, 99])
    return (xlo, xhi, ylo, yhi)


xl, yl = r_pos[side_r] @ a_l, r_pos[side_r] @ b_l
xr_leftbasis, yr_leftbasis = r_pos[~side_r] @ a_l, r_pos[~side_r] @ b_l
bl, br_lb = box(xl, yl), box(xr_leftbasis, yr_leftbasis)
print("\n[2] footprint (1-99 pct, um): left eye, right eye in the LEFT "
      "basis (edge-on, the degenerate case the union mapping had)")
for nm, b in (("L     ", bl), ("R left", br_lb)):
    print(f"  {nm}: x {b[0]:7.1f}..{b[1]:7.1f} (w {b[1]-b[0]:6.1f})  "
          f"y {b[2]:7.1f}..{b[3]:7.1f} (h {b[3]-b[2]:6.1f})  "
          f"aspect w/h {(b[1]-b[0])/(b[3]-b[2]):.2f}")

print("\n[3] orientation via mirror test: reflect right-lobe cells "
      "x -> 2*mid - x, project in LEFT basis")
mir = r_pos[~side_r].copy()
mir[:, 0] = 2 * mid - mir[:, 0]
bm = box(mir @ a_l, mir @ b_l)
print(f"  mirrored-R box: x {bm[0]:7.1f}..{bm[1]:7.1f}  "
      f"y {bm[2]:7.1f}..{bm[3]:7.1f}")
print(f"  left-eye  box: x {bl[0]:7.1f}..{bl[1]:7.1f}  "
      f"y {bl[2]:7.1f}..{bl[3]:7.1f}")
ov_x = max(0, min(bl[1], bm[1]) - max(bl[0], bm[0])) \
    / max(min(bl[1], bm[1]) - max(bl[0], bm[0]), 1e-9)
ov_y = max(0, min(bl[3], bm[3]) - max(bl[2], bm[2])) \
    / max(min(bl[3], bm[3]) - max(bl[2], bm[2]), 1e-9)
# symmetric centers are the meaningful check (footprints may differ in
# size by eye-shape asymmetry, not by orientation)
cx = abs((bl[0] + bl[1]) / 2 - (bm[0] + bm[1]) / 2)
cy = abs((bl[2] + bl[3]) / 2 - (bm[2] + bm[3]) / 2)
print(f"  box overlap x {ov_x:.0%}, y {ov_y:.0%}; center offsets "
      f"{cx:.1f} / {cy:.1f} um (orientation consistent if centers "
      f"coincide; a MIRRORED mapping would offset centers by ~the "
      f"footprint width)")

print("\n[4] preview simulation: mirror the right lobe about x=mid, one "
      "shared LEFT basis, per-eye 1-99 pct mapping, 96x72 grid")
mir = r_pos.copy()
mir[~side_r, 0] = 2 * mid - mir[~side_r, 0]
x_i = mir @ a_l
y_i = mir @ b_l
gw, gh = 96, 72
pxf = np.zeros(len(r_pos))
pyf = np.zeros(len(r_pos))
for m in (side_r, ~side_r):
    xlo, xhi = np.percentile(x_i[m], [1, 99])
    ylo, yhi = np.percentile(y_i[m], [1, 99])
    pxf[m] = (x_i[m] - xlo) / (xhi - xlo) * (gw - 1)
    pyf[m] = (y_i[m] - ylo) / (yhi - ylo) * (gh - 1)
occ = np.zeros((gh, gw), bool)
occ[np.clip(pyf, 0, gh - 1).astype(int),
    np.clip(pxf, 0, gw - 1).astype(int)] = True
cols = np.where(occ.any(axis=0))[0]
rows = np.where(occ.any(axis=1))[0]
print(f"  pooled occupancy spans cols {cols.min()}-{cols.max()} "
      f"({100 * cols.min() / gw:.0f}%-{100 * cols.max() / gw:.0f}% of "
      f"width), rows {rows.min()}-{rows.max()}; "
      f"fill {occ.mean():.0%}")
print("  PASS" if cols.min() <= 2 and cols.max() >= gw - 3 else "  FAIL")
