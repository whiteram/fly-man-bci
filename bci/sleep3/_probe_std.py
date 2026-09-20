"""Careful per-spike probe of the GPU std delivery path (bci/sleep3).

Stamps one heavy CEN_C presynaptic cell, steps, and prints spike_row /
d / y after EVERY stamp-step (never after clear-steps).  Expectation
for U=0.9: d halves-of-nines downward 1 -> 0.1 -> 0.01 -> ... per
spike.
"""
import numpy as np
import cupy as cp

from ffbm import cache as fcache
from ffbm import pipeline as fp
from ffbm.params import cal
from ffbm import gpu as fgpu

circuit = fcache.load_circuit("693698e48fa257ee3380")
circuit["extra_edges"]["CEN_C"]["std"] = [0.9, 20000.0]
st = fp.build_stack(circuit, cal(), np.random.default_rng(1))
t = fgpu.GPUTrial(st)
pg = t.exp["CEN_C"]
local = pg.local.get()
cnts = pg.counts.get()
starts = pg.starts.get()
flat = pg.flat_idx.get()
off = t._mask_offsets["CEN"]
r = int(np.argmax(cnts))
li = int(local[r])
e = int(flat[starts[r]])
print(f"row {r} local {li} edges {cnts[r]} first-edge {e}")
for it in range(4):
    t.mask_all[off + li] = 1
    pg.step(t.K, 0.0)
    sr = int(pg.spike_row.get()[r])
    dv = float(pg.std_d.get()[e])
    yv = float(np.abs(pg.y.get()[e]))
    print(f"spike {it + 1}: spike_row {sr}  d[e] {dv:.6g}  |y[e]| {yv:.4g}")
    t.mask_all[off + li] = 0
    pg.step(t.K, 0.0)
    print(f"   after clear-step: d[e] {float(pg.std_d.get()[e]):.6g}")
