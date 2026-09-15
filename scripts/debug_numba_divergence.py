"""Find the first divergent record/group between numba on/off paths.

Runs the SAME short simulation twice (same seed) in one process by
monkey-toggling simulation._HAVE_NUMBA between runs, snapshotting every
group's edge currents at each record, and reporting the first record +
group where the paths differ.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp015_bilateral"))

import numpy as np

from ffbm import pipeline as fp
from ffbm import regions as freg
import ffbm.simulation as sim

circuit, _ = freg.build_circuit({"visual_bilateral": True})
cal = dict(fp.CAL)

T_MS = 120.0
snaps = {}


def run(tag):
    def record(j, k, t, st, inc_f, sp):
        syn, pops = st["syn"], st["pops"]
        row = {}
        for name in ("RL", "LM") + tuple(f"MT_{m}" for m in cal["MID_TAU_S"]):
            if name == "RL":
                row[name] = syn[name].edge_currents(pops["L"].v).copy()
            elif name == "LM":
                row[name] = syn[name].edge_currents(pops["MID"].v).copy()
            else:
                row[name] = syn[name].edge_currents(pops["T45"].v).copy()
        # also the drive consumed by the NEXT pop
        row["drive_LM"] = np.concatenate(
            [a.ravel() for a in syn["RL"].to_neuron_drive()])
        row["drive_T45"] = np.concatenate(
            [a.ravel() for a in syn["LM"].to_neuron_drive()])
        snaps.setdefault(tag, []).append(row)

    fp.simulate(circuit, cal, lambda t: 0.0, 42, T_MS, on_sample=record)


sim._HAVE_NUMBA = False
run("numpy")
sim._HAVE_NUMBA = True
run("numba")

a, b = snaps["numpy"], snaps["numba"]
print(f"records: {len(a)}")
for j, (ra, rb) in enumerate(zip(a, b)):
    for key in ra:
        if not np.array_equal(ra[key], rb[key]):
            d = np.abs(ra[key].astype(np.float64) - rb[key].astype(np.float64))
            print(f"FIRST DIVERGENCE record {j} group {key}: "
                  f"max {d.max():.3e} at idx {int(d.argmax())} "
                  f"({(d > 0).sum()}/{d.size} bins differ)")
            i = int(d.argmax())
            print(f"  numpy={ra[key][i]!r}  numba={rb[key][i]!r}")
            sys.exit(0)
print("NO DIVERGENCE")
