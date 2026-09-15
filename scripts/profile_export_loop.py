"""A/B benchmark for the export biology loop (planning tool).

Assembles the FULL-region circuit ONCE, then runs the same short
simulation twice inside one process -- pure-numpy path vs numba-kernel
path (toggling ffbm.simulation._HAVE_NUMBA in place) -- timing only the
simulate() call in each mode.  Wall-clock ratios, no cProfile overhead.

Usage:  python scripts/profile_export_loop.py [T_MS]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp015_bilateral"))

import numpy as np

from ffbm import pipeline as fp
from ffbm import regions as freg
import ffbm.simulation as sim

T_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
REGIONS = ("visual_bilateral,vpn_central,ol_rest,central_brain,vnc").split(",")


def main():
    t0 = time.time()
    circuit, _ = freg.build_circuit({r: True for r in REGIONS})
    print(f"assembly: {time.time() - t0:.0f} s (one-time for both runs)",
          flush=True)
    cal = dict(fp.CAL)
    extra_post = {g: s["post"]
                  for g, s in (circuit.get("extra_edges") or {}).items()}

    store = {}

    def record(j, k, t, st, inc_f, sp):
        syn, pops = st["syn"], st["pops"]

        def y_of(name):
            if name == "RL":
                return syn[name].edge_currents(pops["L"].v)
            if name == "LM":
                return syn[name].edge_currents(pops["MID"].v)
            post = extra_post.get(name, "T45")
            return syn[name].edge_currents(pops[post].v)

        for name in (["RL", "LM"]
                     + [g for g in extra_post if extra_post[g] != "VNC"]
                     + [g for g in syn if g.startswith("MT_")]):
            y = y_of(name)
            buf = store.setdefault(name, np.zeros((300, y.size), np.float32))
            if j < buf.shape[0]:
                buf[j] = y

    # warm-up pass: page cache + numba JIT compile (excluded from timing)
    sim._HAVE_NUMBA = True
    t0 = time.time()
    fp.simulate(circuit, cal, lambda t: 0.0, 42, 40.0,
                on_sample=lambda *a: None)
    print(f"warmup (incl. JIT): {time.time() - t0:.0f} s", flush=True)

    results = {}
    for tag, flag in (("numba", True), ("numpy", False)):
        sim._HAVE_NUMBA = flag
        t0 = time.time()
        fp.simulate(circuit, cal, lambda t: 0.0, 42, T_MS, on_sample=record)
        wall = time.time() - t0
        n_rec = int(T_MS)
        results[tag] = wall
        print(f"{tag:6s}: {wall:6.1f} s for {T_MS:.0f} ms "
              f"({n_rec} records) -> full 10.5 s ~= "
              f"{wall * 10500 / T_MS / 60:.1f} min", flush=True)
    sp = results["numpy"] / results["numba"]
    print(f"speedup: {sp:.2f}x  "
          f"(full export est. {results['numba'] * 10500 / T_MS / 60:.0f} min "
          f"vs {results['numpy'] * 10500 / T_MS / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
