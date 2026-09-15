"""Diagnostic profiler for the export biology loop (planning tool).

Splits the measured export runtime (2,843 s for 10.5 s biological time,
full CNS) into its two components:
  (a) per-step biology: LIF pops + exponential synapse decays + noise,
  (b) per-record forward: edge_currents over all kernel groups + f32
      buffer stores (10,500 records at 1 kHz).

Runs a SHORT full-region simulation twice under cProfile (with the same
record work the exporter does) and reports the cumulative split, so the
acceleration roadmap targets the right component.  No pipeline code is
modified.

Usage:  python scripts/profile_export_loop.py [T_MS]
"""

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "exp015_bilateral"))

import numpy as np

from ffbm import pipeline as fp
from ffbm import regions as freg

T_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
REGIONS = ("visual_bilateral,vpn_central,ol_rest,central_brain,vnc").split(",")


def main():
    t0 = time.time()
    circuit, active = freg.build_circuit({r: True for r in REGIONS})
    print(f"assembly: {time.time() - t0:.0f} s (regions: {', '.join(active)})",
          flush=True)
    extra_post = {g: s["post"]
                  for g, s in (circuit.get("extra_edges") or {}).items()}
    cal = dict(fp.CAL)

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
            buf = store.get(name)
            if buf is None:
                buf = store[name] = np.zeros((300, y.size), np.float32)
            if j < buf.shape[0]:
                buf[j] = y

    # warm pass (page cache, allocator)
    fp.simulate(circuit, cal, lambda t: 0.0, 42, 100.0,
                on_sample=lambda *a: None)

    t0 = time.time()
    pr = cProfile.Profile()
    pr.enable()
    fp.simulate(circuit, cal, lambda t: 0.0, 42, T_MS, on_sample=record)
    pr.disable()
    wall = time.time() - t0
    n_rec = int(T_MS / 1000 * 1000)
    print(f"profiled run: {wall:.1f} s for {T_MS:.0f} ms "
          f"({n_rec} records) -> full 10.5 s ~= {wall * 10500 / T_MS:.0f} s "
          "(cProfile inflates; ratios are what matter)", flush=True)

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(28)
    out = s.getvalue()
    print(out)
    # the record/step split: cumulative time under the record frame
    for line in out.splitlines():
        if "record" in line or "simulate" in line:
            print(line)


if __name__ == "__main__":
    main()
