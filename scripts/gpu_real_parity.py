"""P3-3 real-data validation: GPU loop vs CPU simulate() on the actual
visual_bilateral circuit (both lobes, 8 MT groups), bitwise, plus a
first wall-clock measurement of the GPU loop.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from ffbm import regions as freg
from ffbm.pipeline import CAL, build_stack, simulate
from ffbm.gpu import GPUTrial

N_STEPS = 200                    # 100 ms of biology
COMPARE_AT = list(range(0, N_STEPS, 50))
SEED = 20260916

REGIONS = (sys.argv[1] if len(sys.argv) > 1 else "visual_bilateral")
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else N_STEPS
COMPARE_AT = list(range(0, N_STEPS, max(N_STEPS // 4, 1)))


def lum(t):
    return 260.0 * (1.0 + 0.5 * np.sin(2 * np.pi * t / 40.0))


def snapshot(st):
    s = {}
    for name, pop in st["pops"].items():
        s[f"pop.{name}.v"] = pop.v.copy()
    for name, pool in st["syn"].items():
        s[f"syn.{name}.y"] = pool.y.copy()
        if getattr(pool, "delayed", False):
            s[f"syn.{name}.buffer"] = pool.buffer.copy()
            s[f"syn.{name}.ptr"] = int(pool.ptr)
    return s


def main():
    regions = ({r.strip(): True for r in REGIONS.split(",")}
               if REGIONS != "all"
               else {r: True for r in freg.known_regions()})
    circuit, active = freg.build_circuit(regions)
    print(f"regions: {active}; n_r={len(circuit['r_ids'])} "
          f"n_l={len(circuit['l_ids'])} n_mid={len(circuit['mid_ids'])} "
          f"n_t45={len(circuit['t45_ids'])}")

    cpu_snaps = {}
    spikes_seen = {"T45": 0}

    def on_sample(j, k, t, st, inc_f, spikes):
        spikes_seen["T45"] += int(spikes["T45"].sum())
        if k in COMPARE_AT:
            cpu_snaps[k] = snapshot(st)

    t0 = time.perf_counter()
    simulate(circuit, CAL, lum, SEED, N_STEPS * 0.5, on_sample=on_sample)
    t_cpu = time.perf_counter() - t0
    print(f"CPU loop: {t_cpu:.2f} s for {N_STEPS} steps "
          f"({t_cpu / N_STEPS * 1e3:.2f} ms/step); "
          f"T45 spikes/record: {spikes_seen['T45'] / len(COMPARE_AT) * 20 / 2:.1f}")

    rng = np.random.default_rng(SEED)
    st = build_stack(circuit, CAL, rng)
    t0 = time.perf_counter()
    trial = GPUTrial(st)
    t_init = time.perf_counter() - t0

    gpu_snaps = {}

    def hook(k, t):
        if k in COMPARE_AT:
            gpu_snapshot(trial, gpu_snaps, k)

    t0 = time.perf_counter()
    trial.run(lum, N_STEPS, hook=hook)
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    t_gpu = time.perf_counter() - t0
    print(f"GPU init: {t_init:.2f} s; GPU loop: {t_gpu:.2f} s "
          f"({t_gpu / N_STEPS * 1e3:.2f} ms/step) "
          f"-> speedup {t_cpu / t_gpu:.2f}x")

    fails = 0
    for k in COMPARE_AT:
        c, g = cpu_snaps.get(k, {}), gpu_snaps.get(k, {})
        bad = []
        for key, ref in c.items():
            got = g.get(key)
            if got is None:
                bad.append(key + ":missing")
            elif isinstance(ref, (int, np.integer)):
                if int(ref) != int(got):
                    bad.append(f"{key}:{ref}!={got}")
            elif not np.array_equal(ref, got, equal_nan=True):
                d = np.count_nonzero(ref.view(np.uint8)
                                     != np.asarray(got).view(np.uint8))
                bad.append(f"{key}:{d}differ")
        if bad:
            print(f"step {k:4d}: BAD " + "; ".join(bad[:8]))
            fails += 1
    print("PARITY PASS" if not fails else "PARITY FAIL")
    return 1 if fails else 0


def gpu_snapshot(trial, out, k):
    import cupy as cp
    s = {}
    for name, p in trial.pops.items():
        s[f"pop.{name}.v"] = cp.asnumpy(p.v)
    for name, g in trial.graded.items():
        s[f"syn.{name}.y"] = cp.asnumpy(g.y)
    for name, g in trial.exp.items():
        s[f"syn.{name}.y"] = cp.asnumpy(g.y)
        if g.delayed:
            s[f"syn.{name}.buffer"] = cp.asnumpy(g.buffer)
            s[f"syn.{name}.ptr"] = int(g.ptr)
    out[k] = s


if __name__ == "__main__":
    sys.exit(main())
