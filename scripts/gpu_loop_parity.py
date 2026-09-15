"""P3-3 parity harness: GPU loop vs CPU simulate() on a tiny synthetic
circuit (fast iteration), BITWISE on every pool state.

Covers: PhotoCascade (CPU) -> R (explicit LIF) -> RL graded -> L (g32
LIF) -> LM graded -> MID (g32+base) -> 8 MT conductance groups with
delays (T45, g64 accumulation) -> one extra region fed by MID+T45.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from ffbm.pipeline import CAL, build_stack, simulate
from ffbm.gpu import GPUTrial

N_STEPS = 120
COMPARE_AT = list(range(0, N_STEPS, 2))   # on_sample fires on even steps


def make_circuit(seed=7):
    r = np.random.default_rng(seed)
    ids = {}
    for name, n in (("R", 800), ("L", 600), ("MID", 700), ("T45", 900),
                    ("X", 300)):
        ids[name] = np.sort(r.choice(np.arange(1, 5_000_000), n,
                                     replace=False))
    pos = {int(b): r.standard_normal(3) * 50
           for name in ids for b in ids[name]}

    def table(pre_pool, post_pool, m, signed):
        pre = r.choice(ids[pre_pool], m)
        post = r.choice(ids[post_pool], m)
        d = {"body_pre": pre, "body_post": post,
             "weight": r.random(m).astype(np.float32) * 0.05}
        if signed:
            d["sign"] = np.where(r.random(m) < 0.7, 1.0, -1.0)
        return pd.DataFrame(d)

    e_mt = {cell: table("MID", "T45", 2_500, True)
            for cell in CAL["MID_TAU_S"]}
    t45_type = r.choice(["T4a", "T4b", "T5a", "T5b"], len(ids["T45"]))
    return {
        "pre_pos": pos, "post_pos": pos,
        "r_ids": ids["R"], "l_ids": ids["L"], "mid_ids": ids["MID"],
        "t45_ids": ids["T45"], "t45_type": t45_type,
        "e_rl": table("R", "L", 3_000, False),
        "e_lm": table("L", "MID", 4_000, True),
        "e_mt": e_mt,
        "extra_pops": {"X": {"ids": ids["X"], "tau_ms": 15.0,
                             "t_refrac_ms": 2.0, "i_base": 4.0,
                             "ou_sigma": 30.0}},
        "extra_edges": {"XIN": {"pre": ("MID", "T45"), "post": "X",
                                "table": table("MID", "X", 3_500, True),
                                "tau_s": 12.0, "g_unit": 0.01}},
    }


def lum(t):
    # large enough that R.v enters the release map range [-59, -25] mV,
    # so the graded pools and downstream cascade carry nontrivial signal
    return 420.0 * (1.0 + 0.6 * np.sin(2 * np.pi * t / 60.0))


def snapshot(st):
    s = {}
    for name, pop in st["pops"].items():
        s[f"pop.{name}.v"] = pop.v.copy()
        s[f"pop.{name}.refrac"] = pop.refrac.copy()
    for name, pool in st["syn"].items():
        s[f"syn.{name}.y"] = pool.y.copy()
        if hasattr(pool, "s"):
            s[f"syn.{name}.s"] = pool.s.copy()
        if getattr(pool, "delayed", False):
            s[f"syn.{name}.buffer"] = pool.buffer.copy()
            s[f"syn.{name}.ptr"] = int(pool.ptr)
    for name, nz in st["noises"].items():
        s[f"noise.{name}.x"] = nz.x.copy()
    return s


def gpu_snapshot(trial):
    import cupy as cp
    s = {}
    for name, p in trial.pops.items():
        s[f"pop.{name}.v"] = cp.asnumpy(p.v)
        s[f"pop.{name}.refrac"] = cp.asnumpy(p.refrac)
    for name, g in trial.graded.items():
        s[f"syn.{name}.y"] = cp.asnumpy(g.y)
        s[f"syn.{name}.s"] = cp.asnumpy(g.s)
    for name, g in trial.exp.items():
        s[f"syn.{name}.y"] = cp.asnumpy(g.y)
        if g.delayed:
            s[f"syn.{name}.buffer"] = cp.asnumpy(g.buffer)
            s[f"syn.{name}.ptr"] = int(g.ptr)
    for name, (_, _, x) in trial._ou.items():
        s[f"noise.{name}.x"] = cp.asnumpy(x)
    return s


def main():
    seed = 20260916
    circuit = make_circuit()

    cpu_snaps = {}

    def on_sample(j, k, t, st, inc_f, spikes):
        if k in COMPARE_AT:
            cpu_snaps[k] = snapshot(st)

    simulate(circuit, CAL, lum, seed, N_STEPS * 0.5, on_sample=on_sample)

    gpu_snaps = {}

    def hook(k, t):
        if k in COMPARE_AT:
            gpu_snaps[k] = gpu_snapshot(trial)

    rng = np.random.default_rng(seed)
    st = build_stack(circuit, CAL, rng)
    trial = GPUTrial(st)
    trial.run(lum, N_STEPS, hook=hook)

    fails = 0
    for k in COMPARE_AT:
        c, g = cpu_snaps.get(k, {}), gpu_snaps.get(k, {})
        if not c or not g:
            print(f"step {k:4d}: MISSING SNAPSHOT cpu={bool(c)} gpu={bool(g)}")
            fails += 1
            continue
        bad = []
        for key, ref in c.items():
            got = g.get(key)
            if got is None:
                bad.append(f"{key}:missing")
                continue
            if isinstance(ref, (int, np.integer)):
                if int(ref) != int(got):
                    bad.append(f"{key}:{ref}!={got}")
            elif ref.shape != np.shape(got) or not np.array_equal(
                    ref, got, equal_nan=True):
                where = np.nonzero(ref.view(np.uint8)
                                   != np.asarray(got).view(np.uint8))
                n = where[0].size if where else -1
                bad.append(f"{key}:{n}differ")
        if bad:
            print(f"step {k:4d}: BAD " + "; ".join(bad[:6])
                  + (" ..." if len(bad) > 6 else ""))
            fails += 1
    # nontrivial-signal sanity: the graded pools must carry real state
    s_rl = cpu_snaps[COMPARE_AT[-1]].get("syn.RL.s")
    print(f"sanity |RL.s| mean={np.abs(s_rl).mean():.3e} "
        f"max={np.abs(s_rl).max():.3e}; "
        f"T45 v range=[{cpu_snaps[COMPARE_AT[-1]]['pop.T45.v'].min():.1f},"
        f"{cpu_snaps[COMPARE_AT[-1]]['pop.T45.v'].max():.1f}]")
    n_ok = len(COMPARE_AT) - fails
    print(f"{n_ok}/{len(COMPARE_AT)} steps bitwise; "
          + ("PARITY PASS" if not fails else "PARITY FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
