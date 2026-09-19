"""bci/blocking analysis: did A+ pre-training block B?

Scalp: probe net response per trial (phase-1 target probe, then the
B-probe curve across phase-2 trials).  Mechanism: final-state w_scale
decomposed by presynaptic KC subtype (state files carry per-edge pre
ids).  The blocking statistic is the blocked-vs-control difference of
the B-learning measure, with session-label permutation.

Usage:
    python bci/blocking/analyze.py [--pilot]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
A, B = 4300, 4700
SUB = {"A": "KCg-m", "B": "KCab-m", "C": "KCab-s"}


def net(phi, a, b, pre=200):
    return float(np.sqrt((phi[a:b] ** 2).mean())
                 - np.sqrt((phi[a - pre:a] ** 2).mean()))


def w_by_subtype(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    w, types = z["w"], z["pre_type"]
    return {str(t): float(w[types == t].mean())
            for t in np.unique(types) if str(t) != "?"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    prefix = "pilot_" if args.pilot else ""
    meta = json.loads((OUT / f"{prefix}meta.json").read_text(
        encoding="utf-8"))
    n1, n2 = meta["n_p1"], meta["n_p2"]
    curves, wfinal = {}, {}
    for cond in meta["conditions"]:
        curves[cond], wfinal[cond] = [], []
        for r in range(meta["sessions"]):
            p1 = [net(np.load(OUT / f"{prefix}{cond}_r{r}_p1{k}.npy")
                      * 1e6 * 1.7, A, B) for k in range(n1)]
            p2 = [net(np.load(OUT / f"{prefix}{cond}_r{r}_p2{k}.npy")
                      * 1e6 * 1.7, A, B) for k in range(n2)]
            curves[cond].append((p1, p2))
            sp = OUT / "states" / f"{prefix}{cond}_r{r}_p2{n2 - 1}.npz"
            if sp.exists():
                wfinal[cond].append(w_by_subtype(sp))
    for cond in curves:
        print(f"\n== {cond} ==")
        for si, (p1, p2) in enumerate(curves[cond]):
            print(f" session {si} p1({SUB['A' if cond == 'blocked' else 'C']}"
                  f"-probe): " + " ".join(f"{v:+.4f}" for v in p1))
            print(f" session {si} p2(B-probe): "
                  + " ".join(f"{v:+.4f}" for v in p2))
        for si, w in enumerate(wfinal.get(cond, [])):
            print(f" session {si} final w: A({SUB['A']}) "
                  f"{w.get(SUB['A'], float('nan')):.3f}  "
                  f"B({SUB['B']}) "
                  f"{w.get(SUB['B'], float('nan')):.3f}  "
                  f"C({SUB['C']}) "
                  f"{w.get(SUB['C'], float('nan')):.3f}")
    if args.pilot:
        return
    # scalp-level: B-curve mean, blocked vs control
    b_bl = np.array([np.mean(c[1]) for c in curves["blocked"]])
    b_ct = np.array([np.mean(c[1]) for c in curves["control"]])
    obs = b_bl.mean() - b_ct.mean()
    pool = np.concatenate([b_bl, b_ct])
    rng = np.random.default_rng(3)
    n = len(b_bl)
    null = []
    for _ in range(5000):
        pm = rng.permutation(pool)
        null.append(pm[:n].mean() - pm[n:].mean())
    p_sc = (np.sum(np.abs(null) >= abs(obs)) + 1) / 5001
    print(f"\nscalp B-curve mean: blocked {b_bl.mean():+.4f} vs control "
          f"{b_ct.mean():+.4f} uV -> diff {obs:+.4f}, perm-p {p_sc:.4f}")
    # mechanism-level: w_B, blocked vs control
    if wfinal.get("blocked") and wfinal.get("control"):
        w_bl = np.array([w[SUB["B"]] for w in wfinal["blocked"]])
        w_ct = np.array([w[SUB["B"]] for w in wfinal["control"]])
        obs_w = w_bl.mean() - w_ct.mean()
        poolw = np.concatenate([w_bl, w_ct])
        nullw = []
        for _ in range(5000):
            pm = rng.permutation(poolw)
            nullw.append(pm[:n].mean() - pm[n:].mean())
        p_w = (np.sum(np.abs(nullw) >= abs(obs_w)) + 1) / 5001
        print(f"w_B final: blocked {w_bl.mean():.4f} vs control "
              f"{w_ct.mean():.4f} -> diff {obs_w:+.4f}, perm-p {p_w:.4f}")
        w_abl = np.array([w[SUB["A"]] for w in wfinal["blocked"]])
        print(f"w_A (blocked, pre-trained): {w_abl.mean():.4f} "
              f"[expect < 1: A learned in phase 1]")
    (OUT / f"{prefix}summary.json").write_text(json.dumps({
        "b_curve": {"blocked": [c[1] for c in curves["blocked"]],
                    "control": [c[1] for c in curves["control"]]},
        "w_final": wfinal}, indent=1))
    print(f"-> {OUT / f'{prefix}summary.json'}")


if __name__ == "__main__":
    main()
