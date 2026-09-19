"""bci/blocking2 analysis: does the error gate produce blocking?

Same readouts as bci/blocking (B-probe curves + final w by KC
subtype) plus the V-trajectory check (the gate's own bookkeeping).
Headline: blocked vs control final w_B -- with the window gate
(bci/blocking) both floored at 0.0200; with the RW gate the
prediction is blocked ~ 1.0 vs control << 1.

Usage:
    python bci/blocking2/analyze.py [--pilot]
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
    curves, wfinal, vfinal = {}, {}, {}
    for cond in meta["conditions"]:
        curves[cond], wfinal[cond] = [], []
        for r in range(meta["sessions"]):
            p2 = [net(np.load(OUT / f"{prefix}{cond}_r{r}_p2{k}.npy")
                      * 1e6 * 1.7, A, B) for k in range(n2)]
            curves[cond].append(p2)
            sp = OUT / "states" / f"{prefix}{cond}_r{r}_p2{n2 - 1}.npz"
            if sp.exists():
                wfinal[cond].append(w_by_subtype(sp))
        vp = OUT / f"{prefix}V_{cond}_r0.json"
        if vp.exists():
            vfinal[cond] = json.loads(vp.read_text(encoding="utf-8"))["V"]
    for cond in curves:
        print(f"\n== {cond} ==")
        for si, p2 in enumerate(curves[cond]):
            print(f" session {si} p2(B-probe): "
                  + " ".join(f"{v:+.4f}" for v in p2))
        for si, w in enumerate(wfinal.get(cond, [])):
            print(f" session {si} final w: A {w.get(SUB['A'], -1):.3f}"
                  f"  B {w.get(SUB['B'], -1):.3f}"
                  f"  C {w.get(SUB['C'], -1):.3f}")
        if cond in vfinal:
            print(f" final V: {vfinal[cond]}")
    if args.pilot:
        return
    w_bl = np.array([w[SUB["B"]] for w in wfinal["blocked"]])
    w_ct = np.array([w[SUB["B"]] for w in wfinal["control"]])
    obs = w_ct.mean() - w_bl.mean()
    pool = np.concatenate([w_bl, w_ct])
    rng = np.random.default_rng(4)
    n = len(w_bl)
    null = []
    for _ in range(5000):
        pm = rng.permutation(pool)
        null.append(pm[n:].mean() - pm[:n].mean())
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / 5001
    print(f"\nw_B final: blocked {w_bl.mean():.4f} vs control "
          f"{w_ct.mean():.4f} -> BLOCKING effect {obs:+.4f} "
          f"(control learns more), perm-p {p:.4f}")
    print("[reference] window-gate study (bci/blocking): both "
          "conditions floored at 0.0200 (no blocking)")
    (OUT / f"{prefix}summary.json").write_text(json.dumps({
        "b_curve": {c: [p2 for p2 in curves[c]]
                    for c in curves},
        "w_final": wfinal, "w_B_blocked": w_bl.tolist(),
        "w_B_control": w_ct.tolist(), "effect": obs, "p": p,
        "V_final": vfinal}, indent=1))
    print(f"-> {OUT / f'{prefix}summary.json'}")


if __name__ == "__main__":
    main()
