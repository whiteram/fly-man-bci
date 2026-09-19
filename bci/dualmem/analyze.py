"""bci/dualmem analysis: gamma vs alpha/beta extinction dissociation.

Per trial two probe nets: gamma window (4300-4400) and alpha/beta
window (4700-4800), each vs its own 200 ms pre-window.  Curves per
session; statistic = extinction-phase slope difference (gamma minus
alpha/beta), permutation over session-level label swaps of the two
probe curves.  Mechanism trace: w per group from the state files.

Usage:
    python bci/dualmem/analyze.py [--pilot]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WG = (4300, 4400)
WAB = (4700, 4800)


def net(phi, a, b, pre=200):
    return float(np.sqrt((phi[a:b] ** 2).mean())
                 - np.sqrt((phi[a - pre:a] ** 2).mean()))


def slope(y):
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def perm_slope_diff(curves_g, curves_ab, n_perm=5000, seed=5):
    """curves: list of 1-D arrays per session.  Statistic = mean
    session slope(gamma) - mean session slope(ab); null = swapping
    the two labels within each session."""
    obs = (np.mean([slope(c) for c in curves_g])
           - np.mean([slope(c) for c in curves_ab]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        g, ab = [], []
        for cg, cab in zip(curves_g, curves_ab):
            if rng.random() < 0.5:
                cg, cab = cab, cg
            g.append(cg)
            ab.append(cab)
        null[i] = (np.mean([slope(c) for c in g])
                   - np.mean([slope(c) for c in ab]))
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    prefix = "pilot_" if args.pilot else ""
    meta = json.loads((OUT / f"{prefix}meta.json").read_text(
        encoding="utf-8"))
    n1, n2 = meta["n_acq"], meta["n_ext"]
    curves = {"g": [], "ab": []}
    wtraj = []
    for r in range(meta["sessions"]):
        ng, nab, wg_row = [], [], []
        for phase in ("a", "x"):
            for k in range(n1 if phase == "a" else n2):
                p = OUT / f"{prefix}r{r}_{phase}{k}.npy"
                s = OUT / "states" / f"{prefix}r{r}_{phase}{k}.npz"
                if not p.exists():
                    continue
                phi = np.load(p) * 1e6 * 1.7
                ng.append(net(phi, *WG))
                nab.append(net(phi, *WAB))
                if s.exists():
                    z = np.load(s)
                    wg_row.append((float(z["w_KCMg"].mean()),
                                   float(z["w_KCMab"].mean())))
        curves["g"].append(np.array(ng))
        curves["ab"].append(np.array(nab))
        wtraj.append(wg_row)
    for r in range(len(curves["g"])):
        print(f"session {r} gamma-probe:      "
              + " ".join(f"{v:+.3f}" for v in curves["g"][r]))
        print(f"session {r} alpha/beta-probe: "
              + " ".join(f"{v:+.3f}" for v in curves["ab"][r]))
    for r, wt in enumerate(wtraj):
        print(f"session {r} w: KCMg "
              + " ".join(f"{a:.2f}" for a, b in wt) + "   KCMab "
              + " ".join(f"{b:.2f}" for a, b in wt))
    if args.pilot:
        return
    ext_g = [c[n1:] for c in curves["g"]]
    ext_ab = [c[n1:] for c in curves["ab"]]
    obs, p = perm_slope_diff(ext_g, ext_ab)
    print(f"\nextinction slope: gamma {np.mean([slope(c) for c in ext_g]):+.4f}"
          f" vs ab {np.mean([slope(c) for c in ext_ab]):+.4f} uV/trial"
          f" -> diff {obs:+.4f}, perm-p {p:.4f}")
    acq_g = [c[:n1] for c in curves["g"]]
    acq_ab = [c[:n1] for c in curves["ab"]]
    print(f"acquisition slope: gamma "
          f"{np.mean([slope(c) for c in acq_g]):+.4f} vs ab "
          f"{np.mean([slope(c) for c in acq_ab]):+.4f} uV/trial")
    (OUT / f"{prefix}summary.json").write_text(json.dumps({
        "curves": {"g": [c.tolist() for c in curves["g"]],
                   "ab": [c.tolist() for c in curves["ab"]]},
        "w_traj": wtraj, "ext_slope_diff": obs, "p_ext": p}, indent=1))
    print(f"-> {OUT / f'{prefix}summary.json'}")


if __name__ == "__main__":
    main()
