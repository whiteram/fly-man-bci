"""bci/condit2 analysis: acquisition and extinction curves.

Per trial the probe net response (window rms - pre-window 200 ms) --
the same readout as bci/condit.  The session chains trials through
--plastic-state files, so the curves are ordered: acquisition (reward
on) should RISE trial over trial, extinction (reward omitted) should
DECAY, control (never rewarded) should stay flat.  Mechanism trace:
mean KCM w_scale per trial from the state files.

Statistics: per-session OLS slopes, pooled across sessions;
label-permutation null = shuffling trial order WITHIN each session
(keeps session levels, destroys the time structure).  Two-sided.

Usage:
    python bci/condit2/analyze.py            # full study
    python bci/condit2/analyze.py --pilot    # 3-trial chain check
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
A, B = 4300, 4700                      # probe window, ms


def net(phi, a, b, pre=200):
    return float(np.sqrt((phi[a:b] ** 2).mean())
                 - np.sqrt((phi[a - pre:a] ** 2).mean()))


def slope(y):
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def perm_slopes(curves, n_perm=5000, seed=1):
    """curves: list of 1-D arrays (one per session).  Observed = mean
    session slope; null = within-session shuffles."""
    obs = float(np.mean([slope(c) for c in curves]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.mean([slope(rng.permutation(c)) for c in curves])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def curve(meta, cond, r, prefix):
    if cond == "paired":
        tags = ([("a", k) for k in range(meta["n_acq"])]
                + [("x", k) for k in range(meta["n_ext"])])
    else:
        tags = [("c", k) for k in range(meta["n_ctrl"])]
    nets, ws = [], []
    for phase, k in tags:
        p = OUT / f"{prefix}{cond}_r{r}_{phase}{k}.npy"
        s = OUT / "states" / f"{prefix}{cond}_r{r}_{phase}{k}.npz"
        if not p.exists():
            return None, None
        phi = np.load(p) * 1e6 * 1.7
        nets.append(net(phi, A, B))
        if s.exists():
            ws.append(float(np.load(s)["w"].mean()))
    return np.array(nets), (np.array(ws) if ws else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    prefix = "pilot_" if args.pilot else ""
    meta = json.loads((OUT / f"{prefix}meta.json").read_text(
        encoding="utf-8"))
    n_sess = meta["sessions"]
    data = {}
    for cond in ("paired", "ctrl"):
        rows = [curve(meta, cond, r, prefix) for r in range(n_sess)]
        rows = [(n, w) for n, w in rows if n is not None]
        data[cond] = rows
    n_acq, n_ext = meta["n_acq"], meta["n_ext"]
    for cond in data:
        print(f"\n== {cond} ==")
        for si, (nets, ws) in enumerate(data[cond]):
            wtxt = ""
            if ws is not None:
                wtxt = "   w: " + " ".join(f"{v:.2f}" for v in ws)
            print(f" session {si}: "
                  + " ".join(f"{v:+.3f}" for v in nets) + f"  uV{wtxt}")
    res = {"meta": {k: meta[k] for k in
                    ("n_acq", "n_ext", "n_ctrl", "tau_w_ms")}, "curves": {}}
    for cond in data:
        res["curves"][cond] = [n.tolist() for n, _ in data[cond]]
    if not args.pilot:
        acq = [n[:n_acq] for n, _ in data["paired"] if len(n) >= n_acq]
        ext = [n[n_acq:] for n, _ in data["paired"]
               if len(n) >= n_acq + n_ext]
        ctl = [n for n, _ in data["ctrl"]]
        print("\n== slopes (uV per trial, mean over sessions) ==")
        out = {}
        for name, curves in (("acquisition", acq), ("extinction", ext),
                             ("control", ctl)):
            if not curves:
                continue
            obs, p = perm_slopes(curves)
            sess = [slope(c) for c in curves]
            out[name] = {"slope": obs,
                         "session_slopes": sess,
                         "perm_p": p}
            print(f" {name:11s} {obs:+.4f} "
                  f"(sessions {['%+.4f' % s for s in sess]}) "
                  f"perm-p {p:.4f}")
        if acq:
            d_pa = [np.mean(c[-3:]) - np.mean(c[:3]) for c in acq]
            if ctl:
                d_ct = [np.mean(c[-3:]) - np.mean(c[:3]) for c in ctl]
                print(f" early->late: paired {np.mean(d_pa):+.4f} uV vs "
                      f"ctrl {np.mean(d_ct):+.4f} uV")
                out["early_late_delta"] = {
                    "paired": float(np.mean(d_pa)),
                    "ctrl": float(np.mean(d_ct))}
        res["stats"] = out
    (OUT / f"{prefix}summary.json").write_text(json.dumps(res, indent=1))
    print(f"\n-> {OUT / f'{prefix}summary.json'}")


if __name__ == "__main__":
    main()
