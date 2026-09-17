"""bci/motor_imagery analysis: 2-class imagery-hold decoding.

Window table (cue + 3 imagery sub-windows, memoryless raw rms) with
permutation p for the L-R difference -- the exp020 MI-v2 reading --
plus a leave-one-out nearest-centroid decode on the imagery-window
spatial rms pattern (chance 50%).

Usage:
    python bci/motor_imagery/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CUE = (350, 1150)
IMAGERY = ((1300, 2000), (2000, 2700), (2700, 3500))


def load():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    vals = {c: [] for c in meta["classes"]}
    for i, cls in enumerate(meta["classes"]):
        for r in range(meta["repeats"]):
            p = OUT / f"{cls}_r{r}.npy"
            if not p.exists():
                continue
            phi = np.load(p) * 1e6 * 1.7
            vals[cls].append([np.sqrt((phi[a:b] ** 2).mean())
                              for a, b in (CUE,) + IMAGERY])
    return (np.array(vals[meta["classes"][0]]),
            np.array(vals[meta["classes"][1]]), meta["classes"])


def main():
    L, R, classes = load()
    if len(L) < 3 or len(R) < 3:
        raise SystemExit("not enough trials -- run acquire.py first")
    rng = np.random.default_rng(1)
    names = ["cue"] + [f"imagery{a}-{b}" for a, b in IMAGERY]
    print(f"classes: {classes}  trials: {len(L)}+{len(R)}")
    for w, nm in enumerate(names):
        l, r = L[:, w], R[:, w]
        obs = l.mean() - r.mean()
        pool = np.concatenate([l, r])
        n = len(l)
        diff = []
        for _ in range(5000):
            pm = rng.permutation(pool)
            diff.append(pm[:n].mean() - pm[n:].mean())
        p = (np.sum(np.abs(diff) >= abs(obs)) + 1) / 5001
        print(f"{nm:16s}: L {l.mean():.4f}+-{l.std():.4f} | "
              f"R {r.mean():.4f}+-{r.std():.4f} uV | "
              f"diff {obs:+.4f} perm-p={p:.3f}")

    # spatial-pattern decode on the imagery window (17-dim rms)
    # rebuild full-channel trials for the LOO decode
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    X, y = [], []
    for i, cls in enumerate(meta["classes"]):
        for r in range(meta["repeats"]):
            p = OUT / f"{cls}_r{r}.npy"
            if not p.exists():
                continue
            phi = np.load(p) * 1e6 * 1.7
            X.append(np.sqrt((phi[1300:3500] ** 2).mean(axis=0)))
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Z[m & (y == c)].mean(0) for c in (0, 1)])
        ok += int(np.argmin(((cents - Z[i]) ** 2).sum(1)) == y[i])
    acc = ok / len(y)
    perm = []
    for _ in range(2000):
        yp = rng.permutation(y)
        okp = 0
        for i in range(len(y)):
            m = np.arange(len(y)) != i
            cents = np.array([Z[m & (yp == c)].mean(0) for c in (0, 1)])
            okp += int(np.argmin(((cents - Z[i]) ** 2).sum(1)) == yp[i])
        perm.append(okp / len(y))
    p = (np.sum(np.array(perm) >= acc) + 1) / 2001
    print(f"imagery-window spatial LOO: {acc:.0%}  "
          f"perm-p={p:.4f} (chance 50%, "
          f"perm 95th pct {np.percentile(perm, 95):.0%})")


if __name__ == "__main__":
    main()
