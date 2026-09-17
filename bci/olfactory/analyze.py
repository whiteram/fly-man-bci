"""bci/olfactory analysis: 3-class odor-identity decoding.

Same honest pipeline as bci/auditory: memoryless per-channel rms over
the response window (400-1400 ms, covering the pulse + AL decay onset)
-> z-score -> nearest-centroid LOO + 2000-fold label permutation
(chance = 33%).

Usage:
    python bci/olfactory/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (400, 1400)


def load():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    X, y = [], []
    for i, cls in enumerate(meta["classes"]):
        for r in range(meta["repeats"]):
            p = OUT / f"{cls}_r{r}.npy"
            if not p.exists():
                continue
            phi = np.load(p) * 1e6 * 1.7
            X.append(np.sqrt((phi[WIN[0]:WIN[1]] ** 2).mean(axis=0)))
            y.append(i)
    return np.array(X), np.array(y), meta["classes"]


def loo_acc(X, y):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Z[m & (y == c)].mean(0)
                          for c in range(y.max() + 1)])
        ok += int(np.argmin(((cents - Z[i]) ** 2).sum(1)) == y[i])
    return ok / len(y)


def main():
    X, y, classes = load()
    if len(y) < 6:
        raise SystemExit("not enough trials -- run acquire.py first")
    acc = loo_acc(X, y)
    rng = np.random.default_rng(1)
    perm = [loo_acc(X, rng.permutation(y)) for _ in range(2000)]
    p = (np.sum(np.array(perm) >= acc) + 1) / 2001
    resp = np.sqrt((X ** 2).mean(1))
    base = None
    print(f"trials: {len(y)}  classes: {classes}")
    print("mean rms per class (uV):",
          {c: round(float(resp[y == i].mean()), 4)
           for i, c in enumerate(classes)})
    print(f"LOO nearest-centroid: {acc:.0%}  perm-p={p:.4f} "
          f"(chance 33%, perm 95th pct {np.percentile(perm, 95):.0%})")


if __name__ == "__main__":
    main()
