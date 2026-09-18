"""Looming analysis: 3-class escape-signal decoding + response table.

Memoryless per-channel rms features; the response table reports the
time-course contrast (expansion window vs pre-window baseline) per
class.  Classification: 17-dim spatial rms pattern over the expansion
window (300-2300 ms) -> z-score -> nearest-centroid LOO + 2000-fold
label permutation (chance 33%).

Usage:
    python bci/looming/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (300, 2300)                 # expansion window
BASE = (0, 280)                   # pre-stimulus baseline


def load():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    X, y, curves = [], [], {}
    for i, cls in enumerate(meta["classes"]):
        for r in range(meta["repeats"]):
            p = OUT / f"{cls}_r{r}.npy"
            if not p.exists():
                continue
            phi = np.load(p) * 1e6 * 1.7
            X.append(np.sqrt((phi[WIN[0]:WIN[1]] ** 2).mean(axis=0)))
            y.append(i)
        if y and y[-1] == i:
            curves[cls] = np.load(OUT / f"{cls}_r0.npy") * 1e6 * 1.7
    return np.array(X), np.array(y), meta["classes"], curves


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
    X, y, classes, curves = load()
    if len(y) < 6:
        raise SystemExit("not enough trials -- run acquire.py first")
    print("response time-course (rms uV, class means):")
    edges = list(range(0, 3000, 500))
    print("        " + "".join(f"{a}-{a + 500:<7d}"
                              for a in edges[:-1]))
    for cls in classes:
        phi = curves[cls]
        seg = [np.sqrt((phi[a:b] ** 2).mean())
               for a, b in zip(edges[:-1], edges[1:])]
        print(f"{cls:8s}" + "".join(f"{v:<12.3f}" for v in seg))
    acc = loo_acc(X, y)
    rng = np.random.default_rng(1)
    perm = [loo_acc(X, rng.permutation(y)) for _ in range(2000)]
    p = (np.sum(np.array(perm) >= acc) + 1) / 2001
    print(f"trials: {len(y)}")
    print(f"3-class LOO: {acc:.0%}  perm-p={p:.4f} "
          f"(chance 33%, perm 95th pct {np.percentile(perm, 95):.0%})")


if __name__ == "__main__":
    main()
