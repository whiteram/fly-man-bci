"""bci/auditory analysis: 4-class song decoding from clean scalp EEG.

Memoryless features (the filtfilt long-tail lesson, exp018/020):
per-channel broadband rms over the drive window (500-2000 ms) as a
17-dim spatial pattern -> z-scored -> nearest-centroid leave-one-out
accuracy; significance from a 2000-fold label permutation of the same
pipeline (chance = 25%).

Usage:
    python bci/auditory/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (500, 2000)
FS = 1000.0


def load():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    X, y, names = [], [], []
    for i, cls in enumerate(meta["classes"]):
        for r in range(meta["repeats"]):
            p = OUT / f"{cls}_r{r}.npy"
            if not p.exists():
                continue
            phi = np.load(p) * 1e6 * 1.7
            X.append(np.sqrt((phi[WIN[0]:WIN[1]] ** 2)
                             .mean(axis=0)))
            y.append(i)
            names.append(f"{cls}_r{r}")
    return np.array(X), np.array(y), meta["classes"], names


def loo_acc(X, y):
    """nearest-centroid leave-one-out on z-scored features"""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Z[m & (y == c)].mean(0)
                          for c in range(y.max() + 1)])
        ok += int(np.argmin(((cents - Z[i]) ** 2).sum(1)) == y[i])
    return ok / len(y)


def main():
    X, y, classes, names = load()
    if len(y) < 8:
        raise SystemExit("not enough trials -- run acquire.py first")
    acc = loo_acc(X, y)
    rng = np.random.default_rng(1)
    perm = [loo_acc(X, rng.permutation(y)) for _ in range(2000)]
    p = (np.sum(np.array(perm) >= acc) + 1) / 2001
    print(f"trials: {len(y)}  classes: {classes}")
    per = {c: float(np.mean([v for v, yy in zip(
        np.sqrt((X ** 2).mean(1)), y) if yy == i]))
        for i, c in enumerate(classes)}
    print("mean rms per class (uV):",
          {k: round(v, 4) for k, v in per.items()})
    print(f"LOO nearest-centroid: {acc:.0%}  "
          f"perm-p={p:.4f} (chance 25%, "
          f"perm 95th pct {np.percentile(perm, 95):.0%})")


if __name__ == "__main__":
    main()
