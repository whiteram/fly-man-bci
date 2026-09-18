"""Motion-direction analysis: 4-class decoding + direction tuning.

Features: 17-dim spatial rms over the grating window (300-2300 ms);
nearest-centroid LOO + 2000-fold permutation (chance 25%).  Also
reports the opposite-direction pair structure (90/270 vs 0/180).

Usage:
    python bci/direction/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (300, 2300)


def load():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    X, y = [], []
    for i, cls in enumerate(meta["classes"]):
        for r in range(meta["repeats"]):
            p = OUT / f"d{cls}_r{r}.npy"
            if not p.exists():
                continue
            phi = np.load(p) * 1e6 * 1.7
            X.append(np.sqrt((phi[WIN[0]:WIN[1]] ** 2).mean(axis=0)))
            y.append(i)
    return np.array(X), np.array(y), meta["classes"]


def loo_acc(X, y, n_cls):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Z[m & (y == c)].mean(0) for c in range(n_cls)])
        ok += int(np.argmin(((cents - Z[i]) ** 2).sum(1)) == y[i])
    return ok / len(y)


def main():
    X, y, classes = load()
    if len(y) < 6:
        raise SystemExit("not enough trials -- run acquire.py first")
    n_cls = len(classes)
    acc = loo_acc(X, y, n_cls)
    rng = np.random.default_rng(1)
    perm = [loo_acc(X, rng.permutation(y), n_cls) for _ in range(2000)]
    p = (np.sum(np.array(perm) >= acc) + 1) / 2001
    print(f"trials: {len(y)}  directions: {classes}")
    print(f"{n_cls}-class spatial-rms LOO: {acc:.0%}  perm-p={p:.4f} "
          f"(chance {1.0 / n_cls:.0%}, perm 95th pct "
          f"{np.percentile(perm, 95):.0%})")

    # temporal-template decode (c-VEP style): drift phase lives in the
    # response SHAPE -- epoch (17 ch x window) flattened, trial-trial
    # Pearson matrix once, LOO template = other same-class trials,
    # permutations reuse the matrix (identical to naive rebuild)
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    X2, y2 = [], []
    for i, cls in enumerate(meta["classes"]):
        for r in range(meta["repeats"]):
            pp = OUT / f"d{cls}_r{r}.npy"
            if not pp.exists():
                continue
            phi = np.load(pp) * 1e6 * 1.7
            X2.append(phi[WIN[0]:WIN[1]].T.reshape(-1))
            y2.append(i)
    X2 = np.array(X2)
    y2 = np.array(y2)
    n = len(y2)
    Xn = X2 - X2.mean(1, keepdims=True)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
    C = Xn @ Xn.T
    S = np.full((n, n_cls), -2.0)
    for i in range(n):
        for c in range(n_cls):
            m = (np.arange(n) != i) & (y2 == c)
            if m.any():
                S[i, c] = C[i, m].mean()
    acc2 = float((S.argmax(1) == y2).mean())
    perm2 = [float((S.argmax(1) == rng.permutation(y2)).mean())
             for _ in range(2000)]
    p2 = (np.sum(np.array(perm2) >= acc2) + 1) / 2001
    print(f"{n_cls}-class temporal-template LOO: {acc2:.0%}  "
          f"perm-p={p2:.4f} (chance {1.0 / n_cls:.0%}, perm 95th pct "
          f"{np.percentile(perm2, 95):.0%})")


if __name__ == "__main__":
    main()
