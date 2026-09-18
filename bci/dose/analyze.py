"""bci/dose analysis: concentration decoding + dose-response curve.

Usage:
    python bci/dose/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (400, 1400)
BASE = (0, 280)


def load():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    X, y = [], []
    for i, tag in enumerate(meta["levels"]):
        for r in range(meta["repeats"]):
            p = OUT / f"d{tag}_r{r}.npy"
            if p.exists():
                phi = np.load(p) * 1e6 * 1.7
                X.append(np.sqrt((phi[WIN[0]:WIN[1]] ** 2).mean(axis=0)))
                y.append(i)
    return np.array(X), np.array(y), meta


def loo_acc(X, y, n_cls):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Z[m & (y == c)].mean(0) for c in range(n_cls)])
        ok += int(np.argmin(((cents - Z[i]) ** 2).sum(1)) == y[i])
    return ok / len(y)


def main():
    X, y, meta = load()
    if len(y) < 6:
        raise SystemExit("not enough trials -- run acquire.py first")
    n_cls = len(meta["levels"])
    base = np.sqrt((np.load(OUT / f"d{meta['levels'][1]}_r0.npy")
                    [BASE[0]:BASE[1]] ** 2).mean()) * 1e6 * 1.7
    resp = np.sqrt((X ** 2).mean(1))
    print("dose-response (rms uV per level, net of baseline):")
    for i, tag in enumerate(meta["levels"]):
        print(f"  {meta['amps'][i]:.0f} pA ({tag}): "
              f"{resp[y == i].mean():.4f} (net "
              f"{resp[y == i].mean() - base:+.4f})")
    acc = loo_acc(X, y, n_cls)
    rng = np.random.default_rng(1)
    perm = [loo_acc(X, rng.permutation(y), n_cls) for _ in range(2000)]
    p = (np.sum(np.array(perm) >= acc) + 1) / 2001
    print(f"trials: {len(y)}")
    print(f"{n_cls}-class dose LOO: {acc:.0%}  perm-p={p:.4f} "
          f"(chance {1.0 / n_cls:.0%}, perm 95th pct "
          f"{np.percentile(perm, 95):.0%})")


if __name__ == "__main__":
    main()
