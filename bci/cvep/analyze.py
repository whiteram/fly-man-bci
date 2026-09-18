"""c-VEP analysis: code-shape template matching (BETA-style).

The c-VEP decoder matches the epoch against per-target templates --
the target identity lives in the response SHAPE aligned to the code,
not in any frequency line.  Pipeline: epoch = the code period
(500-2617 ms) x 17 leads, flattened per trial; leave-one-out
nearest-template classification by max Pearson correlation;
significance from a 2000-fold label permutation.

Implementation note: templates live in the correlation matrix C[i,j]
(trial-trial Pearson, computed once); the LOO template of class c for
test trial i is exactly mean(C[i, other trials of class c]) -- for
the 2-repeat layout this reduces to the single partner trial, i.e.
"nearest matching trial".  Label permutations reuse the same matrix,
so 2000 permutations are free (the naive per-permutation rebuild of
templates takes ~15 min; this form is mathematically identical).

Usage:
    python bci/cvep/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    a, b = [int(round(v)) for v in meta["epoch_ms"]]
    X, y = [], []
    for i, k in enumerate(meta["targets"]):
        for r in range(meta["repeats"]):
            stem = f"eeg_t{k:02d}" + ("" if meta["repeats"] == 1
                                      else f"_r{r}")
            p = OUT / f"{stem}.npy"
            if not p.exists():
                continue
            phi = np.load(p) * 1e6 * 1.7
            X.append(phi[a:b].T.reshape(-1))
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    n = len(y)
    if n < 6:
        raise SystemExit("not enough trials -- run acquire.py first")
    Xn = X - X.mean(1, keepdims=True)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
    C = Xn @ Xn.T
    classes = np.arange(len(meta["targets"]))
    S = np.full((n, len(classes)), -2.0)
    for i in range(n):
        for c in classes:
            m = (np.arange(n) != i) & (y == c)
            if m.any():
                S[i, c] = C[i, m].mean()

    def acc_of(lab):
        return float((S.argmax(1) == lab).mean())

    acc = acc_of(y)
    rng = np.random.default_rng(1)
    perm = [acc_of(rng.permutation(y)) for _ in range(2000)]
    p = (np.sum(np.array(perm) >= acc) + 1) / 2001
    chance = 1.0 / len(classes)
    print(f"trials: {n}  targets: {len(classes)}")
    print(f"template-matching LOO: {acc:.0%}  perm-p={p:.4f} "
          f"(chance {chance:.1%}, perm 95th pct "
          f"{np.percentile(perm, 95):.0%})")


if __name__ == "__main__":
    main()
