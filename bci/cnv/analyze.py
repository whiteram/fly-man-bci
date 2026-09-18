"""CNV analysis: anticipation-window buildup + L/R decoding.

1. Time-resolved rms (500 ms bins): the CNV analog is the progressive
   rise between S1 (300-600) and S2 (2400-3100).
2. L/R LOO on the anticipation window (1000-2300 ms) spatial pattern
   + 2000-fold permutation (chance 50%) -- decoding expected side
   BEFORE the imperative stimulus.

Usage:
    python bci/cnv/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ANTIC = (1000, 2300)


def load():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    trials = {c: [] for c in meta["classes"]}
    for i, cls in enumerate(meta["classes"]):
        for r in range(meta["repeats"]):
            p = OUT / f"{cls}_r{r}.npy"
            if p.exists():
                trials[cls].append(np.load(p) * 1e6 * 1.7)
    return trials, meta


def main():
    trials, meta = load()
    if any(len(v) < 3 for v in trials.values()):
        raise SystemExit("not enough trials -- run acquire.py first")
    print("time-resolved rms (uV, mean over trials):")
    edges = list(range(0, 3600, 600))
    print("        " + "".join(f"{a}-{a + 600:<9d}"
                               for a in edges[:-1]))
    for cls in meta["classes"]:
        stack = np.array(trials[cls])
        seg = [np.sqrt((stack[:, a:b] ** 2).mean())
               for a, b in zip(edges[:-1], edges[1:])]
        print(f"{cls:8s}" + "".join(f"{v:<13.3f}" for v in seg))
    a, b = ANTIC
    X = np.array([np.sqrt((phi[a:b] ** 2).mean(axis=0))
                  for cls in meta["classes"] for phi in trials[cls]])
    y = np.array([i for i, cls in enumerate(meta["classes"])
                  for _ in trials[cls]])
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Z[m & (y == c)].mean(0) for c in (0, 1)])
        ok += int(np.argmin(((cents - Z[i]) ** 2).sum(1)) == y[i])
    acc = ok / len(y)
    rng = np.random.default_rng(1)
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
    print(f"anticipation L/R LOO ({a}-{b} ms): {acc:.0%}  "
          f"perm-p={p:.4f} (chance 50%, perm 95th pct "
          f"{np.percentile(perm, 95):.0%})")


if __name__ == "__main__":
    main()
