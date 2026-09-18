"""MRCP analysis: readiness buildup + L/R decoding.

1. Time-resolved rms table (500 ms bins, class means) -- the MRCP
   analog is the PROGRESSIVE rise across the ramp window (1300->3500),
   with the movement step at 2800 ms visible on top.
2. 2-class L/R LOO on the PRE-movement ramp window (1500-2700 ms)
   spatial pattern + 2000-fold permutation (chance 50%) -- decoding
   the intended side BEFORE the movement.

Usage:
    python bci/mrcp/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
RAMP = (1500, 2700)


def load():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    trials, curves = {c: [] for c in meta["classes"]}, {}
    for i, cls in enumerate(meta["classes"]):
        for r in range(meta["repeats"]):
            p = OUT / f"{cls}_r{r}.npy"
            if not p.exists():
                continue
            trials[cls].append(np.load(p) * 1e6 * 1.7)
        if trials[cls]:
            curves[cls] = trials[cls][0]
    return trials, curves, meta


def main():
    trials, curves, meta = load()
    if any(len(v) < 3 for v in trials.values()):
        raise SystemExit("not enough trials -- run acquire.py first")

    print("time-resolved rms (uV, mean over trials):")
    edges = list(range(500, 3600, 500))
    print("        " + "".join(f"{a}-{a + 500:<9d}"
                               for a in edges[:-1]))
    for cls in meta["classes"]:
        stack = np.array(trials[cls])
        seg = [np.sqrt((stack[:, a:b] ** 2).mean())
               for a, b in zip(edges[:-1], edges[1:])]
        print(f"{cls:8s}" + "".join(f"{v:<13.3f}" for v in seg))

    a, b = RAMP
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
    print(f"pre-movement ramp L/R LOO ({RAMP[0]}-{RAMP[1]} ms): "
          f"{acc:.0%}  perm-p={p:.4f} (chance 50%, perm 95th pct "
          f"{np.percentile(perm, 95):.0%})")


if __name__ == "__main__":
    main()
