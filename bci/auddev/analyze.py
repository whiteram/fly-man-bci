"""bci/auddev analysis: deviant burst vs adaptation-matched standards.

Per burst b: feature = rms(phi[onset+50 : onset+450]).  The deviance
contrast = deviant burst (index 2) vs the mean of its two neighboring
standard bursts (1 and 3) -- all equally adapted at that point in the
train.  Permutation over runs (labels: deviant vs neighbor role).

Usage:
    python bci/auddev/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    dev_i = meta["deviant_index"]
    feats = []                       # (run, burst) -> rms
    for r in range(meta["runs"]):
        phi = np.load(OUT / f"train_r{r}.npy") * 1e6 * 1.7
        row = []
        for (a, b) in meta["bursts_ms"]:
            row.append(np.sqrt((phi[a + 50:b] ** 2).mean()))
        feats.append(row)
    feats = np.array(feats)
    d = feats[:, dev_i]
    nb = np.concatenate([feats[:, dev_i - 1], feats[:, dev_i + 1]])
    obs = d.mean() - nb.mean()
    pool = np.concatenate([d, nb])
    n = len(d)
    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(5000):
        pm = rng.permutation(pool)
        diffs.append(pm[:n].mean() - pm[n:].mean())
    p = (np.sum(np.array(diffs) >= obs) + 1) / 5001
    print("per-burst rms (uV, mean over runs):",
          {i: round(float(feats[:, i].mean()), 4)
           for i in range(feats.shape[1])})
    print(f"deviant(50 Hz) {d.mean():.4f} vs neighbor standards(25 Hz) "
          f"{nb.mean():.4f} uV | diff {obs:+.4f} perm-p={p:.4f} "
          f"({meta['runs']} runs)")


if __name__ == "__main__":
    main()
