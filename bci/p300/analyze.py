"""P300 oddball analysis: target vs standard event responses.

Per event k: feature = rms(phi[onset+50 : onset+350]) minus baseline
rms(phi[onset-100 : onset]); the deviance response is the target
feature exceeding the adaptation-matched standard distribution.
Statistics: permutation difference test on event features (labels =
target/standard, 5000 shuffles) + single-event AUC.  HONEST framing:
adaptation-driven deviance, not cognitive P300.

Usage:
    python bci/p300/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    fs = int(meta["fs"])
    soa, flash = int(meta["soa_ms"]), int(meta["flash_ms"])
    feats, labs = [], []
    for r in range(meta["runs"]):
        phi = np.load(OUT / f"train_r{r}.npy") * 1e6 * 1.7
        for k in range(meta["events"]):
            on = int((k * soa + 100) + flash)
            seg = phi[on + 50:on + 350]
            base = phi[on - 100:on]
            feats.append(np.sqrt((seg ** 2).mean())
                         - np.sqrt((base ** 2).mean()))
            labs.append(1 if k in meta["targets"] else 0)
    feats = np.array(feats)
    labs = np.array(labs, dtype=bool)
    rng = np.random.default_rng(1)
    obs = feats[labs].mean() - feats[~labs].mean()
    pool = feats.copy()
    diffs = []
    for _ in range(5000):
        pm = rng.permutation(pool)
        diffs.append(pm[labs].mean() - pm[~labs].mean())
    p = (np.sum(np.array(diffs) >= obs) + 1) / 5001
    # single-event AUC: P(target feature > standard feature)
    t, s = feats[labs], feats[~labs]
    auc = float((t[:, None] > s[None, :]).mean())
    print(f"events: {len(labs)}  targets: {int(labs.sum())}  "
          f"runs: {meta['runs']}")
    print(f"target response {feats[labs].mean():+.4f} uV vs standard "
          f"{feats[~labs].mean():+.4f} uV | diff {obs:+.4f} "
          f"perm-p={p:.4f}")
    print(f"single-event AUC (target louder than standard): "
          f"{auc:.2f}")


if __name__ == "__main__":
    main()
