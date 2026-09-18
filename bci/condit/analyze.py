"""bci/condit analysis: did conditioning change the test responses?

Per trial: net responses (window rms - pre-window 200 ms) for train,
test A, test B.  The learning signature = test-A difference between
learn and nolr conditions (depressed KC->MBON -> smaller A response),
with test B as the specificity control (should NOT differ).

Usage:
    python bci/condit/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WINS = {"train": (600, 2400), "probe": (4300, 4700)}


def net(phi, a, b, pre=200):
    return float(np.sqrt((phi[a:b] ** 2).mean())
                 - np.sqrt((phi[a - pre:a] ** 2).mean()))


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for cond in meta["conditions"]:
        rows = []
        for r in range(meta["runs"]):
            p = OUT / f"{cond}_r{r}.npy"
            if p.exists():
                phi = np.load(p) * 1e6 * 1.7
                rows.append([net(phi, *WINS[k]) for k in WINS])
        res[cond] = np.array(rows)
    hdr = "condition  train      probe     (uV, mean+-sd)"
    print(hdr)
    for cond in meta["conditions"]:
        m, sd = res[cond].mean(0), res[cond].std(0)
        print(f"{cond:8s}  " + "  ".join(
            f"{v:+.4f}+-{s:.4f}" for v, s in zip(m, sd)))
    la = res["learn"][:, 1].mean()
    na = res["nolr"][:, 1].mean()
    print(f"learning effect probe: {la:+.4f} (learn) vs {na:+.4f} "
          f"(nolr) -> delta {la - na:+.4f} uV")
    # permutation on the probe-response learn-vs-nolr difference
    pool = np.concatenate([res["learn"][:, 1], res["nolr"][:, 1]])
    n = len(res["learn"])
    obs = res["learn"][:, 1].mean() - res["nolr"][:, 1].mean()
    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(5000):
        pm = rng.permutation(pool)
        diffs.append(pm[:n].mean() - pm[n:].mean())
    p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / 5001
    print(f"probe learn-vs-nolr perm-p (two-sided): {p:.4f}")


if __name__ == "__main__":
    main()
