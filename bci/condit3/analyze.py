"""bci/condit3 analysis: true-odor conditioning + identity specificity.

Per trial: probe net response (window rms - pre-window 200 ms), same
readout as condit/condit2.  Two questions:

  1. learning: does the TRAINED-odor probe (a) change learn vs nolr?
  2. specificity: is the change larger for the trained odor (a) than
     the untrained one (b)?  Statistic = interaction
     (learn-nolr)|a - (learn-nolr)|b, permutation on condition labels.

Usage:
    python bci/condit3/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
A, B = 4300, 4700


def net(phi, a, b, pre=200):
    return float(np.sqrt((phi[a:b] ** 2).mean())
                 - np.sqrt((phi[a - pre:a] ** 2).mean()))


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    res = {}
    for cond in meta["conditions"]:
        for pb in meta["probes"]:
            key = (cond, pb[-1])
            rows = []
            for r in range(meta["runs"]):
                p = OUT / f"{cond}_{pb[-1]}_r{r}.npy"
                if p.exists():
                    phi = np.load(p) * 1e6 * 1.7
                    rows.append(net(phi, A, B))
            res[key] = np.array(rows)
    print("probe net response (uV, mean+-sd)")
    for pb in ("a", "b"):
        for cond in meta["conditions"]:
            v = res[(cond, pb)]
            print(f"  probe {pb} {cond:5s}: {v.mean():+.4f} "
                  f"+- {v.std():.4f}  (n={len(v)})")
    d_a = res[("learn", "a")].mean() - res[("nolr", "a")].mean()
    d_b = res[("learn", "b")].mean() - res[("nolr", "b")].mean()
    print(f"\nlearning effect  trained (a): {d_a:+.4f} uV")
    print(f"learning effect untrained (b): {d_b:+.4f} uV")
    obs_int = d_a - d_b
    print(f"specificity interaction (a-b): {obs_int:+.4f} uV")
    # permutation: shuffle learn/nolr labels within each probe block
    rng = np.random.default_rng(2)
    n = meta["runs"]
    null = []
    for _ in range(5000):
        da = db = 0.0
        for pb in ("a", "b"):
            pool = np.concatenate([res[("learn", pb)], res[("nolr", pb)]])
            pm = rng.permutation(pool)
            d = pm[:n].mean() - pm[n:].mean()
            if pb == "a":
                da = d
            else:
                db = d
        null.append(da - db)
    p_int = (np.sum(np.abs(null) >= abs(obs_int)) + 1) / 5001
    print(f"specificity perm-p (two-sided): {p_int:.4f}")
    # main effect on the trained odor
    pool = np.concatenate([res[("learn", "a")], res[("nolr", "a")]])
    obs = res[("learn", "a")].mean() - res[("nolr", "a")].mean()
    rng = np.random.default_rng(1)
    null_a = []
    for _ in range(5000):
        pm = rng.permutation(pool)
        null_a.append(pm[:n].mean() - pm[n:].mean())
    p_a = (np.sum(np.abs(null_a) >= abs(obs)) + 1) / 5001
    print(f"trained-odor learning perm-p (two-sided): {p_a:.4f}")
    summary = {"probe_net": {f"{c}_{p}": v.tolist()
                             for (c, p), v in res.items()},
               "learning_a": d_a, "learning_b": d_b,
               "interaction": obs_int, "p_interaction": p_int,
               "p_trained": p_a}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
