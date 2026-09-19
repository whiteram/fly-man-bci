"""bci/thermal analysis: warm/cool and dry/moist spatial LOO.

Features: per-trial spatial rms (17 electrodes) over the ramp plateau
(1500-3400 ms), z-scored, nearest-centroid LOO within each block.
Permutation on class labels (5000).

Usage:
    python bci/thermal/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
W0, W1 = 1500, 3400


def loo_acc(X, y):
    Zs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Zs[m & (y == c)].mean(0) for c in (0, 1)])
        ok += int(np.argmin(((cents - Zs[i]) ** 2).sum(1)) == y[i])
    return ok / len(y)


def main():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(9)
    summary = {}
    for block, classes in meta["blocks"].items():
        X, y = [], []
        for ci, chem in enumerate(classes):
            for r in range(meta["runs"]):
                p = OUT / f"{chem}_r{r}.npy"
                if p.exists():
                    phi = np.load(p) * 1e6 * 1.7
                    X.append(np.sqrt((phi[W0:W1] ** 2).mean(0)))
                    y.append(ci)
        if len(y) < 4:
            print(f"[skip] {block}")
            continue
        X, y = np.array(X), np.array(y)
        acc = loo_acc(X, y)
        null = [loo_acc(X, rng.permutation(y)) for _ in range(5000)]
        p = (np.sum(np.array(null) >= acc) + 1) / 5001
        _phi0 = np.load(OUT / f"{classes[0]}_r0.npy") * 1e6 * 1.7
        base = float(np.sqrt((_phi0[100:300] ** 2).mean()))
        ramp = float(np.sqrt((_phi0[W0:W1] ** 2).mean()))
        print(f"{block}: {classes[0]} vs {classes[1]}  LOO {acc:.0%}"
              f" (chance 50%, perm-p {p:.4f})  "
              f"signature {base:.3f} -> {ramp:.3f} uV")
        summary[block] = {"acc": acc, "p": p,
                          "sig_base_uv": base, "sig_ramp_uv": ramp}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
