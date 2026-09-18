"""bci/seizure analysis: ictal onset detection + focus lateralization.

1. Onset detection: per trial, ictal onset = first crossing of
   (baseline rms) * RATIO on a 100 ms-smoothed rms curve; reported
   per class (ctl should stay at ~1, sz classes should explode after
   the 800 ms drive onset).
2. Focus lateralization: sz_L vs sz_R LOO on the ictal window
   (1200-2800 ms) spatial pattern + permutation.
3. Seizure vs control: 2-class LOO on the same window.

Usage:
    python bci/seizure/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ICTAL = (1200, 2800)
BASE_WIN = (0, 700)
RATIO = 1.5


def load():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    data = {}
    for cls in meta["classes"]:
        trials = []
        for r in range(meta["repeats"]):
            p = OUT / f"{cls}_r{r}.npy"
            if p.exists():
                trials.append(np.load(p) * 1e6 * 1.7)
        data[cls] = trials
    return data, meta


def loo_acc(X, y, n_cls):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Z[m & (y == c)].mean(0) for c in range(n_cls)])
        ok += int(np.argmin(((cents - Z[i]) ** 2).sum(1)) == y[i])
    return ok / len(y)


def perm(X, y, n_cls, rng, n=2000):
    accs = [loo_acc(X, rng.permutation(y), n_cls) for _ in range(n)]
    return accs


def main():
    data, meta = load()
    rng = np.random.default_rng(1)
    fs = int(meta["fs"])
    win = 100
    print("ictal onset (ms, per trial; drive 800-2800):")
    onsets = {}
    for cls in meta["classes"]:
        ons = []
        for phi in data[cls]:
            sig = (phi ** 2).mean(axis=1)          # 1-D, over leads
            rms = np.sqrt(np.convolve(sig, np.ones(win) / win, "same"))
            thr = rms[BASE_WIN[0]:BASE_WIN[1]].mean() * RATIO
            idx = np.flatnonzero(rms[fs // 2:] > thr)
            ons.append(int(idx[0]) if len(idx) else -1)
        onsets[cls] = ons
        print(f"  {cls:6s}: {ons}")
    # lateralization + seizure/control on the ictal window
    a, b = ICTAL
    XL = np.array([np.sqrt((phi[a:b] ** 2).mean(axis=0))
                   for phi in data["sz_l"]])
    XR = np.array([np.sqrt((phi[a:b] ** 2).mean(axis=0))
                   for phi in data["sz_r"]])
    XLR = np.concatenate([XL, XR])
    yLR = np.array([0] * len(XL) + [1] * len(XR))
    accL = loo_acc(XLR, yLR, 2)
    perms = perm(XLR, yLR, 2, rng)
    pL = (np.sum(np.array(perms) >= accL) + 1) / (len(perms) + 1)
    XS = np.concatenate([XLR, np.array(
        [np.sqrt((phi[a:b] ** 2).mean(axis=0)) for phi in data["ctl"]])])
    yS = np.array([1] * len(XLR) + [0] * len(data["ctl"]))
    accS = loo_acc(XS, yS, 2)
    perms2 = perm(XS, yS, 2, rng)
    pS = (np.sum(np.array(perms2) >= accS) + 1) / (len(perms2) + 1)
    print(f"focus lateralization (sz_L vs sz_R, {a}-{b} ms): "
          f"{accL:.0%}  perm-p={pL:.4f} (chance 50%)")
    print(f"seizure vs control LOO: {accS:.0%}  perm-p={pS:.4f} "
          f"(chance 50%)")


if __name__ == "__main__":
    main()
