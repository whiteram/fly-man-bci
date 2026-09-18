"""bci/odormix analysis: mixture linearity, identity, and order.

Usage:
    python bci/odormix/analyze.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
WIN = (400, 1400)                 # response window (single + ab)
EARLY = (300, 900)                # first-pulse window (order decode)
BASE = (0, 280)


def load():
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    phis = {}
    for cond in meta["conditions"]:
        phis[cond] = [np.load(OUT / f"{cond}_r{r}.npy") * 1e6 * 1.7
                      for r in range(meta["repeats"])
                      if (OUT / f"{cond}_r{r}.npy").exists()]
    return phis, meta


def rms_win(phis, a, b):
    return np.array([np.sqrt((p[a:b] ** 2).mean()) for p in phis])


def loo_acc(X, y, n_cls):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ok = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        cents = np.array([Z[m & (y == c)].mean(0) for c in range(n_cls)])
        ok += int(np.argmin(((cents - Z[i]) ** 2).sum(1)) == y[i])
    return ok / len(y)


def main():
    phis, meta = load()
    base = rms_win(phis["a"], *BASE).mean()      # same dark baseline
    ra = rms_win(phis["a"], *WIN).mean()
    rb = rms_win(phis["b"], *WIN).mean()
    rab = rms_win(phis["ab"], *WIN).mean()
    pred = base + (ra - base) + (rb - base)
    print("== mixture linearity (response-window rms uV) ==")
    print(f"A {ra:.4f} | B {rb:.4f} | AB {rab:.4f} | "
          f"linear prediction {pred:.4f} | "
          f"ratio AB/pred {rab / pred:.3f} "
          f"({'sub' if rab < pred else 'super'}linear)")
    # 3-class identity {a, b, ab}
    X = np.array([np.sqrt((p[WIN[0]:WIN[1]] ** 2).mean(axis=0))
                  for cond in ("a", "b", "ab") for p in phis[cond]])
    y = np.array([i for i, cond in enumerate(("a", "b", "ab"))
                  for _ in phis[cond]])
    acc3 = loo_acc(X, y, 3)
    rng = np.random.default_rng(1)
    perm = [loo_acc(X, rng.permutation(y), 3) for _ in range(2000)]
    p3 = (np.sum(np.array(perm) >= acc3) + 1) / 2001
    print(f"3-class identity LOO: {acc3:.0%}  perm-p={p3:.4f} "
          f"(chance 33%)")
    # order decoding (temporal template)
    X2 = np.array([p[EARLY[0]:EARLY[1]].T.reshape(-1)
                   for cond in ("abseq", "baseq") for p in phis[cond]])
    y2 = np.array([0] * len(phis["abseq"]) + [1] * len(phis["baseq"]))
    n = len(y2)
    Xn = X2 - X2.mean(1, keepdims=True)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
    C = Xn @ Xn.T
    S = np.full((n, 2), -2.0)
    for i in range(n):
        for c in (0, 1):
            m = (np.arange(n) != i) & (y2 == c)
            if m.any():
                S[i, c] = C[i, m].mean()
    acc2 = float((S.argmax(1) == y2).mean())
    perm2 = [float((S.argmax(1) == rng.permutation(y2)).mean())
             for _ in range(2000)]
    p2 = (np.sum(np.array(perm2) >= acc2) + 1) / 2001
    print(f"order (A->B vs B->A) temporal-template LOO: {acc2:.0%}  "
          f"perm-p={p2:.4f} (chance 50%)")


if __name__ == "__main__":
    main()
