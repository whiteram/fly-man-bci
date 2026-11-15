"""bci/condstate33/balance_check.py — cs32 干净 MI 的类不均衡敏感性（零采集）。

cs32 的头条（干净 I_dec=1.26 bit）带 caveat："naive 池 44 样本
（类不均衡，MI 对 naive 偏保守）"。本脚本用**平衡子抽样**（每态
下采样到 n=44、K=20 重复）直接检验：1.26 是否受不均衡影响。

用法 (conda ffbm, repo root):
    python bci/condstate33/balance_check.py
输出: outputs/balance_check.json (+打印)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "bci" / "condstate14"))
sys.path.insert(0, str(ROOT / "bci" / "condstate32"))
from state_monitor import (ARMS, CLASSES, load_dev, fit_fold,  # noqa: E402
                           predict, build_xy)
from mi_nonoverlap import RECORDS, feats, mi_from_confusion  # noqa: E402


def main():
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    Xtr, ytr, seeds, _ = build_xy(records, 2.0, 1.0)
    fold63 = fit_fold(Xtr[seeds == 63], ytr[seeds == 63])

    X, y = [], []
    for cls in RECORDS:
        for p, t0, t1, cl in RECORDS[cls]:
            F = feats(load_dev(p, t0, t1))
            X.append(F)
            y.append(np.full(len(F), CLASSES.index(cl)))
    X = np.concatenate(X)
    y = np.concatenate(y)
    yp = predict(fold63, X)
    C_full = np.zeros((4, 4), int)
    for t, p_ in zip(y, yp):
        C_full[t, p_] += 1
    mi_full = mi_from_confusion(C_full)
    n_per = np.bincount(y)
    print("全样本 n/类:", n_per.tolist(), "I_dec =", round(mi_full, 4))

    rng = np.random.default_rng(7)
    n0 = int(n_per[0])                    # naive 类样本数 (44)
    mis = []
    for rep in range(20):
        idx = []
        for c in range(4):
            idx.append(rng.choice(np.flatnonzero(y == c), n0,
                                  replace=False))
        idx = np.concatenate(idx)
        C = np.zeros((4, 4), int)
        for t, p_ in zip(y[idx], yp[idx]):
            C[t, p_] += 1
        mis.append(mi_from_confusion(C))
    mis = np.array(mis)
    res = {"full_imbalanced_mi": round(mi_full, 4),
           "n_per_class_full": n_per.tolist(),
           "balanced_n": n0, "reps": 20,
           "balanced_mi_mean": round(float(mis.mean()), 4),
           "balanced_mi_sd": round(float(mis.std()), 4),
           "shift": round(float(mis.mean() - mi_full), 4)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (OUT / "balance_check.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", OUT / "balance_check.json")


if __name__ == "__main__":
    main()
