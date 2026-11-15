"""bci/condstate32/mi_nonoverlap.py — 非重叠窗的决策信道 MI 复测（零采集）。

cs18 的 MI 用 1 s 步进 50% 重叠窗（"窗独立 ~2× 乐观"是手挥注）且
每态仅 ~40 窗（20 s 臂）。本弧用 300 s 长记录（cs15/20/21）+
**非重叠窗（W=hop=2 s，~140 窗/态）**复测：

  1. 决策信道 MI vs σ_bg（粗网格聚焦干净/膝点/11.4）；
  2. **诚实 bits/τ_sig**：I(2 s) × τ/(2 s)——无重叠=无冗余修正
     （对照 cs18 的 293 bit@τ=336 s）；
  3. fold = cs14 seed63 折（62 记录评测，无泄漏）；hll63 作复证。

数据: hll62/qll62/resl62 (300 s) + naive 池（condstate3 ctl62 +
condstate2 ctl62 两独立 trial；condstate4 ctl62 为 cs3ctl 逐位
副本已排除）。

用法 (conda ffbm, repo root):
    python bci/condstate32/mi_nonoverlap.py
输出: outputs/mi_nonoverlap.png, outputs/mi_nonoverlap_stats.json
"""
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "bci" / "condstate14"))
from state_monitor import (ARMS, CLASSES, device, features,  # noqa: E402
                           load_dev, fit_fold, predict, build_xy, make_bg)

W_S, HOP_S = 2.0, 2.0            # 非重叠
SIGMAS = [0.0, 0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.16, 11.4]
DRAWS = 4
TAU = {"qlock": 2988.0, "hlock": 336.0, "resid": 2948.0}
RECORDS = {   # (path, t0_ms, t1_ms, class)
    "naive": [(ROOT / "bci/condstate3/outputs/ctl62_scalp.npy",
               4000, 16500, "naive"),
              (ROOT / "bci/condstate2/outputs/ctl62_scalp.npy",
               4000, 14000, "naive")],
    "hlock": [(ROOT / "bci/condstate15/outputs/hll62_scalp.npy",
               3000, 299000, "hlock")],
    "qlock": [(ROOT / "bci/condstate21/outputs/qll62_scalp.npy",
               3000, 299000, "qlock")],
    "resid": [(ROOT / "bci/condstate20/outputs/resl62_scalp.npy",
               8000, 299000, "resid")],
}


def mi_from_confusion(C):
    P = C / C.sum()
    p, q = P.sum(1), P.sum(0)
    m = P > 0
    return float((P[m] * np.log2(P[m] / (p[:, None] * q[None, :])[m])).sum())


def feats(x):
    wlen, step = int(W_S * 250), int(HOP_S * 250)
    return np.array([features(x[i:i + wlen])
                     for i in range(0, len(x) - wlen + 1, step)])


def main():
    # fold: cs14 seed63 折 (62 记录评测, 无泄漏)
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    Xtr, ytr, seeds, _ = build_xy(records, 2.0, 1.0)
    fold63 = fit_fold(Xtr[seeds == 63], ytr[seeds == 63])

    dev = {k: [load_dev(p, t0, t1) for p, t0, t1, _ in v]
           for k, v in RECORDS.items()}
    res = {"window_s": W_S, "hop_s": HOP_S, "sigmas": SIGMAS,
           "mi": {}, "bits_tau": {}, "n_windows": {}}
    print(f"{'σ_bg':>9} {'I_dec':>8}")
    for sg in SIGMAS:
        X, y = [], []
        for cls in RECORDS:
            for d in range(DRAWS):
                for p, t0, t1, cl in RECORDS[cls]:
                    base = load_dev(p, t0, t1)
                    xb = base + (make_bg(base.shape[0], 17, sg,
                                         8100 + d) if sg > 0 else 0.0)
                    F = feats(xb)
                    X.append(F)
                    y.append(np.full(len(F), CLASSES.index(cl)))
        X = np.concatenate(X)
        y = np.concatenate(y)
        yp = predict(fold63, X)
        C = np.zeros((4, 4), int)
        for t, p_ in zip(y, yp):
            C[t, p_] += 1
        i_dec = mi_from_confusion(C)
        res["mi"][str(sg)] = round(i_dec, 4)
        print(f"{sg:9.4f} {i_dec:8.4f}")

    i_clean = res["mi"]["0.0"]
    for cls, tau in TAU.items():
        res["bits_tau"][cls] = round(i_clean / W_S * tau, 1)
    res["n_windows"] = {cls: int(sum(len(feats(load_dev(p, t0, t1)))
                                     for p, t0, t1, _ in RECORDS[cls]))
                        for cls in RECORDS}
    print("诚实 bits/τ:", res["bits_tau"], "(cs18 参照 293@hlock)")
    print("窗数:", res["n_windows"])

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                             constrained_layout=True)
    ax = axes[0]
    b18 = json.loads((ROOT / "bci/condstate18/outputs"
                      / "capacity_stats.json").read_text(encoding="utf-8"))
    sg18 = [max(float(s), 5e-4) for s in b18["sigmas"]]
    ax.semilogx(sg18, [b18["mi_decision"][str(s)] for s in b18["sigmas"]],
                "o-", color="0.6", label="cs18（重叠窗, 20 s 臂）")
    ax.semilogx([max(s, 5e-4) for s in SIGMAS],
                [res["mi"][str(s)] for s in SIGMAS], "s-",
                color="tab:blue", label="本弧（非重叠, 300 s）")
    ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
    ax.axvline(11.4, color="k", ls=":", lw=1.1)
    ax.set_xlabel("背景 (µV)", fontsize=8)
    ax.set_ylabel("I_dec (bit/2 s 窗)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("MI 曲线：非重叠复测 vs cs18", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[1]
    labs, vals, cols = [], [], []
    for cls, tau in TAU.items():
        labs.append(f"{cls}\nτ={tau:.0f}s")
        vals.append(res["bits_tau"][cls])
        cols.append({"qlock": "tab:cyan", "hlock": "tab:green",
                     "resid": "tab:red"}[cls])
    ax.bar(range(len(vals)), vals, 0.55, color=cols)
    ax.axhline(293, color="0.5", ls="--", lw=1)
    ax.text(1.0, 300, "cs18 参照 293", fontsize=7, color="0.4",
            ha="center")
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("bit / τ_sig（非重叠=无冗余修正）", fontsize=8)
    ax.set_title("诚实 bits/τ_sig", fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    ax = axes[2]
    xs = [max(s, 5e-4) for s in SIGMAS]
    ys = [res["mi"][str(s)] for s in SIGMAS]
    ax.semilogx(xs, ys, "s-", color="tab:blue")
    ax.axhspan(-0.05, 0.08, color="tab:red", alpha=0.12)
    ax.text(1.3e-4, -0.035, "I=0 区", fontsize=7, color="tab:red")
    zero_pt = max(s for s in SIGMAS if res["mi"][str(s)] == 0)
    ax.axvline(zero_pt, color="k", ls=":", lw=1.1)
    ax.text(zero_pt * 1.1, 0.8, f"归零点 {zero_pt:g} µV\n（cs18: 0.1）",
            fontsize=7)
    ax.set_xlabel("背景 (µV)", fontsize=8)
    ax.set_ylabel("I_dec (bit)", fontsize=8)
    ax.set_title("归零点复测", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    fig.suptitle("condstate32 非重叠窗 MI 复测——bits/τ_sig 从假设修正"
                 "变实测（300 s 记录, ~140 窗/态）", fontsize=11)
    fig.savefig(OUT / "mi_nonoverlap.png", dpi=150)
    plt.close(fig)
    (OUT / "mi_nonoverlap_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'mi_nonoverlap.png'}")


if __name__ == "__main__":
    main()
