"""bci/condstate34/analyze.py — naive 300 s 分析：MI 诚实化 + 底稳定性。

1. 确定性: 前 20 s vs cs25 floor_scalp（同暗无 chem 协议）;
2. MI 诚实化: cs32 协议但 naive 用 300 s 记录（~145 非重叠窗，
   fold=cs14 seed63 折）——干净 MI 单值, 关闭 cs33 的 1.26-1.62 带;
3. 底稳定性: 层速率轨迹 + 头皮 8 特征滑窗斜率 → naive τ 下界
   （与三态寿命谱合流）; 监测器保持率（true=naive）。

用法 (conda ffbm, repo root):
    python bci/condstate34/analyze.py
输出: outputs/naive300.png, outputs/naive300_stats.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "outputs"
sys.path.insert(0, str(ROOT / "bci" / "condstate14"))
sys.path.insert(0, str(ROOT / "bci" / "condstate32"))
from state_monitor import (ARMS, CLASSES, ZHCN, COLORS, device,  # noqa: E402
                           features, load_dev, fit_fold, predict,
                           build_xy, make_bg)
from mi_nonoverlap import RECORDS, feats, mi_from_confusion  # noqa: E402

W_S, HOP_S = 2.0, 2.0
SIGMAS = [0.0, 0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.16, 11.4]
DRAWS = 4


def main():
    # ---- 0. 确定性 ------------------------------------------------------
    x_raw = np.load(OUT / "nll62_scalp.npy").astype(np.float64)
    ref = np.load(ROOT / "bci/condstate25/outputs/floor_scalp.npy")
    n20 = min(20000, len(ref), len(x_raw))
    det = float(np.abs(x_raw[:n20] - ref[:n20].astype(np.float64)).max())
    print(f"确定性: max|nll62 - floor| 前 {n20} ms = {det:.2e} V")
    x = device(x_raw * 1e6)

    # ---- 1. MI 诚实化 (naive=300 s 记录) --------------------------------
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    Xtr, ytr, seeds, _ = build_xy(records, 2.0, 1.0)
    fold63 = fit_fold(Xtr[seeds == 63], ytr[seeds == 63])

    recs = {  # cs32 的 RECORDS 但 naive 换成 300 s
        "naive": [(OUT / "nll62_scalp.npy", 3000, 299000, "naive")],
        **{k: v for k, v in RECORDS.items() if k != "naive"}}
    mi = {}
    for sg in SIGMAS:
        X, y = [], []
        for cls in recs:
            for d in range(DRAWS):
                for p, t0, t1, cl in recs[cls]:
                    base = load_dev(p, t0, t1)
                    xb = base + (make_bg(base.shape[0], 17, sg,
                                         9100 + d) if sg > 0 else 0.0)
                    F = feats(xb)
                    X.append(F)
                    y.append(np.full(len(F), CLASSES.index(cl)))
        X = np.concatenate(X)
        y = np.concatenate(y)
        yp = predict(fold63, X)
        C = np.zeros((4, 4), int)
        for t, p_ in zip(y, yp):
            C[t, p_] += 1
        mi[sg] = round(mi_from_confusion(C), 4)
        print(f"σ={sg:g}: I={mi[sg]:.4f}  (naive n={sum(y==0)})")

    # ---- 2. 底稳定性 -----------------------------------------------------
    lay = np.load(OUT / "nll62_layers.npz")
    wlen, step = int(W_S * 250), int(HOP_S * 250)
    Xn, tc = [], []
    for i in range(int(3 * 250), len(x) - wlen + 1, int(1.0 * 250)):
        Xn.append(features(x[i:i + wlen]))
        tc.append((i + wlen / 2) / 250.0)
    Xn, tc = np.array(Xn), np.array(tc)
    nv = np.array([CLASSES[i] for i in ytr]) == "naive"
    mu_n, sd_n = Xtr[nv].mean(0), Xtr[nv].std(0) + 1e-12
    names = ["lgRMS_δ", "lgRMS_bb", "lgRMS_hi", "峭度", "谱指数",
             "事件率", "峰dB", "峰Hz"]
    tt = (tc - tc.mean()) / tc.std()
    slopes = {names[k]: round(float(np.polyfit(tt, Xn[:, k], 1)[0]
                                    / Xn[:, k].std()), 3)
              for k in range(Xn.shape[1])}
    v = float(np.sqrt(sum(s ** 2 for s in slopes.values()))) \
        / float(tc.std())
    tau_naive = 3.0 / v if v > 0 else float("inf")
    pred = predict(fold63, Xn)
    ret_naive = float((pred == 0).mean())

    res = {
        "det_diff_V": det, "mi_honest": {str(k): vv for k, vv in
                                          mi.items()},
        "naive_n_windows": int(sum(1 for _ in feats(x))),
        "feature_slopes": slopes, "v_sig_SD_per_s": round(v, 5),
        "tau_naive_lower_bound_s": (round(tau_naive, 0)
                                    if np.isfinite(tau_naive) else None),
        "naive_retention": round(ret_naive, 4),
        "layers_first/mid/last": {k: [round(float(lay[k][a:b].mean()), 2)
                                      for a, b in ((2000, 10000),
                                                   (100000, 110000),
                                                   (280000, 290000))]
                                  for k in ("R", "L", "MID", "T4", "T5")},
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # ---- 3. 图 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                             constrained_layout=True)
    ax = axes[0]
    b32 = json.loads((ROOT / "bci/condstate32/outputs"
                      / "mi_nonoverlap_stats.json").read_text(
                          encoding="utf-8"))
    xs32 = [max(float(s), 5e-4) for s in b32["sigmas"]]
    ax.semilogx(xs32, [b32["mi"][str(s)] for s in b32["sigmas"]],
                "o-", color="0.6", label="cs32（naive=11 窗）")
    ax.semilogx([max(s, 5e-4) for s in SIGMAS],
                [mi[s] for s in SIGMAS], "s-", color="tab:blue",
                label=f"本弧（naive={res['naive_n_windows']} 窗）")
    ax.axhspan(1.26, 1.62, color="tab:orange", alpha=0.15)
    ax.text(2e-4, 1.44, "cs33 带 1.26-1.62", fontsize=7,
            color="tab:orange")
    ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
    ax.axvline(11.4, color="k", ls=":", lw=1.1)
    ax.set_xlabel("背景 (µV)", fontsize=8)
    ax.set_ylabel("I_dec (bit/2 s)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("MI 诚实化：naive 300 s", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[1]
    for k, c in (("L", "tab:red"), ("MID", "tab:orange"),
                 ("T4", "tab:green"), ("T5", "tab:purple")):
        r = lay[k].astype(np.float64)[2000:]
        rs = np.array([r[i * 2500:(i + 1) * 2500].mean()
                       for i in range(len(r) // 2500)])
        ax.plot(np.arange(len(rs)) * 10 + 5, rs, lw=0.8, color=c,
                label=k)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("层速率 (L=释放%)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("视觉通路强直底 300 s 轨迹（10 s 窗）", fontsize=9)
    ax.grid(True, color="0.93")

    ax = axes[2]
    zmax = np.abs((Xn - mu_n) / sd_n).max(1)
    ax.plot(tc, zmax, lw=0.8, color="tab:blue")
    ax.axhline(3.0, color="k", ls="--", lw=1)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("max |z| vs 训练 naive", fontsize=8)
    tau_txt = (f"{tau_naive:.0f} s" if np.isfinite(tau_naive) else "∞")
    ax.set_title(f"底特征漂移：τ(naive) 下界 {tau_txt}；"
                 f"naive 保持 {ret_naive:.0%}", fontsize=9)
    ax.grid(True, color="0.93")

    fig.suptitle("condstate34 naive 300 s——MI 诚实化 + 底的长时程稳定"
                 f"（确定性差 {det:.1e} V）", fontsize=11)
    fig.savefig(OUT / "naive300.png", dpi=150)
    plt.close(fig)
    (OUT / "naive300_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'naive300.png'}")


if __name__ == "__main__":
    main()
