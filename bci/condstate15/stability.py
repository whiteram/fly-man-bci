"""bci/condstate15/stability.py — 高锁静息签名 300 s 平稳性分析。

数据: acquire.py 的 hll62 (cs4hlnt, DA1@1600 单脉冲, 300 s, seed 62)。
问题三连:
  1. 签名平稳吗? 池率 (ALPN/KC/MBON) + 头皮滑窗特征 (带 RMS/
     峭度/谱指数/事件率) 在 300 s 内的漂移 vs condstate14 训练期
     (8-17 s) 的类内散布;
  2. 监测器时间外推: condstate14 的 seed63 折模型(训练数据完全
     不含本 run)逐窗打标签——高锁保持率/漂移方向;
  3. 主节律漂移: 共模 5-40 Hz 主峰频率的 30 s 分辨率轨迹。

附加: 确定性复证——hll62 前 22 s 应逐位等于 condstate4/hlnt62
(同 chem 同种子, t_end 不影响早期动力学), 差值应为 0。

用法 (conda ffbm, repo root):
    python bci/condstate15/stability.py
输出: outputs/stability_300s.png, outputs/stability_stats.json
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
from state_monitor import (ARMS, CLASSES, ZHCN, COLORS, FS_D,  # noqa: E402
                           device, features, load_dev, fit_fold,
                           predict, build_xy)

W_S, HOP_S = 2.0, 1.0
T_START = 3.0


def main():
    # ---- 0. 确定性复证: 前 22 s vs condstate4/hlnt62 -----------------
    x_raw = np.load(OUT / "hll62_scalp.npy").astype(np.float64)
    hlnt = np.load(ROOT / "bci/condstate4/outputs/hlnt62_scalp.npy")
    n22 = min(22000, len(hlnt), len(x_raw))
    det_diff = float(np.abs(x_raw[:n22] - hlnt[:n22].astype(np.float64))
                     .max())
    print(f"确定性复证: max|hll62 - hlnt62| 前 {n22} ms = "
          f"{det_diff:.2e} V")
    x = device(x_raw * 1e6)                      # (75k, 17) µV @250 Hz

    # ---- 1. 重建 condstate14 seed63 折 (训练不含本 run) ---------------
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    Xtr, ytr, seeds, _ = build_xy(records, W_S, HOP_S)
    fold = fit_fold(Xtr[seeds == 63], ytr[seeds == 63])
    hk = np.array([CLASSES[i] for i in ytr]) == "hlock"
    mu_hk, sd_hk = Xtr[hk].mean(0), Xtr[hk].std(0) + 1e-12
    feat_names = ["lgRMS_δ", "lgRMS_bb", "lgRMS_hi", "峭度", "谱指数",
                  "事件率", "峰dB", "峰Hz"]

    # ---- 2. 长跑滑窗特征 + 监测器时间线 --------------------------------
    wlen, step = int(W_S * FS_D), int(HOP_S * FS_D)
    i0 = int(T_START * FS_D)
    X, tc = [], []
    for i in range(i0, len(x) - wlen + 1, step):
        X.append(features(x[i:i + wlen]))
        tc.append((i + wlen / 2) / FS_D)
    X = np.array(X)
    tc = np.array(tc)
    pred = predict(fold, X)
    zhk = (X - mu_hk) / sd_hk                     # 相对训练期 hlock 的 z
    zmax = np.abs(zhk).max(1)
    ret_all = float((pred == 2).mean())
    print(f"监测器保持率 (hlock): {ret_all:.3f}  "
          f"({(pred == 2).sum()}/{len(pred)} 窗)")
    for cls_i, c in enumerate(CLASSES):
        n = int((pred == cls_i).sum())
        if n:
            print(f"   误漂 -> {c}: {n} 窗 "
                  f"(首个 t={tc[pred == cls_i][0]:.0f}s)")

    # ---- 3. 池率 + 主节律轨迹 ------------------------------------------
    pop = np.load(OUT / "hll62_pop.npz")
    pop_s = {}
    for g in ("ALPN", "Kenyon_Cell", "MBON", "DAN"):
        # pop 是 1 ms 网格, tc 是秒 -> 索引 ×1000
        pop_s[g] = np.array([pop[g][int(t0 * 1000):
                                    int(t0 * 1000) + 2000].mean()
                             for t0 in tc])
    # 30 s 主节律 (共模 PSD 5-40 Hz argmax)
    rfreq, rtimes = [], []
    cm_full = x.mean(1)
    for t0 in np.arange(5.0, 297.0, 10.0):
        a = int(t0 * FS_D)
        seg = cm_full[a:a + int(10 * FS_D)]
        f, P = welch(seg, fs=FS_D, nperseg=500)
        m = (f >= 5) & (f <= 40)
        rfreq.append(f[m][np.argmax(P[m])])
        rtimes.append(t0 + 5.0)
    rfreq = np.array(rfreq)

    # ---- 4. 漂移统计 ----------------------------------------------------
    tt = (tc - tc.mean()) / (tc.std() + 1e-12)
    slopes = {}
    for k in range(X.shape[1]):
        sl = float(np.polyfit(tt, X[:, k], 1)[0] / X[:, k].std())
        slopes[feat_names[k]] = round(sl, 3)      # 每 1 SD(时间) 的 SD(特征)
    # 首末 60 s 对比
    m_e, m_l = tc <= tc[0] + 60, tc >= tc[-1] - 60
    drift = {feat_names[k]: round(float(X[m_l, k].mean()
                                        - X[m_e, k].mean()), 4)
             for k in range(X.shape[1])}
    res = {
        "det_diff_V": det_diff,
        "n_windows": len(tc),
        "hlock_retention": round(ret_all, 4),
        "false_drift": {CLASSES[i]: int((pred == i).sum())
                        for i in range(4) if i != 2 and (pred == i).any()},
        "zmax_median": round(float(np.median(zmax)), 2),
        "zmax_p95": round(float(np.percentile(zmax, 95)), 2),
        "windows_beyond_3sigma": int((zmax > 3).sum()),
        "feature_slope_per_sd_time": slopes,
        "first60_vs_last60": drift,
        "ALPN_first/mid/last": [round(float(pop_s['ALPN'][m].mean()), 2)
                                for m in (m_e,
                                          (tc > 140) & (tc < 160), m_l)],
        "rhythm_hz_start/mid/end": [float(rfreq[0]),
                                    float(rfreq[len(rfreq) // 2]),
                                    float(rfreq[-1])],
        "rhythm_hz_std": round(float(rfreq.std()), 2),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # ---- 5. 图 ----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4),
                             constrained_layout=True)
    ax = axes[0][0]
    for g, c in (("ALPN", "tab:red"), ("Kenyon_Cell", "tab:orange"),
                 ("MBON", "tab:purple")):
        ax.plot(tc, pop_s[g], lw=0.9, color=c, label=g)
    ax.axhspan(60, 140, color="tab:red", alpha=0.08)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("池率 (Hz)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("池率轨迹（2 s 窗；红带=高锁签名带 60-140）",
                 fontsize=9)

    ax = axes[0][1]
    ax.plot(tc, X[:, 1], lw=0.8, color="tab:blue")
    ax.axhspan(mu_hk[1] - 2 * sd_hk[1], mu_hk[1] + 2 * sd_hk[1],
               color="tab:blue", alpha=0.15,
               label="训练期 hlock ±2σ (8-17 s)")
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("lgRMS 2-20 Hz", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("头皮宽带 RMS 轨迹", fontsize=9)

    ax = axes[0][2]
    ax.plot(tc, X[:, 3], lw=0.8, color="tab:green", label="峭度")
    ax.axhspan(mu_hk[3] - 2 * sd_hk[3], mu_hk[3] + 2 * sd_hk[3],
               color="tab:green", alpha=0.12)
    ax2 = ax.twinx()
    ax2.plot(tc, X[:, 5], lw=0.8, color="tab:brown", label="事件率")
    ax2.axhspan(mu_hk[5] - 2 * sd_hk[5], mu_hk[5] + 2 * sd_hk[5],
                color="tab:brown", alpha=0.12)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("峭度", fontsize=8, color="tab:green")
    ax2.set_ylabel(">4σ 事件/s", fontsize=8, color="tab:brown")
    ax.set_ylim(0, np.max(X[:, 3]) * 1.12)
    ax.set_title("形态特征轨迹（带=训练期 ±2σ）", fontsize=9)

    ax = axes[1][0]
    ax.plot(rtimes, rfreq, "o-", ms=3, color="tab:purple")
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("主节律 (Hz)", fontsize=8)
    ax.set_ylim(0, 45)
    ax.set_title(f"共模主节律 30 s 轨迹 (σ={rfreq.std():.1f} Hz)",
                 fontsize=9)

    ax = axes[1][1]
    ax.plot(tc, np.array([(p == 2) for p in pred], float) - 0.02,
            lw=0.4, color="0.8")
    for p_, t0, t1 in zip(pred, tc - HOP_S / 2, tc + HOP_S / 2):
        ax.axvspan(t0, t1, ymin=0.02, ymax=0.98,
                   color=COLORS[CLASSES[p_]], alpha=0.85, lw=0)
    ax.set_yticks(range(4))
    ax.set_yticklabels([ZHCN[c] for c in CLASSES], fontsize=8)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_title(f"condstate14 监测器（seed63 折）时间外推——"
                 f"hlock 保持 {ret_all:.0%}", fontsize=9)

    ax = axes[1][2]
    ax.plot(tc, zmax, lw=0.8, color="tab:red")
    ax.axhline(3.0, color="k", ls="--", lw=1)
    ax.text(150, 3.3, "3σ（训练期 hlock 分布）", fontsize=7)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("max |z|（8 特征）", fontsize=8)
    ax.set_title(f"相对训练分布的最大偏离（中位 "
                 f"{np.median(zmax):.1f}σ / p95 "
                 f"{np.percentile(zmax, 95):.1f}σ）", fontsize=9)

    fig.suptitle("condstate15 高锁静息签名 300 s 平稳性（cs4hlnt, "
                 f"seed 62；确定性复证差 {det_diff:.1e} V）", fontsize=11)
    fig.savefig(OUT / "stability_300s.png", dpi=150)
    plt.close(fig)
    (OUT / "stability_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'stability_300s.png'}")
    print(f"-> {OUT / 'stability_stats.json'}")


if __name__ == "__main__":
    main()
