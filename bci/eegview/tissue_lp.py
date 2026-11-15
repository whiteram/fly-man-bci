"""bci/eegview/tissue_lp.py — 优化轴①试点：组织低通需要多大的 τ？

likeness_audit 的根因一：源是单个紧凑尖峰驱动团 → 锁/闩锁臂谱
被尖峰梳压平（谱指数 −0.48/+0.06 vs 人脑 −1.0~−1.6）、棘波 4 ms
过锐（人脑病理棘波 20-70 ms）、共模去相关 4-16 ms 过快（人脑
数十 ms）。

本脚本零采集回答"要把这些压进人脑范围，正向核需要多少低通"：
在四条审计臂的源信号上施加一阶组织/突触低通
    H(f) = 1 / (1 + (2πfτ)²)      （α-突触核的频响）
τ ∈ {0, 5, 10, 20, 50, 100} ms，过 250 Hz 设备链后重算审计指标：

  - 谱指数 1-40 Hz（Theil-Sen）→ 人脑带 −1.0~−1.6
  - 闩锁臂 >4σ 棘波半峰宽 → 病理对照 20-70 ms
  - 共模自相关 1/e 时间 → 数十 ms
  - SSVEP 14 Hz 存活（解析衰减 1/(1+(2π·14·τ)²) + 实测带 SNR）

用途：给出 in-sim 源端平滑（正向核组织低通）的定标窗口；同时
报告 τ 与蝇突触/膜时间常数的对照（诚实边界：这是后置滤波的
等效量化，不是 in-sim 实现）。

用法 (conda ffbm, repo root):
    python bci/eegview/tissue_lp.py
输出: outputs/lp_sweep.png, outputs/lp_stats.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, welch, lfilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "outputs"
FS_N, FS_D = 1000.0, 250.0

ARMS = [
    ("naive", ROOT / "bci/condstate3/outputs/ctl62_scalp.npy",
     7000, 15000, "tab:blue"),
    ("high-lock", ROOT / "bci/condstate4/outputs/hl62_scalp.npy",
     7000, 15000, "tab:green"),
    ("res/latch", ROOT / "bci/condstate3/outputs/res62_scalp.npy",
     7000, 15000, "tab:red"),
    ("ssvep", ROOT / "bci/condstate13/outputs/ssvep_naive_scalp.npy",
     3500, 10500, "tab:purple"),
]
TAUS = [0.0, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]     # s


def device(x):
    sos = butter(4, [0.5, 100.0], btype="bandpass", fs=FS_N, output="sos")
    return resample_poly(sosfiltfilt(sos, x, axis=0), 1, 4, axis=0)


def lp(x, tau):
    """二阶低通 = 两个单极点 EMA 级联 (幅频 1/(1+(f/f0)²) 型),
    1 kHz 原生率上实现; b/a 由级联传函直接给出. 先去均值——零初始
    状态对直流/漂移分量会注入阶跃瞬态, 漏进设备带内."""
    if tau <= 0:
        return x.copy()
    mu = x.mean(0, keepdims=True)
    xz = x - mu
    a = float(np.exp(-1.0 / (FS_N * tau)))
    b = [(1 - a) ** 2]
    aa = [1.0, -2 * a, a * a]
    return lfilter(b, aa, xz, axis=0) + mu


def spec_fit(f, p, flo=1.0, fhi=40.0):
    from scipy.stats import theilslopes
    from scipy.signal import find_peaks
    m = (f >= flo) & (f <= fhi)
    pm = np.copy(p[m]).astype(float)
    for i in range(1, len(pm) - 1):
        pm[i] = np.median(p[m][i - 1:i + 2])
    lf, lp_ = np.log(f[m]), np.log(np.maximum(pm, 1e-40))
    slope, icpt, _, _ = theilslopes(lp_, lf)
    return float(slope)


def ac_1e_ms(x, max_lag=250):
    x = x - x.mean()
    v = (x * x).mean()
    below = None
    for l in range(1, max_lag + 1):
        if float((x[:-l] * x[l:]).mean()) / v < 1 / np.e:
            below = l
            break
    return (below if below is not None else max_lag) / FS_D * 1000


def spike_fwhm_ms(cm, thr=4.0):
    cm = (cm - cm.mean()) / (cm.std() + 1e-30)
    idx = np.flatnonzero(np.abs(cm) > thr)
    if not len(idx):
        return None
    events = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    widths = []
    for ev in events:
        seg = cm[ev[0]:ev[-1] + 1]
        half = abs(seg[np.argmax(np.abs(seg))]) / 2
        widths.append((np.abs(seg) >= half).sum() / FS_D * 1000)
    return float(np.median(widths))


def band_snr14(pm, f):
    m0 = (f >= 13.3) & (f <= 14.7)
    mb = ((f >= 2) & (f <= 40)) & ~((f >= 13.3) & (f <= 14.7))
    return float(10 * np.log10(pm[m0].max() / np.median(pm[mb])))


def main():
    res = {"taus_ms": [t * 1000 for t in TAUS], "arms": {}}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6),
                             constrained_layout=True)
    expo_ax, fwhm_ax, ac_ax, tr_ax = (axes[0][0], axes[0][1],
                                      axes[1][0], axes[1][1])
    for tag, path, t0, t1, color in ARMS:
        e = np.load(path).astype(np.float64)[t0:t1] * 1e6
        rows = []
        for tau in TAUS:
            x = device(lp(e, tau))
            cm = x.mean(1)
            f, P = welch(x, fs=FS_D, nperseg=min(len(x), 2048), axis=0)
            pm = np.median(P, axis=1)
            expo = spec_fit(f, pm)
            fwhm = spike_fwhm_ms(cm)
            ac = ac_1e_ms(cm)
            r = dict(tau_ms=round(tau * 1000, 1), expo=round(expo, 2),
                     fwhm_ms=(round(fwhm, 1) if fwhm else None),
                     ac_1e_ms=round(ac, 1),
                     band_rms_uv=round(float(np.sqrt(
                         (x ** 2).mean(0)).mean()), 5))
            if tag == "ssvep":
                r["snr14_db"] = round(band_snr14(pm, f), 1)
            rows.append(r)
            print(f"{tag:10s} τ={tau*1000:5.1f}ms expo={expo:+.2f} "
                  f"fwhm={fwhm and round(fwhm,1)} ac1e={ac:.0f}ms "
                  f"rms={r['band_rms_uv']}")
        res["arms"][tag] = rows

        expo_ax.plot([t * 1000 for t in TAUS], [r["expo"] for r in rows],
                     "o-", color=color, label=tag)
        if tag == "res/latch":
            fw = [r["fwhm_ms"] or np.nan for r in rows]
            fwhm_ax.plot([t * 1000 for t in TAUS], fw, "o-", color=color)
            ac_ax.plot([t * 1000 for t in TAUS],
                       [r["ac_1e_ms"] for r in rows], "o-", color=color)
        # 波形示例 (τ=0 vs 50 ms 档), 各臂按自身 τ=0 RMS 归一显示
        x0 = device(lp(e, 0.0))
        x5 = device(lp(e, 5e-2))
        n_show = min(3000, len(x0))
        tt_ = np.arange(n_show) / FS_D
        s0 = np.std(x0[:, 8]) + 1e-30
        tr_ax.plot(tt_, x0[:n_show, 8] / s0,
                  color=color, alpha=0.35, lw=0.7)
        tr_ax.plot(tt_, x5[:n_show, 8] / s0,
                  color=color, lw=0.9)

    expo_ax.axhspan(-1.6, -1.0, color="0.85", zorder=0)
    expo_ax.text(30, -1.35, "人脑 −1.0~−1.6", fontsize=8)
    expo_ax.axhline(0, color="0.7", lw=0.5)
    expo_ax.set_xlabel("组织低通 τ (ms)", fontsize=8)
    expo_ax.set_ylabel("谱指数 1-40 Hz", fontsize=8)
    expo_ax.legend(fontsize=7)
    expo_ax.set_title("谱指数 vs τ（灰带=人脑范围）", fontsize=9)

    fwhm_ax.axhspan(20, 70, color="0.85", zorder=0)
    fwhm_ax.text(52, 30, "人脑病理棘波\n20-70 ms", fontsize=8, va="bottom")
    fwhm_ax.set_xlabel("组织低通 τ (ms)", fontsize=8)
    fwhm_ax.set_ylabel("闩锁棘波半峰宽 (ms)", fontsize=8)
    fwhm_ax.set_title("棘波锐度 vs τ（残留/闩锁臂）", fontsize=9)

    ac_ax.axhspan(20, 100, color="0.85", zorder=0)
    ac_ax.text(52, 115, "人脑数十 ms", fontsize=8, va="top", ha="right")
    ac_ax.set_xlabel("组织低通 τ (ms)", fontsize=8)
    ac_ax.set_ylabel("共模自相关 1/e (ms)", fontsize=8)
    ac_ax.set_title("时间去相关时标 vs τ（残留/闩锁臂）", fontsize=9)

    tr_ax.set_xlabel("时间 (s)", fontsize=8)
    tr_ax.set_ylabel("通道 8（×各臂 τ=0 RMS）", fontsize=8)
    tr_ax.set_title("波形示例：淡=无滤波，实=τ=50 ms", fontsize=9)

    fig.suptitle("优化轴①试点：正向核组织低通定标（α-核二阶低通, "
                 "后置等效量化）", fontsize=11)
    fig.savefig(OUT / "lp_sweep.png", dpi=150)
    plt.close(fig)
    (OUT / "lp_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'lp_sweep.png'}")
    print(f"-> {OUT / 'lp_stats.json'}")


if __name__ == "__main__":
    main()
