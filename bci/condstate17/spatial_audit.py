"""bci/condstate17/spatial_audit.py — 空间维度审计：任何 montage/空间滤波都救不了。

condstate16 封死了时间维度（积分被签名寿命封顶）。最后的逃生
通道是**空间维度**：换导联配置（最佳单导/双极/CAR/最佳加权
组合）能否把共模蝇信号从部分去相关的人脑背景里挖出来？

解析预期（信号模式 g∝1、背景 C=a·J+b·I, a=ρ）：
  匹配滤波 SNR² 增益 = 17/(b+17a) → ρ=0.85: ×1.08, ρ=0.5: ×1.37,
  ρ=0.2: ×2.0 —— 减背景共模的同时等比例杀共模信号，空间维度
  买不到量级。

实测：五类检测器 × 背景共享度 ρ × 强度 σ 的 d'(2 s 窗,
state-vs-naive) 全扫描（零采集，重用 condstate14 数据链）:

  1. best-single  最佳单导（干净信号定导, 固定）
  2. common-mode  17 导均值（eegview/审计口径）
  3. matched      匹配滤波 w ∝ Ĉ_bg⁻¹(μ_state−μ_naive)
                  （背景协方差由 bg-only draw 估计, shrinkage 正则）
  4. best-bipolar 最佳双极对（干净信号定对, 固定）
  5. CAR          全局平均参考后最佳导（已知有害, 对照）

用法 (conda ffbm, repo root):
    python bci/condstate17/spatial_audit.py
输出: outputs/spatial_audit.png, outputs/spatial_stats.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

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
from state_monitor import device, load_dev  # noqa: E402

W_S, HOP_S = 2.0, 1.0
FS_D = 250.0
STATES = {                        # 稳态窗（避开训练/探针段）
    "naive": (ROOT / "bci/condstate3/outputs/ctl62_scalp.npy",
              4000, 16500),
    "hlock": (ROOT / "bci/condstate4/outputs/hl62_scalp.npy",
              3000, 16500),
    "resid": (ROOT / "bci/condstate3/outputs/res62_scalp.npy",
              8000, 16500),
}
SIGMAS = np.array([0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.16, 11.4])
RHOS = (0.85, 0.5, 0.2)
DRAWS = 5
BB = (2.0, 20.0)


def make_bg(n, ch, sigma, rho, seed):
    """1/f² 2-20 Hz RMS=sigma 背景, 共享度 rho (amp 混合)."""
    rng = np.random.default_rng(seed)
    fr = np.fft.rfftfreq(n, 1 / FS_D)
    shape = np.where((fr >= 0.4) & (fr <= 45), 1.0 / (fr + 0.5) ** 2, 0.0)
    shape[0] = 0.0

    def one():
        return np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * shape, n)

    shared = one()
    a, b = np.sqrt(rho), np.sqrt(1 - rho)
    sos = butter(4, list(BB), btype="bandpass", fs=FS_D, output="sos")
    bg = np.zeros((n, ch))
    for c in range(ch):
        y = a * shared + b * one()
        y *= sigma / (np.sqrt((sosfiltfilt(sos, y) ** 2).mean()) + 1e-30)
        bg[:, c] = y
    return bg


def win_rms_vec(x):
    """逐窗 17 导 2-20 Hz RMS -> (n_win, 17)."""
    sos = butter(4, list(BB), btype="bandpass", fs=FS_D, output="sos")
    xb = sosfiltfilt(sos, x, axis=0)
    wlen, step = int(W_S * FS_D), int(HOP_S * FS_D)
    return np.array([np.sqrt((xb[i:i + wlen] ** 2).mean(0))
                     for i in range(0, len(xb) - wlen + 1, step)])


def dprime(a, b):
    return float((a.mean() - b.mean())
                 / np.sqrt(0.5 * (a.var() + b.var()) + 1e-30))


def main():
    # ---- 0. 干净记录 + 信号空间模式 ------------------------------------
    rec = {k: win_rms_vec(load_dev(p, t0, t1))
           for k, (p, t0, t1) in STATES.items()}
    nch = rec["naive"].shape[1]
    mu = {k: v.mean(0) for k, v in rec.items()}
    # 信号模式 g（相对 naive 的每导增量, 归一）——共模占比
    pat = {}
    for k in ("hlock", "resid"):
        g = mu[k] - mu["naive"]
        pat[k] = g / (np.linalg.norm(g) + 1e-30)
        cmf = float((g.mean()) ** 2 * nch / (g @ g))
        print(f"{k} 信号模式: 共模占比 {cmf:.1%}, 每导增益范围 "
              f"{g.min()/g.mean():.2f}~{g.max()/g.mean():.2f}")

    # 干净检测器定标（固定导/对, 不接触测试 draw）
    clean = {}
    for k in ("hlock", "resid"):
        d_single = [dprime(rec[k][:, c], rec["naive"][:, c])
                    for c in range(nch)]
        best_c = int(np.argmax(d_single))
        d_bi, best_pair = -9, None
        for i in range(nch):
            for j in range(i + 1, nch):
                dd = dprime(rec[k][:, i] - rec[k][:, j],
                            rec["naive"][:, i] - rec["naive"][:, j])
                if dd > d_bi:
                    d_bi, best_pair = dd, (i, j)
        clean[k] = dict(best_single_ch=best_c,
                        d_single=float(d_single[best_c]),
                        d_cm=dprime(rec[k].mean(1), rec["naive"].mean(1)),
                        best_pair=best_pair, d_bipolar=float(d_bi))
        rc, rn = rec[k] - rec[k].mean(1, keepdims=True), \
            rec["naive"] - rec["naive"].mean(1, keepdims=True)
        clean[k]["d_car"] = max(
            (dprime(rc[:, c], rn[:, c]) for c in range(nch)))
        print(f"{k} 干净 d': 单导{best_c}={d_single[best_c]:.2f} "
              f"共模={clean[k]['d_cm']:.2f} "
              f"双极{best_pair}={d_bi:.2f} "
              f"CAR最佳导={clean[k]['d_car']:.2f}")

    res = {"signal_pattern": {
        k: {"cm_frac": round(float((mu[k]-mu['naive']).mean()**2 * nch
                                   / ((mu[k]-mu['naive']) @ (mu[k]-mu['naive']))), 3),
            "gain_min/max": [round(float((mu[k]-mu['naive']).min()
                                         / (mu[k]-mu['naive']).mean()), 2),
                             round(float((mu[k]-mu['naive']).max()
                                         / (mu[k]-mu['naive']).mean()), 2)]}
        for k in ("hlock", "resid")},
        "clean_dprime": {k: {kk: (round(vv, 2) if isinstance(vv, float)
                                  else vv)
                             for kk, vv in v.items()}
                         for k, v in clean.items()},
        "knee_sigma": {}, "analytic_mf_gain": {}}

    n_win = {k: len(v) for k, v in rec.items()}
    # 缓存设备链记录（内层循环不再重复加载）
    xdev = {k: load_dev(p, t0, t1) for k, (p, t0, t1) in STATES.items()}
    lens = {k: len(v) for k, v in xdev.items()}

    # ---- 1. 扫描: rho × sigma × 检测器 → d' ------------------------------
    DETS = ("best-single", "common-mode", "matched", "best-bipolar", "CAR")
    dcurves = {(st, rho, det): np.zeros(len(SIGMAS))
               for st in ("hlock", "resid") for rho in RHOS
               for det in DETS}
    for rho in RHOS:
        for si, sg in enumerate(SIGMAS):
            # 背景协方差估计 (bg-only, 100 窗)
            bgR = []
            for d in range(10):
                bg = make_bg(int(20 * FS_D), nch, sg, rho, 7000 + d)
                bgR.append(win_rms_vec(bg))
            bgR = np.concatenate(bgR)
            S = np.cov(bgR.T)
            Sh = 0.9 * S + 0.1 * np.diag(np.diag(S))   # shrinkage
            Cinv = np.linalg.inv(Sh + 1e-12 * np.eye(nch))
            for st in ("hlock", "resid"):
                # 信号+背景 draws（前 2 拟合匹配滤波, 后 3 评测——防 in-sample）
                Rs, Rn = [], []
                for d in range(DRAWS):
                    for tgt, key in ((st, "s"), ("naive", "n")):
                        bg = make_bg(lens[tgt], nch, sg, rho,
                                     100 * d + int(tgt == st))
                        R = win_rms_vec(xdev[tgt] + bg)
                        (Rs if key == "s" else Rn).append(R)
                Rfit = np.concatenate(Rs[:2] + Rn[:2])
                w = Cinv @ (np.concatenate(Rs[:2]).mean(0)
                            - np.concatenate(Rn[:2]).mean(0))
                Rs = np.concatenate(Rs[2:])
                Rn = np.concatenate(Rn[2:])
                bc, bp = clean[st]["best_single_ch"], clean[st]["best_pair"]
                stats = {
                    "best-single": dprime(Rs[:, bc], Rn[:, bc]),
                    "common-mode": dprime(Rs.mean(1), Rn.mean(1)),
                    "matched": dprime(Rs @ w, Rn @ w),
                    "best-bipolar": dprime(Rs[:, bp[0]] - Rs[:, bp[1]],
                                           Rn[:, bp[0]] - Rn[:, bp[1]]),
                    "CAR": dprime((Rs - Rs.mean(1, keepdims=True))[:, bc],
                                  (Rn - Rn.mean(1, keepdims=True))[:, bc]),
                }
                for det in DETS:
                    dcurves[(st, rho, det)][si] = stats[det]
        print(f"rho={rho}: "
              f"hlock d'(11.4µV) = "
              f"{ {det: round(float(dcurves[('hlock', rho, det)][-1]), 2) for det in DETS} }")

    # 膝点 (d'=2 的最大 sigma, 线性插值 log 轴) + 解析增益
    for st in ("hlock", "resid"):
        for rho in RHOS:
            for det in DETS:
                y = dcurves[(st, rho, det)]
                knee = None
                for i in range(len(y)):
                    if y[i] < 2.0:
                        if i == 0:
                            knee = f"<{SIGMAS[0]:g}"   # 干净达标, 首测 σ 即失效
                            break
                        knee = round(float(np.sqrt(SIGMAS[i - 1]
                                                   * SIGMAS[i])), 4)
                        break
                else:
                    knee = None                        # 任何 σ 下都未达 d'=2
                res["knee_sigma"][f"{st}|rho{rho}|{det}"] = knee
        b = 1 - 0.85
        res["analytic_mf_gain"]["rho0.85"] = round(
            float(np.sqrt(17 / (b + 17 * 0.85))), 2)
        for rho in RHOS:
            res["analytic_mf_gain"][f"rho{rho}"] = round(
                float(np.sqrt(17 / ((1 - rho) + 17 * rho))), 2)
    print("膝点表:", json.dumps(res["knee_sigma"], ensure_ascii=False))
    print("解析匹配滤波 d' 增益:", res["analytic_mf_gain"])

    # ---- 2. 图 ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                             constrained_layout=True)
    det_style = {"best-single": ("tab:blue", "-"),
                 "common-mode": ("tab:green", "-"),
                 "matched": ("tab:red", "-"),
                 "best-bipolar": ("tab:orange", "--"),
                 "CAR": ("0.5", ":")}
    ax = axes[0]
    for det, (c, ls) in det_style.items():
        ax.loglog(SIGMAS, dcurves[("hlock", 0.85, det)], ls, color=c,
                  label=det, lw=1.1 if det != "matched" else 1.8)
    ax.axhline(2.0, color="k", ls="--", lw=0.8)
    ax.axvline(11.4, color="k", ls=":", lw=1.1)
    ax.text(12, 8, "项目背景\n11.4 µV", fontsize=7)
    ax.set_xlabel("背景 2-20 Hz RMS (µV)", fontsize=8)
    ax.set_ylabel("d' (hlock vs naive, 2 s 窗)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("五类检测器 × 背景强度（ρ=0.85 项目模型）", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[1]
    x = np.arange(len(RHOS))
    for det, (c, ls) in det_style.items():
        v = [dcurves[("hlock", rho, det)][4] for rho in RHOS]  # σ=0.32
        ax.plot(x, v, "o", color=c, label=det)
    ax.axhline(2.0, color="k", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ρ={r}" for r in RHOS], fontsize=8)
    ax.set_ylabel("d' @ σ=0.32 µV", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("背景共享度越低, 空间滤波增益越大——但量级仍不够",
                 fontsize=9)
    ax.grid(True, color="0.93")

    ax = axes[2]
    g = mu["resid"] - mu["naive"]
    g = g / g.mean()
    ax.bar(range(nch), g, color="tab:red", alpha=0.7)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("电极 E0-E16", fontsize=8)
    ax.set_ylabel("信号增益 (×共模均值)", fontsize=8)
    ax.set_ylim(0, 2.4)
    ax.set_title(f"resid 信号空间模式（共模占比 "
                 f"{res['signal_pattern']['resid']['cm_frac']:.0%}，"
                 f"±2× 导联场起伏）", fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    fig.suptitle("condstate17 空间维度审计——信号近乎全共模 → 空间滤波"
                 "买不到量级（最优膝点 0.018/0.057 µV, 距 11.4 µV 差 200-2000×）",
                 fontsize=11)
    fig.savefig(OUT / "spatial_audit.png", dpi=150)
    plt.close(fig)
    res["n_windows"] = n_win
    (OUT / "spatial_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'spatial_audit.png'}")
    print(f"-> {OUT / 'spatial_stats.json'}")


if __name__ == "__main__":
    main()
