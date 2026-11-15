"""bci/condstate16/budget_map.py — 状态读出的积分预算地图：积分-漂移不相容。

三块既有结果的闭环:
  condstate12/14: 状态检测的背景账——2 s 窗监测器准确率膝点在
     ~0.005 µV 背景, 11.4 µV 下纯随机;
  condstate15: 签名寿命——高锁头皮签名被可塑性慢漂移 (8 维特征
     联合漂移速度 ~0.74 SD / 86 s)。
本弧零采集回答: **积分能不能把可测性买回来?**

  检测侧 (能量检测器, 共模 2-20 Hz 带 RMS, 与 eegview/审计同口径):
      d'(W, σ_bg) = s / σ_est,
      σ_est² = (κ·σ_naive)²/(B·W) + σ_bg²/(B·W)
     s = 态-幼稚带 RMS 差 (实测); B = 18 Hz;
     κ = 估计器自噪声校正 (仿真幼稚信号的长程相关 → 不按 1/√W 收,
        实测标定); 背景项是高斯白带 → 标准 √W 收敛。
     W_req(d'=2) = (2/s)²·(κ²σ_naive² + σ_bg²)/B。
  监测器锚点: condstate14 的实测膝点 (σ≈0.005 µV @ W=2 s) 按
     √W 外推 (σ_knee ∝ √W) 作为多特征检测器参照线。
  漂移侧: 签名寿命 τ_sig = 3σ / 联合漂移速度 (condstate15 斜率)。

**不相容判据**: W_req(σ_bg) > τ_sig → 状态读出原理上不可行
(积分没攒够, 签名已漂走)。临界背景 σ* = s·√(B·τ_sig)/(2·κ_eff)。

用法 (conda ffbm, repo root):
    python bci/condstate16/budget_map.py
输出: outputs/budget_map.png, outputs/budget_stats.json
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
from state_monitor import (ARMS, CLASSES, device, features,  # noqa: E402
                           load_dev, fit_fold, predict, build_xy)

B = 18.0                # 2-20 Hz 带宽 (Hz)
DPRIME = 2.0            # 判据
SIG_BG_PROJECT = 11.4   # 项目背景模型 (µV)
MONITOR_KNEE = 0.005    # condstate14 实测: 2 s 窗监测器膝点 (µV)
MONITOR_W = 2.0         # …@ W=2 s


def band_rms(x, fs=250.0, lo=2.0, hi=20.0):
    f, P = welch(x, fs=fs, nperseg=min(len(x), 512), axis=0)
    pm = np.median(P, axis=1)
    m = (f >= lo) & (f < hi)
    return float(np.sqrt(np.trapezoid(pm[m], f[m])))


def cm_rms_series(x, w_s, lo=2.0, hi=20.0):
    """共模通道 2-20 Hz 带宽后逐窗 RMS（与审计共模口径一致）."""
    sos = butter(4, [lo, hi], btype="bandpass", fs=250.0, output="sos")
    cm = sosfiltfilt(sos, x.mean(axis=1))
    wlen = int(w_s * 250.0)
    idx = np.arange(0, len(cm) - wlen + 1, wlen)
    return np.array([np.sqrt((cm[i:i + wlen] ** 2).mean()) for i in idx])


def main():
    # ---- 1. 估计器自噪声标定 (κ) ---------------------------------------
    # 稳态窗（避开 cs3ctl 的训练 [17,19] s / 探针 [20,20.5] s）
    naive = load_dev(ROOT / "bci/condstate3/outputs/ctl62_scalp.npy",
                     4000, 16500)
    rms_naive = band_rms(naive)
    ser = {w: cm_rms_series(naive, w) for w in (1.0, 2.0, 4.0)}
    kappa = {w: float(s_.std() / s_.mean() * np.sqrt(B * w))
             for w, s_ in ser.items()}
    kap = kappa[2.0]
    print(f"幼稚共模带 RMS = {rms_naive:.5f} µV")
    print(f"κ(自噪声/白噪声): { {w: round(k, 2) for w, k in kappa.items()} }"
          f"  (随 W 增长 → 长程相关, √W 收益打折, 取 κ(2s)={kap:.2f})")

    # ---- 2. 各态信号强度 s (同口径实测) ----------------------------------
    sig = {"naive": rms_naive}
    for tag, path, seed, t0, t1 in ARMS:
        if tag != "naive":
            sig[tag] = band_rms(load_dev(path, t0, t1))
    s_vec = {t: sig[t] - sig["naive"] for t in ("qlock", "hlock", "resid")}
    print("带 RMS:", {t: round(v, 5) for t, v in sig.items()})
    print("信号 s:", {t: round(v, 5) for t, v in s_vec.items()})

    # ---- 3. 漂移侧: 签名寿命 τ_sig (condstate15 斜率) --------------------
    st = json.loads((ROOT / "bci/condstate15/outputs"
                     / "stability_stats.json").read_text(encoding="utf-8"))
    slopes = np.array(list(st["feature_slope_per_sd_time"].values()))
    sd_time = 300.0 / np.sqrt(12)                  # tc 的 SD (~86.6 s)
    v_sig = float(np.sqrt((slopes ** 2).sum())) / sd_time   # SD/s
    tau_sig = 3.0 / v_sig
    # τ_90 (保守运营值, 滚动保持率; 重算一次对齐索引)
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    Xtr, ytr, seeds, _ = build_xy(records, 2.0, 1.0)
    fold = fit_fold(Xtr[seeds == 63], ytr[seeds == 63])
    e = np.load(ROOT / "bci/condstate15/outputs/hll62_scalp.npy") \
        .astype(np.float64) * 1e6
    x300 = device(e)
    wlen, step = 500, 250
    tc, pred = [], []
    for i in range(750, len(x300) - wlen + 1, step):
        tc.append((i + wlen / 2) / 250.0)
        pred.append(int(predict(fold, features(x300[i:i + wlen])[None, :])[0]))
    tc, pred = np.array(tc), np.array(pred)
    keep = (pred == 2).astype(float)
    K = 20
    roll = np.convolve(keep, np.ones(K) / K, "valid")   # roll[i] <-> tc[i+K-1]
    below = np.flatnonzero(roll < 0.9)
    tau_90 = float(tc[below[0] + K - 1]) if len(below) else float(tc[-1])
    print(f"签名寿命: τ_sig(3σ/联合漂移速度) = {tau_sig:.0f} s, "
          f"τ_90(滚动保持率) = {tau_90:.0f} s (保守运营值)")

    # ---- 4. 积分需求地图 + 临界背景 --------------------------------------
    sigmas = np.logspace(-3.3, 1.3, 90)
    var_self = (kap * sig["naive"]) ** 2
    W_req = {t: (DPRIME / s_) ** 2 * (var_self + sigmas ** 2) / B
             for t, s_ in s_vec.items()}
    sig_star = {t: float(np.sqrt(
        (s_ * np.sqrt(B * tau_sig) / DPRIME) ** 2 - var_self))
        for t, s_ in s_vec.items()}
    res = {
        "kappa_2s": round(kap, 2),
        "kappa_per_W": {str(w): round(k, 2) for w, k in kappa.items()},
        "band_rms_uv": {t: round(v, 5) for t, v in sig.items()},
        "s_uv": {t: round(v, 5) for t, v in s_vec.items()},
        "v_sig_SD_per_s": round(v_sig, 5),
        "tau_sig_s": round(tau_sig, 0), "tau_90_s": round(tau_90, 0),
        "sigma_star_uv": {t: round(v, 4) for t, v in sig_star.items()},
        "W_req_11p4_h": {t: round(float(W_req[t][np.argmin(
            np.abs(sigmas - SIG_BG_PROJECT))] / 3600), 1)
            for t in s_vec},
        "monitor_knee_extrap_h_at_11p4": round(
            MONITOR_W * (SIG_BG_PROJECT / MONITOR_KNEE) ** 2 / 3600, 1),
        "monitor_knee_tau_sig_uv": round(
            MONITOR_KNEE * np.sqrt(tau_sig / MONITOR_W), 3),
    }
    print("临界背景 σ*(τ_sig 内):", res["sigma_star_uv"])
    print("11.4 µV 下能量检测器需积分 (h):", res["W_req_11p4_h"])
    print("监测器膝点 √W 外推: 11.4 µV 需", res["monitor_knee_extrap_h_at_11p4"],
          "h; τ_sig 内膝点仅到", res["monitor_knee_tau_sig_uv"], "µV")

    # ---- 5. 图 ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                             constrained_layout=True)
    colors = {"qlock": "tab:cyan", "hlock": "tab:green", "resid": "tab:red"}
    ax = axes[0]
    for tag in ("qlock", "hlock", "resid"):
        ax.loglog(sigmas, W_req[tag], color=colors[tag], label=tag)
        ax.plot(sig_star[tag], tau_sig, "*", ms=14, color=colors[tag])
    # 监测器膝点 √W 外推: σ_knee(W) ∝ √W  ⇒  W_req(σ) = W₀(σ/σ₀)²
    ax.plot(sigmas, MONITOR_W * (sigmas / MONITOR_KNEE) ** 2,
            "--", color="tab:purple", label="监测器膝点 √W 外推")
    ax.plot(MONITOR_KNEE, MONITOR_W, "o", ms=6, color="tab:purple")
    ax.annotate("实测膝点\n0.005 µV @ 2 s", (MONITOR_KNEE, MONITOR_W),
                textcoords="offset points", xytext=(8, -2), fontsize=7,
                color="tab:purple")
    ax.axhline(tau_sig, color="k", ls="--", lw=1.2)
    ax.text(0.0016, tau_sig * 1.2,
            f"签名寿命 τ_sig ≈ {tau_sig:.0f} s (可塑性漂移)", fontsize=8)
    ax.axvline(SIG_BG_PROJECT, color="k", ls=":", lw=1.2)
    ax.text(SIG_BG_PROJECT * 1.25, 3e-3, "项目背景\n11.4 µV", fontsize=8)
    ax.axhspan(tau_sig, 3e9, color="0.92", zorder=0)
    ax.text(0.03, 2e4, "积分需求 > 签名寿命\n（原理上不可行）",
            fontsize=8, color="0.35")
    ax.set_xlabel("背景 2-20 Hz RMS σ_bg (µV)", fontsize=8)
    ax.set_ylabel("d'=2 所需积分 W_req (s)", fontsize=8)
    ax.set_ylim(3e-3, 3e9)
    ax.legend(fontsize=7, loc="upper left", title="能量检测器: 态 vs 幼稚",
              title_fontsize=7)
    ax.set_title("积分预算地图（★=临界背景 σ*）", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[1]
    for w, s_ in ser.items():
        ax.errorbar([w], [s_.mean()], yerr=[s_.std()], fmt="o", ms=4,
                    color="tab:blue", capsize=3)
    ws = np.linspace(0.8, 4.5, 50)
    ax.plot(ws, ser[2.0].mean() * np.sqrt(2.0 / ws) / kap * 1.0,
            "--", color="0.5", label="1/√W 白噪声预测")
    ax.plot(ws, ser[2.0].mean() * np.sqrt(2.0 / ws), ":",
            color="tab:red", label="κ 校正后预测")
    ax.set_xlabel("窗长 W (s)", fontsize=8)
    ax.set_ylabel("幼稚共模带 RMS 估计 (µV)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title(f"估计器自噪声标定 κ(2s)={kap:.1f}（实测 SD 蓝棒）",
                 fontsize=9)
    ax.grid(True, color="0.93")

    ax = axes[2]
    ax.plot(tc, keep, lw=0.3, color="0.85")
    ax.plot(tc[K - 1:], roll, lw=1.0, color="tab:blue",
            label=f"滚动 {K} 窗保持率")
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.axvline(tau_90, color="tab:blue", ls=":", lw=1.2)
    ax.text(tau_90 + 4, 0.55, f"τ_90={tau_90:.0f} s", fontsize=7,
            color="tab:blue")
    ax.axvline(tau_sig, color="tab:red", ls="--", lw=1.2)
    ax.text(tau_sig - 4, 0.35, f"τ_sig={tau_sig:.0f} s", fontsize=7,
            color="tab:red", ha="right")
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("hlock 保持率", fontsize=8)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("签名寿命 (condstate15 记录重算)", fontsize=9)
    ax.grid(True, color="0.93")

    fig.suptitle("condstate16 积分预算地图——积分需求 (∝σ_bg²) 与签名寿命 "
                 f"(τ_sig≈{tau_sig:.0f} s) 不相容：★=各态临界背景 σ*",
                 fontsize=11)
    fig.savefig(OUT / "budget_map.png", dpi=150)
    plt.close(fig)
    (OUT / "budget_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'budget_map.png'}")
    print(f"-> {OUT / 'budget_stats.json'}")


if __name__ == "__main__":
    main()
