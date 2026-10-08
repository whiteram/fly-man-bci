"""bci/eegview: 合成脑电的第一次时域检查 — 250 Hz 设备视角的波形可视化。

此前所有 condstate 弧都只做过频带统计(2-20 Hz band RMS / SNR),从未把
合成头皮电位当成一条时间序列画出来。本弧回答:它长什么样、一台 250 Hz
临床设备采到的是什么、幅值在什么量级、像不像脑电。

数据(零采集,全部盘上现有,seed 62,17 电极 capped-Fibonacci 头皮布局,
通道 0 在眼轴锚点;原生 1 kHz,V 单位 → ×1e6 转 µV):
  naive 静息     condstate3 ctl62 [7000,15000] ms (cs3ctl: 17 s 暗态)
  高锁静息      condstate4 hl62  [7000,15000] ms (DA1@1600 [300,600] 成高锁后)
  残留/闩锁静息  condstate3 res62 [7000,15000] ms (LAL 点火 [800,2800]+×0.5 后)
  SSVEP naive   condstate13 ssvep_naive [3500,10500] ms (14 Hz 闪烁稳态)

设备链:  1 kHz 原生 → 0.5-100 Hz 带通(模拟前端带宽) → ÷4 → 250 Hz。
增益:    统一 ×300 显示增益(把真实亚 µV 信号放到可读幅值),行标题同时
         标注真实值;复合行不增益 — 蝇信号按真实幅值叠加进 11.4 µV 的
         人脑背景示意噪声(1/f², 本项目背景模型),展示设备里它长什么样
         (不可见 — 这正是项目论点,背景为示意合成、非本仿真产物)。

用法 (conda ffbm, repo root):
    python bci/eegview/make_fig.py
输出: outputs/eegwave_250hz.png, outputs/eegwave_250hz_car.png,
      outputs/psd_250hz.png, outputs/stats.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, welch

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

FS_N, FS_D = 1000.0, 250.0          # 原生 / 设备采样率 (Hz)
GAIN = 300.0                        # 统一显示增益 (真实值见各行标注)

ARMS = [
    ("naive 静息 (17 s 暗态)",
     ROOT / "bci/condstate3/outputs/ctl62_scalp.npy", 7000, 15000,
     "tab:blue"),
    ("高锁静息 (DA1@1600 成锁后)",
     ROOT / "bci/condstate4/outputs/hl62_scalp.npy", 7000, 15000,
     "tab:green"),
    ("残留/闩锁静息 (LAL 点火后)",
     ROOT / "bci/condstate3/outputs/res62_scalp.npy", 7000, 15000,
     "tab:red"),
    ("SSVEP 14 Hz 闪烁 (naive)",
     ROOT / "bci/condstate13/outputs/ssvep_naive_scalp.npy",
     3500, 10500, "tab:purple"),
]


def load_uv(p):
    return np.load(p).astype(np.float64) * 1e6


def device(x):
    """1 kHz (n,ch) µV → 模拟前端 0.5-100 Hz → ÷4 → 250 Hz。"""
    sos = butter(4, [0.5, 100.0], btype="bandpass", fs=FS_N, output="sos")
    return resample_poly(sosfiltfilt(sos, x, axis=0), 1, 4, axis=0)


def background(n, ch, seed=7, shared=0.85, target_2_20=11.4):
    """示意人脑背景: 幅值 1/f² 噪声 0.5-40 Hz, 2-20 Hz RMS=target µV,
    shared 比例空间共享。仅用于"设备视角"演示行, 非本仿真产物。"""
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1.0 / FS_D)
    shape = np.where(f >= 0.5, np.maximum(f, 0.5) ** -1.0, 0.0)
    shape[0] = 0.0

    def gen():
        X = (rng.standard_normal(len(f))
             + 1j * rng.standard_normal(len(f))) * shape
        x = np.fft.irfft(X, n)
        sos = butter(4, 40.0, btype="lowpass", fs=FS_D, output="sos")
        return sosfiltfilt(sos, x)

    base = gen()
    ind = np.stack([gen() for _ in range(ch)], axis=1)
    b = (np.sqrt(shared) * base[:, None]
         + np.sqrt(1.0 - shared) * ind)
    sos = butter(4, [2.0, 20.0], btype="bandpass", fs=FS_D, output="sos")
    r = np.sqrt((sosfiltfilt(sos, b, axis=0) ** 2).mean(axis=0)).mean()
    return b * (target_2_20 / r)


def band_rms(x, lo, hi):
    sos = butter(4, [lo, hi], btype="bandpass", fs=FS_D, output="sos")
    xf = sosfiltfilt(sos, x, axis=0)
    return np.sqrt((xf ** 2).mean(axis=0))


def nice_sens(p2p):
    for s in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        if s >= p2p / 1.6:
            return s
    return 1000


def plot_stack(ax, x_uv, label, sublabel, t0_ms):
    """EEG 纸样多通道堆叠: 通道间隔=1 灵敏度格, 1 s 竖网格。"""
    n, ch = x_uv.shape
    dur = n / FS_D
    sens = nice_sens(np.ptp(x_uv, axis=0).max())
    sep = sens
    t = np.arange(n) / FS_D
    ax.axhline(0, color="none")
    for k in range(ch):
        ax.axhline(-k * sep, color="0.88", lw=0.4, zorder=0)
        ax.plot(t, x_uv[:, k] - k * sep, color="0.15", lw=0.5, zorder=2)
    for s in range(int(dur) + 1):
        ax.axvline(s, color="0.88", lw=0.4, zorder=1)
    for k in range(ch):
        ax.text(-0.10, -k * sep, (f"E{k}" if k else "E0 眼轴"),
                ha="right", va="center", fontsize=6.2,
                color="0.35" if k else "k")
    sb = 0.35                       # 定标条: 末尾 0.35 s, 高 1 格
    ax.plot([dur - sb, dur - sb], [-sep, 0], color="k", lw=1.2)
    ax.text(dur - sb / 2, -sep / 2, f"{sens:g} µV", fontsize=6,
            ha="center", va="center", color="k",
            bbox=dict(fc="w", ec="none", pad=0.4))
    ax.set_xlim(-0.62, dur)
    ax.set_ylim(-sep * (ch - 1) - 1.6 * sep, 1.1 * sep)
    ax.set_yticks([])
    ax.set_xticks(range(int(dur) + 1))
    ax.tick_params(labelsize=7)
    ax.set_title(f"{label}   |   真实 RMS {sublabel}   |   "
                 f"灵敏度 {sens:g} µV/格", fontsize=8.5, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    return sens


def main():
    arms, stats = [], {}
    for label, path, t0, t1, color in ARMS:
        e = load_uv(path) * GAIN
        x = device(e[t0:t1])
        true_bb = float(np.median(np.sqrt(
            (x ** 2).mean(axis=0)))) / GAIN             # 设备带内真实 µV
        true_native = float(np.median(np.sqrt(
            ((e[t0:t1] / GAIN) ** 2).mean(axis=0))))    # 原生(含<0.5Hz漂移)
        arms.append((label, path, t0, t1, color, x, true_bb))
        cc = np.corrcoef(x.T)
        iu = np.triu_indices(x.shape[1], 1)
        xc = x - x.mean(axis=1, keepdims=True)      # CAR
        cc_car = np.corrcoef(xc.T)
        cm_frac = 1.0 - (xc ** 2).mean() / (x ** 2).mean()
        f, p = welch(x, fs=FS_D, nperseg=min(len(x), 2048), axis=0)
        pm = np.median(p, axis=1)
        m = (f >= 1.0) & (f <= 30.0)
        slope = np.polyfit(np.log(f[m]), np.log(pm[m]), 1)[0]
        stats[label] = {
            "file": str(path.relative_to(ROOT)),
            "window_ms": [t0, t1],
            "true_bb_rms_uv": round(true_bb, 5),
            "true_native_rms_uv": round(true_native, 5),
            "true_2_20_rms_uv": round(float(np.median(
                band_rms(e[t0:t1] / GAIN, 2.0, 20.0))), 5),
            "mean_interchannel_corr": round(
                float(cc[iu].mean()), 3),
            "car_rms_uv": round(float(np.median(np.sqrt(
                (xc ** 2).mean(axis=0)))) / GAIN, 5),
            "car_common_mode_frac": round(float(cm_frac), 4),
            "car_mean_interchannel_corr": round(
                float(cc_car[iu].mean()), 3),
            "psd_slope_1_30hz": round(float(slope), 2),
        }
        print(f"{label:28s} 真实bbRMS={true_bb:7.4f} µV "
              f"2-20Hz={stats[label]['true_2_20_rms_uv']:7.4f} µV "
              f"ch-corr={stats[label]['mean_interchannel_corr']:+.2f} "
              f"CAR共模={cm_frac * 100:5.1f}% "
              f"CAR后corr={stats[label]['car_mean_interchannel_corr']:+.2f} "
              f"slope={stats[label]['psd_slope_1_30hz']:+.1f}")

    # ---- 图 1: 设备视角波形页 -------------------------------------
    fig = plt.figure(figsize=(13.5, 16.2))
    gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 0.62],
                          hspace=0.42, wspace=0.08, left=0.075,
                          right=0.985, top=0.965, bottom=0.042)
    for r, (label, path, t0, t1, color, x, true_bb) in enumerate(arms):
        ax = fig.add_subplot(gs[r, :])
        plot_stack(ax, x, label,
                   f"{true_bb:.4f} µV ×{GAIN:g} → 显示 "
                   f"{true_bb * GAIN:.1f} µV", t0)
    # 复合行: 真实幅值的蝇信号 + 示意人脑背景 (不增益)
    x_comp = arms[2][5] / GAIN + background(*arms[2][5].shape)
    ax = fig.add_subplot(gs[4, :])
    plot_stack(ax, x_comp,
               "设备视角: 示意人脑背景(11.4 µV) + 残留态蝇信号"
               "(真实幅值, 未增益)",
               f"蝇(0.5-100 Hz)/背底(2-20 Hz) ≈ "
               f"1/{11.4 / arms[2][6]:.0f} → 蝇信号不可见",
               arms[2][2])
    # 左下: 250 Hz 采样细节
    e_res = load_uv(ARMS[2][1])
    sos = butter(4, [0.5, 100.0], btype="bandpass", fs=FS_N, output="sos")
    nat = sosfiltfilt(sos, e_res[7000:15000], axis=0) * GAIN
    brms = band_rms(nat[::4], 2.0, 20.0)
    k = int(np.argmax(brms))
    i0 = 500                          # 设备样本: 窗内 2000-4000 ms
    t_nat = (np.arange(2000) + 2000) / 1000.0
    t_dev = (np.arange(i0, i0 + 500) + 0.5) / FS_D + 2.0
    ax = fig.add_subplot(gs[5, 0])
    ax.plot(t_nat, nat[2000:4000, k], color="0.75", lw=0.7,
            label="模拟前端输出 (1 kHz, 100 Hz 带宽)")
    ax.plot(t_dev, arms[2][5][i0:i0 + 500, k], "o", ms=2.4,
            color="tab:red", label="250 Hz 设备采样点")
    ax.set_xlim(2.0, 4.0)
    ax.set_ylim(-1.3 * np.abs(nat[2000:4000, k]).max(),
                1.3 * np.abs(nat[2000:4000, k]).max())
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("窗口内时间 (s)", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    ax.set_title(f"250 Hz 采样细节 — E{k} (残留态最活跃通道), 2 s",
                 fontsize=8.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # 右下: SSVEP 12-16 Hz 带通 — 节律在时域可见
    x_ss = arms[3][5]
    sos_b = butter(4, [12.0, 16.0], btype="bandpass", fs=FS_D,
                   output="sos")
    xf = sosfiltfilt(sos_b, x_ss, axis=0)
    kb = int(np.argmax(np.sqrt((xf ** 2).mean(axis=0))))
    j0, j1 = 1000, min(2000, len(xf))  # 设备样本: 窗内 4 s 至末尾
    t_b = (np.arange(j0, j1) + 0.5) / FS_D + 3.5
    ax = fig.add_subplot(gs[5, 1])
    ax.plot(t_b, xf[j0:j1, kb], color="tab:purple", lw=0.8)
    ax.set_xlim(t_b[0], t_b[-1])
    ax.set_ylim(-2.6 * np.abs(xf[j0:j1, kb]).max(),
                2.6 * np.abs(xf[j0:j1, kb]).max())
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("窗口内时间 (s)", fontsize=8)
    ax.set_title(f"SSVEP 12-16 Hz 带通 — E{kb}: 14 Hz 节律在时域"
                 "可见(约 71 ms/周期)", fontsize=8.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.suptitle(
        "合成脑电时域检查 — 250 Hz 设备视角 (seed 62, 17 电极, "
        f"显示增益 ×{GAIN:g}, 真实值见各行标注)",
        fontsize=11, y=0.992)
    fig.savefig(OUT / "eegwave_250hz.png", dpi=150)
    plt.close(fig)

    # ---- 图 1b: 全局平均参考 (CAR) — 通道间差异 --------------------
    # CAR: 每时刻减 17 导均值, 去掉共模成分; 剩余即导联场差异。
    fig = plt.figure(figsize=(13.5, 13.6))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.8],
                          hspace=0.46, left=0.075, right=0.985,
                          top=0.958, bottom=0.052)
    for r, (label, path, t0, t1, color, x, true_bb) in enumerate(arms):
        xc = x - x.mean(axis=1, keepdims=True)
        car_uv = float(np.median(np.sqrt(
            (xc ** 2).mean(axis=0)))) / GAIN
        ax = fig.add_subplot(gs[r])
        plot_stack(ax, xc, f"{label} — CAR 平均参考后",
                   f"0.4-3.5 nV 级: {car_uv:.5f} µV"
                   f" (原参考 {true_bb:.4f} µV)", t0)
    # 底行: 每导 CAR RMS 剖面 (相对 17 导均值) — 空间增益花纹
    ax = fig.add_subplot(gs[4])
    ch = arms[0][5].shape[1]
    for label, path, t0, t1, color, x, true_bb in arms:
        xc = x - x.mean(axis=1, keepdims=True)
        prof = np.sqrt((xc ** 2).mean(axis=0))
        ax.plot(range(ch), prof / prof.mean(), "o-", ms=3.5, lw=1.1,
                color=color, label=label)
    ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
    ax.set_xticks(range(ch))
    ax.set_xticklabels([f"E{k}" for k in range(ch)], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_ylabel("每导 CAR RMS / 均值", fontsize=8.5)
    ax.set_xlabel("电极 (E0 = 眼轴锚点)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, color="0.9", lw=0.4)
    ax.set_title("通道间差异 — CAR 后每导 RMS 剖面：偏离 1.0 的幅度"
                 "即该导联场对共模的偏离 (符号翻转=低于平均增益)",
                 fontsize=8.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.suptitle(
        "全局平均参考 (CAR) 视图 — 通道间差异 (共模已减除, "
        "各行灵敏度独立缩放, 真实值见标注)",
        fontsize=11, y=0.99)
    fig.savefig(OUT / "eegwave_250hz_car.png", dpi=150)
    plt.close(fig)

    # ---- 图 2: 频谱 ------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    bands = [(1, 4, "δ"), (4, 8, "θ"), (8, 13, "α"), (13, 30, "β"),
             (30, 80, "γ")]
    for lo, hi, nm in bands:
        ax.axvspan(lo, hi, color="0.93", zorder=0)
    for label, path, t0, t1, color, x, true_bb in arms:
        f, p = welch(x, fs=FS_D, nperseg=min(len(x), 2048), axis=0)
        ax.loglog(f, np.median(p, axis=1), color=color, lw=1.3,
                  label=f"{label}  (2-20 Hz RMS "
                        f"{stats[label]['true_2_20_rms_uv']:.4f} µV)")
    f_ref = np.logspace(np.log10(1.0), np.log10(30.0), 40)
    p_naive = welch(arms[0][5], fs=FS_D, nperseg=min(len(arms[0][5]),
                                                     2048), axis=0)[1]
    p10 = np.median(p_naive, axis=1)[np.argmin(np.abs(f - 10.0))]
    ax.loglog(f_ref, p10 * (10.0 / f_ref) ** 3, "--", color="0.6",
              lw=1.0, label="1/f³ 参考 (人头皮 EEG 典型斜率)")
    y0, y1 = ax.get_ylim()
    for lo, hi, nm in bands:
        ax.text(np.sqrt(lo * hi), y1, nm, ha="center", va="bottom",
                fontsize=8, color="0.45", clip_on=False)
    ax.axvline(14.0, color="tab:purple", ls=":", lw=1.0, alpha=0.7)
    ax.text(14.3, y0 * 1.5, "SSVEP f0=14 Hz", fontsize=7.5,
            color="tab:purple", rotation=90, va="bottom")
    ax.set_xlim(0.5, 100)
    ax.set_xlabel("频率 (Hz)", fontsize=9)
    ax.set_ylabel("PSD (µV²/Hz), 17 通道中位数", fontsize=9)
    ax.set_title("合成脑电频谱 — 250 Hz 设备输出 (显示增益已约掉, "
                 "标注为真实幅值)", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, which="both", color="0.9", lw=0.4, zorder=0)
    fig.savefig(OUT / "psd_250hz.png", dpi=150)
    plt.close(fig)

    stats["_meta"] = {
        "gain_display": GAIN, "fs_native_hz": FS_N, "fs_device_hz": FS_D,
        "device_chain": "0.5-100 Hz Butterworth(4) 带通 + ÷4 重采样",
        "electrodes": "17 电极 capped-Fibonacci (通道 0 = 眼轴锚点)",
        "note": "人脑背景行为示意合成(1/f², 2-20 Hz RMS 11.4 µV), "
                "非本仿真产物; 蝇信号按真实幅值叠加",
    }
    (OUT / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfigures -> {OUT / 'eegwave_250hz.png'}")
    print(f"          {OUT / 'eegwave_250hz_car.png'}")
    print(f"          {OUT / 'psd_250hz.png'}")
    print(f"stats   -> {OUT / 'stats.json'}")


if __name__ == "__main__":
    main()
