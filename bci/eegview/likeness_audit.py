"""bci/eegview/likeness_audit.py — 合成脑电 vs 真实人脑脑电逐项相似度审计。

用户问题: 生成的脑电像不像真实脑电信号?  本脚本把四条状态臂的
250 Hz 设备链输出放到人头皮 EEG 的文献典型指纹旁边逐项对照:

  A. 无周期谱指数 (1/f^exponent, 1-40 Hz, 峰剔除后拟合)
     人脑静息典型 ~1.0-1.6 (specparam 文献 ballpark)
  B. 自发振荡峰 (α 8-13 Hz 是否存在, 显著性 = 谱残差峰 >1.5 dB)
  C. 频带功率层级 (δ/θ/α/β 相对功率, 人脑闭眼 δ、α 最大)
  D. 幅值分布峭度 (人脑近高斯, 超量峭度 ~0-1)
  E. 空间相关: 导间相关均值 + 随角距离衰减 (人脑相邻 0.3-0.7、
     随距离衰减; CAR 后仍残留相邻相关)
  F. 时间长程相关: 2-20 Hz 包络 DFA 指数 (人脑 ~0.6-0.9)
  G. 共模自相关 1/e 时间 (人脑数十 ms 量级)
  H. 棘波形态 (残留闩锁臂 >4σ 事件的半峰宽 ms; 人脑癫痫样棘波
     20-70 ms — 病理对照, 非静息对照)

判读框架: naive 静息臂对照人脑静息; 残留/闩锁臂对照病理 EEG;
SSVEP 臂对照诱发响应。参考值为文献 ballpark, 不是本仿真产物。

用法 (conda ffbm, repo root):
    python bci/eegview/likeness_audit.py
输出: outputs/audit_250hz.png, outputs/audit_stats.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import (butter, sosfiltfilt, resample_poly, welch,
                          hilbert, find_peaks)

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


def device(x):
    sos = butter(4, [0.5, 100.0], btype="bandpass", fs=FS_N, output="sos")
    return resample_poly(sosfiltfilt(sos, x, axis=0), 1, 4, axis=0)


def spec_fit(f, p, flo=1.0, fhi=40.0):
    """Theil-Sen 稳健 log-log 拟合 -> (指数, (lf, lp, line), 峰列表)。

    频率轴先做 3-bin 移动中位平滑压制有限窗锯齿; Theil-Sen 对峰
    稳健且线落在数据云内 (迭代剔除会把线压到下包络); 峰按
    prominence 排序只留前 6 个, 转换用 np.exp (lf 是自然对数)。
    """
    from scipy.stats import theilslopes
    m = (f >= flo) & (f <= fhi)
    pm = np.copy(p[m]).astype(float)
    for i in range(1, len(pm) - 1):          # 3-bin 移动中位
        pm[i] = np.median(p[m][i - 1:i + 2])
    lf, lp = np.log(f[m]), np.log(pm)
    slope, icpt, _, _ = theilslopes(lp, lf)
    line = icpt + slope * lf
    pk, props = find_peaks(lp - line, prominence=0.46, distance=8)
    order = np.argsort(-props["prominences"])[:6]
    peaks = [(float(np.exp(lf[i])),
              float(props["prominences"][j] / 0.23))
             for j, i in zip(order, pk[order])]   # 0.23 ln ≈ 1 dB
    return float(slope), (lf, lp, line), peaks


def band_pows(f, p):
    out = {}
    for nm, lo, hi in (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13),
                       ("beta", 13, 30)):
        m = (f >= lo) & (f < hi)
        out[nm] = float(np.trapezoid(p[m], f[m]))
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def dfa_exponent(x, lo=8, hi=300):
    """振幅包络 (2-20 Hz Hilbert) 的 DFA 指数。"""
    sos = butter(4, [2.0, 20.0], btype="bandpass", fs=FS_D, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    env = env - env.mean()
    Y = np.cumsum(env)
    scales = np.unique(np.logspace(np.log10(lo), np.log10(hi), 10)
                       .astype(int))
    Flk, ss = [], []
    for s in scales:
        m = len(Y) // s
        if m < 4:
            break
        t = np.arange(s)
        A = np.vstack([t, np.ones(s)]).T
        Yr = Y[:m * s].reshape(m, s)
        coef, *_ = np.linalg.lstsq(A, Yr.T, rcond=None)
        F2 = ((Yr - (A @ coef).T) ** 2).mean()
        Flk.append(np.sqrt(F2))
        ss.append(s)
    return float(np.polyfit(np.log(ss), np.log(Flk), 1)[0]), ss, Flk


def ac_1e_ms(x, max_lag=250):
    x = x - x.mean()
    v = (x * x).mean()
    n = len(x)
    ac = np.array([1.0] + [float((x[:-l] * x[l:]).mean()) / v
                           for l in range(1, max_lag + 1)])
    below = np.where(ac < 1 / np.e)[0]
    return float(below[0] / FS_D * 1000) if len(below) else float(max_lag)


def spike_fwhm_ms(cm, thr=4.0):
    cm = (cm - cm.mean()) / cm.std()
    m = np.abs(cm) > thr
    # 合并相邻越限样本为一事件, 事件内半峰宽
    idx = np.flatnonzero(m)
    if not len(idx):
        return None, 0
    splits = np.where(np.diff(idx) > 1)[0]
    events = np.split(idx, splits + 1)
    widths = []
    for ev in events:
        seg = cm[ev[0]:ev[-1] + 1]
        peak = np.argmax(np.abs(seg))
        half = abs(seg[peak]) / 2
        above = np.abs(seg) >= half
        widths.append(above.sum() / FS_D * 1000)
    return float(np.median(widths)), len(events)


def main():
    # 电极方向 (默认 capped-Fibonacci, 与四臂导出一致)
    live = json.loads((ROOT / "viz_live/data/viz_data.json")
                      .read_text(encoding="utf-8"))
    dirs = np.array(live["meta"]["scalp"]["elec_dirs"])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(dirs @ dirs.T, -1, 1)))
    iu = np.triu_indices(len(dirs), 1)

    res = {}
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.6),
                             constrained_layout=True)
    for a, (tag, path, t0, t1, color) in enumerate(ARMS):
        e = np.load(path).astype(np.float64) * 1e6
        x = device(e[t0:t1])
        ch = x.shape[1]
        rms = np.sqrt((x ** 2).mean(axis=0))
        kur = float(np.median(((x - x.mean(0)) ** 4).mean(0)
                              / rms ** 4 - 3))
        cc = np.corrcoef(x.T)[iu]
        xc = x - x.mean(axis=1, keepdims=True)
        cc_car = np.corrcoef(xc.T)[iu]
        sl_sp = float(np.corrcoef(ang[iu], cc)[0, 1])
        f, P = welch(x, fs=FS_D, nperseg=min(len(x), 2048), axis=0)
        pm = np.median(P, axis=1)
        expo, (lf, lp, line), peaks = spec_fit(f, pm)
        bp = band_pows(f, pm)
        dfa, ss, Fl = dfa_exponent(x[:, int(np.argmax(rms))])
        act = float(ac_1e_ms(x.mean(axis=1)))
        fwhm, nev = spike_fwhm_ms(x.mean(axis=1))
        # 2f0 谐波 (ssvep)
        h2 = None
        if tag == "ssvep":
            m28 = (f >= 27.5) & (f <= 28.5)
            m30 = (f >= 29) & (f <= 31.5)
            h2 = float(10 * np.log10(pm[m28].max()
                                     / np.median(pm[m30])))
        res[tag] = dict(
            rms_uv=round(float(np.median(rms)), 4),
            kurtosis=round(kur, 2),
            exponent_1_40=round(expo, 2),
            peaks_hz_db=[(round(h, 1), round(db, 1)) for h, db in peaks],
            rel_band={k: round(v, 3) for k, v in bp.items()},
            interch_corr=round(float(cc.mean()), 2),
            interch_corr_car=round(float(cc_car.mean()), 2),
            corr_vs_dist_r=round(sl_sp, 2),
            dfa=round(dfa, 2),
            ac_1e_ms=round(act, 1),
            spike_fwhm_ms=(round(fwhm, 1) if fwhm else None),
            spike_events=nev,
            ssvep_2f0_db=(round(h2, 1) if h2 is not None else None),
        )
        print(f"{tag:10s} exp={expo:+.2f} peaks={[(round(h), round(db)) for h, db in peaks]} "
              f"kurt={kur:+.1f} icc={cc.mean():+.2f}/CAR {cc_car.mean():+.2f} "
              f"spatial_r={sl_sp:+.2f} DFA={dfa:.2f} ac1e={act:.0f}ms "
              f"fwhm={fwhm and round(fwhm, 1)}ms({nev})")

        ax = axes[0][a]
        ax.loglog(f, pm, color=color, lw=1.0)
        ax.loglog(np.exp(lf), np.exp(line), "--", color="0.35", lw=1.1,
                  label=f"1/f^{abs(expo):.2f}")
        for h, db in peaks:
            ax.annotate(f"{h:.0f}Hz", (h, np.exp(np.interp(
                np.log(h), lf, lp))), fontsize=7, rotation=90,
                ha="right", color=color)
        ax.set_xlim(0.5, 100)
        ax.set_title(f"{tag} — 谱指数 {expo:+.2f}", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, which="both", color="0.9", lw=0.35)
        ax.set_xlabel("Hz", fontsize=8)

    # 下排: 幅值分布 / 空间相关-距离 / DFA / 棘波形态
    e = np.load(ARMS[0][1]).astype(np.float64) * 1e6
    cm = device(e[ARMS[0][2]:ARMS[0][3]]).mean(axis=1)
    z = (cm - cm.mean()) / cm.std()
    ax = axes[1][0]
    ax.hist(z, bins=80, density=True, color="tab:blue", alpha=0.75)
    g = np.linspace(-4, 4, 200)
    ax.plot(g, np.exp(-g * g / 2) / np.sqrt(2 * np.pi), "k--", lw=1.0)
    ax.set_title("naive 共模幅值分布 vs 高斯", fontsize=9)
    ax.set_xlabel("z", fontsize=8)

    angp, ccp = ang[iu], cc
    ax = axes[1][1]
    for tag, path, t0, t1, color in ARMS:
        x = device((np.load(path).astype(np.float64) * 1e6)[t0:t1])
        c2 = np.corrcoef(x.T)[iu]
        ax.plot(angp, c2, ".", ms=3, color=color, alpha=0.6, label=tag)
    ax.set_ylim(-0.35, 1.05)
    ax.set_title("导间相关 vs 角距离 (17 导, 136 对)", fontsize=9)
    ax.set_xlabel("角距离 (°)", fontsize=8)
    ax.legend(fontsize=7)

    ax = axes[1][2]
    dfa, ss, Fl = dfa_exponent(cm)
    ax.loglog(ss, Fl, "o-", color="tab:blue", ms=3)
    ax.set_title(f"naive DFA (2-20 Hz 包络) = {dfa:.2f}", fontsize=9)
    ax.grid(True, which="both", color="0.9", lw=0.35)
    ax.set_xlabel("窗口 (样本)", fontsize=8)

    e = np.load(ARMS[2][1]).astype(np.float64) * 1e6
    cml = device(e[ARMS[2][2]:ARMS[2][3]]).mean(axis=1)
    zl = (cml - cml.mean()) / cml.std()
    fwhm_l, nev_l = spike_fwhm_ms(cml)
    ax = axes[1][3]
    evm = np.flatnonzero(np.abs(zl) > 4)
    for k, i0 in enumerate(evm[:20]):
        seg = zl[max(0, i0 - 15):i0 + 25]
        ax.plot(np.arange(len(seg)) - 15, seg / abs(seg.max()),
                color="tab:red", alpha=0.5, lw=0.8)
    ax.axvspan(-fwhm_l / 2, fwhm_l / 2, color="0.85", zorder=0)
    ax.set_title(f"闩锁棘波形态 (n={len(evm)}, 中位半峰宽 "
                 f"{fwhm_l:.0f} ms)", fontsize=9)
    ax.set_xlabel("相对事件 (样本, 4 ms)", fontsize=8)

    fig.suptitle("合成脑电 vs 真实人脑脑电相似度审计 (250 Hz 设备链, "
                 "seed 62)", fontsize=11)
    fig.savefig(OUT / "audit_250hz.png", dpi=150)
    plt.close(fig)

    res["_reference"] = {
        "note": "人脑典型值为文献 ballpark (specparam/DFA/体积传导经典结论), 非本仿真产物",
        "exponent_1_40": "1.0-1.6 (静息)",
        "alpha_peak": "多数人闭眼存在 8-13 Hz 峰",
        "kurtosis": "0 到 +1 (近高斯)",
        "interch_corr": "0.3-0.7 且随距离衰减",
        "dfa": "0.6-0.9",
        "spike_fwhm_ms": "20-70 (癫痫样棘波)",
    }
    (OUT / "audit_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\naudit -> {OUT / 'audit_250hz.png'}")
    print(f"       {OUT / 'audit_stats.json'}")


if __name__ == "__main__":
    main()
