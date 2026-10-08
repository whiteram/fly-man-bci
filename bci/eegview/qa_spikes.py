"""bci/eegview/qa_spikes.py — 三个追问的定量回答 (全零采集)。

Q1  为什么高锁/残留闩锁静息那么多尖峰而 naive 静息像脑电?
    -> 每状态: 头皮共模的峭度 / 尖峰事件率 / p2p÷RMS 系数, 加
    「共模 vs 群体速率 (ALPN/MBON/KC/DAN)」的滞后互相关归因。
Q2  SSVEP 经 CAR 后还有 14 Hz 成分吗 (波形看着像静息)?
    -> CAR 前后的 14 Hz FFT 幅值与正典带 SNR (f0±1 bin vs 2-40 Hz
    中位), 以静息臂作 13-15 Hz 对照。
Q3  正常 SSVEP 分类是不是不能做 CAR?
    -> 由 Q2 数字 + 源几何直接推论。

用法 (conda ffbm, repo root):  python bci/eegview/qa_spikes.py
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, welch
from scipy.stats import kurtosis

ROOT = "bci"
FS_N, FS_D = 1000.0, 250.0


def device(x):
    sos = butter(4, [0.5, 100.0], btype="bandpass", fs=FS_N, output="sos")
    return resample_poly(sosfiltfilt(sos, x, axis=0), 1, 4, axis=0)


def smooth(a, ms):
    k = np.hanning(max(int(ms) | 1, 3))
    return np.convolve(a, k / k.sum(), mode="same")


ARMS = [
    ("naive", f"{ROOT}/condstate3/outputs/ctl62_scalp.npy",
     f"{ROOT}/condstate3/outputs/ctl62_pop.npz", 7000, 15000),
    ("high-lock", f"{ROOT}/condstate4/outputs/hl62_scalp.npy",
     f"{ROOT}/condstate4/outputs/hl62_pop.npz", 7000, 15000),
    ("res/latch", f"{ROOT}/condstate3/outputs/res62_scalp.npy",
     f"{ROOT}/condstate3/outputs/res62_pop.npz", 7000, 15000),
    ("ssvep", f"{ROOT}/condstate13/outputs/ssvep_naive_scalp.npy",
     f"{ROOT}/condstate13/outputs/ssvep_naive_pop.npz", 3500, 10500),
]

print("=" * 78)
print("Q1  尖峰从哪来 — 共模形态统计 + 群体速率归因 (lag 扫描 ±120 ms)")
print("=" * 78)
hdr = (f"{'arm':10s} {'ALPN':>5s} {'KC':>5s} {'MBON':>5s} {'DAN':>5s} "
       f"{'峭度':>6s} {'p2p/RMS':>7s} {'>4σ/s':>6s} "
       f"{'最佳r(群)':>12s}")
print(hdr)
for tag, ps, pp, t0, t1 in ARMS:
    e = np.load(ps).astype(np.float64) * 1e6
    x = device(e[t0:t1])
    cm = x.mean(axis=1)                       # 共模 = 17 导均值
    cm = (cm - cm.mean()) / cm.std()
    z = np.load(pp)
    w = slice(t0, min(t1, z["ALPN"].size))
    pr = {k: float(z[k][w].mean()) for k in z.files}
    kur = float(kurtosis(cm))
    p2p = float(np.ptp(cm) / (6 * cm.std()))
    ev = float((np.abs(cm) > 4).sum() / (cm.size / FS_D))
    # 主节奏: 共模 PSD 峰 (0.3-40 Hz)
    f, pw = welch(cm, fs=FS_D, nperseg=1024)
    m = (f >= 0.3) & (f <= 40.0)
    fpk = float(f[m][np.argmax(pw[m])])
    # 事件触发平均 (ETA): 共模尖峰时刻对齐群体速率 (z 分数)
    evm = np.flatnonzero(np.abs(cm) > 4)
    eta_s = ""
    if evm.size >= 4:
        tg = np.arange(t0, t1, 1000.0 / FS_D)
        off = np.arange(-62, 63)                # ±248 ms
        for g in ("ALPN", "Kenyon_Cell", "MBON", "DAN"):
            r = smooth(z[g][w].astype(float), 15.0)
            r = np.interp(tg, np.arange(t0, t1), r)
            r = (r - r.mean()) / (r.std() + 1e-12)
            seg = np.stack([r[e + off] for e in evm
                            if 62 <= e < r.size - 62])
            mu = seg.mean(axis=0)
            i = int(np.argmax(np.abs(mu)))
            if abs(mu[i]) > 1.0:                # 只报 ETA 峰 >1 z
                eta_s += (f"  {g}:ETA峰值{mu[i]:+.1f}z@"
                          f"{off[i] * 4:+d}ms")
    print(f"{tag:10s} {pr['ALPN']:5.1f} {pr['Kenyon_Cell']:5.1f} "
          f"{pr['MBON']:5.1f} {pr['DAN']:5.1f} {kur:6.1f} "
          f"{p2p:7.2f} {ev:6.1f} 主节律{fpk:5.1f}Hz"
          f" (n={evm.size}){eta_s}")

print()
print("=" * 78)
print("Q2  SSVEP 14 Hz 在 CAR 前后 (幅值 / 正典带 SNR; 静息臂为对照)")
print("=" * 78)


def band_snr(x):
    """通道均值 PSD 的 f0±1 bin vs 2-40 Hz 中位 (±0.7 Hz 除外), dB。"""
    f, p = welch(x, fs=FS_D, nperseg=1024, axis=0)
    pm = p.mean(axis=1)
    sig = pm[(f >= 13) & (f <= 15)].max()
    nz = pm[(f >= 2) & (f <= 40) & (np.abs(f - 14) > 0.7)]
    return 10 * np.log10(sig / np.median(nz)), f, pm


for tag, ps, pp, t0, t1 in ARMS:
    e = np.load(ps).astype(np.float64) * 1e6
    x = device(e[t0:t1])
    xc = x - x.mean(axis=1, keepdims=True)
    s0, f, pm = band_snr(x)
    s1, _, pmc = band_snr(xc)
    i14 = np.argmin(np.abs(f - 14.0))
    a0 = np.sqrt(pm[i14])
    a1 = np.sqrt(pmc[i14])
    print(f"{tag:10s}  14Hz幅值: {a0:7.4f} -> {a1:7.4f} µV "
          f"(x{a0 / max(a1, 1e-12):5.1f} 衰减)   "
          f"带SNR@14: {s0:+6.1f} -> {s1:+6.1f} dB")
