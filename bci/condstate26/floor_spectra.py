"""bci/condstate26/floor_spectra.py — naive 底的光谱指纹归因（零采集）。

cs25 定了底的"谁"（视觉通路暗适应层：L 强直→MID 18 Hz→T4/T5）；
eegview 审计说 naive 底谱是最像人脑的（指数 −1.06）且 12 Hz 平台
=视觉自发节律。本弧把两层接上：底谱形状由哪一层的涨落谱塑造？

  1. 底谱：floor_scalp（cs25 暗态无 chem）设备链 → 共模 PSD；
  2. 层谱：floor_layers（R/L/MID/T4/T5 @1 ms）重采样 250 Hz →
     各层速率涨落 PSD；
  3. 归因：谱指数对照 + 岭回归（底 log-PSD ← 各层 log-PSD）权重
     + 峰位对照（12 Hz 平台的层来源）。

用法 (conda ffbm, repo root):
    python bci/condstate26/floor_spectra.py
输出: outputs/floor_spectra.png, outputs/floor_spectra_stats.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch, resample_poly

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
from state_monitor import device  # noqa: E402
sys.path.insert(0, str(ROOT / "bci" / "eegview"))
from likeness_audit import spec_fit  # noqa: E402

W0, W1 = 2000, 19000        # ms, 稳态窗（避开启动瞬态）
FS1K, FS = 1000.0, 250.0
LAYERS = ("R", "L", "MID", "T4", "T5")


def psd_1_40(x, fs=FS):
    f, P = welch(x, fs=fs, nperseg=min(len(x), 2048))
    m = (f >= 1) & (f <= 40)
    return f[m], P[m]


def main():
    # ---- 1. 底谱 --------------------------------------------------------
    e = np.load(ROOT / "bci/condstate25/outputs/floor_scalp.npy") \
        .astype(np.float64)[W0:W1] * 1e6
    x = device(e)
    cm = x.mean(axis=1)
    f_s, P_s = psd_1_40(cm)
    expo_s, (lf, lp, line), peaks = spec_fit(
        np.hstack([[0.0], f_s]), np.hstack([[1e-30], P_s]))
    print(f"底谱: 指数 {expo_s:+.2f}, 峰 "
          f"{[(round(h,1), round(db,1)) for h, db in peaks[:3]]}")

    # ---- 2. 层谱 --------------------------------------------------------
    lay = np.load(ROOT / "bci/condstate25/outputs/floor_layers.npz")
    expo_l, P_l, f_l = {}, {}, None
    const_layers = []
    for k in LAYERS:
        r = lay[k][W0:W1].astype(np.float64)
        r = r - r.mean()
        if r.std() < 1e-9:                      # R=0 恒定 / L=100% 恒定
            const_layers.append(k)
            expo_l[k] = None
            P_l[k] = None
            print(f"层 {k:4s}: 恒定（零方差——不参与谱归因）")
            continue
        r25 = resample_poly(r, 1, 4)
        f_l, P = psd_1_40(r25)
        P_l[k] = P
        e_l, _, peaks_l = spec_fit(np.hstack([[0.0], f_l]),
                                   np.hstack([[1e-30], P]))
        expo_l[k] = e_l
        print(f"层 {k:4s}: 指数 {e_l:+.2f}, 峰 "
              f"{[(round(h,1), round(db,1)) for h, db in peaks_l[:2]]}")
    var_layers = [k for k in LAYERS if P_l[k] is not None]

    # ---- 3. 归因: 岭回归 底 log-PSD <- 涨落层 log-PSD --------------------
    A = np.column_stack([np.log(P_l[k] + 1e-30) for k in var_layers])
    y = np.log(P_s + 1e-30)
    A_z = (A - A.mean(0)) / (A.std(0) + 1e-12)
    y_z = (y - y.mean()) / (y.std() + 1e-12)
    lam = 0.5
    w = np.linalg.solve(A_z.T @ A_z + lam * np.eye(len(var_layers)),
                        A_z.T @ y_z)
    r2 = 1 - ((A_z @ w - y_z) ** 2).sum() / (y_z ** 2).sum()
    w_all = {k: (round(float(v), 2) if k in var_layers else None)
             for k, v in zip(var_layers, w)}
    for k in const_layers:
        w_all[k] = None
    print("归因权重(标准化, 岭 λ=0.5, 仅涨落层):",
          {k: v for k, v in w_all.items() if v is not None},
          f"R²={r2:.2f}")

    res = {
        "floor_exponent": round(expo_s, 2),
        "floor_peaks_hz_db": [(round(h, 1), round(db, 1))
                              for h, db in peaks[:4]],
        "layer_exponents": {k: (round(v, 2) if v is not None
                                else "constant") for k, v in
                            expo_l.items()},
        "constant_layers": const_layers,
        "attribution_weights": w_all,
        "attribution_r2": round(float(r2), 2),
        "window_ms": [W0, W1],
    }

    # ---- 4. 图 ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             constrained_layout=True)
    ax = axes[0]
    ax.loglog(f_s, P_s, lw=1.4, color="k", label="头皮底谱")
    cols = {"R": "0.7", "L": "tab:red", "MID": "tab:orange",
            "T4": "tab:green", "T5": "tab:purple"}
    for k in var_layers:
        ax.loglog(f_l, P_l[k] / P_l[k].max() * P_s.max(), lw=0.8,
                  alpha=0.75, color=cols[k],
                  label=f"{k} (exp {expo_l[k]:+.1f})")
    ax.set_xlabel("Hz", fontsize=8)
    ax.set_ylabel("PSD (任意归一)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title(f"底谱 vs 层速率涨落谱（R/L 恒定无涨落；"
                 f"底指数 {expo_s:+.2f}）", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[1]
    vw = [w_all[k] for k in var_layers]
    ax.bar(range(len(var_layers)), vw, 0.55,
           color=[cols[k] for k in var_layers])
    ax.set_xticks(range(len(var_layers)))
    ax.set_xticklabels(var_layers, fontsize=9)
    ax.set_ylabel("标准化回归权重", fontsize=8)
    ax.text(0.02, 0.95, f"R²={r2:.2f}（岭 λ=0.5）", fontsize=8,
            transform=ax.transAxes, va="top")
    ax.set_title("层归因失败：平均率解释不了底谱形状", fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    ax = axes[2]
    labs = ["头皮底"] + list(LAYERS)
    expo_ax = [expo_s]
    for k in LAYERS:
        expo_ax.append(expo_l[k] if expo_l[k] is not None else 0.0)
    cols2 = ["k"] + [cols[k] for k in LAYERS]
    ax.bar(range(len(expo_ax)), expo_ax, 0.55, color=cols2)
    for i, k in enumerate(LAYERS, start=1):
        if expo_l[k] is None:
            ax.text(i, 0.05, "恒定", ha="center", fontsize=7,
                    rotation=90, color="0.4")
    ax.axhspan(-1.6, -1.0, color="0.88", zorder=0)
    ax.text(3.4, -1.45, "人脑 −1.0~−1.6", fontsize=7, color="0.35")
    ax.set_xticks(range(len(expo_ax)))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("谱指数 1-40 Hz", fontsize=8)
    ax.set_title("指数对照：底 −0.99 近人脑带；尖峰层源平坦",
                 fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    fig.suptitle("condstate26 底的光谱指纹归因——哪一层塑造了 "
                 "naive 底谱（cs25×audit 接口）", fontsize=11)
    fig.savefig(OUT / "floor_spectra.png", dpi=150)
    plt.close(fig)
    (OUT / "floor_spectra_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'floor_spectra.png'}")
    print(f"-> {OUT / 'floor_spectra_stats.json'}")


if __name__ == "__main__":
    main()
