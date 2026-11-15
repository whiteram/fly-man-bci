"""bci/condstate35/analyze.py — 尖峰时序层的底谱归因（cs26 方法论关闭）。

cs26 用群体平均率做谱归因得 R²=0.02 假阴性，警告"需尖峰级中间量"。
本弧用新仪表 `--spike-trace 128`（export 层只读钩子，位式不变已证）
取得的采样尖峰时刻完成归因：

  1. 各池采样尖峰列车（1 ms 事件计数序列）的 PSD 1-40 Hz；
  2. 底谱（同 trial 头皮）对照；
  3. 岭回归 log S_floor ← log S_pop(s) → 层归因权重 + R²；
  4. 首批结构性发现：R/L 在此机制下零尖峰（L 的 100% 是分级
     释放非放电）——底的偶极源= MID/T45/OLR 尖峰。

用法 (conda ffbm, repo root):
    python bci/condstate35/analyze.py
输出: outputs/spike_attribution.png, outputs/spike_attribution.json
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
sys.path.insert(0, str(ROOT / "bci" / "eegview"))
from state_monitor import device  # noqa: E402
from likeness_audit import spec_fit  # noqa: E402

T_MS = 20000
W0, W1 = 2000, 19000
POPS = ("MID", "T45", "OLR")          # 事件>0 的池（R/L/CEN/ORN 零尖峰）


def binned_counts(z, pop, n_samp):
    t = z[f"{pop}_t"].astype(np.int64)
    cnt = np.bincount(t, minlength=T_MS)
    return cnt.astype(np.float64), float(len(t)) / n_samp / T_MS * 1000


def main():
    z = np.load(OUT / "_spike_trace.npz")
    # 底谱
    e = np.load(OUT / "floor_scalp.npy").astype(np.float64)[W0:W1] * 1e6
    cm = device(e).mean(axis=1)
    f, P = welch(cm, fs=250.0, nperseg=min(len(cm), 2048))
    m = (f >= 1) & (f <= 40)
    f_s, P_s = f[m], P[m]
    expo_f, _, peaks_f = spec_fit(np.hstack([[0.0], f_s]),
                                  np.hstack([[1e-30], P_s]))
    print(f"底谱: 指数 {expo_f:+.2f}, 峰 "
          f"{[(round(h,1), round(d,1)) for h, d in peaks_f[:3]]}")

    # 尖峰池谱
    spec = {}
    for pop in POPS:
        n_samp = len(z[f"{pop}_ids"])
        cnt, hz = binned_counts(z, pop, n_samp)
        c = cnt[W0:W1] - cnt[W0:W1].mean()
        # 1 ms 网格 → 每 4 样本均值到 250 Hz 再 welch（与底同轴）
        c25 = c.reshape(-1, 4).mean(1)
        f2, P2 = welch(c25, fs=250.0, nperseg=min(len(c25), 2048))
        m2 = (f2 >= 1) & (f2 <= 40)
        e_p, _, pk = spec_fit(np.hstack([[0.0], f2[m2]]),
                              np.hstack([[1e-30], P2[m2]]))
        spec[pop] = dict(f=f2[m2], P=P2[m2], expo=e_p,
                         rate_hz=round(hz, 2),
                         peak=(round(pk[0][0], 1), round(pk[0][1], 1))
                         if pk else None)
        print(f"{pop:4s}: {hz:6.2f} Hz/neuron, 指数 {e_p:+.2f}, "
              f"峰 {spec[pop]['peak']}")

    # 岭归因
    A = np.column_stack([np.log(spec[p]["P"] + 1e-30) for p in POPS])
    y = np.log(P_s + 1e-30)
    Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
    yz = (y - y.mean()) / (y.std() + 1e-12)
    lam = 0.5
    w = np.linalg.solve(Az.T @ Az + lam * np.eye(len(POPS)), Az.T @ yz)
    r2 = 1 - ((Az @ w - yz) ** 2).sum() / (yz ** 2).sum()
    print("尖峰级归因权重:", dict(zip(POPS, np.round(w, 2))),
          f"R²={r2:.2f}")

    res = {
        "floor_exponent": round(expo_f, 2),
        "floor_peaks": [(round(h, 1), round(d, 1))
                        for h, d in peaks_f[:3]],
        "pop": {p: dict(rate_hz=spec[p]["rate_hz"],
                        expo=round(spec[p]["expo"], 2),
                        peak=spec[p]["peak"]) for p in POPS},
        "attribution_weights": {p: round(float(v), 2)
                                for p, v in zip(POPS, w)},
        "attribution_r2": round(float(r2), 2),
        "zero_spike_pops": ["R", "L", "CEN", "ORN"],
        "note": "R/L 零尖峰: L 的 100% 为分级释放(k_release)非放电",
    }
    (OUT / "spike_attribution.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                             constrained_layout=True)
    ax = axes[0]
    ax.loglog(f_s, P_s, lw=1.5, color="k", label=f"头皮底谱 "
              f"(exp {expo_f:+.2f})")
    cols = {"MID": "tab:orange", "T45": "tab:green", "OLR": "tab:purple"}
    for p in POPS:
        s = spec[p]
        ax.loglog(s["f"], s["P"] / s["P"].max() * P_s.max(), lw=0.8,
                  alpha=0.8, color=cols[p],
                  label=f"{p} 尖峰 (exp {s['expo']:+.2f})")
    ax.set_xlabel("Hz", fontsize=8)
    ax.set_ylabel("PSD (归一)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("底谱 vs 采样尖峰列车谱（128 神经元/池）", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[1]
    ax.bar(range(len(POPS)), w, 0.55, color=[cols[p] for p in POPS])
    ax.set_xticks(range(len(POPS)))
    ax.set_xticklabels([f"{p}\n{spec[p]['rate_hz']} Hz" for p in POPS],
                       fontsize=8)
    ax.set_ylabel("标准化权重", fontsize=8)
    ax.text(0.02, 0.95, f"R²={r2:.2f}", fontsize=9, transform=ax.transAxes,
            va="top")
    ax.set_title("尖峰级归因（cs26 率级 R²=0.02 的对照）", fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    ax = axes[2]
    expo_ax = [expo_f] + [spec[p]["expo"] for p in POPS]
    ax.bar(range(1 + len(POPS)), expo_ax, 0.55,
           color=["k"] + [cols[p] for p in POPS])
    ax.axhspan(-1.6, -1.0, color="0.88", zorder=0)
    ax.set_xticks(range(1 + len(POPS)))
    ax.set_xticklabels(["头皮底"] + list(POPS), fontsize=8)
    ax.set_ylabel("谱指数", fontsize=8)
    ax.set_title("指数对照：谁的尖峰谱像底谱？", fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    fig.suptitle("condstate35 尖峰时序归因——--spike-trace 仪表"
                 "（位式不变已证）关闭 cs26 方法论缺口", fontsize=11)
    fig.savefig(OUT / "spike_attribution.png", dpi=150)
    plt.close(fig)
    print(f"-> {OUT / 'spike_attribution.png'}")


if __name__ == "__main__":
    main()
