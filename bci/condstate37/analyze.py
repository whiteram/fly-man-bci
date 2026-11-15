"""bci/condstate37/analyze.py — v 谱归因：cs36"时间滤波"再定位的直接检验。

cs36 把底 1/f 机制再定位到"前向时间滤波"。零引擎替代假设：**前向
投影的是阈下 v 动力学**（v = 膜电位的低通/OU 谱，天然负指数）。
本弧用新仪表 `--v-trace 128`（同 --spike-trace 只读钩子，位式不变
已证：3 s 开关逐位同、20 s 底跑校验和与 cs25/27/35 一致）检验：

  1. 各池采样 v 均值的谱指数（vs 底谱 −0.99）；
  2. 逐神经元 v 谱指数分布（异质率下是否同形）；
  3. 若 v 谱 ≈ 底谱形状 → "时间滤波"实体 = 阈下 v 的 OU/膜动力学
     ——机制链闭环（尖峰平谱 + v 负谱 + 前向投影 v）。

用法 (conda ffbm, repo root):
    python bci/condstate37/analyze.py
输出: outputs/v_attribution.png, outputs/v_attribution.json
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

W0, W1 = 2000, 19000
POPS = ("R", "L", "MID", "T45", "OLR")


def expo_of(x, fs=1000.0):
    f, P = welch(x, fs=fs, nperseg=min(len(x), 4096))
    m = (f >= 1) & (f <= 40)
    e, _, pk = spec_fit(np.hstack([[0.0], f[m]]),
                        np.hstack([[1e-30], P[m]]))
    return float(e), f[m], P[m], (pk[0] if pk else None)


def main():
    z = np.load(OUT / "_v_trace.npz")
    e = np.load(OUT / "floor_scalp.npy").astype(np.float64)[W0:W1] * 1e6
    cm = device(e).mean(axis=1)
    e_f, f_f, P_f, pk_f = expo_of(cm, fs=250.0)
    print(f"底谱: {e_f:+.2f}, 峰 {pk_f}")

    res = {"floor_expo": round(e_f, 2),
           "floor_peak": [round(pk_f[0], 1), round(pk_f[1], 1)] if pk_f
           else None, "pops": {}}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                             constrained_layout=True)
    cols = {"R": "0.6", "L": "tab:red", "MID": "tab:orange",
            "T45": "tab:green", "OLR": "tab:purple"}
    ax0 = axes[0]
    ax0.loglog(f_f, P_f, lw=1.6, color="k",
               label=f"头皮底谱 ({e_f:+.2f})")
    for pop in POPS:
        V = z[f"{pop}_v"][W0:W1].astype(np.float64)   # (T, n)
        vm = V.mean(1)
        if vm.std() < 1e-9:
            res["pops"][pop] = dict(v_mean_expo=None,
                                    note="v 恒定")
            print(f"{pop}: v 恒定")
            continue
        e_v, f_v, P_v, pk_v = expo_of(vm - vm.mean())
        # 逐神经元
        nes = []
        for j in range(V.shape[1]):
            x = V[:, j]
            if x.std() < 1e-9:
                continue
            ej, _, _, _ = expo_of(x - x.mean())
            nes.append(ej)
        res["pops"][pop] = dict(
            v_mean_expo=round(e_v, 2),
            v_peak=[round(pk_v[0], 1), round(pk_v[1], 1)] if pk_v
            else None,
            n_neuron=len(nes),
            neuron_expo_med=round(float(np.median(nes)), 2),
            neuron_expo_p10=round(float(np.percentile(nes, 10)), 2),
            neuron_expo_p90=round(float(np.percentile(nes, 90)), 2),
            v_rms_mv=round(float(np.sqrt((V ** 2).mean())), 2))
        print(f"{pop}: v均值谱 {e_v:+.2f} (峰 {pk_v}), 个体中位 "
              f"{np.median(nes):+.2f} (n={len(nes)})")
        ax0.loglog(f_v, P_v / P_v.max() * P_f.max(), lw=0.8,
                   alpha=0.8, color=cols[pop],
                   label=f"{pop} v ({e_v:+.2f})")
    ax0.set_xlabel("Hz", fontsize=8)
    ax0.set_ylabel("PSD (归一)", fontsize=8)
    ax0.legend(fontsize=7)
    ax0.set_title("底谱 vs 各池 v 均值谱（128 采样）", fontsize=9)
    ax0.grid(True, which="both", color="0.93")

    # 面板2: v 指数 vs 底
    ax = axes[1]
    names = [p for p in POPS if res["pops"].get(p, {}).get("v_mean_expo")
             is not None]
    vals = [res["pops"][p]["v_mean_expo"] for p in names]
    ax.bar(range(len(names)), vals, 0.55,
           color=[cols[p] for p in names])
    ax.axhline(e_f, color="k", ls="--", lw=1.3)
    ax.text(len(names) - 0.45, e_f + 0.05, f"底 {e_f:+.2f}",
            fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("v 均值谱指数", fontsize=8)
    ax.set_title("谁的 v 谱像底谱？", fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    # 面板3: 逐神经元 v 指数分布 (合并)
    ax = axes[2]
    for pop in names:
        V = z[f"{pop}_v"][W0:W1].astype(np.float64)
        nes = []
        for j in range(V.shape[1]):
            x = V[:, j]
            if x.std() < 1e-9:
                continue
            ej, _, _, _ = expo_of(x - x.mean())
            nes.append(ej)
        ax.hist(nes, bins=24, alpha=0.5, color=cols[pop],
                label=pop, density=True)
    ax.axvline(e_f, color="k", lw=1.4, ls="--")
    ax.set_xlabel("逐神经元 v 谱指数", fontsize=8)
    ax.set_ylabel("密度", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title(f"个体 v 指数分布（底 {e_f:+.2f}）", fontsize=9)
    ax.grid(True, color="0.93")

    fig.suptitle("condstate37 v 谱归因——阈下动力学是否携带底的 1/f"
                 "（--v-trace 仪表, 位式不变已证）", fontsize=11)
    fig.savefig(OUT / "v_attribution.png", dpi=150)
    plt.close(fig)
    (OUT / "v_attribution.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'v_attribution.png'}")


if __name__ == "__main__":
    main()
