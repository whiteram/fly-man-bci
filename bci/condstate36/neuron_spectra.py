"""bci/condstate36/neuron_spectra.py — 逐神经元谱 + 合成加权聚合（零采集）。

cs35 机制命题："底 −0.99 指数产生于加权聚合本身（异质率×核权重）"。
本弧用 cs35 的尖峰轨迹直接检验：

  1. 异质率定量（每池 top-10% 事件份额、静默占比）；
  2. 逐神经元谱（≥100 事件者）：指数分布——个体是平还是调制？
  3. 合成聚合：128 训练 × 对数正态权重（核增益样）求和 → 谱指数；
     对照：不加权池化（cs35 实测 +0.36 平）与底谱（−0.99）。
     ×50 权重抽样 → 指数分布。

用法 (conda ffbm, repo root):
    python bci/condstate36/neuron_spectra.py
输出: outputs/neuron_spectra.png, outputs/neuron_spectra.json
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
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "bci" / "condstate14"))
sys.path.insert(0, str(ROOT / "bci" / "eegview"))
from state_monitor import device  # noqa: E402
from likeness_audit import spec_fit  # noqa: E402

T_MS, W0, W1 = 20000, 2000, 19000
POPS = ("MID", "T45", "OLR")


def train_matrix(z, pop):
    """(T_MS, n_samp) 0/1 尖峰矩阵."""
    ids = z[f"{pop}_ids"]
    M = np.zeros((T_MS, len(ids)), np.float32)
    t, i = z[f"{pop}_t"].astype(np.int64), z[f"{pop}_i"]
    M[t, i] = 1.0
    return M


def expo_of(x, flo=1.0, fhi=40.0, fs=250.0):
    f, P = welch(x, fs=fs, nperseg=min(len(x), 2048))
    m = (f >= flo) & (f <= fhi)
    e, _, _ = spec_fit(np.hstack([[0.0], f[m]]),
                       np.hstack([[1e-30], P[m]]))
    return float(e), f[m], P[m]


def main():
    z = np.load(ROOT / "bci/condstate35/outputs/_spike_trace.npz")
    e = np.load(ROOT / "bci/condstate35/outputs/floor_scalp.npy") \
        .astype(np.float64)[W0:W1] * 1e6
    cm = device(e).mean(axis=1)
    e_floor, f_f, P_f = expo_of(cm)
    print(f"底谱指数 {e_floor:+.2f}")

    res = {"floor_expo": round(e_floor, 2), "pops": {}}
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                             constrained_layout=True)
    cols = {"MID": "tab:orange", "T45": "tab:green", "OLR": "tab:purple"}

    all_neuron_expos = []
    for pop in POPS:
        M = train_matrix(z, pop)[W0:W1]          # (17000, n)
        cnt = M.sum(0)
        order = np.argsort(-cnt)
        top10 = cnt[order[:max(1, len(cnt) // 10)]].sum()
        share = float(top10 / cnt.sum())
        # 逐神经元指数（≥100 事件）
        ne = []
        for j in np.flatnonzero(cnt >= 100):
            x = M[:, j]
            if x.mean() < 1e-6:
                continue
            ej, _, _ = expo_of(x - x.mean())
            ne.append(ej)
        ne = np.array(ne)
        # 合成加权聚合 ×50（对数正态权重, sigma=1 —— 核增益样异质）
        synth = []
        pooled = M.mean(1)
        ep_pool, fp, Pp = expo_of(pooled - pooled.mean())
        Mz = M - M.mean(0, keepdims=True)
        for rep in range(50):
            w = rng.lognormal(0, 1.0, M.shape[1])
            ep, _, _ = expo_of(Mz @ w)
            synth.append(ep)
        synth = np.array(synth)
        res["pops"][pop] = dict(
            n=len(cnt), silent=int((cnt == 0).sum()),
            top10_share=round(share, 3),
            rate_med_hz=round(float(np.median(cnt) / (W1 - W0) * 1000), 2),
            rate_max_hz=round(float(cnt.max() / (W1 - W0) * 1000), 1),
            n_neuron_expos=int(len(ne)),
            neuron_expo_med=round(float(np.median(ne)), 2) if len(ne)
            else None,
            pooled_expo=round(ep_pool, 2),
            synth_expo_med=round(float(np.median(synth)), 2),
            synth_expo_p10=round(float(np.percentile(synth, 10)), 2),
            synth_expo_p90=round(float(np.percentile(synth, 90)), 2),
        )
        all_neuron_expos += list(ne)
        print(pop, res["pops"][pop])
        ax = axes[0]
        ax.hist(ne, bins=24, alpha=0.55, color=cols[pop],
                label=f"{pop} (n={len(ne)})", density=True)

    # 面板1: 逐神经元指数分布
    ax = axes[0]
    ax.axvline(e_floor, color="k", lw=1.4, ls="--",
               label=f"底谱 {e_floor:+.2f}")
    ax.set_xlabel("逐神经元谱指数（≥100 事件）", fontsize=8)
    ax.set_ylabel("密度", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title(f"个体指数分布（合并中位 "
                 f"{np.median(all_neuron_expos):+.2f}）", fontsize=9)
    ax.grid(True, color="0.93")

    # 面板2: 合成聚合指数 vs 底
    ax = axes[1]
    for pop, off in zip(POPS, (0, 1, 2)):
        d = res["pops"][pop]
        ax.errorbar(off, d["synth_expo_med"],
                    yerr=[[d["synth_expo_med"] - d["synth_expo_p10"]],
                          [d["synth_expo_p90"] - d["synth_expo_med"]]],
                    fmt="o", ms=9, capsize=4, color=cols[pop])
        ax.annotate(f"{pop}\n不加权 {d['pooled_expo']:+.2f}",
                    (off, d["synth_expo_med"]),
                    textcoords="offset points", xytext=(10, -4),
                    fontsize=7.5, color=cols[pop])
    ax.axhline(e_floor, color="k", ls="--", lw=1.2)
    ax.text(-0.4, e_floor + 0.04, f"底谱 {e_floor:+.2f}", fontsize=8)
    ax.set_xticks([])
    ax.set_ylabel("合成对数正态加权和的谱指数（×50 抽样）", fontsize=8)
    ax.set_title("加权聚合能否产生负指数？", fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    # 面板3: top-10% 份额条
    ax = axes[2]
    shares = [res["pops"][p]["top10_share"] for p in POPS]
    silents = [res["pops"][p]["silent"] for p in POPS]
    x = np.arange(len(POPS))
    ax.bar(x, shares, 0.55, color=[cols[p] for p in POPS])
    for i, (s, sl) in enumerate(zip(shares, silents)):
        ax.text(i, s + 0.02, f"{s:.0%}\n静默 {sl}/128", ha="center",
                fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(POPS, fontsize=9)
    ax.set_ylabel("top-10% 神经元事件份额", fontsize=8)
    ax.set_title("异质率：事件由极小尾部扛", fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    fig.suptitle("condstate36 逐神经元谱 + 合成加权聚合——底 1/f 指数的"
                 "机制检验（cs35 命题）", fontsize=11)
    fig.savefig(OUT / "neuron_spectra.png", dpi=150)
    plt.close(fig)
    (OUT / "neuron_spectra.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'neuron_spectra.png'}")


if __name__ == "__main__":
    main()
