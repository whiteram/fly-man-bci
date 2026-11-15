"""bci/condstate38/analyze.py — v-机制的跨态泛化检验（零额外跑）。

cs37: naive 底谱(−0.93) = T45 阈下 v 谱(−1.10)。若"头皮谱 = 投影的
主导池 v 谱"普遍成立, 高锁臂(头皮谱平, audit: −0.48)的 v 谱也应
相应变平。数据: 本弧 hlock v-trace 跑（配方=condstate4 完整旗标,
与 hlnt62 差 9e-13 V ≈ 逐位）。

用法 (conda ffbm, repo root):
    python bci/condstate38/analyze.py
输出: outputs/v_generalization.png, outputs/v_generalization.json
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

W0, W1 = 7000, 15000      # 与 audit/naive-cs37 同代表窗
POPS = ("MID", "T45", "OLR", "ALPN", "CEN")


def expo_of(x, fs=1000.0):
    f, P = welch(x, fs=fs, nperseg=min(len(x), 4096))
    m = (f >= 1) & (f <= 40)
    e, _, pk = spec_fit(np.hstack([[0.0], f[m]]),
                        np.hstack([[1e-30], P[m]]))
    return float(e), f[m], P[m], (pk[0] if pk else None)


def main():
    z = np.load(OUT / "_v_trace.npz")
    e = np.load(OUT / "hl_scalp.npy").astype(np.float64)[W0:W1] * 1e6
    cm = device(e).mean(axis=1)
    e_s, f_s, P_s, pk_s = expo_of(cm, fs=250.0)
    print(f"hlock 头皮谱: {e_s:+.2f} (audit 参照 −0.48)")

    res = {"scalp_expo": round(e_s, 2),
           "scalp_peak": [round(pk_s[0], 1), round(pk_s[1], 1)]
           if pk_s else None, "pops": {}}
    # naive 参照: cs37 的原始 1 kHz 分析结果 (json) —— 本弧仓库内的
    # 250 Hz 压缩件为朴素 [::4] 抽取, 存在混叠, 其指数不可比
    cs37 = json.loads((ROOT / "bci/condstate37/outputs"
                       / "v_attribution.json").read_text(encoding="utf-8"))
    for pop in POPS:
        row = {}
        nref = cs37["pops"].get(pop)
        row["naive"] = (dict(expo=nref["v_mean_expo"],
                             peak=nref["v_peak"][0]
                             if nref.get("v_peak") else None)
                        if nref and nref.get("v_mean_expo") is not None
                        else None)
        k = f"{pop}_v"
        if k in z.files:
            V = z[k][W0:W1].astype(np.float64)
            vm = V.mean(1)
            if vm.std() < 1e-9:
                row["hlock"] = None
            else:
                e_v, _, _, pk = expo_of(vm - vm.mean())
                row["hlock"] = dict(expo=round(e_v, 2),
                                    peak=round(pk[0], 1) if pk else None)
        else:
            row["hlock"] = None
        res["pops"][pop] = row
        print(f"{pop:5s} naive v {row.get('naive')} | hlock v "
              f"{row.get('hlock')}")

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3),
                             constrained_layout=True)
    ax = axes[0]
    names = [p for p in POPS
             if res["pops"][p].get("naive") and res["pops"][p].get("hlock")]
    x = np.arange(len(names))
    nv = [res["pops"][p]["naive"]["expo"] for p in names]
    hv = [res["pops"][p]["hlock"]["expo"] for p in names]
    ax.bar(x - 0.17, nv, 0.34, color="0.6", label="naive")
    ax.bar(x + 0.17, hv, 0.34, color="tab:green", label="hlock")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.axhline(-0.93, color="0.4", ls=":", lw=1)
    ax.text(len(names) - 0.5, -0.90, "naive 底 −0.93", fontsize=7,
            color="0.4")
    ax.axhline(e_s, color="tab:green", ls=":", lw=1)
    ax.text(len(names) - 0.5, e_s + 0.06, f"hlock 头皮 {e_s:+.2f}",
            fontsize=7, color="tab:green")
    ax.set_ylabel("v 均值谱指数", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("v 谱指数: naive vs hlock", fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    ax = axes[1]
    ax.loglog(f_s, P_s, lw=1.3, color="k",
              label=f"hlock 头皮 ({e_s:+.2f})")
    cols = {"MID": "tab:orange", "T45": "tab:green", "OLR": "tab:purple",
            "ALPN": "tab:red", "CEN": "tab:blue"}
    for p in names:
        V = z[f"{p}_v"][W0:W1].astype(np.float64)
        vm = V.mean(1)
        if vm.std() < 1e-9:
            continue
        _, fv, Pv, _ = expo_of(vm - vm.mean())
        ax.loglog(fv, Pv / Pv.max() * P_s.max(), lw=0.8, alpha=0.8,
                  color=cols[p],
                  label=f"{p} v ({res['pops'][p]['hlock']['expo']:+.2f})")
    ax.set_xlabel("Hz", fontsize=8)
    ax.set_ylabel("PSD (归一)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("hlock 头皮谱 vs 各池 v 谱", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    fig.suptitle("condstate38 v-机制跨态泛化——高锁臂的 v 谱是否"
                 "相应变平（cs37 的证伪检验）", fontsize=10.5)
    fig.savefig(OUT / "v_generalization.png", dpi=150)
    plt.close(fig)
    (OUT / "v_generalization.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'v_generalization.png'}")


if __name__ == "__main__":
    main()
