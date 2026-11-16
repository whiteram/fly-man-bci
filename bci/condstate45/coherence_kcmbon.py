"""bci/condstate45/coherence_kcmbon.py — cs45b 补充：招募池（KC/MBON）的相干度。

cs45 的诚实边界"仅 ALPN 池"：resid 的 21 nV 最强签名被判为
"招募为主 + 1.5× 微弱同步"，但招募主体（Kenyon_Cell/MBON）的
同步性从未测过。本补充用**同一批 300 s 档案**（cs15/20/30/41/
21/43）的 KC/MBON 池率做稀疏 Fano 分析——池率极稀（KC 0.0005
Hz × 4050 = 2 spikes/s；MBON 0.0135 × 96 = 1.3/s），改用
10 s 平稳块 × 长窗（0.5-2 s），FF>1（下四分位 >1）判同步。

用法 (conda ffbm, repo root):
    python bci/condstate45/coherence_kcmbon.py
输出: outputs/coherence_kcmbon.png, outputs/coherence_kcmbon_stats.json
"""
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "outputs"

TRACES = {
    "qlock·62": (ROOT / "bci/condstate21/outputs/qll62_pop.npz", 3.0),
    "qlock·63": (ROOT / "bci/condstate43/outputs/qll63_300_pop.npz", 3.0),
    "hlock·62": (ROOT / "bci/condstate15/outputs/hll62_pop.npz", 3.0),
    "hlock·63": (ROOT / "bci/condstate30/outputs/hll63_pop.npz", 3.0),
    "resid·62": (ROOT / "bci/condstate20/outputs/resl62_pop.npz", 8.0),
    "resid·63": (ROOT / "bci/condstate41/outputs/resl63_pop.npz", 8.0),
}
POOLS = {"Kenyon_Cell": 4050, "MBON": 96}
WS = (500, 1000, 2000)      # ms（稀疏池长窗）
BLOCK = 10000                # ms 平稳块


def fano_sparse(counts):
    nblocks = len(counts) // BLOCK
    out = {}
    for w in WS:
        per = BLOCK // w
        ffs = []
        for b in range(nblocks):
            seg = counts[b * BLOCK:(b + 1) * BLOCK]
            s = seg.reshape(per, w).sum(1)
            mu = s.mean()
            if mu > 0.2:               # 均值过低的块无意义
                ffs.append(s.var() / mu)
        if ffs:
            out[w] = (float(np.median(ffs)), float(np.percentile(ffs, 25)),
                      len(ffs))
        else:
            out[w] = (float("nan"),) * 2 + (0,)
    return out


def main():
    res = {}
    for tag, (path, skip) in TRACES.items():
        z = np.load(path)
        res[tag] = {}
        for pool, n in POOLS.items():
            v = z[pool]
            c = np.rint(v[int(skip * 1000):] * (n * 0.001)).astype(np.int64)
            fs = fano_sparse(c)
            # 取均值计数最接近 2 的窗作代表
            best = max(fs, key=lambda w: fs[w][2])
            med, q25, nb = fs[best]
            res[tag][pool] = {
                "mean_rate_hz": round(float(c.mean() / (n * 0.001)), 6),
                "ff_median": round(med, 2) if med == med else None,
                "ff_q25": round(q25, 2) if q25 == q25 else None,
                "w_rep_ms": best, "n_blocks": nb,
                "synchronous": bool(q25 > 1.0),
            }
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # 汇总: 每池跨 6 档案的最大 FF 中位
    summ = {}
    for pool in POOLS:
        meds = [(t, res[t][pool]["ff_median"]) for t in res
                if res[t][pool]["ff_median"] is not None]
        mx = max((m for _, m in meds), default=float("nan"))
        summ[pool] = {"max_ff_median": mx,
                      "any_synchronous": any(res[t][pool]["synchronous"]
                                             for t in res)}
    res["_summary"] = summ
    print(json.dumps(summ, ensure_ascii=False))

    # ---- 图 (1x2) --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             constrained_layout=True)
    ax = axes[0]
    cols = {"qlock": "tab:cyan", "hlock": "tab:green", "resid": "tab:red"}
    tags = list(TRACES)
    x = np.arange(len(tags))
    for i, pool in enumerate(POOLS):
        vals = [res[t][pool]["ff_median"] or 0 for t in tags]
        ax.bar(x + i * 0.38 - 0.19, vals, 0.38, label=pool,
               color="tab:orange" if i == 0 else "tab:purple")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.text(len(tags) - 0.5, 1.08, "泊松 FF=1", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=8, rotation=20)
    ax.set_ylabel("FF 中位（代表窗）", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title("招募池稀疏 Fano：同步（>1）还是异步？", fontsize=10)

    ax = axes[1]
    ax.axis("off")
    rows = [("档案", "池", "均率(Hz)", "FF", "判定")]
    for t in tags:
        for pool in POOLS:
            r = res[t][pool]
            rows.append((t, pool, f"{r['mean_rate_hz']:.4f}",
                         f"{r['ff_median']}",
                         "同步" if r["synchronous"] else "异步"))
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center",
                  loc="center", colWidths=[0.2, 0.22, 0.2, 0.14, 0.14])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    tb.scale(1, 1.15)
    ax.set_title("KC/MBON 池率与 Fano（cs45b，零新采集）", fontsize=10)

    fig.suptitle("condstate45b 招募池相干度补充——招募阶梯的同步性检验",
                 fontsize=11)
    fig.savefig(OUT / "coherence_kcmbon.png", dpi=150)
    plt.close(fig)
    (OUT / "coherence_kcmbon_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'coherence_kcmbon.png'}")


if __name__ == "__main__":
    main()
