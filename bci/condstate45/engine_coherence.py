"""bci/condstate45/engine_coherence.py — 引擎相干度实测：生物学自身同步化 vs 解除要求的距离。

问题: cs22b 解除地图判定"四轴全封，唯一剩余自由度是相干源
（5-78×）"——但蝇自身的状态引擎（AL 复发环路群齐射）已经提供
了部分同步化。本弧从 6 条 300 s ALPN 1-ms 池率档案（3 态 ×
2 种子）用 Fano 因子实测每态的**同步集团大小**，换算相干幅度
增益 g=√FF，与解除要求对照。

数据（全部已有，零新采集）:
  qlock62 = cs21/qll62   qlock63 = cs43/qll63_300
  hlock62 = cs15/hll62   hlock63 = cs30/hll63
  resid62 = cs20/resl62  resid63 = cs41/resl63
（qlock63@300 pA、resid 带 6000:CEN_C=0.5 调度——各自正确配方）

方法: 池率 v → 每毫秒计数 c = round(v·0.68)（680 神经元）；
在 1 s 平稳块内对窗宽 w ∈ {1,2,5,10,20,50} ms 计非重叠和的
Fano FF=σ²/µ，块间取中位。独立泊松 FF=1；同步齐射（复合泊松、
集团大小 k̄）FF≈k̄；相对独立发放的幅度增益 g=√FF。

用法 (conda ffbm, repo root):
    python bci/condstate45/engine_coherence.py
输出: outputs/engine_coherence.png, outputs/engine_coherence_stats.json
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
N_POOL = 680
WS = (1, 2, 5, 10, 20, 50)      # ms
BLOCK = 1000                     # ms 平稳块
UNSEAL = (5.0, 78.0)             # 相干源解除需求（cs22b/43 总账）


def fano_curve(counts):
    """counts: 1-ms 计数序列 → 每窗宽的块内中位 Fano。"""
    nblocks = len(counts) // BLOCK
    out = {}
    for w in WS:
        per = BLOCK // w
        ffs = []
        for b in range(nblocks):
            seg = counts[b * BLOCK:(b + 1) * BLOCK]
            s = seg.reshape(per, w).sum(1)
            mu = s.mean()
            if mu > 0:
                ffs.append(s.var() / mu)
        out[w] = float(np.median(ffs)) if ffs else float("nan")
    return out


def main():
    res = {}
    curves = {}
    for tag, (path, skip) in TRACES.items():
        v = np.load(path)["ALPN"]
        c = np.rint(v[int(skip * 1000):] * (N_POOL * 0.001)).astype(np.int64)
        fc = fano_curve(c)
        curves[tag] = fc
        wmax = max(fc, key=fc.get)
        res[tag] = {
            "mean_rate_hz": round(float(c.mean() / (N_POOL * 0.001)), 3),
            "fano": {str(w): round(fc[w], 2) for w in WS},
            "ff_max": round(fc[wmax], 2), "w_at_ffmax_ms": wmax,
            "gain_sqrt_ff": round(float(np.sqrt(fc[wmax])), 2),
            "group_fraction_of_pool": round(fc[wmax] / N_POOL, 4),
        }
        print(tag, res[tag])

    # 距解除要求的比值（用各态两种子的 g 均值）
    states = {}
    for st in ("qlock", "hlock", "resid"):
        gs = [res[f"{st}·{s}"]["gain_sqrt_ff"] for s in ("62", "63")]
        states[st] = {"g_mean": round(float(np.mean(gs)), 2),
                      "g_range": [round(min(gs), 2), round(max(gs), 2)]}
        states[st]["unseal_coverage"] = [
            round(float(np.mean(gs)) / UNSEAL[1], 4),
            round(float(np.mean(gs)) / UNSEAL[0], 4)]
    res["_summary"] = {"unseal_requirement_x": UNSEAL, "states": states,
                       "rms_nV_ref": {"naive": 1.0, "qlock": 2.8,
                                      "hlock": 7.0, "resid": 21.0}}
    print(json.dumps(res["_summary"], ensure_ascii=False, indent=1))

    # ---- 图 (2x2) --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6),
                             constrained_layout=True)
    ax = axes[0][0]
    cols = {"qlock": "tab:cyan", "hlock": "tab:green", "resid": "tab:red"}
    for tag, fc in curves.items():
        st = tag.split("·")[0]
        ax.plot(WS, [fc[w] for w in WS], "o-", ms=4, lw=0.9,
                color=cols[st], alpha=0.55 if tag.endswith("63") else 1.0,
                label=tag)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.text(20, 1.15, "独立泊松 FF=1", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("窗宽 w (ms)", fontsize=9)
    ax.set_ylabel("Fano 因子 FF(w)", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("ALPN 池率涨落：同步集团大小 k_eff≈FF", fontsize=10)

    ax = axes[0][1]
    tags = list(curves)
    gs = [res[t]["gain_sqrt_ff"] for t in tags]
    ax.bar(range(len(tags)), gs, 0.6,
           color=[cols[t.split("·")[0]] for t in tags])
    ax.axhspan(UNSEAL[0], UNSEAL[1], color="0.85", alpha=0.5)
    ax.axhline(UNSEAL[0], color="k", ls=":", lw=1.2)
    ax.axhline(UNSEAL[1], color="k", ls=":", lw=1.2)
    ax.text(0.1, UNSEAL[0] * 1.3, "解除需求 5×", fontsize=8)
    ax.text(0.1, UNSEAL[1] * 1.1, "78×", fontsize=8)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, fontsize=8, rotation=20)
    ax.set_ylabel("相干幅度增益 g=√FF", fontsize=9)
    ax.set_title("生物学自身同步化 vs 解除需求（灰带）", fontsize=10)
    for i, g in enumerate(gs):
        ax.text(i, g + 0.15, f"{g:.1f}×", ha="center", fontsize=8)

    ax = axes[1][0]
    for i, (tag, (path, skip)) in enumerate(TRACES.items()):
        v = np.load(path)["ALPN"]
        seg = v[int((skip + 5) * 1000):int((skip + 6) * 1000)]
        seg = seg / max(seg.max(), 1e-9) * 40.0   # 各段峰值归一（hlock ~150 Hz 超出公共量程）
        ax.plot(np.arange(len(seg)) / 1000.0 + i * 1.1, seg, lw=0.4,
                color=cols[tag.split("·")[0]], alpha=0.8)
        ax.text(i * 1.1 + 0.5, 46, tag, fontsize=7, ha="center")
    ax.set_xlim(-0.05, 6.55)
    ax.set_ylim(-1, 50)
    ax.set_xlabel("各态 1 s 示例段（错开显示，各段峰值归一）", fontsize=9)
    ax.set_ylabel("ALPN 池率（归一）", fontsize=9)
    ax.set_yticks([])
    ax.set_title("引擎放电结构（齐射 vs 平稳）", fontsize=10)

    ax = axes[1][1]
    ax.axis("off")
    rows = [("态（种子均值）", "k_eff=FF_max", "g=√FF", "解除覆盖"),
            *[(f"{st}", f"{states[st]['g_mean']**2:.0f}",
               f"{states[st]['g_mean']:.1f}×",
               f"{states[st]['unseal_coverage'][1]*100:.0f}-"
               f"{states[st]['unseal_coverage'][0]*100:.0f}%")
              for st in ("qlock", "hlock", "resid")]]
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center",
                  loc="center", colWidths=[0.3, 0.2, 0.2, 0.26])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.7)
    ax.set_title("相干增益对照（解除需求 5-78×）", fontsize=10)

    fig.suptitle("condstate45 引擎相干度实测（6 条 300 s ALPN 档案，零新采集）"
                 "——生物学自身同步化距解除需求还有量级差距", fontsize=11)
    fig.savefig(OUT / "engine_coherence.png", dpi=150)
    plt.close(fig)
    (OUT / "engine_coherence_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'engine_coherence.png'}")
    print(f"-> {OUT / 'engine_coherence_stats.json'}")


if __name__ == "__main__":
    main()
