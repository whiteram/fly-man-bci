"""bci/condstate44/ignition_threshold.py — 静默锁诱导阈值的剂量扫描定位。

数据: 5 个 20 s 剂量臂（cs44d{175,200,225,250,275}, seed 63,
dangate2 全套旗标）+ 三个已知端点（cs42 150 不燃 / cs2 300 燃 /
cs21 62@150 燃）。

问题: cs42/43 留下"阈值在 (150,300] pA 内未定位"——本弧补 5 点
定位阈值区间，并检验点火的**全有全无性**（双稳闩锁 vs 连续剂量
响应）。

点火判据: 脉冲后 2-20 s ALPN 均值 ≥ 10 Hz（62@150 燃 ≈19.7 /
null 臂 = 0.000——双峰分布，阈值取中）。

用法 (conda ffbm, repo root):
    python bci/condstate44/ignition_threshold.py
输出: outputs/ignition_threshold.png, outputs/ignition_threshold_stats.json
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
IGN_TH = 10.0   # Hz, 脉冲后 2-20 s 均值（双峰中间）


def arm_metrics(alpn):
    """alpn: 1-ms ALPN 池率数组（>=20 s）。"""
    return {
        "pulse_peak_hz": round(float(alpn[250:700].max()), 2),
        "post_mean_hz": round(float(alpn[2000:20000].mean()), 3),
        "post_max_hz": round(float(alpn[2000:].max()), 2),
        "post_min_hz": round(float(alpn[2000:].min()), 3),
    }


def main():
    doses = {}
    for a in (175, 200, 225, 250, 275):
        p = np.load(OUT / f"d{a}_pop.npz")["ALPN"]
        doses[a] = arm_metrics(p)
    # 已知端点（各取前 20 s）
    p150 = np.load(ROOT / "bci/condstate42/outputs/qll63_pop.npz")["ALPN"]
    doses[150] = arm_metrics(p150)
    p300 = np.load(ROOT / "bci/condstate43/outputs/qll63_300_pop.npz")["ALPN"]
    doses[300] = arm_metrics(p300)
    p62 = np.load(ROOT / "bci/condstate21/outputs/qll62_pop.npz")["ALPN"]
    s62 = arm_metrics(p62)

    ignited = {a: m["post_mean_hz"] >= IGN_TH for a, m in doses.items()}
    no = [a for a in doses if not ignited[a]]
    yes = [a for a in doses if ignited[a]]
    lo = max(no) if no else None      # 最高不燃剂量
    hi = min(yes) if yes else None    # 最低燃剂量
    # 中间剂量（若存在不燃<燃之间的过渡点）
    mids = {a: doses[a]["post_mean_hz"] for a in doses
            if 0.5 < doses[a]["post_mean_hz"] < IGN_TH}

    res = {
        "doses_seed63": {str(a): doses[a] for a in sorted(doses)},
        "seed62_at_150": s62,
        "ignition_threshold_hz": IGN_TH,
        "ignited": {str(a): ignited[a] for a in sorted(doses)},
        "threshold_bracket_pa": [lo, hi],
        "intermediate_states": mids,
        "allornone": len(mids) == 0,
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # ---- 图 (2x2) --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6),
                             constrained_layout=True)
    ax = axes[0][0]
    aa = sorted(doses)
    pm = [doses[a]["post_mean_hz"] for a in aa]
    cs = ["tab:red" if not ignited[a] else "tab:blue" for a in aa]
    ax.plot(aa, pm, "o-", color="0.7", lw=0.8, zorder=1)
    ax.scatter(aa, pm, c=cs, s=70, zorder=2)
    ax.axhline(IGN_TH, color="k", ls="--", lw=1)
    ax.axhline(19.7, color="0.5", ls=":", lw=1)
    ax.text(230, 20.6, "62@150 燃 19.7 Hz", fontsize=7, color="0.4")
    ax.annotate(f"阈值区间 ({lo},{hi}] pA", ((lo + hi) / 2, IGN_TH / 2),
                fontsize=9, ha="center",
                xytext=((lo + hi) / 2, IGN_TH / 2 + 2),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("DA1 剂量 (pA)", fontsize=9)
    ax.set_ylabel("脉冲后 2-20 s ALPN 均值 (Hz)", fontsize=9)
    ax.set_title("点火曲线（蓝=燃，红=不燃）", fontsize=10)

    ax = axes[0][1]
    for a in aa:
        p = (np.load(OUT / f"d{a}_pop.npz")["ALPN"] if a in (175, 200, 225, 250, 275)
             else {150: p150, 300: p300}[a])[:20000]
        ax.plot(np.arange(20000) / 1000.0, p, lw=0.5,
                color="tab:blue" if ignited[a] else "tab:red",
                alpha=0.9, label=f"{a} pA")
    ax.set_xlabel("时间 (s)", fontsize=9)
    ax.set_ylabel("ALPN 池率 (Hz)", fontsize=9)
    ax.set_xlim(0, 20)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("ALPN 轨迹（蓝=燃，红=不燃）", fontsize=10)

    ax = axes[1][0]
    pk = [doses[a]["pulse_peak_hz"] for a in aa]
    ax.plot(aa, pk, "o-", color="tab:purple")
    ax.plot([150], [s62["pulse_peak_hz"]], "s", ms=8, color="0.4",
            label="62@150（燃）")
    ax.set_xlabel("DA1 剂量 (pA)", fontsize=9)
    ax.set_ylabel("脉冲窗峰 (Hz)", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_title("脉冲窗瞬态峰 vs 剂量（输入端仍连续）", fontsize=10)

    ax = axes[1][1]
    ax.axis("off")
    rows = [("剂量", "脉冲后均值", "点火"),
            *[(f"{a} pA", f"{doses[a]['post_mean_hz']} Hz",
               "燃" if ignited[a] else "不燃") for a in aa]]
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center",
                  loc="center", colWidths=[0.28, 0.36, 0.28])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.6)
    ax.set_title(f"种子 63 剂量表——阈值 ({lo},{hi}] pA，"
                 f"全有全无：{res['allornone']}", fontsize=10)

    fig.suptitle("condstate44 静默锁诱导阈值扫描（seed 63；点火判据："
                 f"脉冲后 2-20 s ALPN ≥{IGN_TH} Hz）", fontsize=11)
    fig.savefig(OUT / "ignition_threshold.png", dpi=150)
    plt.close(fig)
    (OUT / "ignition_threshold_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'ignition_threshold.png'}")
    print(f"-> {OUT / 'ignition_threshold_stats.json'}")


if __name__ == "__main__":
    main()
