"""bci/condstate22/unseal_map.py — 解除条件地图：读出封死的设计空间语言。

总账（docs/MEASURABILITY.md）证明读出封死。本弧把负结果翻成
"什么能解除它"的定量地图——四条逃逸轴各需多大倍数（零采集，
全部数字来自总账 json + 头模型几何 meta）：

  1. 背景轴: 需背景 ÷G（G=11.4/σ*）；等效"皮层静默比例"
     （独立发生器背景 ∝ √f → 需静默 ≥1−1/G²）；
  2. 源轴（放大悖论）: 需相干源 ×G（非相干 ×G²）——蝇 1.4×10⁵
     神经元 → 需 7×10⁵~7.6×10⁶ 相干——比人类皮层 8.6×10¹⁰
     仍低 4 个量级（"差一个量级而不是四个"的悲剧性发现）；
  3. 距离轴: 远场 ~1/r² → 需源-电极距离 ÷√G；**当前蝇在头球正
     心（shift=0，到电极环 90.6 mm——深源最坏位，也是近全共模
     的构造原因）**；贴骨放置 ~15-20 mm 可得 ~25-34×——
     唯一物理可达的解除轴（需核几何改动，超当前范围）；
  4. 时间轴: ×(W_req/τ_sig)——已被寿命封死（cs16/20/21）。

用法 (conda ffbm, repo root):
    python bci/condstate22/unseal_map.py
输出: outputs/unseal_map.png, outputs/unseal_map_stats.json
"""
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

SIG_BG = 11.4
N_FLY = 1.4e5
N_HUMAN = 8.6e10
# 头模型几何 (viz_data meta): 4 球 78-92 mm, 电极环 90.62 mm, 蝇 shift=0
R_NOW = 90.62          # mm, 中心源到电极
R_SUBSKIN = 17.5       # mm, 贴骨放置（皮下 ~15-20 mm 中值）
STATES = {             # s (µV), σ* (µV), τ_sig (s), W_req@11.4 (s)
    "qlock": dict(s=0.00181, star=0.210, tau=2988.0, wreq=2289 * 3600),
    "hlock": dict(s=0.00598, star=0.233, tau=336.0, wreq=209 * 3600),
    "resid": dict(s=0.01997, star=2.30, tau=2948.0, wreq=18.7 * 3600),
}
# 几何事实（fit_scale_shift 语义 + meta 实证, 2026-11-15 修正）:
#   SCALE×202 把蝇脑云放大到极端点 0.98·R_BRAIN≈76.4 mm（长轴骑直径
#   穿头心，宽云居中 shift=0）——思想实验已把脑球填满，**无放置自由度**；
#   距离轴结构上已用尽（cs22 首版的"正心小源/贴骨增益 25-34×"撤回）。


def main():
    rows = {}
    for k, v in STATES.items():
        G = SIG_BG / v["star"]
        rows[k] = dict(
            gap=G,
            bg_div=round(G, 1),
            silence_frac=round(1 - 1 / G ** 2, 4),
            src_coherent_x=round(G, 1),
            src_coherent_n=int(N_FLY * G),
            src_incoherent_x=round(G ** 2, 0),
            dist_div=round(np.sqrt(G), 2),
            r_needed_mm=round(R_NOW / np.sqrt(G), 1),
            time_x=round(v["wreq"] / v["tau"], 0),
        )
        print(k, rows[k])

    res = {
        "geometry": {"scale": 202.0, "native_head_r_um": 387.2,
                     "cloud_extreme_mm": 76.4,
                     "note": "SCALE×202 把蝇脑云放大到 0.98·R_BRAIN"
                             "（长轴骑直径穿头心）——脑球已填满, "
                             "距离轴结构上用尽, 无放置自由度",
                     "correction": "首版'正心小源 r=90.6/贴骨 25-34×'"
                                   "判定撤回 (2026-11-15)"},
        "states": rows,
        "verdict": {
            "resid": "距离轴 ✗ 结构用尽（极端源已至 76.4 mm ≥ 所需 "
                     "40.7 mm, 而实测 σ* 仍差 5×——1/r² 高估分布式"
                     "源收益）；源轴 ×5 相干（悖论）",
            "hlock": "四轴皆封（背景÷49/距离结构用尽/相干×49）",
            "qlock": "四轴皆封（背景÷54/距离结构用尽/相干×54）",
            "time": "全态被封死（×23~×2758 超寿命）",
        },
        "amplification_paradox": {
            "fly": N_FLY, "needed_range": [7e5, 7.6e6],
            "human_cortex": N_HUMAN,
            "note": "解除只需 5-54× 相干源——蝇距可读差一个量级"
                    "而非四个；但那是'果蝇脑在人头皮下'思想实验"
                    "的终点（再放大就不再是蝇脑）"},
    }
    (OUT / "unseal_map_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 图 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9),
                             constrained_layout=True)
    # A. 逃逸轴需求（每态 4 轴, log）
    ax = axes[0]
    tags = ["qlock", "hlock", "resid"]
    axx = [("bg_div", "背景 ÷", "tab:orange"),
           ("dist_div", "距离 ÷", "tab:green"),
           ("src_coherent_x", "相干源 ×", "tab:red"),
           ("time_x", "时间 ×", "tab:purple")]
    ys = np.arange(len(tags))[::-1]
    for j, (key, lab, c) in enumerate(axx):
        xs = np.full(3, j) + (np.arange(3) - 1) * 0.09
        vals = [rows[t][key] for t in tags]
        ax.scatter(xs, ys, s=70, color=c, zorder=3)
        for x, y, v in zip(xs, ys, vals):
            ax.text(x, y + 0.24, f"{v:g}", ha="center", fontsize=7)
    ax.axhspan(-0.5, 2.9, color="0.97", zorder=0)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{l}倍数\n(log)" for _, l, _ in axx], fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels(tags, fontsize=9)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.5, 2.9)
    ax.set_title("四轴解除需求（标注=倍数；时间轴全态封死、"
                 "距离轴结构用尽）", fontsize=9)

    # B. 放大悖论
    ax = axes[1]
    pts = [("蝇脑", N_FLY, "tab:blue"),
           ("解除下限\n(resid 相干)", 7e5, "tab:red"),
           ("解除上限\n(qlock 相干)", 7.6e6, "tab:red"),
           ("人类皮层", N_HUMAN, "0.4")]
    for i, (lab, n, c) in enumerate(pts):
        ax.plot([0, 1], [i, i], "-", color=c, lw=1.0, alpha=0.5)
        ax.plot([np.log10(n)], [i], "o", ms=9, color=c)
        ax.text(np.log10(n) + 0.06, i, f" 10^{np.log10(n):.1f}",
                va="center", fontsize=8, color=c)
        ax.text(-0.35, i, lab, ha="right", va="center", fontsize=8)
    ax.set_xlim(3.6, 12.2)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.set_xticks(range(4, 13))
    ax.set_xlabel("log10 神经元数", fontsize=8)
    ax.set_title("放大悖论：解除只差 5-54× 相干源（一个量级，非四个）\n"
                 "但再放大就不再是蝇脑", fontsize=9)
    ax.grid(True, axis="x", color="0.93")

    # C. 几何事实：蝇脑云已填满脑球（修正版）
    ax = axes[2]
    ax.set_aspect("equal")
    ax.axis("off")
    for r, lab in ((92.0, "头皮 92"), (78.0, "脑 78")):
        ax.add_patch(Circle((0, 0), r, fill=False, color="0.55", lw=1.2))
        a = np.deg2rad(135)
        ax.text(r * np.cos(a) * 0.93, r * np.sin(a) * 0.93, lab,
                fontsize=7, color="0.35", ha="center", rotation=45)
    # 源云: 长轴骑直径, 极端 76.4 mm, 中部密 (示意 250 点)
    rng = np.random.default_rng(3)
    t_ = rng.uniform(-1, 1, 250)
    rad_off = rng.normal(0, 9, 250) * (1 - t_ ** 2) ** 0.5
    ang = rng.uniform(0, 2 * np.pi, 250)
    ax.scatter(76.4 * t_ * np.cos(np.deg2rad(8))
               + rad_off * np.cos(ang),
               76.4 * t_ * np.sin(np.deg2rad(8))
               + rad_off * np.sin(ang),
               s=2.5, color="tab:red", alpha=0.55, lw=0)
    ax.plot([-76.4, 76.4], [-76.4 * np.tan(np.deg2rad(8)),
                            76.4 * np.tan(np.deg2rad(8))],
            color="0.8", lw=0.5)
    th = np.deg2rad(55)
    ax.plot(90.62 * np.cos(th), 90.62 * np.sin(th), "o", ms=5,
            color="tab:blue")
    ax.plot([0, 90.62 * np.cos(th)], [0, 90.62 * np.sin(th)], "-",
            color="tab:blue", lw=1.0, alpha=0.6)
    ax.text(-92, -68, "SCALE×202（387 µm→78 mm）\n"
            "蝇脑云占满脑球：极端 0.98·R=76.4 mm\n"
            "→ 距离轴结构用尽，无放置自由度",
            fontsize=7, color="tab:red")
    ax.text(52, 66, "电极环 90.6", fontsize=7, color="tab:blue")
    ax.set_xlim(-95, 95)
    ax.set_ylim(-95, 95)
    ax.set_title("几何事实（修正）：源云已填满脑球\n"
                 "近全共模(cs17)发生在源充满球时——活跃子群+球导联场平滑",
                 fontsize=9)

    fig.suptitle("condstate22 解除条件地图——四轴 × 三态的封死/可解判定"
                 "（缺口 G=11.4/σ* = 5~54×）", fontsize=11)
    fig.savefig(OUT / "unseal_map.png", dpi=150)
    plt.close(fig)
    print(f"-> {OUT / 'unseal_map.png'}")
    print(f"-> {OUT / 'unseal_map_stats.json'}")


if __name__ == "__main__":
    main()
