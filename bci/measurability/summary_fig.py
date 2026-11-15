"""bci/measurability/summary_fig.py — 可测性边界总账图（六弧头条数字合成一张图）。

数据源（全部为已完成弧的 outputs/*.json，不重算）:
  condstate16 budget_stats.json  — 信号强度 s, σ*, W_req 公式参数
  condstate17 spatial_stats.json — 五类空间检测器膝点
  condstate18 capacity_stats.json— 决策信道互信息曲线
  condstate12/14/15 README 数字 — 事件账/监测器膝点/签名寿命（引注）

面板:
  A 幅值阶梯——所有"可读阈值"聚在 0.003-0.8 µV，不可约参考在 11.4 µV
  B 积分-漂移不相容——W_req 曲线族 vs τ_sig 水平线
  C 空间维度膝点——五类检测器（对数轴）vs 11.4 µV
  D 信道容量——I_dec vs σ_bg（0.1 µV 归零）

用法 (conda ffbm, repo root):
    python bci/measurability/summary_fig.py
输出: bci/measurability/outputs/measurability_summary.png
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
OUT.mkdir(exist_ok=True)
B16 = json.loads((ROOT / "bci/condstate16/outputs/budget_stats.json")
                 .read_text(encoding="utf-8"))
B17 = json.loads((ROOT / "bci/condstate17/outputs/spatial_stats.json")
                 .read_text(encoding="utf-8"))
B18 = json.loads((ROOT / "bci/condstate18/outputs/capacity_stats.json")
                 .read_text(encoding="utf-8"))
# cs20/21 修正: 各态实测 τ_sig 与修正后 σ* (cs16 json 保留历史值)
B20 = json.loads((ROOT / "bci/condstate20/outputs"
                  / "stability_resid_stats.json").read_text(
                      encoding="utf-8"))
B21 = json.loads((ROOT / "bci/condstate21/outputs"
                  / "stability_qlock_stats.json").read_text(
                      encoding="utf-8"))
TAU = {"qlock": B21["tau_sig_qlock_s"], "hlock": 336.0,
       "resid": B20["tau_sig_resid_s"]}
STAR = {"qlock": round(B21["sigma_star_new_uv"], 2), "hlock":
        B16["sigma_star_uv"]["hlock"], "resid":
        round(B20["sigma_star_new_uv"], 2)}

TAU_SIG = B16["tau_sig_s"]   # 历史值 336 (hlock); 各态修正见 TAU
B_BW, DPRIME = 18.0, 2.0
SIG_BG = 11.4


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4),
                             constrained_layout=True)

    # ---- A. 幅值阶梯 -------------------------------------------------
    rows = [  # (标签, µV, 颜色组)
        ("naive 静息带RMS", B16["band_rms_uv"]["naive"], "sig"),
        ("qlock 静默锁", B16["band_rms_uv"]["qlock"], "sig"),
        ("hlock 高锁", B16["band_rms_uv"]["hlock"], "sig"),
        ("resid 残留/闩锁 (最强)", B16["band_rms_uv"]["resid"], "sig"),
        ("I≥1bit 背景上限 (cs18)", B18["sigma_1bit_max_uv"], "knee"),
        ("监测器膝点 2s (cs14)", 0.005, "knee"),
        ("空间最优膝点 hlock (cs17)", 0.018, "knee"),
        ("空间最优膝点 resid (cs17)", 0.057, "knee"),
        ("σ* qlock (cs16+21 τ修正)", STAR["qlock"], "knee"),
        ("σ* hlock (cs16)", STAR["hlock"], "knee"),
        ("σ* resid (cs16+20 τ修正)", STAR["resid"], "knee"),
        ("实验室级放大器噪声", 0.1, "ref"),
        ("项目人脑背景模型", 11.4, "ref"),
    ]
    cmap = {"sig": "tab:blue", "knee": "tab:red", "ref": "0.45"}
    ax = axes[0][0]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, g) in zip(ys, rows):
        ax.plot([1e-4, v], [y, y], "-", color=cmap[g], lw=1.2, alpha=0.6)
        ax.plot(v, y, "o", ms=7, color=cmap[g])
        ax.text(v * 1.15, y, f" {v:g}", va="center", fontsize=7,
                color=cmap[g])
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 40)
    ax.axvline(SIG_BG, color="k", ls=":", lw=1.2)
    ax.set_xlabel("µV (2-20 Hz RMS)", fontsize=8)
    ax.set_title("幅值阶梯：全部可读阈值聚在 0.003-2.3 µV；\n"
                 "不可约参考（人脑背景 11.4）差 5-3600×", fontsize=9)
    ax.grid(True, axis="x", which="both", color="0.93")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=cmap[g], marker="o", ls="-",
                              label=t) for g, t in
                       (("sig", "信号（态-态）"), ("knee", "可读阈值"),
                        ("ref", "参考电平"))], fontsize=7, loc="upper right")

    # ---- B. 积分-漂移不相容 ------------------------------------------
    ax = axes[0][1]
    sigmas = np.logspace(-3.3, 1.3, 90)
    var_self = (B16["kappa_2s"] * B16["band_rms_uv"]["naive"]) ** 2
    colors = {"qlock": "tab:cyan", "hlock": "tab:green", "resid": "tab:red"}
    for tag, c in colors.items():
        s_ = B16["s_uv"][tag]
        ax.loglog(sigmas, (DPRIME / s_) ** 2 * (var_self + sigmas ** 2)
                  / B_BW, color=c, label=f"{tag}: s={s_*1000:.0f} nV")
    ax.plot(sigmas, 2.0 * (sigmas / 0.005) ** 2, "--", color="tab:purple",
            label="监测器 √W 外推 (cs14 锚点)")
    # 各态实测寿命线 + 曲线与其寿命交点 (=修正后 σ*, cs20/21)
    for tag, c in colors.items():
        ax.axhline(TAU[tag], color=c, ls=":", lw=0.9, alpha=0.7)
        ax.plot(STAR[tag], TAU[tag], "*", ms=11, color=c, zorder=5)
    ax.text(0.0015, 336 * 1.5, "τ(hlock)=336s", fontsize=7,
            color="tab:green")
    ax.text(0.0015, 2988 * 1.5, "τ(qlock)~2988s / τ(resid)~2948s "
            "(外推下限)", fontsize=7, color="0.35")
    ax.axvline(SIG_BG, color="k", ls=":", lw=1.2)
    ax.set_ylim(1e-2, 1e9)
    ax.set_xlabel("背景 2-20 Hz RMS (µV)", fontsize=8)
    ax.set_ylabel("d'=2 所需积分 (s)", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("积分-漂移不相容 (cs16+20/21)：11.4 µV 需 18.7~2289 h\n"
                 "vs 各态实测寿命（★=曲线∩寿命=修正 σ*）", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    # ---- C. 空间膝点 ---------------------------------------------------
    ax = axes[1][0]
    det_zh = {"best-single": "最佳单导", "common-mode": "共模均值",
              "matched": "匹配滤波", "best-bipolar": "最佳双极",
              "CAR": "CAR"}
    rows = []
    for det in det_zh:
        for st in ("hlock", "resid"):
            k = B17["knee_sigma"].get(f"{st}|rho0.85|{det}")
            if isinstance(k, (int, float)):
                rows.append((f"{det_zh[det]} · {st}", k,
                             colors[st] if st in colors else "tab:red"))
    rows.sort(key=lambda r: r[1])
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, c) in zip(ys, rows):
        ax.plot([1e-4, v], [y, y], "-", color=c, lw=1.2, alpha=0.6)
        ax.plot(v, y, "o", ms=7, color=c)
        ax.text(v * 1.15, y, f" {v:g}", va="center", fontsize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 40)
    ax.axvline(SIG_BG, color="k", ls=":", lw=1.2)
    ax.set_xlabel("膝点背景 (µV)", fontsize=8)
    ax.set_title(f"空间维度膝点 (cs17)：双极/CAR 低于首测 σ；\n"
                 f"最优 {max(r[1] for r in rows):g} µV，差 "
                 f"{SIG_BG / max(r[1] for r in rows):.0f}×", fontsize=9)
    ax.grid(True, axis="x", which="both", color="0.93")

    # ---- D. 信道容量 ----------------------------------------------------
    ax = axes[1][1]
    sg = [max(float(s), 1e-4) for s in B18["sigmas"]]
    mi = [B18["mi_decision"][str(s)] for s in B18["sigmas"]]
    ax.semilogx(sg, mi, "o-", color="tab:blue")
    ax.axhline(2.0, color="0.6", ls=":", lw=1)
    ax.text(1.3e-4, 2.03, "4 态上限 2 bit", fontsize=7, color="0.4")
    ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
    ax.axvline(SIG_BG, color="k", ls=":", lw=1.2)
    ax.axhspan(-0.1, 0.08, color="tab:red", alpha=0.12)
    ax.text(1.3e-4, -0.09, "I = 0（0.1 µV 起精确归零）", fontsize=7,
            color="tab:red")
    ax.set_xlabel("背景 2-20 Hz RMS (µV)", fontsize=8)
    ax.set_ylabel("I(S; X) (bit / 2 s 窗)", fontsize=8)
    ax.set_ylim(-0.12, 2.1)
    ax.set_title("信道容量 (cs18)：干净 1.75 bit/窗（293 bit/τ_sig）；\n"
                 "11.4 µV 下 I=0.000 bit", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    fig.suptitle("变蝇人 可测性边界总账——闭环与修正（事件 cs12 / 状态 cs14 / "
                 "寿命 cs15+20+21 / 积分 cs16 / 空间 cs17 / 容量 cs18；"
                 "解除条件 cs22 / 解剖 cs23）", fontsize=12)
    fig.savefig(OUT / "measurability_summary.png", dpi=150)
    plt.close(fig)
    print(f"-> {OUT / 'measurability_summary.png'}")


if __name__ == "__main__":
    main()
