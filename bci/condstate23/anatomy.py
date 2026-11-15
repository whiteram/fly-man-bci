"""bci/condstate23/anatomy.py — 活跃群体的头球解剖定位（零采集）。

condstate22b 的再解读"近全共模 = 活跃子群体积小 + 球形导联场
平滑"仍属推断。本弧把它变成实测数字：

  1. 载入连接组注释 + 神经元原生坐标（ffbm.data，与导出链同一
     数据源）；按云最大半径归一到头球坐标（极端 = 0.98·R_BRAIN
     = 76.4 mm，与 fit_scale_shift 一致）；
  2. 各记录组（ALPN/Kenyon_Cell/MBON/DAN）+ 参考类（visual/
     CX/olfactory）的：n、半径中位/p95、质心半径、角散布
     （resultant 长度）、空间 RMS 散布；
  3. 状态有效源云（按池率加权）：naive→visual（未记录外周的
     代表）、qlock→ALPN@20、hlock→ALPN@99、resid→
     ALPN99+KC17+MBON22 —— 有效散布 vs 实测共模占比
     （eegview CAR 方差占比 + cs17 信号模式共模占比）的排序
     对照：**源分布越宽，头皮模式越平（共模越高）**。

用法 (conda ffbm, repo root):
    python bci/condstate23/anatomy.py
输出: outputs/anatomy.png, outputs/anatomy_stats.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

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

R_EXTREME_MM = 76.4          # 0.98 · R_BRAIN (fit_scale_shift 目标)
GROUPS = ["ALPN", "Kenyon_Cell", "MBON", "DAN", "visual", "CX",
          "olfactory"]
ZH = {"ALPN": "触角叶 PN", "Kenyon_Cell": "蕈体 KC", "MBON": "MBON",
      "DAN": "DAN", "visual": "视觉", "CX": "中央复合体",
      "olfactory": "嗅觉 ORN"}
# 状态有效云: (组, 池率 Hz)——cs15/20/21 采集读数
STATE_SETS = {
    "naive(视觉底)": [("visual", 1.0)],
    "qlock": [("ALPN", 20.0)],
    "hlock": [("ALPN", 99.0)],
    "resid": [("ALPN", 99.0), ("Kenyon_Cell", 17.0),
              ("MBON", 22.0)],
}
# 实测共模 (eegview CAR 方差占比 / cs17 信号模式占比)
CM_MEASURED = {  # (variance-cm %, pattern-cm %)
    "naive(视觉底)": (95.1, None), "qlock": (None, None),
    "hlock": (87.0, 95.8), "resid": (99.7, 99.8),
}


def main():
    from ffbm import data as fdata
    ann = fdata.load_annotations()
    pos = fdata.neuron_positions(ann)          # {bodyId: xyz}
    pdf = pd.DataFrame.from_dict(pos, orient="index",
                                 columns=["x", "y", "z"])
    pdf.index.name = "bodyId"
    cls = ann.set_index("bodyId")["class"]
    pdf["cls"] = cls.reindex(pdf.index)

    rel = pdf[["x", "y", "z"]].to_numpy(float)
    rel = rel - rel.mean(0)                    # 云质心
    r = np.linalg.norm(rel, axis=1)
    scale = R_EXTREME_MM / r.max()             # mm per native unit
    pdf["r_mm"] = r * scale
    pdf[["sx", "sy", "sz"]] = rel * scale
    print(f"位置 {len(pdf)} 神经元; 云极端 → {R_EXTREME_MM} mm "
          f"(scale {scale:.1f} mm/unit ≈ ×{1/scale:.0f} native)")

    stats = {}
    for g in GROUPS:
        sub = pdf[pdf["cls"] == g]
        if not len(sub):
            continue
        P = sub[["sx", "sy", "sz"]].to_numpy()
        cen = P.mean(0)
        u = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        R = np.linalg.norm(u.mean(0))          # resultant 长度
        ang_spread = np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9))))
        stats[g] = dict(
            n=int(len(sub)),
            r_med=round(float(sub["r_mm"].median()), 1),
            r_p95=round(float(sub["r_mm"].quantile(0.95)), 1),
            cen_r=round(float(np.linalg.norm(cen)), 1),
            ang_spread_deg=round(float(ang_spread), 1),
            rms_spread=round(float(np.sqrt((
                (P - cen) ** 2).sum(1).mean())), 1),
        )
        print(f"{g:12s} n={stats[g]['n']:5d} r_med={stats[g]['r_med']:5.1f}"
              f" cen_r={stats[g]['cen_r']:5.1f} 角散布="
              f"{stats[g]['ang_spread_deg']:5.1f}° 空间RMS="
              f"{stats[g]['rms_spread']:5.1f} mm")

    # ---- 状态有效云 ---------------------------------------------------
    state_eff = {}
    for st, members in STATE_SETS.items():
        pts, w = [], []
        for g, rate in members:
            sub = pdf[pdf["cls"] == g]
            pts.append(sub[["sx", "sy", "sz"]].to_numpy())
            w.append(np.full(len(sub), rate))
        P = np.concatenate(pts)
        w = np.concatenate(w)
        cen = (P * w[:, None]).sum(0) / w.sum()
        state_eff[st] = dict(
            eff_cen_r=round(float(np.linalg.norm(cen)), 1),
            eff_spread=round(float(np.sqrt((w * ((P - cen) ** 2)
                                            .sum(1)).sum() / w.sum())), 1),
            cm_var=CM_MEASURED[st][0], cm_pat=CM_MEASURED[st][1],
        )
        print(f"{st:14s} 有效质心 r={state_eff[st]['eff_cen_r']:5.1f}"
              f" 有效散布={state_eff[st]['eff_spread']:5.1f} mm"
              f"  实测共模: 方差{state_eff[st]['cm_var']}%/"
              f"模式{state_eff[st]['cm_pat']}%")

    res = {"r_extreme_mm": R_EXTREME_MM, "groups": stats,
           "state_effective": state_eff,
           "note": "云归一: 极端=0.98·R_BRAIN (fit_scale_shift 目标);"
                   " naive 底的'visual'为代表代理(未记录外周含 AL 局部)"}

    # ---- 图 ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0),
                             constrained_layout=True)
    ax = axes[0]
    for g in GROUPS:
        sub = pdf[pdf["cls"] == g]
        if not len(sub):
            continue
        ax.scatter(sub["sx"], sub["sz"], s=2, alpha=0.35, label=ZH[g])
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(78 * np.cos(th), 78 * np.sin(th), color="0.5", lw=1.2)
    ax.plot(92 * np.cos(th), 92 * np.sin(th), color="0.75", lw=1.0)
    ax.set_xlabel("x (mm, 头球坐标)", fontsize=8)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.legend(fontsize=6, loc="upper left", markerscale=4)
    ax.set_aspect("equal")
    ax.set_title("各组在放大头球内的位置（x-z 投影，内圈=脑球）",
                 fontsize=9)

    ax = axes[1]
    ys = np.arange(len(GROUPS))[::-1]
    for y, g in zip(ys, GROUPS):
        if g not in stats:
            continue
        s = stats[g]
        ax.plot([s["cen_r"], s["cen_r"]], [y, y], "|", color="0.6",
                ms=14, mew=2)
        ax.errorbar(s["r_med"], y, xerr=[[s["r_med"]], [max(
            s["r_p95"] - s["r_med"], 0)]], fmt="o", ms=5, capsize=3,
            color="tab:blue")
        ax.text(80, y, f"角散{s['ang_spread_deg']:.0f}° RMS "
                f"{s['rms_spread']:.0f}mm", fontsize=6.5, va="center")
    ax.axvline(76.4, color="r", ls=":", lw=1.1)
    ax.text(77, -0.7, "云极端 76.4", fontsize=7, color="r")
    ax.set_yticks(ys)
    ax.set_yticklabels([ZH[g] for g in GROUPS], fontsize=8)
    ax.set_xlim(0, 130)
    ax.set_xlabel("头球半径 (mm)", fontsize=8)
    ax.set_title("各组径向分布（点=中位, 误差条→p95, |=质心）",
                 fontsize=9)
    ax.grid(True, axis="x", color="0.93")

    ax = axes[2]
    # 可靠两态连线 (hlock/resid); naive 的 visual 代理仅 28 个位置
    # 条目——空心弱代理标记, 不进趋势
    xs, ys = [], []
    for st in ("hlock", "resid"):
        s = state_eff[st]
        xs.append(s["eff_spread"])
        ys.append(s["cm_var"])
    ax.plot(xs, ys, "o-", ms=8, color="tab:blue")
    for st in ("hlock", "resid"):
        s = state_eff[st]
        ax.annotate(f"{st}\n散布{s['eff_spread']:.1f}mm→"
                    f"方差{s['cm_var']:.1f}%",
                    (s["eff_spread"], s["cm_var"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=7)
        ax.plot(s["eff_spread"], s["cm_pat"], "s", ms=8,
                color="tab:red", alpha=0.75)
        ax.annotate(f"模式{s['cm_pat']:.1f}%",
                    (s["eff_spread"], s["cm_pat"]),
                    textcoords="offset points", xytext=(8, -12),
                    fontsize=7, color="tab:red")
    s0 = state_eff["naive(视觉底)"]
    ax.plot(s0["eff_spread"], s0["cm_var"], "o", ms=9,
            mfc="none", mec="tab:blue", mew=1.4)
    ax.annotate("naive(弱代理: visual 类仅 28 条位置)",
                (s0["eff_spread"], s0["cm_var"]),
                textcoords="offset points", xytext=(10, 6), fontsize=7,
                color="0.45")
    ax.set_xlabel("状态有效源云散布 (mm)", fontsize=8)
    ax.set_ylabel("实测共模占比 (%)", fontsize=8)
    ax.set_ylim(84, 101)
    ax.set_title("源越宽 → 头皮模式越平（蓝=方差占比, 红=模式占比）\n"
                 "22b 再解读的定量版", fontsize=9)
    ax.grid(True, color="0.93")

    fig.suptitle("condstate23 活跃群体解剖定位——共模的解剖学基础"
                 f"（SCALE 归一, 云极端 {R_EXTREME_MM} mm）", fontsize=11)
    fig.savefig(OUT / "anatomy.png", dpi=150)
    plt.close(fig)
    (OUT / "anatomy_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'anatomy.png'}")
    print(f"-> {OUT / 'anatomy_stats.json'}")


if __name__ == "__main__":
    main()
