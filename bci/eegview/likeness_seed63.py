"""bci/eegview/likeness_seed63.py — 相似度审计的 seed-63 复算（零采集）。

likeness_audit 的"单种子（seed 62）"限制从未关闭（cs19 复核的是
监测器/带 RMS 数字，不是审计的 10 项指标）。本脚本在同一窗口约定
下对 seed 63 的三臂（naive/hl/res；ssvep 无 63 臂，剔除并注明）
复算审计指标并与 62 并排对照：

  预期：平稳长窗统计量跨种子收敛（cs19 已见带 RMS 逐位一致）；
  任何系统性偏差如实报告。

用法 (conda ffbm, repo root):
    python bci/eegview/likeness_seed63.py
输出: outputs/audit_seed63.json, outputs/audit_seed63.png
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

ARMS63 = [
    ("naive", ROOT / "bci/condstate3/outputs/ctl63_scalp.npy",
     7000, 15000),
    ("high-lock", ROOT / "bci/condstate4/outputs/hl63_scalp.npy",
     7000, 15000),
    ("res/latch", ROOT / "bci/condstate3/outputs/res63_scalp.npy",
     7000, 15000),
]
# 从 likeness_audit 复用全部度量（spec_fit/dfa/ac_1e/spike_fwhm/bands）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "la", ROOT / "bci/eegview/likeness_audit.py")
la = importlib.util.module_from_spec(spec)
spec.loader.exec_module(la)


def audit_arm(path, t0, t1):
    e = np.load(path).astype(np.float64) * 1e6
    x = la.device(e[t0:t1])
    cm = x.mean(axis=1)
    rms = np.sqrt((x ** 2).mean(axis=0))
    kur = float(np.median(((x - x.mean(0)) ** 4).mean(0) / rms ** 4 - 3))
    iu = np.triu_indices(x.shape[1], 1)
    cc = float(np.corrcoef(x.T)[iu].mean())
    f, P = la.welch(x, fs=250.0, nperseg=min(len(x), 2048), axis=0)
    pm = np.median(P, axis=1)
    expo, (lf, lp, line), peaks = la.spec_fit(f, pm)
    bp = la.band_pows(f, pm)
    dfa, _, _ = la.dfa_exponent(x[:, int(np.argmax(rms))])
    act = la.ac_1e_ms(cm)
    fwhm, nev = la.spike_fwhm_ms(cm)
    return dict(rms_uv=round(float(np.median(rms)), 4),
                kurtosis=round(kur, 2), expo=round(expo, 2),
                peak=(round(peaks[0][0], 1), round(peaks[0][1], 1))
                if peaks else None,
                icc=round(cc, 2), dfa=round(dfa, 2),
                ac1e=round(act, 1),
                fwhm=(round(fwhm, 1) if fwhm else None),
                rel_alpha=bp["alpha"])


def main():
    ref = json.loads((OUT / "audit_stats.json").read_text(
        encoding="utf-8"))
    res, table = {}, []
    for tag, path, t0, t1 in ARMS63:
        r63 = audit_arm(path, t0, t1)
        r62 = ref.get(tag, {})
        res[tag] = {"seed62": r62, "seed63": r63}
        print(f"{tag:10s} 62: expo={r62.get('exponent_1_40')} "
              f"kurt={r62.get('kurtosis')} icc={r62.get('interch_corr')} "
              f"| 63: expo={r63['expo']} kurt={r63['kurtosis']} "
              f"icc={r63['icc']}")
        table.append((tag, r62, r63))

    # 图: 关键指标 62 vs 63 成对条形
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9),
                             constrained_layout=True)
    tags = [t for t, _, _ in table]
    metrics = [("谱指数", lambda r: r.get("exponent_1_40", r.get("expo"))),
               ("峭度", lambda r: min(r.get("kurtosis", 0), 70)),
               ("导间相关", lambda r: r.get("interch_corr",
                                            r.get("icc"))),
               ("DFA", lambda r: r.get("dfa"))]
    for ax, (name, get) in zip(axes, metrics):
        v62 = [get(r62) for _, r62, _ in table]
        v63 = [get(r63) for _, _, r63 in table]
        x = np.arange(len(tags))
        ax.bar(x - 0.17, v62, 0.34, color="0.6", label="seed 62")
        ax.bar(x + 0.17, v63, 0.34, color="tab:blue", label="seed 63")
        ax.set_xticks(x)
        ax.set_xticklabels(tags, fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.grid(True, axis="y", color="0.93")
        ax.legend(fontsize=7)
    fig.suptitle("likeness audit 的 seed-63 复算——平稳统计量跨种子"
                 "收敛（ssvep 无 63 臂除外）", fontsize=11)
    fig.savefig(OUT / "audit_seed63.png", dpi=150)
    res["_note"] = "ssvep 臂无 seed-63 文件（condstate13 单种子），剔除"
    (OUT / "audit_seed63.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'audit_seed63.png'}")
    print(f"-> {OUT / 'audit_seed63.json'}")


if __name__ == "__main__":
    main()
