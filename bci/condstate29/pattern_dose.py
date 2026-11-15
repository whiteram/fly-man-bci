"""bci/condstate29/pattern_dose.py — 图案尺度剂量响应（环境误差棒的稳健性）。

cs28 用单一图案（12 px 棋盘）给出 s 参照的环境误差棒 ±23%。本弧
补两个尺度剂量点（6 px 细 / 24 px 粗，同 50% 占空比），检验：
  1. ±23% 是饱和值还是随图案尺度缩放？
  2. 细图案（接近小网膜采样尺度）是否被混叠衰减（R 驱动下降）？
     ——环境误差棒的上确界在哪里。

协议与 cs25/27/28 完全相同（无 chem、seed 62、20 s、层速率+群速率）。
四条件对照：dark（cs25）/ c6 / c12（cs28）/ c24。

用法 (conda ffbm, repo root):
    python bci/condstate29/pattern_dose.py    # 2 跑 ~5.2 min GPU, 幂等
输出: outputs/dose.json, dose.png, c6_*/c24_*.npy
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,Kenyon_Cell,MBON,DAN,CX"
TYPES = "lLN1_bc"
LAYERS = ("R", "L", "MID", "T4", "T5")
T_END = 20000.0
W0, W1 = 2000, 19000
CONDS = {"c6": "checker_c6", "c24": "checker_c24"}


def run(tag, vis):
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{tag}_layers.npz").exists():
        print(f"[acquire] {tag}: exists, skip", flush=True)
        return
    trial = out / f"_trial_{tag}"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", vis,
           "--pop-rate", GROUPS, "--pop-rate-type", TYPES,
           "--t-end", str(T_END), "--seed", "62",
           "--out", str(trial), "--gpu"]
    print(f"[acquire] {tag}: {vis}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"export failed for {tag}")
    shutil.copy(trial / "_debug_phi_scalp.npy", out / f"{tag}_scalp.npy")
    vd = json.loads((trial / "viz_data.json").read_text(encoding="utf-8"))
    np.savez(out / f"{tag}_layers.npz",
             **{k: np.asarray(v, dtype=np.float32)
                for k, v in vd["rates"].items() if k in LAYERS})
    time.sleep(1.0)
    shutil.rmtree(trial, ignore_errors=True)
    print(f"[acquire] {tag}: done {(time.time()-t0)/60:.1f} min", flush=True)


def metrics(scalp_path, layers_path):
    sys.path.insert(0, str(ROOT / "bci" / "condstate14"))
    from state_monitor import device
    sys.path.insert(0, str(ROOT / "bci" / "eegview"))
    from likeness_audit import spec_fit
    from scipy.signal import welch
    e = np.load(scalp_path).astype(np.float64)[W0:W1] * 1e6
    cm = device(e).mean(axis=1)

    def brms(lo, hi):
        f, P = welch(cm, fs=250.0, nperseg=min(len(cm), 512))
        m = (f >= lo) & (f < hi)
        return float(np.sqrt(np.trapezoid(P[m], f[m])))
    f, P = welch(cm, fs=250.0, nperseg=min(len(cm), 2048))
    m = (f >= 1) & (f <= 40)
    expo, _, peaks = spec_fit(np.hstack([[0.0], f[m]]),
                              np.hstack([[1e-30], P[m]]))
    lay = np.load(layers_path)
    return dict(bb=round(brms(2, 20), 5), delta=round(brms(1, 4), 5),
                expo=round(expo, 2),
                peak=round(peaks[0][0], 1) if peaks else None,
                R=round(float(lay["R"][W0:W1].mean()), 1),
                L=round(float(lay["L"][W0:W1].mean()), 1),
                MID=round(float(lay["MID"][W0:W1].mean()), 2))


def main():
    for tag, vis in CONDS.items():
        run(tag, vis)
    # 四条件对照
    rows = {"dark": metrics(ROOT / "bci/condstate25/outputs/floor_scalp.npy",
                            ROOT / "bci/condstate25/outputs/floor_layers.npz"),
            "c12": metrics(ROOT / "bci/condstate28/outputs/patternfloor_scalp.npy",
                           ROOT / "bci/condstate28/outputs/patternfloor_layers.npz")}
    for tag in CONDS:
        rows[tag] = metrics(HERE / "outputs" / f"{tag}_scalp.npy",
                            HERE / "outputs" / f"{tag}_layers.npz")
    res = {"conditions_px": {"dark": None, "c6": 6, "c12": 12, "c24": 24},
           "metrics": rows,
           "bb_ratios_vs_dark": {k: round(v["bb"] / rows["dark"]["bb"], 2)
                                 for k, v in rows.items()}}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (HERE / "outputs" / "dose.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    order = ["dark", "c6", "c12", "c24"]
    lab = {"dark": "暗", "c6": "6px", "c12": "12px", "c24": "24px"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2),
                             constrained_layout=True)
    ax = axes[0]
    ax.plot([0, 6, 12, 24], [rows[k]["R"] for k in order], "o-",
            color="tab:red")
    ax.set_xlabel("棋盘格尺寸 (px; 0=暗)", fontsize=8)
    ax.set_ylabel("R 感光器率 (Hz)", fontsize=8)
    ax.set_title("R 驱动 vs 图案尺度", fontsize=9)
    ax.grid(True, color="0.93")
    ax = axes[1]
    ax.plot([0, 6, 12, 24], [rows[k]["bb"] * 1000 for k in order], "o-",
            color="tab:blue")
    for x, k in zip((0, 6, 12, 24), order):
        ax.annotate(f"×{res['bb_ratios_vs_dark'][k]:g}", (x, rows[k]["bb"]*1000),
                    textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xlabel("棋盘格尺寸 (px; 0=暗)", fontsize=8)
    ax.set_ylabel("底 bb RMS (nV)", fontsize=8)
    ax.set_title("底幅度 vs 图案尺度（标注=×暗）", fontsize=9)
    ax.grid(True, color="0.93")
    ax = axes[2]
    for k, c in zip(order, ("0.5", "tab:orange", "tab:red", "tab:green")):
        ax.plot([0, 6, 12, 24][order.index(k)], rows[k]["peak"] or 0, "o",
                ms=9, color=c, label=lab[k])
    ax.set_xlabel("棋盘格尺寸 (px; 0=暗)", fontsize=8)
    ax.set_ylabel("底主节律 (Hz)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("底节律 vs 图案尺度", fontsize=9)
    ax.grid(True, color="0.93")
    fig.suptitle("condstate29 图案尺度剂量——环境误差棒的稳健性（50% 占空, "
                 "无 chem 20 s, seed 62）", fontsize=10.5)
    fig.savefig(HERE / "outputs" / "dose.png", dpi=150)
    print("->", HERE / "outputs" / "dose.png")


if __name__ == "__main__":
    main()
