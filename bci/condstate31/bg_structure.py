"""bci/condstate31/bg_structure.py — 距离衰减空间结构背景下的检测（零采集）。

总账限制清单第一条"背景为合成模型（无距离结构；ρ 扫描已括住）"
的关闭弧：把全局共享度 ρ 升级为**距离衰减协方差**
C_ij = exp(−d_ij/λ)（真实人脑背景相关随导联间距离衰减），
检验 cs17 的膝点/缺口结论在结构真实背景下是否移动：

  1. 生成：17 维空间高斯（Cholesky of C_unit）→ 逐通道 1/f² 时间
     形状 → 带内 RMS 归一 σ（与既有 make_bg 同口径）；
  2. 检测器：最佳单导/共模/匹配滤波（**全协方差** Ĉ 由 bg draws
     估计——匹配滤波现在能利用距离结构）；
  3. 扫描：λ ∈ {8, 40, 80} mm × σ 网格（×1.41 步），hlock/resid
     vs naive（seed 62，协议同 cs17/17b）；
  4. 对照：全局 ρ 参照（cs17b 细化膝点 0.0126/0.0404 µV）。

预期：距离衰减膝点落在全局 ρ=0.2~0.85 曲线之间（已被括住）；
缺口 280-900× 不动。

用法 (conda ffbm, repo root):
    python bci/condstate31/bg_structure.py
输出: outputs/bg_structure.png, outputs/bg_structure_stats.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

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
sys.path.insert(0, str(ROOT / "bci" / "condstate17"))
from state_monitor import load_dev  # noqa: E402
from spatial_audit import win_rms_vec, dprime  # noqa: E402

FS = 250.0
SIGMAS = [0.0032, 0.0045, 0.0063, 0.009, 0.0127, 0.0178, 0.025,
          0.035, 0.05, 0.071, 0.1]
LAMBDAS = (8.0, 40.0, 80.0)          # mm
DRAWS = 4
STATES = {
    "hlock": ("bci/condstate4/outputs/hl62_scalp.npy", 3000, 16500,
              "tab:green", 0.0126),
    "resid": ("bci/condstate3/outputs/res62_scalp.npy", 8000, 16500,
              "tab:red", 0.0404),
}
NAIVE = ("bci/condstate3/outputs/ctl62_scalp.npy", 4000, 16500)
DETS = ("best-single", "common-mode", "matched")
REF_KNEE_GLOBAL_RHO = {"hlock": 0.0126, "resid": 0.0404}  # cs17b


def elec_chord_mm():
    d = json.loads((ROOT / "viz_live/data/viz_data.json")
                   .read_text(encoding="utf-8"))
    D = np.array(d["meta"]["scalp"]["elec_dirs"])
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(D @ D.T, -1, 1)))
    return 2 * 92.0 * np.sin(np.radians(ang) / 2)      # mm


def make_bg_dist(n, ch, sigma, lam_mm, dists, seed):
    """距离衰减空间协方差 + 1/f² 时间形状的背景."""
    rng = np.random.default_rng(seed)
    C = np.exp(-dists / lam_mm)
    C += 1e-6 * np.eye(ch)
    L = np.linalg.cholesky(C)
    Z = rng.standard_normal((n, ch)) @ L.T
    fr = np.fft.rfftfreq(n, 1 / FS)
    shape = np.where((fr >= 0.4) & (fr <= 45), 1.0 / (fr + 0.5) ** 2, 0.0)
    shape[0] = 0.0
    sos = butter(4, [2.0, 20.0], btype="bandpass", fs=FS, output="sos")
    bg = np.zeros((n, ch))
    for c in range(ch):
        y = np.fft.irfft(np.fft.rfft(Z[:, c]) * shape, n)
        y *= sigma / (np.sqrt((sosfiltfilt(sos, y) ** 2).mean()) + 1e-30)
        bg[:, c] = y
    return bg


def knee_interp(y, thr=2.0):
    for i in range(len(y)):
        if y[i] < thr:
            if i == 0:
                return f"<{SIGMAS[0]:g}"
            f0, f1 = SIGMAS[i - 1], SIGMAS[i]
            y0, y1 = y[i - 1], y[i]
            return round(float(f0 + (thr - y0) * (f1 - f0) / (y1 - y0)),
                         4)
    return None


def main():
    dists = elec_chord_mm()
    xn = load_dev(ROOT / NAIVE[0], NAIVE[1], NAIVE[2])
    nch = xn.shape[1]
    rn = win_rms_vec(xn)
    rh = win_rms_vec(load_dev(*STATES["hlock"][:3]))
    best_c = int(np.argmax([dprime(rh[:, c], rn[:, c])
                            for c in range(nch)]))
    res = {"lambdas_mm": LAMBDAS, "knees": {}, "curves": {}}
    for st, (p, t0, t1, col, ref) in STATES.items():
        x = load_dev(ROOT / p, t0, t1)
        for lam in LAMBDAS:
            cur = {d: [] for d in DETS}
            for sg in SIGMAS:
                bgR = np.concatenate([
                    win_rms_vec(make_bg_dist(5000, nch, sg, lam, dists,
                                             7400 + d))
                    for d in range(8)])
                S = np.cov(bgR.T)
                Sh = 0.9 * S + 0.1 * np.diag(np.diag(S))
                Cinv = np.linalg.inv(Sh + 1e-12 * np.eye(nch))
                Rs, Rn = [], []
                for d in range(DRAWS):
                    Rn.append(win_rms_vec(
                        xn + make_bg_dist(len(xn), nch, sg, lam, dists,
                                          300 + d)))
                    Rs.append(win_rms_vec(
                        x + make_bg_dist(len(x), nch, sg, lam, dists,
                                         400 + d)))
                w = Cinv @ (np.concatenate(Rs[:2]).mean(0)
                            - np.concatenate(Rn[:2]).mean(0))
                Rs, Rn = np.concatenate(Rs[2:]), np.concatenate(Rn[2:])
                cur["best-single"].append(
                    dprime(Rs[:, best_c], Rn[:, best_c]))
                cur["common-mode"].append(
                    dprime(Rs.mean(1), Rn.mean(1)))
                cur["matched"].append(dprime(Rs @ w, Rn @ w))
            res["curves"][f"{st}|lam{lam:g}"] = {
                d: [round(v, 3) for v in cur[d]] for d in DETS}
            for d in DETS:
                res["knees"][f"{st}|lam{lam:g}|{d}"] = \
                    knee_interp(cur[d])
        print(st, {k: v for k, v in res["knees"].items()
                   if k.startswith(st) and "matched" in k})

    # 图: 匹配滤波膝点 vs λ, 全局 ρ 参照线
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                             constrained_layout=True)
    ax = axes[0]
    for st, (p, t0, t1, col, ref) in STATES.items():
        ks = []
        for lam in LAMBDAS:
            k = res["knees"][f"{st}|lam{lam:g}|matched"]
            ks.append(k if isinstance(k, float) else 0.003)
        ax.plot(LAMBDAS, ks, "o-", color=col, label=st)
        ax.axhline(ref, color=col, ls=":", lw=1.1, alpha=0.7)
        ax.annotate(f"全局ρ 参照 {ref}", (LAMBDAS[0], ref),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=7, color=col)
        ax.annotate(f"缺口 {11.4/ (ks[-1] if isinstance(ks[-1],float) else ref):.0f}×",
                    (LAMBDAS[-1], ks[-1]), textcoords="offset points",
                    xytext=(-8, 8), fontsize=7, color=col, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(LAMBDAS)
    ax.set_xticklabels([f"{l:g}mm" for l in LAMBDAS], fontsize=8)
    ax.set_xlabel("相关长度 λ (d_ij 弦距 0-43 mm)", fontsize=8)
    ax.set_ylabel("匹配滤波膝点 (µV)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("距离衰减背景下的膝点", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[1]
    for st, (p, t0, t1, col, ref) in STATES.items():
        for lam, mk in zip(LAMBDAS, "osd"):
            cv = res["curves"][f"{st}|lam{lam:g}"]["matched"]
            ax.semilogx([max(s, 5e-4) for s in SIGMAS], cv, mk + "-",
                        ms=4, lw=0.9, color=col,
                        label=f"{st} λ={lam:g}" if mk == "o" else None)
        y62 = res["curves"][f"{st}|lam{LAMBDAS[1]:g}"]["matched"]
    ax.axhline(2.0, color="k", ls="--", lw=0.8)
    ax.axvline(11.4, color="k", ls=":", lw=1.1)
    ax.set_xlabel("背景 2-20 Hz RMS (µV)", fontsize=8)
    ax.set_ylabel("d' (matched, 2 s 窗)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("匹配滤波 d' 曲线（λ=8/40/80 重合于带内）", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    fig.suptitle("condstate31 距离衰减空间结构背景——限制清单第一条的"
                 "实测关闭", fontsize=11)
    fig.savefig(OUT / "bg_structure.png", dpi=150)
    plt.close(fig)
    (OUT / "bg_structure_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("膝点表:", json.dumps(res["knees"], ensure_ascii=False))
    print(f"-> {OUT / 'bg_structure.png'}")


if __name__ == "__main__":
    main()
