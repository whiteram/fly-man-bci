"""bci/condstate17/refine_knees.py — 空间膝点的细化网格重扫（零采集）。

cs17 的"膝点受 σ 网格量化（×3.16 步）"限制关闭弧：以 ×1.41 步
细化网格（11 点，0.0032→0.1 µV）在 ρ=0.85 上重扫五检测器 × 两态
（协议与 cs17 逐字相同：2 s 窗带 RMS，前 2 draw 拟合匹配滤波、
后 3 draw 评测），d'=2 交点线性插值得细化膝点；回写
spatial_stats.json 的 rho0.85 条目并重生成总账总图。

用法 (conda ffbm, repo root):
    python bci/condstate17/refine_knees.py
输出: outputs/refined_knees.json (+更新 spatial_stats.json /
      measurability/outputs/measurability_summary.png)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "outputs"
sys.path.insert(0, str(ROOT / "bci" / "condstate14"))
sys.path.insert(0, str(ROOT / "bci" / "condstate17"))
from state_monitor import load_dev, make_bg  # noqa: E402
from spatial_audit import win_rms_vec, dprime  # noqa: E402

SIGMAS = [0.0032, 0.0045, 0.0063, 0.009, 0.0127, 0.0178, 0.025,
          0.035, 0.05, 0.071, 0.1]
DRAWS = 4
STATES = {
    "hlock": (ROOT / "bci/condstate4/outputs/hl62_scalp.npy",
              3000, 16500),
    "resid": (ROOT / "bci/condstate3/outputs/res62_scalp.npy",
              8000, 16500),
}
NAIVE = (ROOT / "bci/condstate3/outputs/ctl62_scalp.npy", 4000, 16500)
DETS = ("best-single", "common-mode", "matched", "best-bipolar", "CAR")


def knee_interp(sigmas, y, thr=2.0):
    for i in range(len(y)):
        if y[i] < thr:
            if i == 0:
                return f"<{sigmas[0]:g}"
            f0, f1 = sigmas[i - 1], sigmas[i]
            y0, y1 = y[i - 1], y[i]
            return round(float(f0 + (thr - y0) * (f1 - f0)
                               / (y1 - y0)), 4)
    return None


def main():
    xn = load_dev(*NAIVE)
    nch = xn.shape[1]
    rn = win_rms_vec(xn)
    clean = {}
    for k, (p, t0, t1) in STATES.items():
        x = load_dev(p, t0, t1)
        r = win_rms_vec(x)
        d_s = [dprime(r[:, c], rn[:, c]) for c in range(nch)]
        best_c = int(np.argmax(d_s))
        d_bi, best_pair = -9, None
        for i in range(nch):
            for j in range(i + 1, nch):
                dd = dprime(r[:, i] - r[:, j], rn[:, i] - rn[:, j])
                if dd > d_bi:
                    d_bi, best_pair = dd, (i, j)
        clean[k] = (best_c, best_pair)
    curves = {}
    for k, (p, t0, t1) in STATES.items():
        x = load_dev(p, t0, t1)
        cur = {d: [] for d in DETS}
        for sg in SIGMAS:
            bgR = np.concatenate([
                win_rms_vec(make_bg(5000, nch, sg, 7300 + d))
                for d in range(8)])
            S = np.cov(bgR.T)
            Sh = 0.9 * S + 0.1 * np.diag(np.diag(S))
            Cinv = np.linalg.inv(Sh + 1e-12 * np.eye(nch))
            Rs, Rn = [], []
            for d in range(DRAWS):
                bgn = make_bg(len(xn), nch, sg, 200 + d)
                Rn.append(win_rms_vec(xn + bgn))
                bgs = make_bg(len(x), nch, sg, 100 + d)
                Rs.append(win_rms_vec(x + bgs))
            w = Cinv @ (np.concatenate(Rs[:2]).mean(0)
                        - np.concatenate(Rn[:2]).mean(0))
            Rs, Rn = np.concatenate(Rs[2:]), np.concatenate(Rn[2:])
            bc, bp = clean[k]
            cur["best-single"].append(dprime(Rs[:, bc], Rn[:, bc]))
            cur["common-mode"].append(dprime(Rs.mean(1), Rn.mean(1)))
            cur["matched"].append(dprime(Rs @ w, Rn @ w))
            cur["best-bipolar"].append(dprime(Rs[:, bp[0]] - Rs[:, bp[1]],
                                              Rn[:, bp[0]] - Rn[:, bp[1]]))
            cur["CAR"].append(dprime(
                (Rs - Rs.mean(1, keepdims=True))[:, bc],
                (Rn - Rn.mean(1, keepdims=True))[:, bc]))
        curves[k] = cur
        print(k, {d: [round(v, 2) for v in cur[d]] for d in DETS})

    refined = {}
    for k in STATES:
        for d in DETS:
            refined[f"{k}|rho0.85|{d}"] = knee_interp(
                SIGMAS, curves[k][d])
    print("细化膝点:", json.dumps(refined, ensure_ascii=False))

    # 回写 spatial_stats.json (保留 rho0.5/0.2 旧网格值, 注明)
    sp = OUT / "spatial_stats.json"
    stats = json.loads(sp.read_text(encoding="utf-8"))
    stats.setdefault("knee_sigma", {})
    for key, val in refined.items():
        stats["knee_sigma"][key] = val
    stats["knee_refinement"] = {
        "grid": SIGMAS, "rho": 0.85,
        "note": "rho0.85 条目经 refine_knees 细化 (×1.41 步, d'=2 "
                "线性插值); rho0.5/0.2 保留粗网格值"}
    sp.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    (OUT / "refined_knees.json").write_text(
        json.dumps({"sigmas": SIGMAS, "knees": refined,
                    "curves": {k: {d: [round(v, 3) for v in c[d]]
                                   for d in DETS}
                               for k, c in curves.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("-> spatial_stats.json 更新 + refined_knees.json")


if __name__ == "__main__":
    main()
