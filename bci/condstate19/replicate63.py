"""bci/condstate19/replicate63.py — 可测性闭环头条数字的跨种子复核算。

总账（docs/MEASURABILITY.md）的诚实边界之一：cs16/17/18 的关键
数字主要在 seed 62 上计算。本弧零采集在 seed 63（四臂全在盘上）
复算三类头条数字并逐项对照：

  1. 信号强度 s（态−幼稚带 RMS）与 σ*（τ_sig 内可读上限，
     cs16 公式, κ 在 naive63 上重标定）；
  2. 空间五检测器膝点（ρ=0.85, d'=2, cs17 协议）；
  3. 决策信道互信息曲线 I_dec(σ_bg)（cs14 的 seed62 折模型
     → seed63 臂评测, 天然跨种子方向, cs18 协议）。

判据：63 与 62 的数字差在网格/抽样不确定度内（预期 ≤3×）
即"复现"；任何系统性偏差如实报告。

用法 (conda ffbm, repo root):
    python bci/condstate19/replicate63.py
输出: outputs/replicate63.png, outputs/replicate63_stats.json
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
sys.path.insert(0, str(ROOT / "bci" / "condstate16"))
sys.path.insert(0, str(ROOT / "bci" / "condstate17"))
from state_monitor import (device, features, load_dev, fit_fold,  # noqa: E402
                           predict, build_xy, make_bg)
from budget_map import cm_rms_series, band_rms  # noqa: E402

ARMS63 = {
    "naive": (ROOT / "bci/condstate3/outputs/ctl63_scalp.npy",
              4000, 16500),
    "qlock": (ROOT / "bci/condstate2/outputs/lock63_scalp.npy",
              3000, 14000),
    "hlock": (ROOT / "bci/condstate4/outputs/hl63_scalp.npy",
              3000, 16500),
    "resid": (ROOT / "bci/condstate3/outputs/res63_scalp.npy",
              8000, 16500),
}
B, DPRIME, TAU_SIG, SIG_BG = 18.0, 2.0, 336.0, 11.4
SIGMAS = [0.0, 0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.16, 11.4]
DRAWS = 4
BB = (2.0, 20.0)


def mi_from_confusion(C):
    P = C / C.sum()
    p, q = P.sum(1), P.sum(0)
    m = P > 0
    return float((P[m] * np.log2(P[m] / (p[:, None] * q[None, :])[m])).sum())


def knee_of(sigmas, y, thr=2.0):
    for i, v in enumerate(y):
        if v < thr:
            if i == 0:
                return f"<{sigmas[0]:g}"
            return round(float(np.sqrt(sigmas[i - 1] * sigmas[i])), 4)
    return None


def main():
    # ---- 0. seed62 参照数字 (总账 json) ---------------------------------
    B16 = json.loads((ROOT / "bci/condstate16/outputs/budget_stats.json")
                     .read_text(encoding="utf-8"))
    B17 = json.loads((ROOT / "bci/condstate17/outputs/spatial_stats.json")
                     .read_text(encoding="utf-8"))
    B18 = json.loads((ROOT / "bci/condstate18/outputs/capacity_stats.json")
                     .read_text(encoding="utf-8"))

    x63 = {k: load_dev(p, t0, t1) for k, (p, t0, t1) in ARMS63.items()}
    nch = x63["naive"].shape[1]

    # ---- 1. s 与 σ* ------------------------------------------------------
    rms63 = {k: band_rms(v) for k, v in x63.items()}
    ser = {w: cm_rms_series(x63["naive"], w) for w in (1.0, 2.0, 4.0)}
    kap63 = float(ser[2.0].std() / ser[2.0].mean() * np.sqrt(B * 2.0))
    var_self = (kap63 * rms63["naive"]) ** 2
    s63 = {t: rms63[t] - rms63["naive"] for t in ("qlock", "hlock", "resid")}
    star63 = {t: float(np.sqrt(max(
        (s_ * np.sqrt(B * TAU_SIG) / DPRIME) ** 2 - var_self, 0)))
        for t, s_ in s63.items()}
    print(f"seed63: κ={kap63:.2f}  带RMS="
          f"{ {k: round(v, 5) for k, v in rms63.items()} }")
    print(f"seed63: s={ {k: round(v, 5) for k, v in s63.items()} }"
          f"  σ*={ {k: round(v, 4) for k, v in star63.items()} }")
    print(f"seed62: s={B16['s_uv']}  σ*={B16['sigma_star_uv']}")

    # ---- 2. 空间膝点 (ρ=0.85, 5 检测器) ---------------------------------
    from spatial_audit import win_rms_vec, dprime  # noqa: E402
    rec = {k: win_rms_vec(v) for k, v in x63.items()}
    mu = {k: v.mean(0) for k, v in rec.items()}
    clean = {}
    for k in ("hlock", "resid"):
        d_s = [dprime(rec[k][:, c], rec["naive"][:, c]) for c in range(nch)]
        bc = int(np.argmax(d_s))
        clean[k] = dict(best_single_ch=bc, d_single=float(d_s[bc]),
                        d_cm=dprime(rec[k].mean(1), rec["naive"].mean(1)))
    SP_SIG = np.array([0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.16, 11.4])
    DETS = ("best-single", "common-mode", "matched")
    knee63 = {}
    for k in ("hlock", "resid"):
        cur = {d: [] for d in DETS}
        for sg in SP_SIG:
            bgR = np.concatenate([
                win_rms_vec(make_bg(5000, nch, sg, 7000 + d))
                for d in range(8)])
            S = 0.9 * np.cov(bgR.T) + 0.1 * np.diag(np.diag(np.cov(bgR.T)))
            Cinv = np.linalg.inv(S + 1e-12 * np.eye(nch))
            Rs, Rn = [], []
            for d in range(DRAWS):
                for tgt, key in ((k, "s"), ("naive", "n")):
                    bg = make_bg(len(x63[tgt]), nch, sg,
                                 100 * d + int(tgt == k))
                    (Rs if key == "s" else Rn).append(
                        win_rms_vec(x63[tgt] + bg))
            w = Cinv @ (np.concatenate(Rs[:2]).mean(0)
                        - np.concatenate(Rn[:2]).mean(0))
            Rs, Rn = np.concatenate(Rs[2:]), np.concatenate(Rn[2:])
            bc = clean[k]["best_single_ch"]
            cur["best-single"].append(dprime(Rs[:, bc], Rn[:, bc]))
            cur["common-mode"].append(dprime(Rs.mean(1), Rn.mean(1)))
            cur["matched"].append(dprime(Rs @ w, Rn @ w))
        for d in DETS:
            knee63[f"{k}|{d}"] = knee_of(SP_SIG, cur[d])
        ref = {d: B17["knee_sigma"].get(f"{k}|rho0.85|{d}") for d in DETS}
        print(f"seed63 {k}: 膝点 "
              f"{ {d: knee63[f'{k}|{d}'] for d in DETS} }  (62: {ref})")

    # ---- 3. MI 曲线 (cs14 seed62 折 → 63 臂) -----------------------------
    from state_monitor import ARMS as ARMS14
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS14]
    Xc, yc, sc, _ = build_xy(records, 2.0, 1.0)
    fold62 = fit_fold(Xc[sc == 62], yc[sc == 62])
    rec63 = [("naive", 63, x63["naive"]), ("qlock", 63, x63["qlock"]),
             ("hlock", 63, x63["hlock"]), ("resid", 63, x63["resid"])]
    mi63 = {}
    for sg in SIGMAS:
        X, y = [], []
        for d in range(DRAWS):
            rec_d = [(cls, seed,
                      x + (make_bg(x.shape[0], x.shape[1], sg, 9000 + d)
                           if sg > 0 else 0.0)) for cls, seed, x in rec63]
            Xd, yd, _, _ = build_xy(rec_d, 2.0, 1.0)
            X.append(Xd), y.append(yd)
        X, y = np.concatenate(X), np.concatenate(y)
        yp = predict(fold62, X)
        C = np.zeros((4, 4), int)
        for t, p_ in zip(y, yp):
            C[t, p_] += 1
        mi63[sg] = round(mi_from_confusion(C), 4)
    print("seed63 MI:", mi63)
    print("seed62 MI:", B18["mi_decision"])
    above = [s for s in SIGMAS if mi63[s] >= 1.0]
    one_bit63 = max(above) if above else 0.0

    res = {
        "kappa_63": round(kap63, 2),
        "band_rms_uv_63": {k: round(v, 5) for k, v in rms63.items()},
        "s_uv_63": {k: round(v, 5) for k, v in s63.items()},
        "sigma_star_uv_63": {k: round(v, 4) for k, v in star63.items()},
        "spatial_knees_63": knee63,
        "mi_decision_63": {str(k): v for k, v in mi63.items()},
        "sigma_1bit_63": one_bit63,
        "ref_62": {"s": B16["s_uv"], "sigma_star": B16["sigma_star_uv"],
                   "knees": {f"{k}|{d}":
                             B17["knee_sigma"][f"{k}|rho0.85|{d}"]
                             for k in ("hlock", "resid")
                             for d in DETS},
                   "mi": B18["mi_decision"],
                   "sigma_1bit": B18["sigma_1bit_max_uv"]},
    }

    # ---- 图 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6),
                             constrained_layout=True)
    ax = axes[0]
    tags = ["qlock", "hlock", "resid"]
    xx = np.arange(3)
    ax.bar(xx - 0.18, [B16["s_uv"][t] * 1000 for t in tags], 0.34,
           color="0.6", label="seed 62")
    ax.bar(xx + 0.18, [s63[t] * 1000 for t in tags], 0.34,
           color="tab:blue", label="seed 63")
    ax.set_yscale("log")
    ax.set_xticks(xx)
    ax.set_xticklabels(tags, fontsize=9)
    ax.set_ylabel("s = 态−幼稚 带 RMS (nV)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("信号强度 s：62 vs 63", fontsize=9)
    ax.grid(True, axis="y", which="both", color="0.93")

    ax = axes[1]
    rows = []
    for k in ("hlock", "resid"):
        for d in DETS:
            v62 = res["ref_62"]["knees"][f"{k}|{d}"]
            v63 = knee63[f"{k}|{d}"]
            rows.append((f"{k}·{d}", v62, v63))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v62, v63) in zip(ys, rows):
        for v, c, mk, lb in ((v62, "0.6", "o", "62"), (v63, "tab:blue",
                                                       "s", "63")):
            if isinstance(v, (int, float)):
                ax.plot(v, y, mk, ms=7, color=c)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xscale("log")
    ax.axvline(11.4, color="k", ls=":", lw=1.2)
    ax.set_xlim(1e-3, 20)
    ax.set_xlabel("膝点背景 (µV)", fontsize=8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="0.6", marker="o", ls="",
                              label="seed 62"),
                       Line2D([], [], color="tab:blue", marker="s", ls="",
                              label="seed 63")], fontsize=8)
    ax.set_title("空间膝点：62 vs 63", fontsize=9)
    ax.grid(True, axis="x", which="both", color="0.93")

    ax = axes[2]
    ax.semilogx([max(s, 1e-4) for s in SIGMAS],
                [B18["mi_decision"][str(s)] for s in SIGMAS], "o-",
                color="0.6", label="seed 62 (cs18)")
    ax.semilogx([max(s, 1e-4) for s in SIGMAS],
                [mi63[s] for s in SIGMAS], "s--", color="tab:blue",
                label="seed 63 (复算)")
    ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
    ax.axvline(11.4, color="k", ls=":", lw=1.2)
    ax.set_xlabel("背景 2-20 Hz RMS (µV)", fontsize=8)
    ax.set_ylabel("I_dec (bit/2 s 窗)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title(f"决策信道 MI：62 vs 63（I≥1bit 上限 "
                 f"{B18['sigma_1bit_max_uv']:g} / {one_bit63:g} µV）",
                 fontsize=9)
    ax.grid(True, which="both", color="0.93")

    fig.suptitle("condstate19 跨种子复核算——可测性闭环头条数字 "
                 "seed 62 vs 63", fontsize=11)
    fig.savefig(OUT / "replicate63.png", dpi=150)
    plt.close(fig)
    (OUT / "replicate63_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'replicate63.png'}")
    print(f"-> {OUT / 'replicate63_stats.json'}")


if __name__ == "__main__":
    main()
