"""bci/condstate20/stability_resid.py — 残留态签名 300 s 漂移分析。

数据: acquire.py 的 resl62 (cs3resnt + x0.5@6s, 300 s, seed 62)。
1. 确定性复证: 前 22 s vs condstate3/resnt62 (同 chem+调度+种子);
2. 签名漂移: 池率 + 头皮 8 特征滑窗斜率 → τ_sig(resid)=3σ/联合速度;
3. σ*(resid) 修正: σ* = √((s√(Bτ)/2)² − (κσ_n)²) 用 τ_sig(resid)
   替换 hlock 的 336 s, 对照总账数字;
4. 机制: w_A 22 s vs 300 s + DAN 调制积分 (对比 hlock 的 5110 ms);
5. 监测器时间外推: condstate14 seed63 折逐窗打 resid 标签。

用法 (conda ffbm, repo root):
    python bci/condstate20/stability_resid.py
输出: outputs/stability_resid.png, outputs/stability_resid_stats.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "outputs"
sys.path.insert(0, str(ROOT / "bci" / "condstate14"))
from state_monitor import (ARMS, CLASSES, ZHCN, COLORS, device,  # noqa: E402
                           features, load_dev, fit_fold, predict,
                           build_xy)

W_S, HOP_S, T_START = 2.0, 1.0, 8.0
B, DPRIME = 18.0, 2.0
S_RESID, KAP, SIG_N = 0.01997, 0.95, 0.00104   # cs16 seed62 数字


def main():
    # ---- 0. 确定性复证 --------------------------------------------------
    x_raw = np.load(OUT / "resl62_scalp.npy").astype(np.float64)
    ref = np.load(ROOT / "bci/condstate3/outputs/resnt62_scalp.npy")
    n22 = min(22000, len(ref), len(x_raw))
    det = float(np.abs(x_raw[:n22] - ref[:n22].astype(np.float64)).max())
    print(f"确定性复证: max|resl62 - resnt62| 前 {n22} ms = {det:.2e} V")
    x = device(x_raw * 1e6)

    # ---- 1. 折与训练分布 -------------------------------------------------
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    Xtr, ytr, seeds, _ = build_xy(records, W_S, HOP_S)
    fold = fit_fold(Xtr[seeds == 63], ytr[seeds == 63])
    rk = np.array([CLASSES[i] for i in ytr]) == "resid"
    mu_r, sd_r = Xtr[rk].mean(0), Xtr[rk].std(0) + 1e-12
    names = ["lgRMS_δ", "lgRMS_bb", "lgRMS_hi", "峭度", "谱指数",
             "事件率", "峰dB", "峰Hz"]

    # ---- 2. 滑窗特征 + 漂移 ---------------------------------------------
    wlen, step = int(W_S * 250), int(HOP_S * 250)
    i0 = int(T_START * 250)
    X, tc = [], []
    for i in range(i0, len(x) - wlen + 1, step):
        X.append(features(x[i:i + wlen]))
        tc.append((i + wlen / 2) / 250.0)
    X, tc = np.array(X), np.array(tc)
    pred = predict(fold, X)
    zmax = np.abs((X - mu_r) / sd_r).max(1)
    ret = float((pred == 3).mean())

    tt = (tc - tc.mean()) / tc.std()
    slopes = {names[k]: round(float(np.polyfit(tt, X[:, k], 1)[0]
                                    / X[:, k].std()), 3)
              for k in range(X.shape[1])}
    v = float(np.sqrt(sum(s ** 2 for s in slopes.values()))) \
        / float(tc.std())
    tau_resid = 3.0 / v
    tau_hlock = 336.0
    var_self = (KAP * SIG_N) ** 2
    star_old = float(np.sqrt(max(
        (S_RESID * np.sqrt(B * tau_hlock) / DPRIME) ** 2 - var_self, 0)))
    star_new = float(np.sqrt(max(
        (S_RESID * np.sqrt(B * tau_resid) / DPRIME) ** 2 - var_self, 0)))

    # ---- 3. 池率 + 机制 --------------------------------------------------
    pop = np.load(OUT / "resl62_pop.npz")
    tc_ms = tc * 1000
    pop_s = {g: np.array([pop[g][int(t):int(t) + 2000].mean()
                          for t in tc_ms])
             for g in ("ALPN", "Kenyon_Cell", "MBON", "DAN")}
    st = np.load(OUT / "states" / "resl62.npz")
    w300 = float(st["w_KCM"][st["pre_type"] == "KCg-m"].mean())
    st22 = np.load(ROOT / "bci/condstate3/outputs/states/resnt62.npz")
    w22 = float(st22["w_KCM"][st22["pre_type"] == "KCg-m"].mean())
    tr = np.load(OUT / "resl62_dan.npy")
    mod_int = float(tr[:, 2].sum() * 0.5)

    res = {
        "det_diff_V": det, "n_windows": len(tc),
        "resid_retention": round(ret, 4),
        "false_drift": {CLASSES[i]: int((pred == i).sum())
                        for i in range(4) if i != 3 and (pred == i).any()},
        "feature_slopes": slopes,
        "v_sig_SD_per_s": round(v, 5),
        "tau_sig_resid_s": round(tau_resid, 0),
        "tau_sig_hlock_ref": tau_hlock,
        "w_A_22s": round(w22, 5), "w_A_300s": round(w300, 5),
        "dan_mod_integral_ms": round(mod_int, 1),
        "sigma_star_old_uv": round(star_old, 4),
        "sigma_star_new_uv": round(star_new, 4),
        "gap_to_11p4_old": round(11.4 / star_old, 1),
        "gap_to_11p4_new": round(11.4 / star_new, 1),
        "ALPN_first/mid/last": [round(float(pop_s["ALPN"][m].mean()), 2)
                                for m in (tc <= tc[0] + 60,
                                          (tc > 140) & (tc < 160),
                                          tc >= tc[-1] - 60)],
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # ---- 4. 图 (2x3) -----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4),
                             constrained_layout=True)
    ax = axes[0][0]
    for g, c in (("ALPN", "tab:red"), ("Kenyon_Cell", "tab:orange"),
                 ("MBON", "tab:purple")):
        ax.plot(tc, pop_s[g], lw=0.9, color=c, label=g)
    ax.axhspan(60, 140, color="tab:red", alpha=0.08)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("池率 (Hz)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("池率轨迹（2 s 窗；红带=残留签名带）", fontsize=9)

    ax = axes[0][1]
    ax.plot(tc, X[:, 1], lw=0.8, color="tab:blue")
    ax.axhspan(mu_r[1] - 2 * sd_r[1], mu_r[1] + 2 * sd_r[1],
               color="tab:blue", alpha=0.15,
               label="训练期 resid ±2σ (8-17 s)")
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("lgRMS 2-20 Hz", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("头皮宽带 RMS 轨迹", fontsize=9)

    ax = axes[0][2]
    ax.plot(tc, X[:, 3], lw=0.8, color="tab:green", label="峭度")
    ax.axhspan(mu_r[3] - 2 * sd_r[3], mu_r[3] + 2 * sd_r[3],
               color="tab:green", alpha=0.12)
    ax.set_ylim(0, np.max(X[:, 3]) * 1.12)
    ax2 = ax.twinx()
    ax2.plot(tc, X[:, 5], lw=0.8, color="tab:brown", label="事件率")
    ax2.axhspan(mu_r[5] - 2 * sd_r[5], mu_r[5] + 2 * sd_r[5],
                color="tab:brown", alpha=0.12)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("峭度", fontsize=8, color="tab:green")
    ax2.set_ylabel(">4σ 事件/s", fontsize=8, color="tab:brown")
    ax.set_title("形态特征轨迹（带=训练期 ±2σ）", fontsize=9)

    cm = x.mean(1)
    rfreq, rtimes = [], []
    for t0 in np.arange(10.0, 297.0, 10.0):
        a = int(t0 * 250)
        f, P = welch(cm[a:a + 2500], fs=250.0, nperseg=500)
        m = (f >= 5) & (f <= 40)
        rfreq.append(f[m][np.argmax(P[m])])
        rtimes.append(t0 + 5.0)
    ax = axes[1][0]
    ax.plot(rtimes, rfreq, "o-", ms=3, color="tab:purple")
    ax.set_ylim(0, 45)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("主节律 (Hz)", fontsize=8)
    ax.set_title(f"共模主节律 30 s 轨迹 (σ={np.std(rfreq):.1f} Hz)",
                 fontsize=9)

    ax = axes[1][1]
    for p_, t0, t1 in zip(pred, tc - HOP_S / 2, tc + HOP_S / 2):
        ax.axvspan(t0, t1, ymin=0.02, ymax=0.98,
                   color=COLORS[CLASSES[p_]], alpha=0.85, lw=0)
    ax.set_yticks(range(4))
    ax.set_yticklabels([ZHCN[c] for c in CLASSES], fontsize=8)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_title(f"condstate14 监测器（seed63 折）——resid 保持 "
                 f"{ret:.0%}", fontsize=9)

    ax = axes[1][2]
    ax.plot(tc, zmax, lw=0.8, color="tab:red")
    ax.axhline(3.0, color="k", ls="--", lw=1)
    ax.text(150, 3.3, "3σ（训练期 resid 分布）", fontsize=7)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("max |z|（8 特征）", fontsize=8)
    ax.set_title(f"相对训练分布最大偏离（τ_sig(resid)="
                 f"{tau_resid:.0f} s vs hlock 336 s）", fontsize=9)

    fig.suptitle("condstate20 残留态签名 300 s 漂移（cs3resnt+x0.5@6s, "
                 f"seed 62；确定性差 {det:.1e} V）——σ*(resid) "
                 f"{star_old:.3f}→{star_new:.3f} µV（τ 修正）", fontsize=11)
    fig.savefig(OUT / "stability_resid.png", dpi=150)
    plt.close(fig)
    (OUT / "stability_resid_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'stability_resid.png'}")
    print(f"-> {OUT / 'stability_resid_stats.json'}")


if __name__ == "__main__":
    main()
