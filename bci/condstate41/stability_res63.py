"""bci/condstate30/stability63.py — hlock 300 s 寿命的 seed-63 复制分析。

数据: acquire.py 的 resl63 (cs3resnt, seed 63, 300 s)。
1. 确定性复证: 前 22 s vs condstate4/hlnt63 (同 chem 同种子);
2. τ_sig(63) = 3σ/8 特征联合漂移速度, 对照 62 的 336 s;
3. 机制: w_A 漂移 + DAN 调制积分 (62 参照: +0.0109 / 5110 ms);
4. 监测器 (cs14 seed62 折——跨种子方向) 保持率与失败模式。

用法 (conda ffbm, repo root):
    python bci/condstate30/stability63.py
输出: outputs/stability_res63.png, outputs/stability_res63_stats.json
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

W_S, HOP_S, T_START = 2.0, 1.0, 3.0
B, DPRIME = 18.0, 2.0
S_RESID, KAP, SIG_N = 0.01997, 0.95, 0.00104   # cs16 seed62
VAR_SELF = (KAP * SIG_N) ** 2
TAU_62 = 2948.0
W0REF = 0.03


def main():
    # ---- 0. 确定性复证 --------------------------------------------------
    x_raw = np.load(OUT / "resl63_scalp.npy").astype(np.float64)
    ref = np.load(ROOT / "bci/condstate3/outputs/resnt63_scalp.npy")
    n22 = min(22000, len(ref), len(x_raw))
    det = float(np.abs(x_raw[:n22] - ref[:n22].astype(np.float64)).max())
    print(f"确定性复证: max|resl63 - hlnt63| 前 {n22} ms = {det:.2e} V")
    x = device(x_raw * 1e6)

    # ---- 1. 折 (seed62 训练 → 63 评测, 跨种子方向) -----------------------
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    Xtr, ytr, seeds, _ = build_xy(records, W_S, HOP_S)
    fold = fit_fold(Xtr[seeds == 62], ytr[seeds == 62])
    rk = np.array([CLASSES[i] for i in ytr]) == "resid"
    mu_h, sd_h = Xtr[rk].mean(0), Xtr[rk].std(0) + 1e-12
    names = ["lgRMS_δ", "lgRMS_bb", "lgRMS_hi", "峭度", "谱指数",
             "事件率", "峰dB", "峰Hz"]

    # ---- 2. 滑窗特征 + 漂移 ---------------------------------------------
    wlen, step = int(W_S * 250), int(HOP_S * 250)
    X, tc = [], []
    for i in range(int(T_START * 250), len(x) - wlen + 1, step):
        X.append(features(x[i:i + wlen]))
        tc.append((i + wlen / 2) / 250.0)
    X, tc = np.array(X), np.array(tc)
    pred = predict(fold, X)
    zmax = np.abs((X - mu_h) / sd_h).max(1)
    ret = float((pred == 3).mean())

    tt = (tc - tc.mean()) / tc.std()
    slopes = {names[k]: round(float(np.polyfit(tt, X[:, k], 1)[0]
                                    / X[:, k].std()), 3)
              for k in range(X.shape[1])}
    v = float(np.sqrt(sum(s ** 2 for s in slopes.values()))) \
        / float(tc.std())
    tau63 = 3.0 / v

    # ---- 3. 机制 ---------------------------------------------------------
    pop = np.load(OUT / "resl63_pop.npz")
    pop_s = {g: np.array([pop[g][int(t * 1000):int(t * 1000) + 2000]
                          .mean() for t in tc])
             for g in ("ALPN", "Kenyon_Cell", "MBON", "DAN")}
    st = np.load(OUT / "states" / "resl63.npz")
    w300 = float(st["w_KCM"][st["pre_type"] == "KCg-m"].mean())
    st22 = np.load(ROOT / "bci/condstate3/outputs/states/resnt63.npz")
    w22 = float(st22["w_KCM"][st22["pre_type"] == "KCg-m"].mean())
    tr = np.load(OUT / "resl63_dan.npy")
    mod_int = float(tr[:, 2].sum() * 0.5)

    star62 = float(np.sqrt(max(
        (S_RESID * np.sqrt(B * TAU_62) / DPRIME) ** 2 - VAR_SELF, 0)))
    star63 = float(np.sqrt(max(
        (S_RESID * np.sqrt(B * tau63) / DPRIME) ** 2 - VAR_SELF, 0)))

    res = {
        "det_diff_V": det, "n_windows": len(tc),
        "resid_retention": round(ret, 4),
        "false_drift": {CLASSES[i]: int((pred == i).sum())
                        for i in range(4) if i != 3 and (pred == i).any()},
        "feature_slopes": slopes,
        "v_sig_SD_per_s": round(v, 5),
        "tau_sig_s63": round(tau63, 0), "tau_sig_s62_ref": TAU_62,
        "w_A_22s": round(w22, 5), "w_A_300s": round(w300, 5),
        "dan_mod_integral_ms": round(mod_int, 1),
        "sigma_star_uv_62": round(star62, 3),
        "sigma_star_uv_63": round(star63, 3),
        "ALPN_first/mid/last": [round(float(pop_s["ALPN"][m].mean()), 2)
                                for m in (tc <= tc[0] + 60,
                                          (tc > 140) & (tc < 160),
                                          tc >= tc[-1] - 60)],
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # ---- 4. 图 (2x3, 与 cs15/20/21 同构) ---------------------------------
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
    ax.set_title("池率轨迹（红带=高锁签名带）", fontsize=9)

    ax = axes[0][1]
    ax.plot(tc, X[:, 1], lw=0.8, color="tab:blue")
    ax.axhspan(mu_h[1] - 2 * sd_h[1], mu_h[1] + 2 * sd_h[1],
               color="tab:blue", alpha=0.15, label="训练期 ±2σ")
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("lgRMS 2-20 Hz", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("头皮宽带 RMS 轨迹", fontsize=9)

    ax = axes[0][2]
    ax.plot(tc, X[:, 3], lw=0.8, color="tab:green")
    ax.axhspan(mu_h[3] - 2 * sd_h[3], mu_h[3] + 2 * sd_h[3],
               color="tab:green", alpha=0.12)
    ax.set_ylim(0, np.max(X[:, 3]) * 1.12)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("峭度", fontsize=8, color="tab:green")
    ax.set_title("峭度轨迹（带=训练期 ±2σ）", fontsize=9)

    cm = x.mean(1)
    rf, rt = [], []
    for t0 in np.arange(5.0, 297.0, 10.0):
        a = int(t0 * 250)
        f, P = welch(cm[a:a + 2500], fs=250.0, nperseg=500)
        m = (f >= 5) & (f <= 40)
        rf.append(f[m][np.argmax(P[m])])
        rt.append(t0 + 5.0)
    ax = axes[1][0]
    ax.plot(rt, rf, "o-", ms=3, color="tab:purple")
    ax.set_ylim(0, 45)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("主节律 (Hz)", fontsize=8)
    ax.set_title(f"共模主节律 (σ={np.std(rf):.1f} Hz)", fontsize=9)

    ax = axes[1][1]
    for p_, t0, t1 in zip(pred, tc - HOP_S / 2, tc + HOP_S / 2):
        ax.axvspan(t0, t1, ymin=0.02, ymax=0.98,
                   color=COLORS[CLASSES[p_]], alpha=0.85, lw=0)
    ax.set_yticks(range(4))
    ax.set_yticklabels([ZHCN[c] for c in CLASSES], fontsize=8)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_title(f"监测器（seed62 折→63 评测）hlock 保持 {ret:.0%}",
                 fontsize=9)

    ax = axes[1][2]
    ax.plot(tc, zmax, lw=0.8, color="tab:red")
    ax.axhline(3.0, color="k", ls="--", lw=1)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("max |z|", fontsize=8)
    ax.set_title(f"τ_sig(63)={tau63:.0f} s vs 62 的 {TAU_62:.0f} s",
                 fontsize=9)

    fig.suptitle("condstate30 hlock 300 s 寿命的 seed-63 复制（cs3resnt, "
                 f"seed 63；确定性差 {det:.1e} V）", fontsize=11)
    fig.savefig(OUT / "stability_res63.png", dpi=150)
    plt.close(fig)
    (OUT / "stability_res63_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'stability_res63.png'}")


if __name__ == "__main__":
    main()
