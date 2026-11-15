"""bci/condstate21/stability_qlock.py — 静默锁签名 300 s 漂移分析。

数据: acquire.py 的 qll62 (cs21locknt, DA1@150, 300 s, seed 62)。
1. 确定性复证: 前 15 s vs condstate2/lock62 (同 DA1 脉冲同种子,
   cs2lock 训练 15 s 才开始——此前应逐位相同);
2. 签名漂移: 池率 + 8 特征滑窗斜率 → τ_sig(qlock);
3. 三态剂量-响应: DAN 调制积分 vs τ_sig (hlock 5110ms/336s,
   resid 362ms/2948s, qlock ?) 检验"可塑性写入速率决定签名寿命";
4. σ*(qlock) 修正 + 监测器时间外推 (true class = qlock)。

用法 (conda ffbm, repo root):
    python bci/condstate21/stability_qlock.py
输出: outputs/stability_qlock.png, outputs/stability_qlock_stats.json
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
S_QLOCK, KAP, SIG_N = 0.00181, 0.95, 0.00104    # cs16 seed62
DOSE = {"hlock": (5110.0, 336.0), "resid": (361.8, 2948.0)}
W0 = 0.03


def main():
    # ---- 0. 确定性复证 (前 15 s, cs2lock 训练开始前) --------------------
    x_raw = np.load(OUT / "qll62_scalp.npy").astype(np.float64)
    ref = np.load(ROOT / "bci/condstate2/outputs/lock62_scalp.npy")
    n15 = min(15000, len(ref), len(x_raw))
    det = float(np.abs(x_raw[:n15] - ref[:n15].astype(np.float64)).max())
    print(f"确定性复证: max|qll62 - lock62| 前 {n15} ms = {det:.2e} V")
    x = device(x_raw * 1e6)

    # ---- 1. 折与训练分布 -------------------------------------------------
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    Xtr, ytr, seeds, _ = build_xy(records, W_S, HOP_S)
    fold = fit_fold(Xtr[seeds == 63], ytr[seeds == 63])
    qk = np.array([CLASSES[i] for i in ytr]) == "qlock"
    mu_q, sd_q = Xtr[qk].mean(0), Xtr[qk].std(0) + 1e-12
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
    zmax = np.abs((X - mu_q) / sd_q).max(1)
    ret = float((pred == 1).mean())

    tt = (tc - tc.mean()) / tc.std()
    slopes = {names[k]: round(float(np.polyfit(tt, X[:, k], 1)[0]
                                    / X[:, k].std()), 3)
              for k in range(X.shape[1])}
    v = float(np.sqrt(sum(s ** 2 for s in slopes.values()))) \
        / float(tc.std())
    tau_qlock = 3.0 / v
    var_self = (KAP * SIG_N) ** 2
    star_old = float(np.sqrt(max(
        (S_QLOCK * np.sqrt(B * 336.0) / DPRIME) ** 2 - var_self, 0)))
    star_new = float(np.sqrt(max(
        (S_QLOCK * np.sqrt(B * tau_qlock) / DPRIME) ** 2 - var_self, 0)))

    # ---- 3. 池率 + 机制 --------------------------------------------------
    pop = np.load(OUT / "qll62_pop.npz")
    pop_s = {g: np.array([pop[g][int(t * 1000):int(t * 1000) + 2000]
                          .mean() for t in tc])
             for g in ("ALPN", "Kenyon_Cell", "MBON", "DAN")}
    st = np.load(OUT / "states" / "qll62.npz")
    w300 = float(st["w_KCM"][st["pre_type"] == "KCg-m"].mean())
    tr = np.load(OUT / "qll62_dan.npy")
    mod_int = float(tr[:, 2].sum() * 0.5)

    res = {
        "det_diff_V": det, "n_windows": len(tc),
        "qlock_retention": round(ret, 4),
        "false_drift": {CLASSES[i]: int((pred == i).sum())
                        for i in range(4) if i != 1 and (pred == i).any()},
        "feature_slopes": slopes,
        "v_sig_SD_per_s": round(v, 5),
        "tau_sig_qlock_s": round(tau_qlock, 0),
        "w_A_300s_minus_w0": round(w300 - W0, 5),
        "dan_mod_integral_ms": round(mod_int, 1),
        "sigma_star_old_uv": round(star_old, 4),
        "sigma_star_new_uv": round(star_new, 4),
        "ALPN_first/mid/last": [round(float(pop_s["ALPN"][m].mean()), 2)
                                for m in (tc <= tc[0] + 60,
                                          (tc > 140) & (tc < 160),
                                          tc >= tc[-1] - 60)],
        "dose_response": {**DOSE, "qlock": (round(mod_int, 1),
                                            round(tau_qlock, 0))},
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # ---- 4. 图 (2x4) -----------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(19.5, 8.4),
                             constrained_layout=True)
    ax = axes[0][0]
    for g, c in (("ALPN", "tab:red"), ("Kenyon_Cell", "tab:orange"),
                 ("MBON", "tab:purple")):
        ax.plot(tc, pop_s[g], lw=0.9, color=c, label=g)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("池率 (Hz)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("池率轨迹（静默锁 AL~20 Hz）", fontsize=9)

    ax = axes[0][1]
    ax.plot(tc, X[:, 1], lw=0.8, color="tab:blue")
    ax.axhspan(mu_q[1] - 2 * sd_q[1], mu_q[1] + 2 * sd_q[1],
               color="tab:blue", alpha=0.15,
               label="训练期 qlock ±2σ (3-14 s)")
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("lgRMS 2-20 Hz", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("头皮宽带 RMS 轨迹", fontsize=9)

    ax = axes[0][2]
    ax.plot(tc, X[:, 3], lw=0.8, color="tab:green")
    ax.axhspan(mu_q[3] - 2 * sd_q[3], mu_q[3] + 2 * sd_q[3],
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
    ax = axes[0][3]
    ax.plot(rt, rf, "o-", ms=3, color="tab:purple")
    ax.set_ylim(0, 45)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("主节律 (Hz)", fontsize=8)
    ax.set_title(f"共模主节律 (σ={np.std(rf):.1f} Hz)", fontsize=9)

    ax = axes[1][0]
    for p_, t0, t1 in zip(pred, tc - HOP_S / 2, tc + HOP_S / 2):
        ax.axvspan(t0, t1, ymin=0.02, ymax=0.98,
                   color=COLORS[CLASSES[p_]], alpha=0.85, lw=0)
    ax.set_yticks(range(4))
    ax.set_yticklabels([ZHCN[c] for c in CLASSES], fontsize=8)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_title(f"监测器（seed63 折）——qlock 保持 {ret:.0%}", fontsize=9)

    ax = axes[1][1]
    ax.plot(tc, zmax, lw=0.8, color="tab:red")
    ax.axhline(3.0, color="k", ls="--", lw=1)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("max |z|（8 特征）", fontsize=8)
    ax.set_title(f"相对训练分布最大偏离（中位 "
                 f"{np.median(zmax):.1f}σ）", fontsize=9)

    # 剂量-响应: DAN 积分 vs τ_sig (三态)
    ax = axes[1][2]
    dose = {**DOSE, "qlock": (mod_int, tau_qlock)}
    for (tag, (mi, tau)), c in zip(
            dose.items(), ("tab:green", "tab:red", "tab:cyan")):
        ax.loglog(mi, tau, "o", ms=9, color=c, label=tag)
        ax.annotate(f"{tag}\n{mi:.0f}ms→{tau:.0f}s", (mi, tau),
                    textcoords="offset points", xytext=(6, -4),
                    fontsize=7, color=c)
    xs = np.array([m_ for m_, _ in dose.values()])
    ys = np.array([t_ for _, t_ in dose.values()])
    sl = np.polyfit(np.log(xs), np.log(ys), 1)[0]
    xx = np.logspace(np.log10(300), np.log10(6000), 20)
    ax.loglog(xx, ys.mean() * (xx / xs.mean()) ** sl, "--", color="0.6",
              lw=0.9, label=f"斜率 {sl:.2f}")
    ax.set_xlabel("DAN 调制积分 (ms / 300 s)", fontsize=8)
    ax.set_ylabel("τ_sig (s)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("剂量-响应：可塑写速率 vs 签名寿命", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[1][3]
    ax.bar([0, 1], [star_old, star_new], 0.55,
           color=["0.6", "tab:cyan"])
    ax.axhline(11.4, color="k", ls=":", lw=1.2)
    ax.text(0.5, 12.5, "项目背景 11.4 µV", fontsize=7, ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"τ=336s\n(旧, 按 hlock)", f"τ={tau_qlock:.0f}s\n"
                        f"(qlock 实测)"], fontsize=8)
    ax.set_ylabel("σ*(qlock) (µV)", fontsize=8)
    ax.set_title(f"σ* 修正: {star_old:.3f} → {star_new:.3f} µV",
                 fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    fig.suptitle("condstate21 静默锁签名 300 s 漂移（cs21locknt, seed 62；"
                 f"确定性差 {det:.1e} V）——τ_sig 剂量-响应第三点",
                 fontsize=11)
    fig.savefig(OUT / "stability_qlock.png", dpi=150)
    plt.close(fig)
    (OUT / "stability_qlock_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'stability_qlock.png'}")
    print(f"-> {OUT / 'stability_qlock_stats.json'}")


if __name__ == "__main__":
    main()
