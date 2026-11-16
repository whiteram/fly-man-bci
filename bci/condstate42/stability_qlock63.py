"""bci/condstate42/stability_qlock63.py — 静默锁 seed-63 点火失败臂分析（null 臂）。

数据: acquire 的 qll63 (cs21locknt, DA1@150 pA, 300 s, seed 63)。
设计意图是 cs21 的 τ 复制（62 校准剂量逐字复制），结果静默锁在
seed 63 上根本不点燃——本脚本量化这个点火失败：

1. 确定性诊断: vs condstate2/lock63（cs2lock300 = DA1@**300** pA，
   condstate2 的 seed-63 成锁剂量）——前 300 ms 逐位相同、分歧
   起点 ~332 ms = 脉冲上升沿内：7e-7 V 差纯为剂量差，确定性框架
   无损（也反证 condstate2 当年已知 150 pA 点不着 63）;
2. 点火失败: 脉冲窗 ALPN 峰 1.47 Hz（62@同剂量 29.4 Hz，20×），
   脉冲后 299.4 s 引擎 **0.000 Hz**;
3. 监测器（cs14 seed62 折）全程判 naive（99.7%）——未点火轨迹
   与幼稚静息不可分;
4. null 轨迹漂移 τ_null=1399 s（次级观察：naive 样轨迹的 63 侧写）;
5. 剂量表: 150 pA → 62 成锁 / 63 不燃; 300 pA → 63 成锁
   （condstate2 已证; τ 真复制 = condstate43, cs42locknt300）。

用法 (conda ffbm, repo root):
    python bci/condstate42/stability_qlock63.py
输出: outputs/stability_qlock63.png, outputs/stability_qlock63_stats.json
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
DOSE = {"hlock": (5110.0, 336.0), "resid": (361.8, 2948.0),
        "qlock62": (0.9, 2988.0)}


def main():
    # ---- 0. 确定性诊断 (vs lock63@300pA——差值应为纯剂量差) -----------
    x_raw = np.load(OUT / "qll63_scalp.npy").astype(np.float64)
    ref = np.load(ROOT / "bci/condstate2/outputs/lock63_scalp.npy")
    n15 = min(15000, len(ref), len(x_raw))
    d15 = np.abs(x_raw[:n15] - ref[:n15].astype(np.float64)).max(1)
    det = float(d15.max())
    onset = int(np.argmax(d15 > 1e-12))
    pre300 = float(d15[:300].max())
    print(f"确定性诊断: max|qll63(150pA) - lock63(300pA)| 前 {n15} ms"
          f" = {det:.2e} V; 分歧起点 {onset} ms, 前 300 ms {pre300:.2e} V")
    x = device(x_raw * 1e6)

    # ---- 1. 折与训练分布 (seed62 训练 → 63 评测, 跨种子方向) -------------
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    Xtr, ytr, seeds, _ = build_xy(records, W_S, HOP_S)
    fold = fit_fold(Xtr[seeds == 62], ytr[seeds == 62])
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
    ret_nv = float((pred == 0).mean())

    tt = (tc - tc.mean()) / tc.std()
    slopes = {names[k]: round(float(np.polyfit(tt, X[:, k], 1)[0]
                                    / X[:, k].std()), 3)
              for k in range(X.shape[1])}
    v = float(np.sqrt(sum(s ** 2 for s in slopes.values()))) \
        / float(tc.std())
    tau_null = 3.0 / v

    # ---- 3. 点火定量 -----------------------------------------------------
    pop = np.load(OUT / "qll63_pop.npz")
    pop_s = {g: np.array([pop[g][int(t * 1000):int(t * 1000) + 2000]
                          .mean() for t in tc])
             for g in ("ALPN", "Kenyon_Cell", "MBON", "DAN")}
    p63 = pop["ALPN"]
    p62 = np.load(ROOT / "bci/condstate21/outputs/qll62_pop.npz")["ALPN"]
    ign = {
        "pulse_peak_hz_63": round(float(p63[250:700].max()), 3),
        "pulse_peak_hz_62_ref": round(float(p62[250:700].max()), 2),
        "post_pulse_max_hz_63": round(float(p63[700:].max()), 3),
        "post_pulse_max_hz_62_ref": round(float(p62[700:].max()), 2),
        "alpn_mean_hz_63": round(float(p63.mean()), 6),
        "alpn_mean_hz_62_ref": round(float(p62.mean()), 3),
        "dose_locks_63_pa": 300.0,
    }
    st = np.load(OUT / "states" / "qll63.npz")
    w300 = float(st["w_KCM"][st["pre_type"] == "KCg-m"].mean())
    tr = np.load(OUT / "qll63_dan.npy")
    mod_int = float(tr[:, 2].sum() * 0.5)

    res = {
        "det_diff_V": det, "divergence_onset_ms": onset,
        "pre_300ms_diff_V": pre300, "n_windows": len(tc),
        "ignition": ign,
        "qlock_retention": round(ret, 4),
        "naive_classification": round(ret_nv, 4),
        "false_drift": {CLASSES[i]: int((pred == i).sum())
                        for i in range(4) if i != 1 and (pred == i).any()},
        "feature_slopes": slopes,
        "v_sig_SD_per_s": round(v, 5),
        "tau_null_naivelike_s63": round(tau_null, 0),
        "w_A_300s_minus_w0": round(w300 - 0.03, 5),
        "dan_mod_integral_ms": round(mod_int, 1),
        "ALPN_first/mid/last": [round(float(pop_s["ALPN"][m].mean()), 4)
                                for m in (tc <= tc[0] + 60,
                                          (tc > 140) & (tc < 160),
                                          tc >= tc[-1] - 60)],
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # ---- 4. 图 (2x4) -----------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(19.5, 8.4),
                             constrained_layout=True)
    ax = axes[0][0]
    ax.plot(np.arange(len(p63)) / 1000.0, p63, lw=0.5, color="tab:red",
            label="63@150 pA（本弧）")
    ax.plot(np.arange(len(p62)) / 1000.0, p62, lw=0.5, color="0.6",
            alpha=0.7, label="62@150 pA（cs21 参照）")
    ax.set_xlim(0, 30)
    ax.set_ylim(-1, 47)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("ALPN 池率 (Hz)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title(f"点火失败：脉冲窗峰 {ign['pulse_peak_hz_63']} Hz"
                 f" vs 62 的 {ign['pulse_peak_hz_62_ref']} Hz，"
                 f"此后 0.000 Hz", fontsize=9)

    ax = axes[0][1]
    ax.plot(tc, X[:, 1], lw=0.8, color="tab:blue")
    ax.axhspan(mu_q[1] - 2 * sd_q[1], mu_q[1] + 2 * sd_q[1],
               color="tab:blue", alpha=0.15,
               label="训练期 qlock ±2σ (3-14 s)")
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("lgRMS 2-20 Hz", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("头皮宽带 RMS（全程低于 qlock 带——未点火）", fontsize=9)

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
    ax.set_title(f"监测器（seed62 折→63 评测）：{ret_nv:.1%} 判 naive"
                 f"——未点火与幼稚静息不可分", fontsize=9)

    ax = axes[1][1]
    ax.plot(tc, zmax, lw=0.8, color="tab:red")
    ax.axhline(3.0, color="k", ls="--", lw=1)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("max |z|（8 特征）", fontsize=8)
    ax.set_title(f"相对 qlock 训练分布偏离（中位 "
                 f"{np.median(zmax):.1f}σ）", fontsize=9)

    # 剂量-响应: 三参照点 + null 臂（灰 x，不入拟合）
    ax = axes[1][2]
    for (tag, (mi, tau)), c in zip(
            DOSE.items(), ("tab:green", "tab:red", "tab:cyan")):
        ax.loglog(mi, tau, "o", ms=9, color=c, label=tag)
        ax.annotate(f"{tag}\n{mi:.0f}ms→{tau:.0f}s", (mi, tau),
                    textcoords="offset points", xytext=(6, -4),
                    fontsize=7, color=c)
    ax.loglog(mod_int, tau_null, "x", ms=9, mew=2, color="0.5",
              label="63@150（null，未点火）")
    xs = np.array([m_ for m_, _ in DOSE.values()])
    ys = np.array([t_ for _, t_ in DOSE.values()])
    sl = np.polyfit(np.log(xs), np.log(ys), 1)[0]
    xx = np.logspace(np.log10(0.3), np.log10(6000), 20)
    ax.loglog(xx, ys.mean() * (xx / xs.mean()) ** sl, "--", color="0.6",
              lw=0.9, label=f"斜率 {sl:.2f}")
    ax.set_xlabel("DAN 调制积分 (ms / 300 s)", fontsize=8)
    ax.set_ylabel("τ_sig (s)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("剂量-响应（灰 x = 未点火，不入拟合）", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    # 点火剂量表
    ax = axes[1][3]
    ax.axis("off")
    rows = [("DA1 剂量", "seed 62", "seed 63"),
            ("150 pA", "成锁 19.7 Hz", "不燃 0.000 Hz"),
            ("300 pA", "（未测）", "成锁（cs2）"),
            ("τ_sig", "≥2988 s", "待 condstate43")]
    tb = ax.table(cellText=rows[1:], colLabels=rows[0],
                  cellLoc="center", loc="center",
                  colWidths=[0.3, 0.35, 0.35])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.7)
    ax.set_title("点火剂量表：静默锁诱导是种子脆弱的", fontsize=9)

    fig.suptitle("condstate42 静默锁 seed-63 点火失败（cs21locknt@150 pA"
                 f" 逐字复制；确定性诊断：前 300 ms 逐位同，分歧 {onset} ms"
                 " = 剂量差）", fontsize=11)
    fig.savefig(OUT / "stability_qlock63.png", dpi=150)
    plt.close(fig)
    (OUT / "stability_qlock63_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'stability_qlock63.png'}")
    print(f"-> {OUT / 'stability_qlock63_stats.json'}")


if __name__ == "__main__":
    main()
