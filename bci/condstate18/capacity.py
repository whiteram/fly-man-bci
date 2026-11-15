"""bci/condstate18/capacity.py — 蝇→头皮读出信道的容量总结算（信息论收口）。

五层可测性闭环（condstate12/14/15/16/17）之后的收口弧：把
"事件不可测 / 状态膝点 nV 级 / 签名寿命分钟级 / 积分被封顶 /
空间被封死" 的账目换算成**信道容量语言**——

  I(S; X) 随背景 σ_bg 的曲线（S = 4 态, X = 2 s 窗 8 维特征）：
    - 决策信道互信息（下界）：跨种子混淆矩阵 p(i,j) 的
      I = Σ p_ij log2(p_ij/(p_i q_j))——判别器可实现的速率；
    - 高斯模型互信息（估计）：类条件高斯（对角+shrinkage）下
      Monte-Carlo E[log2 p(s|X)]，半样本拟合/半样本评测降 plug-in 偏置。
  换算：每签名寿命 τ_sig=336 s 的比特数 = (I/W)·τ_sig；
  头条：I ≥ 1 bit（二分"态在/不在"）所需的背景上限 vs 11.4 µV。

零采集，重用 condstate14 数据链与背景生成器（ρ=0.85 项目模型）。

用法 (conda ffbm, repo root):
    python bci/condstate18/capacity.py
输出: outputs/capacity_curve.png, outputs/capacity_stats.json
"""
import json
import sys
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
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "bci" / "condstate14"))
from state_monitor import (ARMS, CLASSES, device, features,  # noqa: E402
                           load_dev, fit_fold, predict, build_xy, make_bg)

W_S, HOP_S = 2.0, 1.0
SIGMAS = [0.0, 0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.16, 11.4]
DRAWS = 4
TAU_SIG = 336.0          # condstate15 签名寿命 (宽估)
RHO = 0.85


def mi_from_confusion(C):
    """决策信道互信息 (bit): 混淆矩阵 -> I(S; Ŷ)."""
    P = C / C.sum()
    p, q = P.sum(1), P.sum(0)
    m = P > 0
    return float((P[m] * np.log2(P[m] / (p[:, None] * q[None, :])[m])).sum())


def mi_gaussian_mc(X, y, n_mc=20000, seed=0):
    """类条件对角高斯模型的 I(S;X)，半样本拟合/半样本评测."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    halves = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        halves.append(idx)
    I = []
    for rep in range(2):                     # 两个半样本交换拟合/评测
        fit_idx = np.concatenate([h[rep::2] for h in halves])
        ev_idx = np.concatenate([h[1 - rep::2] for h in halves])
        mu = np.array([X[(y == c) & np.isin(np.arange(len(y)), fit_idx)]
                       .mean(0) for c in classes])
        sd = np.array([X[(y == c) & np.isin(np.arange(len(y)), fit_idx)]
                       .std(0) for c in classes]) + 1e-9
        sd = 0.9 * sd + 0.1 * sd.mean()      # 跨类收缩
        Xev = X[ev_idx]
        dev = ((Xev[:, None, :] - mu[None]) / sd[None]) ** 2   # (n,K,D)
        logp = -0.5 * dev.sum(-1) - np.log(sd).sum(-1)[None]   # (n,K)
        logp = logp - logp.max(1, keepdims=True)
        post = np.exp(logp)
        post /= post.sum(1, keepdims=True)
        yev = y[ev_idx]
        H = -np.log2(np.clip(post[np.arange(len(yev)), yev], 1e-12, 1))
        I.append(np.log2(len(classes)) - H.mean())
    return float(np.mean(I))


def main():
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    Xc, yc, sc, _ = build_xy(records, W_S, HOP_S)
    folds = {tr: fit_fold(Xc[sc == tr], yc[sc == tr]) for tr in (62, 63)}

    res = {"sigmas": SIGMAS, "W_s": W_S, "tau_sig_s": TAU_SIG,
           "mi_decision": {}, "mi_gaussian": {},
           "bits_per_tau_sig": {}, "confusion": {}}
    print(f"{'σ_bg (µV)':>10} {'I_dec (bit)':>11} {'I_gauss (bit)':>13}"
          f" {'acc':>6} {'bits/τ_sig':>10}")
    for sg in SIGMAS:
        X, y, s = [], [], []
        for d in range(DRAWS):
            rec_d = [(cls, seed,
                      x + (make_bg(x.shape[0], x.shape[1], sg, 9000 + d)
                           if sg > 0 else 0.0))
                     for cls, seed, x in records]
            Xd, yd, sd_, _ = build_xy(rec_d, W_S, HOP_S)
            X.append(Xd), y.append(yd), s.append(sd_)
        X, y, s = np.concatenate(X), np.concatenate(y), np.concatenate(s)
        yp = np.empty_like(y)
        for tr in (62, 63):
            m = s != tr
            yp[m] = predict(folds[tr], X[m])
        C = np.zeros((4, 4), int)
        for t, p_ in zip(y, yp):
            C[t, p_] += 1
        i_dec = mi_from_confusion(C)
        i_gau = mi_gaussian_mc(X, y)
        acc = float((yp == y).mean())
        bps = i_dec / W_S * TAU_SIG
        res["mi_decision"][sg] = round(i_dec, 4)
        res["mi_gaussian"][sg] = round(i_gau, 4)
        res["bits_per_tau_sig"][sg] = round(bps, 1)
        res["confusion"][sg] = C.tolist()
        print(f"{sg:10.4f} {i_dec:11.4f} {i_gau:13.4f} {acc:6.3f}"
              f" {bps:10.1f}")

    # I>=1 bit 的背景上限 (决策信道, 线性插值)
    sig_np = np.array(SIGMAS)
    mi_np = np.array([res["mi_decision"][s] for s in SIGMAS])
    above = sig_np[mi_np >= 1.0]
    sigma_1bit = float(above.max()) if len(above) else 0.0
    res["sigma_1bit_max_uv"] = sigma_1bit
    print(f"I≥1 bit 的背景上限: {sigma_1bit} µV "
          f"(距 11.4 µV 差 {11.4 / max(sigma_1bit, 1e-9):.0f}×)")

    # ---- 图 ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6),
                             constrained_layout=True)
    ax = axes[0]
    ax.semilogx([max(s, 1e-4) for s in SIGMAS],
                [res["mi_decision"][s] for s in SIGMAS], "o-",
                color="tab:blue", label="决策信道 I (可实现下界)")
    # 高斯模型估计在高噪声端失真 (可为负) —— 只入 JSON, 详见 README
    ax.axhline(2.0, color="0.6", ls=":", lw=1)
    ax.text(1e-4, 2.05, "4 态上限 2 bit", fontsize=7, color="0.4")
    ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
    ax.axvline(11.4, color="k", ls=":", lw=1.1)
    ax.text(12, 1.4, "项目背景\n11.4 µV", fontsize=7)
    ax.set_xlabel("背景 2-20 Hz RMS (µV)", fontsize=8)
    ax.set_ylabel("I(S; X) (bit / 2 s 窗)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("状态信道互信息 vs 背景", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[1]
    ax.semilogx([max(s, 1e-4) for s in SIGMAS],
                [res["bits_per_tau_sig"][s] for s in SIGMAS], "o-",
                color="tab:purple")
    ax.axvline(11.4, color="k", ls=":", lw=1.1)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlabel("背景 2-20 Hz RMS (µV)", fontsize=8)
    ax.set_ylabel(f"bit / 签名寿命 τ_sig={TAU_SIG:.0f} s", fontsize=8)
    ax.set_title("每签名寿命的可用比特数（决策信道）", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[2]
    C0 = np.array(res["confusion"][0.0])          # 干净混淆
    Cn = np.array(res["confusion"][11.4])         # 11.4 µV 混淆
    im = ax.imshow(np.log2(Cn / Cn.sum() + 1e-6) * 0 + Cn / Cn.sum(),
                   cmap="Blues", vmin=0, vmax=0.5)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{Cn[i, j] / Cn.sum():.2f}", ha="center",
                    va="center", fontsize=8,
                    color="white" if Cn[i, j] / Cn.sum() > 0.25 else "black")
    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASSES, fontsize=8, rotation=20)
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("预测", fontsize=8)
    ax.set_ylabel("真值", fontsize=8)
    ax.set_title(f"11.4 µV 下混淆矩阵（I_dec="
                 f"{res['mi_decision'][11.4]:.3f} bit ≈ 0）", fontsize=9)

    fig.suptitle("condstate18 信道容量总结算——蝇→头皮状态信道在项目背景 "
                 f"11.4 µV 下 I≈0 bit（I≥1 bit 上限 "
                 f"{sigma_1bit:g} µV，差 {11.4 / max(sigma_1bit, 1e-9):.0f}×）",
                 fontsize=11)
    fig.savefig(OUT / "capacity_curve.png", dpi=150)
    plt.close(fig)
    (OUT / "capacity_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'capacity_curve.png'}")
    print(f"-> {OUT / 'capacity_stats.json'}")


if __name__ == "__main__":
    main()
