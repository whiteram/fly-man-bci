"""bci/condstate14/state_monitor.py — 头皮-only 状态监测器（零采集）.

condstate12/13 提出"事件检测应由状态读出门控"，但项目里从未有过
一个真正能用的状态判别器。本弧把 eegview 相似度审计验证过的特征
（带 RMS / 峭度 / 谱指数 / >4σ 事件率）接成 4 类监督判别器：

  naive（幼稚静息）/ qlock（静默锁 ~20 Hz）/ hlock（高锁 ~100 Hz）
  / resid（残留/闩锁，KC/MBON 招募）

数据全部盘上现有（condstate2/3/4 头皮导出，双种子 62/63）：

  1. 段级特征（250 Hz 设备链，2 s 窗）→ 跨种子 2 折逻辑回归，
     混淆矩阵 + 幼稚复证（condstate2 ctl 独立 trial）；
  2. 决策时延：窗长扫描 0.5/1/2/4 s；
  3. 状态翻转检测：unlk 臂（8 s 溶锁）时间线——判别器能否从头皮
     单独读出"锁→幼稚"的翻转、延迟多少；
  4. 背景可测性边界：叠 1/f² 人脑背景（2-20 Hz RMS 扫 σ），
     准确率跌到几成——把"状态本身还能不能被设备读到"接入项目
     主论点的可测性账。

特征严格头皮-only（不触碰 pop npz）；状态标签来自协议臂（监督
训练标签）。背景为示意模型（同 eegview 复合行），仅用于退化测试。

用法 (conda ffbm, repo root):
    python bci/condstate14/state_monitor.py
输出: outputs/monitor_main.png, outputs/monitor_unlk_bg.png,
      outputs/monitor_stats.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import (butter, sosfiltfilt, resample_poly, welch,
                          find_peaks)
from scipy.stats import theilslopes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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
FS_N, FS_D = 1000.0, 250.0

CLASSES = ["naive", "qlock", "hlock", "resid"]
ZHCN = {"naive": "幼稚静息", "qlock": "静默锁", "hlock": "高锁",
        "resid": "残留/闩锁"}
COLORS = {"naive": "tab:blue", "qlock": "tab:cyan", "hlock": "tab:green",
          "resid": "tab:red"}

# (class, path, seed, t0_ms, t1_ms) —— 稳态窗，避开诱导脉冲/调度/训练窗
ARMS = [
    ("naive", ROOT / "bci/condstate3/outputs/ctl62_scalp.npy", 62, 4000, 16500),
    ("naive", ROOT / "bci/condstate3/outputs/ctl63_scalp.npy", 63, 4000, 16500),
    ("qlock", ROOT / "bci/condstate2/outputs/lock62_scalp.npy", 62, 3000, 14000),
    ("qlock", ROOT / "bci/condstate2/outputs/lock63_scalp.npy", 63, 3000, 14000),
    ("hlock", ROOT / "bci/condstate4/outputs/hl62_scalp.npy", 62, 3000, 16500),
    ("hlock", ROOT / "bci/condstate4/outputs/hl63_scalp.npy", 63, 3000, 16500),
    ("resid", ROOT / "bci/condstate3/outputs/res62_scalp.npy", 62, 8000, 16500),
    ("resid", ROOT / "bci/condstate3/outputs/res63_scalp.npy", 63, 8000, 16500),
]
# 幼稚复证：condstate2 ctl 是独立 trial 的暗态（训练窗 15-17 s 已避开），
# 且完全不在训练臂里 —— 真复证
REPL = [
    ("naive", ROOT / "bci/condstate2/outputs/ctl62_scalp.npy", 62, 4000, 14000),
    ("naive", ROOT / "bci/condstate2/outputs/ctl63_scalp.npy", 63, 4000, 14000),
]
UNLK = [
    (ROOT / "bci/condstate2/outputs/unlk62_scalp.npy", 62),
    (ROOT / "bci/condstate2/outputs/unlk63_scalp.npy", 63),
]
FEATS = ["lgRMS_δ(1-4)", "lgRMS_bb(2-20)", "lgRMS_hi(30-80)", "峭度",
         "谱指数(1-40)", ">4σ事件/s", "峰显著dB", "峰频率Hz"]
MELT_MS = 8000          # unlk 溶锁时刻 (CEN_C ×0.5)


def device(x):
    sos = butter(4, [0.5, 100.0], btype="bandpass", fs=FS_N, output="sos")
    return resample_poly(sosfiltfilt(sos, x, axis=0), 1, 4, axis=0)


def load_dev(path, t0, t1):
    e = np.load(path).astype(np.float64)[t0:t1] * 1e6
    return device(e)


def features(x):
    """x: (n,17) µV @250 Hz -> 8 维特征（头皮-only）."""
    n = len(x)
    cm = x.mean(axis=1)
    z = (cm - cm.mean()) / (cm.std() + 1e-30)
    f, P = welch(x, fs=FS_D, nperseg=min(n, 512), axis=0)
    pm = np.median(P, axis=1)

    def brms(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(np.sqrt(np.trapezoid(pm[m], f[m])))

    mu, sd = x.mean(0), x.std(0) + 1e-30
    kur = float(np.median(((x - mu) ** 4).mean(0) / sd ** 4 - 3.0))
    m = (f >= 1) & (f <= 40)
    pmS = np.copy(pm[m]).astype(float)
    for i in range(1, len(pmS) - 1):          # 3-bin 移动中位
        pmS[i] = np.median(pm[m][i - 1:i + 2])
    lf = np.log(f[m])
    lp = np.log(np.maximum(pmS, 1e-40))
    slope, icpt, _, _ = theilslopes(lp, lf)
    pk, props = find_peaks(lp - (icpt + slope * lf), prominence=0.46,
                           distance=4)
    if len(pk):
        j = int(pk[np.argmax(props["prominences"])])
        peak_hz = float(np.exp(lf[j]))
        peak_db = float(props["prominences"].max() / 0.23)
    else:
        peak_hz, peak_db = 0.0, 0.0
    idx = np.flatnonzero(np.abs(z) > 4.0)      # >4σ 事件（合并相邻）
    ev = (len(np.where(np.diff(idx) > 1)[0]) + 1) if len(idx) else 0
    return np.array([
        np.log10(brms(1, 4) + 1e-30), np.log10(brms(2, 20) + 1e-30),
        np.log10(brms(30, 80) + 1e-30), kur, slope,
        ev / (n / FS_D), peak_db, peak_hz,
    ])


def seg_bounds(n, w_s, hop_s):
    wlen, step = int(w_s * FS_D), int(hop_s * FS_D)
    return [(i, i + wlen) for i in range(0, n - wlen + 1, step)]


def make_bg(n, ch, sigma, seed):
    """示意人脑背景: 1/f² 0.5-40 Hz, 85% 空间共享, 2-20 Hz RMS=sigma."""
    rng = np.random.default_rng(seed)
    fr = np.fft.rfftfreq(n, 1 / FS_D)
    shape = np.where((fr >= 0.4) & (fr <= 45), 1.0 / (fr + 0.5) ** 2, 0.0)
    shape[0] = 0.0

    def one():
        return np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * shape, n)

    shared = one()
    sosd = butter(4, [2.0, 20.0], btype="bandpass", fs=FS_D, output="sos")
    a, b = np.sqrt(0.85), np.sqrt(0.15)
    bg = np.zeros((n, ch))
    for c in range(ch):
        y = a * shared + b * one()
        y *= sigma / (np.sqrt((sosfiltfilt(sosd, y) ** 2).mean()) + 1e-30)
        bg[:, c] = y
    return bg


def fit_fold(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    clf = LogisticRegression(max_iter=4000, C=1.0)
    clf.fit((Xtr - mu) / sd, ytr)
    return dict(mu=mu, sd=sd, clf=clf)


def predict(fold, X):
    return fold["clf"].predict((X - fold["mu"]) / fold["sd"])


def posterior(fold, X):
    return fold["clf"].predict_proba((X - fold["mu"]) / fold["sd"])


def build_xy(records, w_s, hop_s, degrade=None):
    """records: [(cls, seed, x_dev)] -> X, y, seeds, seg_t0 (窗中心 s)."""
    X, y, seeds, tc = [], [], [], []
    for cls, seed, x in records:
        if degrade is not None:
            x = x + degrade(x.shape, seed)
        for i, j in seg_bounds(len(x), w_s, hop_s):
            X.append(features(x[i:j]))
            y.append(CLASSES.index(cls))
            seeds.append(seed)
            tc.append((i + j) / 2 / FS_D)
    return np.array(X), np.array(y), np.array(seeds), np.array(tc)


def main():
    records = [(cls, seed, load_dev(p, t0, t1))
               for cls, p, seed, t0, t1 in ARMS]
    repl_rec = [(cls, seed, load_dev(p, t0, t1))
                for cls, p, seed, t0, t1 in REPL]
    res = {"protocol": "4类×双种子, 跨种子2折, 特征=8维头皮-only"}

    # ---- 1. 主表: 2 s 窗, 跨种子 2 折 --------------------------------
    X, y, seeds, _ = build_xy(records, 2.0, 1.0)
    folds = {}
    for tr in (62, 63):
        m = seeds == tr
        folds[tr] = fit_fold(X[m], y[m])
    yp = np.empty_like(y)
    for tr in (62, 63):
        te = seeds != tr
        yp[te] = predict(folds[tr], X[te])
    acc = float((yp == y).mean())
    conf = np.zeros((4, 4), int)
    for t, p in zip(y, yp):
        conf[t, p] += 1
    per_class = {CLASSES[i]: round(float(conf[i, i] / conf[i].sum()), 3)
                 for i in range(4)}
    # 幼稚复证 (独立 trial ctl)
    Xr, yr, sr, _ = build_xy(repl_rec, 2.0, 1.0)
    ypr = np.concatenate([predict(folds[62], Xr[sr == 62]),
                          predict(folds[63], Xr[sr == 63])])
    repl_acc = float((ypr == yr).mean())
    print(f"主表 2 s 窗: 跨种子 acc={acc:.3f}  per-class={per_class}  "
          f"复证acc={repl_acc:.3f}")
    res["main_2s"] = dict(acc=round(acc, 3), per_class=per_class,
                          n_seg=len(y), repl_acc=round(repl_acc, 3),
                          confusion=conf.tolist())

    # ---- 2. 窗长扫描 ------------------------------------------------
    sweep = {}
    for w in (0.5, 1.0, 2.0, 4.0):
        Xw, yw, sw, _ = build_xy(records, w, w / 2)
        ypw = np.empty_like(yw)
        for tr in (62, 63):
            te = sw != tr
            ypw[te] = predict(folds[tr], Xw[te])
        sweep[w] = round(float((ypw == yw).mean()), 3)
        print(f"  W={w:4.1f}s  acc={sweep[w]:.3f}  (n={len(yw)})")
    res["window_sweep"] = sweep

    # ---- 3. unlk 状态翻转检测 ----------------------------------------
    unl = {}
    fig2, ax2 = plt.subplots(1, 3, figsize=(15.5, 4.4),
                             constrained_layout=True)
    for k, (path, seed) in enumerate(UNLK):
        x = load_dev(path, 0, 14000)             # 0-14 s，避开训练窗
        Xu, _, _, tc = build_xy([("naive", seed, x)], 1.0, 0.5)
        fold = folds[63 if seed == 62 else 62]   # 用另一种子的模型
        pred = predict(fold, Xu)
        post = posterior(fold, Xu)
        # 翻转延迟: 熔断后首个"naive 且此后不再变回"的窗中心
        lat = None
        for i in range(len(tc)):
            if tc[i] > MELT_MS / 1000 and (pred[i:] == 0).all():
                lat = float(tc[i] - MELT_MS / 1000)
                break
        fa = float((pred[tc < MELT_MS / 1000 - 0.5] == 0).mean())
        unl[f"seed{seed}"] = dict(latency_s=(round(lat, 2) if lat else None),
                                  pre_melt_naive_frac=round(fa, 3))
        print(f"  unlk seed{seed}: 溶锁检出延迟={lat and round(lat, 2)}s "
              f"(窗中心-8s), 熔断前误报naive={fa:.2f}")
        if k == 0:                                # 画 seed62 时间线
            ax = ax2[0]
            ax.axvspan(1.0, 8.0, color="0.88", zorder=0)
            ax.axvspan(9.5, 14.0, color="0.95", zorder=0)
            for p_, t0, t1 in zip(pred, tc - 0.5, tc + 0.5):
                ax.axvspan(t0, t1, ymin=0.02, ymax=0.98,
                           color=COLORS[CLASSES[p_]], alpha=0.85, lw=0)
            ax.axvline(8.0, color="k", ls="--", lw=1)
            ax.text(8.1, 3.1, "溶锁 (×0.5)", fontsize=8)
            ax.text(2.0, 3.1, "真值: 锁", fontsize=8)
            ax.text(11.0, 3.1, "真值: 幼稚", fontsize=8)
            ax.set_ylim(0, 4)
            ax.set_yticks(range(4))
            ax.set_yticklabels([ZHCN[c] for c in CLASSES], fontsize=8)
            ax.set_xlabel("时间 (s)", fontsize=8)
            ax.set_title("unlk62 判别时间线 (1 s 窗, 对侧种子模型)",
                         fontsize=9)
            axp = ax2[1]
            axp.plot(tc, post[:, 0], "o-", color="tab:blue", ms=3,
                     label="P(幼稚)")
            axp.plot(tc, post[:, 1], "o-", color="tab:cyan", ms=3,
                     label="P(静默锁)")
            axp.axvline(8.0, color="k", ls="--", lw=1)
            axp.set_ylim(-0.05, 1.05)
            axp.set_xlabel("时间 (s)", fontsize=8)
            axp.legend(fontsize=8)
            axp.set_title("后验轨迹 (锁→幼稚翻转)", fontsize=9)
    res["unlk"] = unl

    # ---- 4. 背景可测性边界 --------------------------------------------
    sigmas = [0.0, 0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.16, 11.4]
    bg_curve = {}
    for sg in sigmas:
        accs = []
        for draw in range(3):
            Xb, yb, sb, _ = build_xy(
                records, 2.0, 1.0,
                degrade=(None if sg == 0 else
                         lambda sh, s, sg=sg, d=draw:
                         make_bg(sh[0], sh[1], sg, 1000 * d + s)))
            ypb = np.empty_like(yb)
            for tr in (62, 63):
                te = sb != tr
                ypb[te] = predict(folds[tr], Xb[te])
            accs.append(float((ypb == yb).mean()))
        bg_curve[sg] = round(float(np.mean(accs)), 3)
        print(f"  背景 σ={sg:6.3f} µV -> acc={bg_curve[sg]:.3f}")
    res["bg_sweep"] = bg_curve
    res["_bg_note"] = ("1/f² 示意背景 85% 共享 (同 eegview 复合行), "
                       "训练于干净信号, 测试于退化信号")

    # ---- 图 1 ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2),
                             constrained_layout=True)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    pca = PCA(2).fit(Z)
    Zp = pca.transform(Z)
    ax = axes[0][0]
    for i, c in enumerate(CLASSES):
        for s, mk in ((62, "o"), (63, "^")):
            m = (y == i) & (seeds == s)
            ax.plot(Zp[m, 0], Zp[m, 1], mk, ms=5, color=COLORS[c],
                    alpha=0.8, label=f"{ZHCN[c]} s{s}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("特征空间 PCA（viz 用，非判别输入）", fontsize=9)
    ax.grid(True, color="0.92")

    ax = axes[0][1]
    confn = conf / conf.sum(1, keepdims=True)
    im = ax.imshow(confn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{confn[i, j]:.2f}\n({conf[i, j]})",
                    ha="center", va="center", fontsize=8,
                    color="white" if confn[i, j] > 0.5 else "black")
    ax.set_xticks(range(4))
    ax.set_xticklabels([ZHCN[c] for c in CLASSES], fontsize=8, rotation=20)
    ax.set_yticks(range(4))
    ax.set_yticklabels([ZHCN[c] for c in CLASSES], fontsize=8)
    ax.set_xlabel("预测", fontsize=8)
    ax.set_ylabel("真值", fontsize=8)
    ax.set_title(f"混淆矩阵（跨种子 2 折，acc={acc:.1%}）", fontsize=9)

    ax = axes[1][0]
    ws = sorted(sweep)
    ax.plot(ws, [sweep[w] for w in ws], "o-", color="tab:purple")
    ax.axhline(0.25, color="0.6", ls=":", lw=1)
    ax.text(2.5, 0.27, "随机 25%", fontsize=7, color="0.5")
    for w in ws:
        ax.annotate(f"{sweep[w]:.2f}", (w, sweep[w]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(ws)
    ax.set_xticklabels([f"{w:g}" for w in ws])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("窗长 (s)", fontsize=8)
    ax.set_ylabel("跨种子准确率", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_title("决策时延：准确率 vs 窗长", fontsize=9)
    ax.grid(True, color="0.92")

    ax = axes[1][1]
    M = np.array([[X[y == i, k].mean() for i in range(4)]
                  for k in range(len(FEATS))])
    Mz = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-12)
    im = ax.imshow(Mz, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels([ZHCN[c] for c in CLASSES], fontsize=8)
    ax.set_yticks(range(len(FEATS)))
    ax.set_yticklabels(FEATS, fontsize=8)
    for i in range(len(FEATS)):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(Mz[i, j]) > 1.3 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="z(类均值)")
    ax.set_title("特征 × 类均值剖面（色=z 分数，标注原值）", fontsize=9)

    fig.suptitle("condstate14 头皮-only 状态监测器 — 4 态 × 双种子，"
                 "跨种子 2 折（250 Hz 设备链）", fontsize=11)
    fig.savefig(OUT / "monitor_main.png", dpi=150)
    plt.close(fig)

    # ---- 图 2c --------------------------------------------------------
    ax = ax2[2]
    xs = sorted(bg_curve)
    ax.semilogx(xs, [bg_curve[s] for s in xs], "o-", color="tab:red")
    ax.axhline(0.25, color="0.6", ls=":", lw=1)
    ax.text(0.0013, 0.27, "随机 25%", fontsize=7, color="0.5")
    ax.axvspan(0.0019, 0.065, color="tab:orange", alpha=0.18)
    ax.text(0.008, 0.04, "蝇信号带RMS\n0.002-0.065 µV", fontsize=7,
            color="tab:orange", ha="center")
    ax.axvline(11.4, color="k", ls="--", lw=1)
    ax.text(9.0, 0.5, "项目背景模型\n11.4 µV", fontsize=7, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("背景 2-20 Hz RMS (µV, 1/f² 示意)", fontsize=8)
    ax.set_ylabel("跨种子准确率", fontsize=8)
    ax.set_title("可测性边界：准确率 vs 背景水平", fontsize=9)
    ax.grid(True, which="both", color="0.92")
    fig2.suptitle("状态翻转检测 (unlk) 与背景退化", fontsize=11)
    fig2.savefig(OUT / "monitor_unlk_bg.png", dpi=150)
    plt.close(fig2)

    res["feature_means"] = {
        FEATS[k]: {CLASSES[i]: round(float(M[k, i]), 4) for i in range(4)}
        for k in range(len(FEATS))}
    res["lr_coef"] = {  # 种子62折, 类 vs naive 基线的标准化系数
        CLASSES[i]: {FEATS[k]: round(float(folds[62]["clf"].coef_[i - 1, k]), 2)
                     for k in range(len(FEATS))}
        for i in range(1, 4)}
    (OUT / "monitor_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT / 'monitor_main.png'}")
    print(f"-> {OUT / 'monitor_unlk_bg.png'}")
    print(f"-> {OUT / 'monitor_stats.json'}")


if __name__ == "__main__":
    main()
