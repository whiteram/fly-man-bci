"""bci/condstate24/ssvep_cca.py — SSVEP 正典 CCA 分类器 × 背景扫描（应用层收官）。

condstate13 用 SNR/trials 账量化了 SSVEP，但从未跑过**正典分类器**。
本弧补上：标准 CCA（参考集 = sin/cos @ f 与 2f；监视集 8-17 Hz
10 个整数频点，目标 14 Hz）× 背景 σ 扫描 × 窗长扫描 × 静息假阳性
对照——应用层版本的 cs18：状态信道死了，最好的应用（SSVEP，
项目内最佳预算）也死得一样。

  预期: 干净仿真内高精度; 膝点 nV 级; 11.4 µV 下精度=随机(10%)
        且静息对照同样"随机"——应用与状态同一死法。

用法 (conda ffbm, repo root):
    python bci/condstate24/ssvep_cca.py
输出: outputs/ssvep_cca.png, outputs/ssvep_cca_stats.json
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
from state_monitor import device, load_dev, make_bg  # noqa: E402

FS = 250.0
FREQS = np.arange(8.0, 18.0, 1.0)      # 监视集 10 频点
TARGET = 14.0
SIGMAS = [0.0, 0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.16, 11.4]
DRAWS = 3
ARMS = {
    "naive": ("bci/condstate13/outputs/ssvep_naive_scalp.npy",
              3000, 17000, "tab:purple"),
    "lock": ("bci/condstate13/outputs/ssvep_lock_scalp.npy",
             3000, 17000, "tab:cyan"),
    "high-lock": ("bci/condstate13/outputs/ssvep_hl_scalp.npy",
                  3000, 17000, "tab:green"),
}
REST = ("bci/condstate3/outputs/ctl62_scalp.npy", 4000, 16500)


def cca_max_corr(X, Y):
    """第一大典范相关 (numpy QR+SVD 实现)."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    Qx, _ = np.linalg.qr(Xc)
    Qy, _ = np.linalg.qr(Yc)
    return float(np.linalg.svd(Qx.T @ Qy, compute_uv=False)[0])


def ref_matrix(t, f):
    return np.column_stack([np.sin(2 * np.pi * f * t),
                            np.cos(2 * np.pi * f * t),
                            np.sin(4 * np.pi * f * t),
                            np.cos(4 * np.pi * f * t)])


def classify_window(x, t):
    """返回 (预测频率, 全频点相关谱)."""
    rs = np.array([cca_max_corr(x, ref_matrix(t, f)) for f in FREQS])
    return float(FREQS[np.argmax(rs)]), rs


def windows(x, w_s, hop_s=0.5):
    wlen, step = int(w_s * FS), int(hop_s * FS)
    for i in range(0, len(x) - wlen + 1, step):
        yield i, x[i:i + wlen]


def run_cond(x, w_s):
    t = np.arange(int(w_s * FS)) / FS
    preds, spectra = [], []
    for _, w in windows(x, w_s):
        p, rs = classify_window(w, t)
        preds.append(p)
        spectra.append(rs)
    return np.array(preds), np.array(spectra)


def main():
    res = {"freqs": FREQS.tolist(), "target": TARGET,
           "chance": 1.0 / len(FREQS), "acc": {}, "rest_acc": {},
           "knee50": {}, "knee70": {}}
    WS = (1.0, 2.0, 4.0)
    xdev = {k: load_dev(ROOT / p, t0, t1)
            for k, (p, t0, t1, _) in ARMS.items()}
    rest = load_dev(ROOT / REST[0], REST[1], REST[2])

    for w in WS:
        for tag in ARMS:
            accs = {s: [] for s in SIGMAS}
            for sg in SIGMAS:
                hits, tot = 0, 0
                for d in range(DRAWS):
                    x = xdev[tag] + (make_bg(len(xdev[tag]), 17, sg,
                                             4100 + d) if sg > 0 else 0.0)
                    preds, _ = run_cond(x, w)
                    hits += int((preds == TARGET).sum())
                    tot += len(preds)
                accs[sg] = hits / tot
            res["acc"][f"{tag}|W{w:g}"] = {str(s): round(accs[s], 3)
                                            for s in SIGMAS}
            print(f"W={w:g}s {tag:10s} " + " ".join(
                f"{s:g}:{accs[s]:.2f}" for s in SIGMAS))
        # 静息假阳性 (naive 底 + bg, 无 14 Hz 驱动)
        for sg in (0.0, 0.032, 11.4):
            hits, tot = 0, 0
            for d in range(DRAWS):
                x = rest + (make_bg(len(rest), 17, sg, 4200 + d)
                            if sg > 0 else 0.0)
                preds, _ = run_cond(x, w)
                hits += int((preds == TARGET).sum())
                tot += len(preds)
            res["rest_acc"][f"W{w:g}|sg{sg:g}"] = round(hits / tot, 3)
        # 膝点 (naive 臂)
        acc_n = [res["acc"][f"naive|W{w:g}"][str(s)] for s in SIGMAS]
        for thr, key in ((0.5, "knee50"), (0.7, "knee70")):
            knee = None
            for i, a in enumerate(acc_n):
                if a < thr:
                    if i == 0 or SIGMAS[i - 1] == 0:
                        knee = f"<{SIGMAS[i]:g}"
                    else:
                        knee = round(float(np.sqrt(
                            SIGMAS[i - 1] * SIGMAS[i])), 4)
                    break
            res[key][f"W{w:g}"] = knee
    print("静息假阳性:", res["rest_acc"])
    print("膝点:", res["knee50"], res["knee70"])

    # ---- 图 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7),
                             constrained_layout=True)
    ax = axes[0]
    for w, c in zip(WS, ("tab:blue", "tab:orange", "tab:red")):
        ys = [res["acc"][f"naive|W{w:g}"][str(s)] for s in SIGMAS]
        ax.semilogx([max(s, 5e-4) for s in SIGMAS], ys, "o-", color=c,
                    label=f"W={w:g}s")
    ax.axhline(0.1, color="0.6", ls="--", lw=0.9)
    ax.text(1.2, 0.12, "随机 10%", fontsize=7, color="0.5")
    ax.axvline(11.4, color="k", ls=":", lw=1.1)
    ax.set_xlabel("背景 2-20 Hz RMS (µV)", fontsize=8)
    ax.set_ylabel("14 Hz 分类精度 (naive 臂)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("CCA 精度 vs 背景 × 窗长", fontsize=9)
    ax.grid(True, which="both", color="0.93")

    ax = axes[1]
    t2 = np.arange(int(2.0 * FS)) / FS
    for lab, sg, c in (("干净", 0.0, "tab:purple"),
                       ("σ=0.032 µV", 0.032, "tab:orange"),
                       ("11.4 µV", 11.4, "0.4")):
        x = xdev["naive"][:int(2.0 * FS)] \
            + (make_bg(int(2.0 * FS), 17, sg, 99) if sg > 0 else 0.0)
        _, rs = classify_window(x, t2)
        ax.plot(FREQS, rs, "o-", ms=4, color=c, label=lab)
    ax.axvline(TARGET, color="tab:purple", ls=":", lw=1)
    ax.set_xlabel("候选频率 (Hz)", fontsize=8)
    ax.set_ylabel("典范相关 ρ", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("相关谱示例（单 2 s 窗）", fontsize=9)
    ax.grid(True, color="0.93")

    ax = axes[2]
    labels, vals = [], []
    zh = {"naive": "na", "lock": "lk", "high-lock": "hl"}
    for w in WS:
        for tag in ARMS:
            labels.append(f"{zh[tag]} W={w:g}")
            vals.append(res["acc"][f"{tag}|W{w:g}"]["11.4"])
    for w in WS:
        labels.append(f"rest W={w:g}")
        vals.append(res["rest_acc"][f"W{w:g}|sg11.4"])
    cols = (["tab:purple", "tab:cyan", "tab:green"] * 3
            + ["0.6"] * 3)
    ax.bar(range(len(vals)), vals, 0.6, color=cols)
    ax.axhline(0.1, color="k", ls="--", lw=0.9)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45,
                       ha="right")
    ax.set_ylabel("11.4 µV 下精度", fontsize=8)
    ax.set_ylim(0, 0.5)
    ax.set_title("项目背景下：三臂与静息对照同为随机（~10%）",
                 fontsize=9)
    ax.grid(True, axis="y", color="0.93")

    fig.suptitle("condstate24 SSVEP 正典 CCA——应用层收官：膝点 nV 级，"
                 "11.4 µV 下与静息同为随机", fontsize=11)
    fig.savefig(OUT / "ssvep_cca.png", dpi=150)
    plt.close(fig)
    (OUT / "ssvep_cca_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'ssvep_cca.png'}")
    print(f"-> {OUT / 'ssvep_cca_stats.json'}")


if __name__ == "__main__":
    main()
