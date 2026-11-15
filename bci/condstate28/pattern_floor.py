"""bci/condstate28/pattern_floor.py — 图案适应 naive 底对照（1 跑探针）。

所有 s 值（态−naive）都以**暗适应底**为参照（cs25：视觉通路强直）。
真实场景环境光可变——本弧测底的光敏感性：新目录条目 `light`
（单白帧，与 dark 单黑帧同构）+ 与 cs25 完全相同的协议（无 chem、
seed 62、20 s、pop-rate 类级+CX/lLN1_bc、层速率提取），回答：

  1. 层速率：图案适应下 R/L/MID/T4/T5 如何变化（暗：0/100/18/6.9/1.4）；
  2. 头皮底：带 RMS（δ/bb/hi）与谱指数暗/亮比——s 参照的环境
     误差棒；
  3. 结论：底变化 ×N → σ* 的环境包络（结论是否移动）。

用法 (conda ffbm, repo root):
    python bci/condstate28/pattern_floor.py    # ~3 min GPU, 幂等
数据: outputs/patternfloor_scalp.npy, patternfloor_pop.npz,
      patternfloor_layers.npz, meta.json, compare.json
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

REGIONS = "visual_bilateral,ol_rest,central_brain,olfactory"
GROUPS = "ALPN,Kenyon_Cell,MBON,DAN,CX"
TYPES = "lLN1_bc"
LAYERS = ("R", "L", "MID", "T4", "T5")
T_END = 20000.0
W0, W1 = 2000, 19000
KEYS = (GROUPS + "," + TYPES).split(",")


def run():
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "patternfloor_pop.npz").exists():
        print("[acquire] patternfloor: exists, skip", flush=True)
        return
    trial = out / "_trial_light"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "checker_static",
           "--pop-rate", GROUPS, "--pop-rate-type", TYPES,
           "--t-end", str(T_END), "--seed", "62",
           "--out", str(trial), "--gpu"]
    print("[acquire] pattern floor: static checkerboard, no chem", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit("export failed")
    shutil.copy(trial / "_debug_phi_scalp.npy", out / "patternfloor_scalp.npy")
    shutil.copy(trial / "_pop_rate.npz", out / "patternfloor_pop.npz")
    vd = json.loads((trial / "viz_data.json").read_text(encoding="utf-8"))
    np.savez(out / "patternfloor_layers.npz",
             **{k: np.asarray(v, dtype=np.float32)
                for k, v in vd["rates"].items() if k in LAYERS})
    time.sleep(1.0)
    shutil.rmtree(trial, ignore_errors=True)
    print(f"[acquire] done in {(time.time()-t0)/60:.1f} min", flush=True)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"pattern_floor": {"regions": REGIONS, "chem": None,
                                 "visual": "light (single white frame)",
                                 "dur_ms": T_END, "seed": 62,
                                 "question": "naive 底的明/暗敏感性"
                                             "（s 参照的环境误差棒）"}})
    meta_p.write_text(json.dumps(meta, indent=1))


def band_rms(x, fs=250.0, lo=2.0, hi=20.0):
    from scipy.signal import welch
    f, P = welch(x, fs=fs, nperseg=min(len(x), 512), axis=0)
    pm = np.median(P, axis=1) if P.ndim == 2 else P
    m = (f >= lo) & (f < hi)
    return float(np.sqrt(np.trapezoid(pm[m], f[m])))


def main():
    run()
    sys.path.insert(0, str(ROOT / "bci" / "condstate14"))
    from state_monitor import device
    sys.path.insert(0, str(ROOT / "bci" / "eegview"))
    from likeness_audit import spec_fit

    res = {"layers": {}, "group_rates": {}, "scalp": {}}
    # 层与群速率（暗 = cs25 参照; 亮 = 本弧）
    for cond, base in (("dark", ROOT / "bci/condstate25/outputs"),
                       ("light", HERE / "outputs")):
        lay = np.load(base / f"{ 'floor' if cond == 'dark' else 'patternfloor' }_layers.npz")
        z = np.load(base / f"{ 'floor' if cond == 'dark' else 'patternfloor' }_pop.npz")
        res["layers"][cond] = {k: round(float(
            lay[k][W0:W1].mean()), 2) for k in LAYERS}
        res["group_rates"][cond] = {k: round(float(
            z[k][W0:W1].mean()), 3) for k in KEYS if k in z}
    # 头皮带 RMS 与谱指数
    from scipy.signal import butter, sosfiltfilt
    for cond, base in (("dark", ROOT / "bci/condstate25/outputs"),
                       ("light", HERE / "outputs")):
        pref = "floor" if cond == "dark" else "patternfloor"
        e = np.load(base / f"{pref}_scalp.npy") \
            .astype(np.float64)[W0:W1] * 1e6
        x = device(e)
        cm = x.mean(axis=1)
        bb = band_rms(cm)
        dl = band_rms(cm, lo=1.0, hi=4.0)
        hi = band_rms(cm, lo=30.0, hi=80.0)
        from scipy.signal import welch as _w
        f, P = _w(cm, fs=250.0, nperseg=min(len(cm), 2048))
        m = (f >= 1) & (f <= 40)
        expo, _, peaks = spec_fit(np.hstack([[0.0], f[m]]),
                                  np.hstack([[1e-30], P[m]]))
        res["scalp"][cond] = {"bb_rms_uv": round(bb, 5),
                              "delta_rms_uv": round(dl, 5),
                              "hi_rms_uv": round(hi, 5),
                              "exponent": round(expo, 2),
                              "peak_hz": round(peaks[0][0], 1)
                              if peaks else None}
    r_bb = res["scalp"]["light"]["bb_rms_uv"] / res["scalp"]["dark"]["bb_rms_uv"]
    res["bb_ratio_light_dark"] = round(r_bb, 2)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (HERE / "outputs" / "compare.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", HERE / "outputs" / "compare.json")


if __name__ == "__main__":
    main()
