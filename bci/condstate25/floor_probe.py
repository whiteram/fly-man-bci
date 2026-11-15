"""bci/condstate25/floor_probe.py — naive 静息底的发生器定位（探针跑）。

每个缺口数字都以 naive 底为参照（s = 态−naive），但"暗态平滑底
是谁在放电"从未直接测过（eegview QA 只证明四个记录池静默；
cs23 的 visual 代理因外周缺 soma 位置失败）。本弧跑一次暗态
无 chem 20 s（seed 62），用类型级 pop-rate 记录 12 个候选群
（视觉链 R1-R6/L1/L2/L3/Mi1/Tm3/T4a/T5a + 嗅觉 ORN_DA1 +
AL 局部 lLN1_bc）+ 类级参照（ALPN/KC/MBON/DAN），回答：

  1. 暗态下哪些群活跃、速率多少 → 底的组成；
  2. 与 ctl62（cs3ctl 前 17 s 纯暗段）的确定性对照（无 chem
     是否改变 RNG 流）。

用法 (conda ffbm, repo root):
    python bci/condstate25/floor_probe.py      # ~3 min GPU, 幂等
数据: outputs/floor_scalp.npy, floor_pop.npz, meta.json
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
GROUPS = "ALPN,Kenyon_Cell,MBON,DAN,CX"      # 类级（CEN 空间可解析）
TYPES = "lLN1_bc"                            # 类型级仅 CEN(cb_*) 可解析;
                                             # 外周/视叶(ol_intrinsic)不可录
LAYERS = ("R", "L", "MID", "T4", "T5")       # 底模四层速率在 viz_data.rates
T_END = 20000.0
KEYS = (GROUPS + "," + TYPES).split(",")


def run():
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "floor_pop.npz"
    if dst.exists() and (out / "floor_scalp.npy").exists():
        print("[acquire] floor: exists, skip", flush=True)
        return
    trial = out / "_trial_floor"
    cmd = [sys.executable, str(ROOT / "viz" / "export_data.py"),
           "--regions", REGIONS, "--visual-input", "dark",
           "--pop-rate", GROUPS, "--pop-rate-type", TYPES,
           "--t-end", str(T_END), "--seed", "62",
           "--out", str(trial), "--gpu"]
    print(f"[acquire] floor: dark no-chem {T_END/1000:.0f} s, "
          f"types={TYPES}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit("export failed")
    shutil.copy(trial / "_debug_phi_scalp.npy", out / "floor_scalp.npy")
    shutil.copy(trial / "_pop_rate.npz", dst)
    # 底模层速率 (R/L/MID/T4/T5) 只在 viz_data.json 里
    vd = json.loads((trial / "viz_data.json").read_text(encoding="utf-8"))
    np.savez(out / "floor_layers.npz",
             **{k: np.asarray(v, dtype=np.float32)
                for k, v in vd["rates"].items() if k in LAYERS})
    time.sleep(1.0)
    shutil.rmtree(trial, ignore_errors=True)
    print(f"[acquire] done in {(time.time()-t0)/60:.1f} min", flush=True)
    # meta load-merge-rewrite
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) \
        if meta_p.exists() else {}
    meta.update({"floor_probe": {"regions": REGIONS, "chem": None,
                                 "dur_ms": T_END, "seed": 62,
                                 "pop_groups": GROUPS,
                                 "pop_types": TYPES,
                                 "question": "暗态 naive 底的发生器"
                                             "组成（速率账）"}})
    meta_p.write_text(json.dumps(meta, indent=1))


def main():
    run()
    z = np.load(HERE / "outputs" / "floor_pop.npz")
    print("\n暗态各群速率 (Hz/neuron, [4,17] s 窗):")
    rows = {}
    for k in KEYS:
        if k in z:
            r = float(z[k][4000:17000].mean())
            rows[k] = round(r, 3)
            print(f"  {k:10s} {r:7.3f}")
    lay = np.load(HERE / "outputs" / "floor_layers.npz")
    print("\n底模层速率 (R/L=释放率%, 其余 Hz):")
    layers = {}
    for k in LAYERS:
        if k in lay:
            r = float(lay[k][4000:17000].mean())
            layers[k] = round(r, 2)
            print(f"  {k:4s} {r:8.2f}")
    # 确定性对照: vs ctl62 前 17 s (cs3ctl 训练在 17 s 后)
    a = np.load(HERE / "outputs" / "floor_scalp.npy")
    b = np.load(ROOT / "bci/condstate3/outputs/ctl62_scalp.npy")
    n = min(17000, len(a), len(b))
    det = float(np.abs(a[:n].astype(np.float64)
                       - b[:n].astype(np.float64)).max())
    print(f"\n确定性对照: max|floor - ctl62| 前 {n} ms = {det:.2e} V "
          f"({'逐位同' if det == 0 else '不同(无 chem 改变 RNG 流?)'})")
    (HERE / "outputs" / "floor_rates.json").write_text(
        json.dumps({"group_rates_hz": rows, "layer_rates": layers,
                    "det_diff_V": det,
                    "window_ms": [4000, 17000]},
                   ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
