# bci/condstate43：静默锁 τ_sig 的 seed-63 真复制（300 pA 成锁剂量）——最后一个单种子寿命关闭

## 问题

cs42 证明 150 pA（62 校准剂量）在 seed 63 上不点火后，本弧用
condstate2 记载的 63 成锁剂量（DA1@**300 pA**，新 chem 条目
cs42locknt300 = cs2lock300 无训练版）做 τ_sig(qlock) 的真复制：
300 s、dangate2 全套旗标（57.6 min GPU）。

## 结果（outputs/stability_qlock300_stats.json）

| 量 | seed 62@150（cs21）| **seed 63@300（本弧）** |
|---|---|---|
| 确定性 vs lock63 | 逐位 0（前 15 s）| **逐位 0（前 15 s）** |
| 点火 | 脉冲峰 29.4 Hz | **脉冲峰 35.3 Hz，自持 47.1 Hz** |
| ALPN 300 s 均值 | 19.724 Hz | **19.718 Hz（几乎重合）** |
| ALPN 首/中/尾 | 19.9/19.6/19.7 Hz | **19.8/20.1/19.7 Hz** |
| τ_sig（3σ/联合漂移）| ≥2988 s | **1452 s（0.00207 SD/s）** |
| 监测器保持（62 折）| 74.7% | **92.2%** |
| w_A 漂移 / DAN 积分 | −0.0 / 0.9 ms | **−0.0 / 0.1 ms** |
| σ*(qlock) | 0.210 µV | **0.146 µV** |

**判定**：cs21 全部结论经 seed-63 成锁剂量复现——自持引擎
（~20 Hz 恒定 300 s）、近零可塑写（DAN 积分 0.1 ms、w 不动）、
监测器可用。**τ 下限存在 2× 种子散布**（1452 vs ≥2988 s，联合
漂移 0.00207 vs 0.001 SD/s）——远小于 resid 的 4×（cs41），
排序不变（hlock 284/336 ≪ qlock 1452~≥2988 ≈ resid 741~≥2948 s）。

## 跨状态种子趋势（三点合成的次级观察）

seed 63 的签名漂移在两个可塑态上**一致地快于** 62：qlock
0.001→0.00207（2×）、resid 0.00102→0.00405（4×）；而 hlock
（写通道最简单）两种子近确定（284/336 s，DAN 积分差 0.1%）。
**慢写时钟的种子敏感度随回路异质性增长**（hlock ≈确定性 →
qlock 2× → resid 4×）——与 cs41 的机制结论同向。

## 总账更新

- **寿命谱收官**：qlock τ_sig = **1452~≥2988 s（双种子带）**——
  关闭表最后一个单种子寿命关闭；全部关键寿命数字现为双种子或
  定量包络（hlock 284/336、qlock 1452/≥2988、resid 741/≥2948、
  naive ≥1985/1399）；
- **σ*(qlock) = 0.146-0.210 µV（双种子带）**，对 11.4 µV 缺口
  **54-78×**（qlock 本就是最弱态——封死结论再加固）；
- 剂量-响应第四点落位（0.1 ms→1452 s，近零写角落双种子）。

## 诚实边界

- 单 trial/种子；1452 在 4σ 外推内较可靠，≥2988 是地平线外
  下限；诱导剂量种子特定（62@150 / 63@300，阈值未扫——
  cs42 已注）；
- 监测器 23 窗漂移（18 naive + 2 hlock + 3 resid）——92.2% 保持
  但低于 hlock 臂的 94-97%。

## 复现

```bash
# conda ffbm, repo root（绝对路径启动——后台 shell 不继承 env, pitfall #28）
python viz/export_data.py --regions visual_bilateral,ol_rest,central_brain,olfactory \
  --visual-input dark --chem-input cs42locknt300 --pop-rate ALPN,Kenyon_Cell,MBON,DAN \
  --plastic-mb --plastic-lr=-2e-7 --plastic-w0 0.03 --plastic-kcm-gain 0.015 \
  --mbon-dan-gain 178 --plastic-dan-gate 150,0.02,400 \
  --t-end 300000 --seed 63 --out <trial> --gpu --plastic-state-out <states>/qll63_300.npz
python bci/condstate43/stability_qlock300.py
```
