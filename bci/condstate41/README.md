# bci/condstate41：残留态寿命的 seed-63 复制——诚实修正：τ 下限的种子散布 4×

## 问题

cs20 的 τ_sig(resid)≥2948 s 一直单种子（关闭表"qlock/resid 仍为
单种子下限"）。本弧复制：cs3resnt + 调度 seed **63**，t_end 300 s
（57.6 min GPU；配方逐字同 condstate3/20；**确定性 diff=0** vs
resnt63）。

## 结果（outputs/）

| 量 | seed 62（cs20）| **seed 63（本弧）** |
|---|---|---|
| τ_sig(3σ/联合漂移) | ≥2948 s | **741 s** |
| 联合漂移速度 | 0.00102 SD/s | **0.00405 SD/s（4×）** |
| w_A 300 s 漂移 | +0.0029 | **+0.0009（3× 小）** |
| DAN 调制积分 | 361.8 ms | **370.4 ms（几乎相同）** |
| 监测器保持 | 97.3% | **94.3%**（漂移 17 窗→qlock/hlock 各半）|
| σ*(resid) | 2.30 µV | **1.15 µV** |

**τ 下限存在 4× 的种子散布**：62 的近地平线（0.001 SD/s）不是
普适的——63 的残留签名以 4× 速率漂移（τ 下限 741 s，虽仍 ≥2×
hlock 的 284-336 s）。机制注脚：DAN 积分两种子几乎相同（370 ms）
而 w 漂移差 3×、特征漂移差 4×——**慢写时钟在残留态是种子敏感
的**（与 hlock 的近确定性相反；hlock 写通道结构简单，残留态的
KC/MBON 异质回路对实现更敏感）。

## 总账修正

- **σ*(resid) = 1.15-2.30 µV（种子带）**，对 11.4 µV 缺口
  **5-10×**（原单种子记 5×）——封死结论不动（≥5×），但最强态
  余量必须记为带；
- 寿命谱更新：hlock 284/336 s（双种子，唯一有界）/ qlock ≥2988 s
  （单种子）/ **resid 741~≥2948 s（双种子带）**——排序不变
  （hlock ≪ qlock ≈ resid）。

## 诚实边界

- 单 trial/种子；τ 为斜率外推下限（741 在 300 s 记录的 4σ 外推
  内，较可靠；≥2948 是地平线外下限）；
- 63 的 4× 漂移可能部分来自窗口斜率估计噪声（296 窗），但量级
  差异超出抽样散布。

## 复现

```bash
# conda ffbm, repo root
python viz/export_data.py --regions visual_bilateral,ol_rest,central_brain,olfactory \
  --visual-input dark --chem-input cs3resnt --pop-rate ALPN,Kenyon_Cell,MBON,DAN \
  --plastic-mb --plastic-lr=-2e-7 --plastic-w0 0.03 --plastic-kcm-gain 0.015 \
  --mbon-dan-gain 178 --plastic-dan-gate 150,0.02,400 \
  --t-end 300000 --seed 63 --out <trial> --gpu --gain-schedule 6000:CEN_C=0.5
python bci/condstate41/stability_res63.py
```
