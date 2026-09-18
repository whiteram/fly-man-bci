# bci/looming：威胁/逃跑检测（3 类）

经典 looming（碰撞逼近）范式：膨胀暗盘驱动 LC4/LPLC→DN 巨纤维逃跑
通路——连接组里**最大的具名 DN 驱动类型**（LC4+LPLC4/2/1 合计
~82k 突触直连 DN，dn_probe.py）。视频输入走标准级联，区域集
visual_bilateral,ol_rest,central_brain（DN 在 CEN，forward=True）。

## 刺激（make_stimulus.py，3 s @ 60 fps 全场）

| 类别 | 描述 |
|---|---|
| loom | 暗盘膨胀（300–2300 ms 内 2→60 px，然后保持） |
| recede | 暗盘缩小（对照） |
| static | 恒定 30 px 暗盘（亮度匹配对照） |

## 结果（18 试次，17 通道空间模式 LOO + 置换）

| 解码 | 准确率 | perm-p | chance |
|---|---|---|---|
| 3 类（loom/recede/static） | **72%** | **0.0005** | 33%（perm95 56%）|
| 运动 vs 静止（loom+recede vs static） | **100%** | — | 50% |
| loom vs recede（膨胀方向） | 58% | — | 50% |

时间过程：loom 与 recede 几乎相同（~0.93 µV），static 显著更低
（0.67→0.75 µV）。

**诚实解读**：头皮上可分的是"**视野变化 vs 静止**"（100%），而
**膨胀方向不外显**（loom≈recede）。T4/T5/LPLC 的方向选择性在
视网膜级联里有，但在这个工作点上没存活到 17 导头皮 rms 读出——
逃跑相关的"有威胁事件检测"成立，"逼近方向解码"不成立。若要方向
外显，需要（i）更强的 LPLC→DN 复归增益或（ii）偏侧双目刺激设计，
列为后续。

## 用法

```bash
conda activate ffbm
python bci/looming/make_stimulus.py --all
python bci/looming/acquire.py        # 3 类 × 6（~70 s/试次）
python bci/looming/analyze.py
```
