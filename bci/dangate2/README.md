# bci/dangate2 — 电路误差门 + MBON 余量恢复 (mechanism 15 续)

前作 [bci/dangate](../dangate/README.md) 达成电路级 Kamin 阻断但很弱
（w_B 比值 1.11×、误差抑制 12%），瓶颈诊断：**未训练的基线 MBON 放电
（~43 Hz）已经把 MBMD 抑制推到饱和** —— blocked 与 control 的 DAN 轨迹
几乎相同（8.5 vs 8.8 Hz，mod 地板 0.42），V 再增长也压不出差异。
门在 V 分化之前就饱和了。

## 机制修复：`--mbon-bias`

一条紧张性超极化 chem 通道（group `MBON*`，96 细胞，全程脉冲，
pA 直通符号）把未训练 MBON 压到 w-膝点以下，**让膝点本身成为
trained/untrained 的判别器**。运行时幅度、不进任何缓存键（与
`--chem-amp-scale` 同类），实现约 20 行。

## 校准（outputs/scan_summary.json，静态 w，seed 1011）

传递链 w → MBON 率 → DAN_err 率 r：

| bias (pA) | w=0.03 | w=0.2 | r(w=0.03) | r(w=0.2..1.0) | r 对比 |
|---|---|---|---|---|---|
| 0   | 43.3 Hz | 160.7 Hz | 15.5 Hz | 9.4-10 Hz | 1.6×（饱和） |
| −300 | 19.6 Hz | 120.6 Hz | **51.2 Hz** | **10-11.5 Hz** | **~5×** |
| −600 | 18.8 Hz | — | （曲线整体右推，w≥1.0 才关门） | | |

- bias 0：未训练点就在膝上，r 全程 9-15 Hz → mod 0.42-0.5，无余量。
- bias −300：未训练 r≈51（门全开），训练端（w≥0.3）r≤11.5。
- 剂量 >150 pA 前平坦（兴奋驱动远超偏置），−600 把整条传递曲线
  右推、丢掉训练端 —— **−300 是甜点**。
- 门增益相应 0.05 → 0.02：未训练 r≈51 → mod 饱和 1.0，训练端
  r≈10 → mod≈0.2（每步学习率对比 5:1，旧配置 1.05:1）。

参数：lr −2e-7，w0 0.03，kcm 0.015，mbmd 178，gate `150,0.02,400`，
bias −300，seed blocked 1042+ / control 1092+。

## 试点（blocked 臂，2+3 trial）

p1 内 **电路内误差递减**首次清晰出现：mod 积分 1808 → 1117 ms（两个
A+ trial 内）；p2 的 w_B 增速 +0.032/trial。上轮（bias 0）p1 内 mod
几乎平坦 —— 余量恢复的直接证据。

## 全跑结果（2 臂 × 2 会话 × 4+6 trial）

| 指标 | dangate（bias 0） | **dangate2（bias −300）** |
|---|---|---|
| w_B 终值 blocked / control | 0.267 / 0.296 | **0.211 / 0.297** |
| 阻断比 w_B | 1.11× | **1.41×** |
| 早相分离（首个 AB+ trial） | ~1.5× | **1.8×**（0.061 vs 0.111） |
| p2 误差抑制（mod_ms 655 vs 974） | 12% | **33%** |
| p1 电路内误差递减 | 平坦（~1950） | **1808→792 ms**（4 trial） |
| control p2 误差递减 | 平坦 | **1778→646 ms**（6 trial） |

结构完整涌现：误差随 V 增长逐 trial 收缩（两臂都是），blocked p2
门稳定在 mod≈0.32 —— w_A 平衡在 ~0.24（增长 = 向 w0 衰减），这是
**带遗忘项的 Rescorla-Wagner 固定点**，全部在电路内。

诚实记录：control p1 的 C（KCab-s）权重不增长（mod 平坦 ~1930）——
沿袭 dangate 的 compartments 发现（KCab-s 驱动的 MBON 不投射
DAN_err），control 实际是「全新 AB+」而非「C 预训练 AB+」；经典
Kamin 设计仍然成立（blocked 有 V_A、control 无任何先验 V）。
剩余与实验层 4.8× 的差距：单室误差求和 vs 真实果蝇逐室架构、
w 平衡点由恢复项设定。

## 结论链（三级→四级）

| 门实现 | 阻断对比 | 出处 |
|---|---|---|
| 固定窗口（无误差） | 0×（无阻断） | bci/blocking |
| 实验层 Rescorla-Wagner | 4.8× | bci/blocking2 |
| 电路门，bias 0 | 1.11×（12% 误差抑制） | bci/dangate |
| 电路门 + 余量恢复 | **1.41×（33% 误差抑制 + 电路内 RW 递减）** | bci/dangate2（本目录） |

诚实框架：偏置本身是建模选择（真实果蝇 MBON 的分级性来自其自身
生物物理，此处用外加电流代理）；它做的事是把「饱和的抑制传递」
恢复成「有动态范围的传递」，使已存在的 MBMD↔DAN 电路能够表达
误差。奖励仍为 chem 注入 800 pA（非真实强化神经元）。

复现：

```bash
python bci/dangate2/acquire.py --pilot   # ~7 min
python bci/dangate2/acquire.py           # ~50 min
python bci/dangate2/analyze.py
```
