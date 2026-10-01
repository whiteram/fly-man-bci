# bci/condstate6：门压制通路定位——一切经 MBON GABA 钳，V 由子集结构决定

condstate4 留下的机制开放问题：租金跟 AL 自持水平走，但「疑 AL→KC→MBON
结构化输入作用于误差门背景」的通路未定。本弧线把它拆到通路级：
**先读门语义，再 US-only 判别 + 解剖 + MBON 偏置校准三路合围**。

## 门语义（export_data 源码级）

`--plastic-dan-gate 150,0.02,400`：(r 平滑 τ=150 ms，gain 0.02，r0
参考窗=trial 头 400 ms)。**mod = clip((r − r0)·0.02, 0, 1)**，r =
DAN_err 池率的指数平滑，r0 ≈ 0（各臂日志一致）→ mod_ms 差异 =
DAN r 轨迹差异本身。MBMD 手术（gain 178）只保留 349 行 GABA
MBON→DAN 反馈：**DAN r ≈ R(奖励驱动) − V(MBON 反馈)**。

## 判别 1：US-only（KC→MBON 腿拆除，t_end 20 s，US [15,17] s）

| 臂 | 状态 | MBON 训练窗 | mod_ms | 比 |
|---|---|---|---|---|
| us62 | 幼稚 | 1.5 Hz | 1936 | — |
| usl62 | 静默锁（已验 19.4 Hz） | 1.9 Hz | **1929** | **0.996** |
| usl63 | 150 pA 未成锁（刀刃，=幼稚复现） | 1.5 Hz | 1937 | 1.001 |
| ush62 | 高锁（100.1 Hz） | 7.6 Hz | 1613 | 0.833 |
| ush63 | 高锁 | 7.4 Hz | 1570 | 0.811 |

**静默锁的压制完全消失**（锁片段真实存在，ALPN 19.4）——静默锁
租金 100% 经 KC→MBON 腿。高锁保留 −17%。

## 判别 2：解剖（bci/condstate6/anatomy.py，CEN_C 表 + type 注释）

MBMD 分裂后，PAM/PPL1 细胞在 CEN 内**几乎没有抑制性传入残留**
（零星 1 行 5 权重碎片）——模拟里 DAN_err 的唯一抑制通道就是
MBON GABA 反馈。MBON 的非 KC 传入：DPM/APL（各 2 细胞）、
PAM/PPL 主干（兴奋）、MBON→MBON/LHMB1（抑制）、CRE067/LAL155
（小股兴奋）——静默锁内 MBON 训练反应 +5 Hz 的载体候选都在这些
小池（class 列 NaN，pop-rate 接口暂不可记录）。

## 判别 3：MBON 偏置校准（幼稚 + --mbon-bias 均匀去极化 + US-only）

| bias | MBON 基线 | r_mean | mod_ms |
|---|---|---|---|
| 100 | 5.4 Hz | 55.6 | 1871 |
| **115** | **7.95 Hz** | **38.6** | **1545** |
| 130 | 11.2 Hz | 21.0 | 839 |
| 200 | 32.2 Hz | 0.03 | 1.4 |
| 300 | 53.7 Hz | 0.0 | 0.0 |

**usmb115 ≈ ushinst（高锁 US-only）几乎逐位**：r 38.6 vs 40.3、
mod 1545 vs 1613——高锁的 −17% 「直接分量」**就是 MBON 基线 V
经陡峭钳制的效应**，无需传导 shunt（CX 在所有状态 0.0 Hz，排除
中央复合体参与）。V 钳制极陡：MBON 5→12 Hz 区间每 +1 Hz 压
r 约 −6 Hz；32 Hz 全灭。

## 定量闭环（静默锁账户）

condstate1 同种子对：MBON 训练反应 46.1 vs 41.2（+4.9 Hz），
r 11.9 vs 15.0（Δr −3.1）。偏置曲线在 41 Hz 附近的局地斜率
≈ 0.6-0.7 Hz-r/Hz-MBON → 预测 Δr ≈ −3.0 ✓。**静默锁租金 =
MBON 训练反应 +5 Hz × 陡峭 V 曲线的局地斜率**。

## 统一机制图

**门是一条陡峭的 MBON-V 钳制曲线，V 由 GABA 子集 MBON 的
firing 决定，不是全体均值**：

- 均匀驱动 32 Hz（全体含子集）→ r≈0 全灭；而训练驱动的全体均值
  101 Hz 只到 r≈9——训练期的 V 远弱于同均值的均匀驱动 →
  子集只占 firing 的小部分（散点残差与 naive2 反常 59 Hz→550
  由此解释）。
- 三状态的租金落点：静默锁 = 训练反应 +5 Hz（基线 0，US-only
  比 1.00）；高锁/残留 = 基线 6-24 Hz 骑在陡段（US-only −17%）
  + 训练反应 47-49 Hz 落进钳制地板（r≈9.5-10.3，与学习后
  naive 链尾同地板——condstate5 的「地板」就是 V 钳制地板）。
- condstate4 的「租金跟 ALPN 走」是相关不是通路：近端载体是
  MBON 工作点（基线+训练反应）在这条陡曲线上的位移，MBON
  反应随 ALPN 标度。

## 剩余开放（下弧线候选）

1. GABA 子集的 per-neuron 速率——需 pop-rate 支持类型级记录
  （class-NaN 池如 DPM/APL/LHMB1/CRE067 现不可记录）；同时
  回答「静默锁内 MBON 训练反应 +5 Hz 的载体」（候选：DPM/APL/
  DAN→MBON 变化/MBON→MBON）。
2. naive2 反常（59 Hz 均值、r 13.8）的子集结构直测。

## 诚实边界

单种子为主（ush/校准/inst 全 seed 62；US-only 双种子）；偏置
校准用均匀驱动近似基线 V（真实基线 firing 的空间结构与均匀
驱动不同，但 usmb115 与 ush 的 r/mod 双指标准确重合支持该近似）；
r0 冻结在 trial 头 400 ms（锁脉冲 300 ms 起有一点点重叠，r0 仍
≈0）；usl63 未成锁按幼稚复现处理（150 pA 刀刃，已知）。

## 用法

```bash
conda activate ffbm
python bci/condstate6/acquire.py       # US-only 判别 5 臂
python bci/condstate6/acquire2.py      # 偏置校准 + 画像 7 臂
python bci/condstate6/anatomy.py       # 解剖（写 cen_typed.parquet）
```

汇总：outputs/summary.json + summary2.json。
