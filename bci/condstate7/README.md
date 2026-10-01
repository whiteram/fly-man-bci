# bci/condstate7：GABA 子集直测——租金载体定位到具体 MBON 类型

condstate6 判明「一切经 MBON GABA 钳，V 由子集结构决定」但子集本身
没测过（class-NaN 池不可记录）。本弧线给 export_data 加
`--pop-rate-type`（注解类型级记录，记录被动、无缓存键盐），把
精确 349 行 GABA 子集（class-MBON→class-DAN）放到显微镜下。

## 臂（seed 62，类型级记录 35 组）

| 臂 | 配方 | r_mean | mod_ms | Vsub_int* | MBON 均值 |
|---|---|---|---|---|---|
| ctl62t | cs2ctl 幼稚 | 14.89 | 595.7 | 447,912 | 41.2 |
| lock62t | cs2lock 静默锁 | 12.00 | 479.9 | 489,799 | 46.1 |
| res62t | cs3res+调度 残留态 | 9.31 | 372.5 | 483,710 | 49.2 |
| hl62t | cs4hl 高锁 | 9.50 | 379.9 | 484,451 | 47.5 |
| naive2r | cs5t+w(naive1) 学习态 | 13.75 | 549.9 | 792,406 | 59.3 |

*Vsub_int = Σ_type W_type × r_type(t) 训练窗积分（精确 349 行按
pre 类型聚合的权重 × 类型平均率；原始 w×Hz 单位）。
res62t 修复调度后与 condstate3 的 res62 逐位一致（9.31/372.5）；
naive2r 与 condstate5 naive2 逐位一致（13.75/549.9）——记录旗标
不进仿真 ✓。

## 结果

### 1. 载体定位（主结果）：静默锁的多余 MBON firing 浓集在 GABA 子集

训练窗类型级差分（lock − naive，前 15 名）：

| 类型 | ΔHz/neuron | lock 绝对 | 子集? |
|---|---|---|---|
| **MBON09** | **+77.6** | 234.6 | ✔ |
| **MBON30** | **+24.8** | 113.0 | ✔ |
| **MBON11** | **+24.5** | 241.8 | ✔ |
| ALPN | +20.9 | 21.2 | （无 ALPN→MBON 边，附带） |
| LHMB1 | +19.0 | 69.0 | （MBON 抑制物） |
| **MBON03** | **+10.8** | 85.8 | ✔ |
| **MBON31** | **+10.3** | 31.0 | ✔ |
| MBON04 | −5.3 | 17.5 | ✔（反向） |
| MBON02 | +5.0 | 38.3 | ✔ |
| MBON（class 均值） | +4.9 | 46.1 | — |

**类均值 +4.9 Hz 掩盖了子集内 +10~+78 Hz 的浓集与非子集的反向
（MBON04/32 下降）**——静默锁租金的载体是 MBON09/30/11/03/31
这批 2-4 细胞微池，它们在锁内被推到 100-240 Hz 的饱和 firing。

### 2. 深状态共享同一子集工作点

res 与 hl 的 Vsub_int 几乎相同（483.7k vs 484.5k）、mod 几乎相同
（372.5 vs 379.9）——condstate4 的「残留态=高锁同租金」在子集级
复现：两态把子集推到同一工作点。

### 3. V→mod 传递在状态级 firing 上饱和/压缩

naive2r（学习态）的子集被打到 140-610 Hz/型（MBON05 610、
MBON09 212），Vsub 792k 比 ctl 高 77%，但 mod 只从 596→550——
w×Hz 线性外推在饱和区失效；定量前向模型需要内核突触传递函数
（开放）。这同时解释 condstate5 散点的 naive2 反常（均值 59、
r 只微降）：均值与子集、子集与 V、V 与 r 三级都是非线性映射。

### 4. APL 是常数背景，非差分载体

APL 在所有臂的训练窗都处于 ~986-989 Hz 饱和（差分 +3.3）——
恒定钳制背景；DPM 不在差分前 15（小变化）。

## 语义收束

静默锁租金的完整因果链现在有名字了：**锁 → GABA 子集微池
（MBON09/30/11/03/31）饱和性增 firing → 349 行 GABA 反馈加权
和 V 上移 → DAN r 压低 → mod 缩水**。类均值 MBON 是这个链条的
糟糕读数（掩盖反向移动）；ALPN 是更上游的相关量。dstate/condstate
两线的「租金」账目从此有了突触级的记账科目。

## 诚实边界

类型平均率近似细胞级率（2-4 细胞微池内一个饱和一个静默会被
平均掉——子集浓集方向的下界）；V→mod 传递函数未建模（饱和区
定量开放）；单种子 seed 62（naive2r 借 condstate5 naive1 的 w
状态）；res62t 首跑漏 `--gain-schedule` 成全闩锁，修复后与
condstate3 逐位一致（教训：复用 cs3res 必带调度）。

## 用法

```bash
conda activate ffbm
python bci/condstate7/acquire.py     # 5 臂（类型级记录）
python bci/condstate7/analyze.py     # 子集前向 + 载体表
```

汇总：outputs/summary.json。新旗标：`--pop-rate-type TYPES`。
