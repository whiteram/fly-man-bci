# bci/comp1：逐室误差门 vs 全局误差门

dangate2 的电路误差回路用**单一标量 mod** 门控全部 33,398 条
可塑 KC→MBON 边；真实果蝇的 MBON→DAN 反馈是**小室特异性**的
（export_data 里 bci/dangate 起就写着这条注释）。本研究落地
`--plastic-comp-reward`：**小室 = 向同一个 DAN 型反馈（MBMD GABA
行）的 MBON 集合**——奖励通道与门监视只看该型细胞，mod 变为
逐边向量、小室外全零。

机制改动（全链路）：`k_plast_adv` 标量 mod → `const float* mod`
逐边向量（标量情形广播同值——逐边运算顺序不变，全局门位级不变）；
CPU 孪生广播共享同式；kernel 缓存键随源码自动更新。

## 试点小室：PAM07

14 个 DAN 细胞 ← 14 个 MBON 反馈，4582/33398 = **13.72%** 可塑边。
门增益重标定 0.02→0.16：PAM07 细胞基线 r0≈0、逐细胞匹配的 800 pA
US 下只到 ~6.3 Hz（DAN_err 混合池 ~51 Hz），0.16 使 mod_max≈1.0
与全局臂同动态范围（pilot.log 记录了 0.02 版 mod_max 0.126 的
欠标定基线）。

## 结果（2 臂 × blocked/control × 2 会话，dangate2 协议与标定）

| 读出 | global（US→DAN_err 154 细胞） | comp（US→PAM07 14 细胞） |
|---|---|---|
| 阻断比（w_B final control/blocked） | **1.41×** | **1.00×** |
| 误差压制（mod_ms p2, control−blocked） | **33%** | 3% |
| p2 边生长占比 grown_frac | 75%/89%（block/control） | **12.1%/12.1%** |
| w_B final（blocked/control） | 0.211/0.297 | 0.0306/0.0306 |

三个结论：

1. **全局臂逐会话复现 dangate2**（1.41×/33%；r0/r1 近位级一致
   ——逐边向量核的重构在标量路径上算术等价，直接验证）。
2. **架构限制生效**：comp 臂生长封顶 12.1%（≤13.7% 小室上限；
   两会话一致）——学习只发生在小室内、且仅「小室内 ∩ 被训练
   气味驱动」的边；全局臂生长铺到 75-89% 的边（eligibility 稠密）。
3. **阻断是小室回路层面的现象，不是逐室现象**：comp 臂阻断
   1.00×/压制 3% ——小室私有 V（14 个 MBON、13.7% 边）压不动
   被强驱的小室门。关键证据：p1(A+)→p2(AB) 的 mod 积分坍缩
   （1972→150 ms）在 blocked/control 完全相同、且从 p2 第 0 试
   就在——这是复合气味聚合 MBON 驱动对 PAM07 的即时压制，
   与学习无关；A 预训练的增量压制（阻断的本体）在全局臂
   是 33%，在小室臂趋零。

机制意义：**聚合误差（全局门）产生阻断；小室私有误差支持
独立联结但不支持阻断**——本电路把文献里「阻断是全局还是
小室性的」之争变成了可测的实现层命题。全 MBON 群的 V 才能关
掉门；这也解释了为什么真实果蝇的阻断实验结果随 US 通路的
小室范围而异。

诚实边界：小室门增益按 PAM07 响应重标定（0.16，跨池速率
尺度不同）；comp 臂的 V 读出被稀释（w_B 是全池子型均值，
小室边均 dw≈0.044 被摊薄 7×）；双小室双奖励的独立联结检验
（逐室架构的真正能力展示）是直接后续。

## 用法

```bash
conda activate ffbm
python bci/comp1/acquire.py --pilot --arms comp   # 冒烟
python bci/comp1/acquire.py                       # 全量 80 trial
python bci/comp1/analyze.py
```

汇总：outputs/summary.json（逐 trial w/grown/mod/probe + 对比）。
