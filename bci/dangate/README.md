# bci/dangate — 误差门的电路化：多巴胺本身编码预测误差

机制 15。bci/blocking2 证明了"阻断是门的性质"——但那里的误差项
（error = R − ΣV）是在实验层用每气味 V 表算出来的标量。本研究把
这个算术**搬进电路**，让多巴胺神经元自己的放电率成为误差信号：

```
奖励（US）  ：chem 注入 DAN_err 分室 800 pA（dangateA/AB/C 条目）
预测（V）   ：KC→MBON 电位化读出（负 lr + w0 低基线）
              ——配对后 MBON 对气味反应增强
误差减法    ：MBON→DAN 的 349 条 GABA 边分裂为 MBMD 组（绝对增益、
              use_sign 启用符号列），学到的 V 抑制被奖励驱动的分室
门          ：mod = clip((r − r0)·gain, 0, 1)，r 为分室低通滤波群体
              速率，逐时间步从上一步放电读回（因果）
```

预测（Kamin 阻断的电路版本）：blocked 臂（A+ ×4 → AB+ ×6）的
V_A 增长使 AB+ 阶段 DAN 误差反应被压低 → B 学得少；control 臂
（C+ ×4 → AB+ ×6）AB+ 阶段无 V 抵消 → B 学得多。

读出：① 状态文件 w 按突触前 KC 亚型分解（w_B 阻断比）；② 每试次
`_dan_trace.npy`（t, r, mod）——mod 积分 = 电路内误差，pilot 已见
其随 V 增长单调下降（1082 → 788 ms / 5 试次）。

## 校准过程中确定的三个架构事实（pilot 1-5）

1. **符号潜伏缺口**：extra-edge 组从不把 sign 列传给突触池——中枢
   所有突触实际全兴奋运行（exp017 README 声明的共识递质符号是
   潜伏的）。MBMD 通过 edge-spec 新键 `"use_sign": True` 定向启用；
   全局启用=重新校准一切，留作独立立项。
2. **驱动力不对称**：混合符号的 MBON→DAN（349 抑/244 兴奋）在电导
   模式下净效应为**兴奋**（兴奋行 |E_rev−v|≈55 mV vs 抑制行 ≈25 mV，
   实测 r max 209→521 Hz）。最小负反馈架构：MBMD 只保留 GABA 行
   （349 条；建模选择，非连接组声明），DAN_err 分室 = 其 154 个
   突触后 DAN。
3. **MBON 二值工作点**：MBON（96 个，~348 KC 汇聚/个）在近阈值
   基电流上，输入降 6.7× 率只 342→243 Hz——V 不是渐变而是开关。
   校准结论：kcm 0.015 + w0 0.03 让 control 臂（w=w0）落在开关下、
   blocked 臂（w→1）跨过开关；MBMD 178 使满 V 大致对消 800 pA
   奖励（窗口 [174,184]，实测全范围 15→40→120→300 都全压）。

## 仪器化副产品

`--pop-rate "CLS,..."`：CEN 内按注释 class 的分群速率记录
（GPU/CPU 双路径，`_pop_rate.npz`，VALIDATION #5 的仪器）。
`--chem-amp-scale`：化学通道幅度运行时缩放（强度剂量扫描）。

## 参数（pilot 5 校准值）

| 参数 | 值 | 说明 |
|---|---|---|
| --plastic-kcm-gain | 0.015 | KC→MBON 绝对增益（MBON 二值开关定位）|
| --plastic-w0 | 0.03 | 基线权重（control 臂 V₀ 在开关下）|
| --mbon-dan-gain | 178 | GABA MBMD 增益（满 V ≈ 对消奖励）|
| --plastic-dan-gate | 150,0.05,400 | τ_ms, 门增益, 基线窗 ms |
| --plastic-lr | −1e-6 | 负=电位化 |

## 全量结果（2 臂 × 2 session × 4+6 试次，lr −2e-7）

| 量 | blocked | control | 比 |
|---|---|---|---|
| w_B 终值（KCab-m→MBON）| 0.267 | 0.296 | 1.11× |
| p2 误差积分（mod ms）| 860 | 979 | 压制 12% |
| p2 首试次误差 | ~850 | 1320 | **早期分离 55%** |

三个电路原生现象成立：

1. **误差随 V 单调下降**（blocked p1：1082→853 pilot / 全量
   p2 平台 ~860 vs 无 V 的 1950）——多巴胺分室放电率编码 R−V；
2. **分室特异性**：control 臂 p1（C+）误差纹丝不动（~1950 平线，
   V_C 不入 DAN_err 分室——KCab-s 驱动的 MBON 不投射该分室），
   p2 里 A 一出现误差立刻掉到 1320；
3. **阻断涌现但弱**（1.11×/12% vs RW 门的 4.8×）：瓶颈=MBON
   二值开关把门的地板限制在 ~41% 残余（797/1950）——B 在 40%
   门占空比下 6 次配对仍累积到 control 的 90%。晚期两臂收敛
  （control 的 A 在 p2 内也学起来了）。

**与 blocking/blocking2 合并的结论链**：窗门无阻断 → RW 误差门
4.8× 阻断 → 电路化误差门 1.11× 阻断。阻断随误差项的实现层级
（实验层算术 → 电路分室放电）**变弱但不消失**——弱的根源不是
概念而是 MBON 工作点的二值性。增强路径（新立项）：MBON 超极化
偏置（负幅值 chem 通道）使其进入分级区、或降低奖励幅度收窄 DAN
动态范围。

## 用法

```bash
conda activate ffbm
python bci/dangate/acquire.py --pilot        # blocked 臂 2+3 试次
python bci/dangate/acquire.py --lr=-2e-7 --w0 0.03 --kcm-gain 0.015 --mbmd-gain 178
python bci/dangate/analyze.py
```
