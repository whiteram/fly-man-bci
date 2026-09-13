# exp013：机制板层——真实生物物理替换 exp009 代理

## 文献依据（2026-09-13 核实）

- L1/2 暗静息 **−38.4 ± 3.2 mV**；判据"静息低于 −30mV 且对光超极化"
  （[Rusanen & Weckström 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4869513/)；
  [JEB 2018](https://journals.biologists.com/jeb/article/221/12/jeb179085/)）
- 光反应 = **分级超极化 10–25 mV**，经组胺门控氯通道
  （[Hardie 1987/89](https://europepmc.org/article/pmc/3694283)；
  [Skingsley & Laughlin 1994](https://link.springer.com/article/10.1007/BF01021581)）
- **物理反解（重要纠正）**：光使释放增加、LMC 却超极化 ⇒ E_Cl 必在
  暗电位**之下**。用双电导稳态解（漏通道 v_K + 氯通道 E_Cl）拟合
  暗点位与光深度、暗/光释放比 ~4：**E_Cl ≈ −70 mV（常规极性！），
  v_K ≈ −25 mV**。"inverted chloride" 传说适用于其他昆虫中间神经元，
  不适用于 L1/L2 板层。

## 模型

R1-6 与 L1/2 均为分级单元（LIF v_th=∞）；R 释放率 r∈[0,1] 映射其膜
电位；`GradedSynapsePool` 承载组胺氯电导（逐突触 t-bar/PSD 几何）；
g_unit 一维扫描自动定暗电位。

## 结果（两次运行，outputs/summary.json 为 v2）

| 配置 | L1/2 暗 | 光超极化 | tbar off/on | neurite off/on | meanpos off/on |
|---|---|---|---|---|---|
| v1（g=0.3，对比度 6×） | −30.2 | 6.8 mV | 0.0 | **0.34 ✓** | 0.0 |
| v2（g=1.6，对比度 4.5×） | −37.2 ✓ | 4.6 mV | 0.0 | 0.085 | 0.0 |

**结论（诚实）**：
1. 机制板层成立：暗电位与超极化方向均复现，无任何基流 hack；
2. **off 瞬变首次变得可达**——代理板层（exp012/012b）在任何配置下
   off/on 恒为 0；机制板层 v1 中 neurite 回流几何通过该锚点
   （0.34 ∈ [0.3,1]），且 tbar/meanpos 仍为 0——**判别信号指向
   neurite 几何**；
3. 但判别**参数敏感**：加深暗钳制（v2）压低 off 幅度。稳健判别需要
   在 (g_unit, 释放映射) 二维参数区间上扫描 off/on 的等值线，并让
   LMC 分数锚点在 L/R 反号几何下重新定义（>1 的"分数"是度量伪影）。
4. 绝对幅值标定仍未闭合；方向性证据 = neurite 几何。

## 下一步

二维参数扫描（g_unit × 释放映射）+ 锚点度量修正；若 neurite 在
暗 −38.4±3、光深度 10–25mV 的联合约束内稳健通过 off/on，则回流
几何判定为 neurite，绝对幅值按其核范数标定（≈ 均值位置的 1.7×）。
