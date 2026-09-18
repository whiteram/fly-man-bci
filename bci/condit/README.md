# bci/condit：嗅觉条件化——模型的第一个学习实验

机制③落地：**DAN 门控的 KC→MBON 突触可塑性**（果蝇三因子学习
规则的模型实现）。这是整个项目第一次"经历改变后续响应"的实验。

## 机制（进主管线，opt-in）

- `--plastic-mb`：运行时把 **KC→MBON（33,398 对 / 40 万突触）**
  从 CEN_C 复归表拆出为可塑边组 KCM（盐化 kernel key，重建一次
  后缓存；电路缓存不动）；
- 每条边持有 eligibility（突触前 KC 尖峰积累，tau 400 ms）与
  weight scale；`--plastic-window a,b` 内 mod=1（奖励窗 = PAM
  多巴胺驱动的 proxy）时 **w ← max(w×(1−lr·mod·elig), 0.02)**
  ——抑郁型三因子规则（KC 活动 × 强化 × 权重更新），CPU/GPU 双
  路径，plast-off 位级不变（回归测试通过）。

## 范式（驱动条件化，8 s，learn lr=3e-6 vs nolr lr=0，各 4 runs）

| 时段 | 内容 |
|---|---|
| 训练 0.5–2.5 s | KC 主簇双侧 1600 pA（气味 KC 表征注入）+ PAM 双侧 800 pA（奖励），学习窗开 |
| 探针 4.2–4.7 s | 同 KC 驱动 500 ms，读出通路级变化 |

## 结果（8 runs）

| 指标 | learn | nolr | delta |
|---|---|---|---|
| 探针净响应 | **+0.2511 µV** | +0.1221 µV | **+0.129 µV（+105%），perm-p=0.026** |

**学习签名头皮可见且显著**。方向为正是机制一致的：KC→MBON 被
抑郁 → MBON（果蝇 MBON 相当部分抑制性）对下游的去抑制 → 净响应
升高——与果蝇嗅觉学习的 MBON 去抑制机制同构。

## 诊断与修复记录（三步排查，全部保留为文档）

1. 首版头皮无差异 → 加 w_scale/elig 打印：**学习未发生**（w 全 1）；
2. KC 直驱验证发现 **PN→KC 通路约束**：AL 回路在 CEN_C 内、被
   chem 工作点（0.002）压到极低，气味身份到不了 KC——协议改为
   驱动条件化（KC 表征注入，文档化的降级）；
3. GPU 路径 **trial.run 漏传 mod_fn**（静默 None）修复后学习
   打通；lr 0.0015 过学习（全边触底）→ 按 elig 稳态标定 3e-6
   （2 s 窗 ~50% 抑郁）。

## 限制与后续

- 身份特异性降级为"被驱动子集特异性"（通路约束所致）；真气味
  条件化需要 AL 子回路增益独立化（机制方向）；
- 学习跨试次不持久（每试次重置）——试次间记忆需要权重状态外
  持久化（管线设计变更，列为后续）。

## 用法

```bash
conda activate ffbm
python bci/condit/acquire.py        # learn + nolr 各 4（~9 min/试次）
python bci/condit/analyze.py
```
