# bci/touch：体表触觉侧别解码（2 类）

新区域 `touch`（exp022，`experiments/exp022_touch_haltere/circuit7.py`）：
**TC 体表触觉群 2,558**（class `mechanosensory_tactile`，SNta* 类型，
rootSide L 1,264 / R 1,294，全部有 PreSyn 末梢）。连边：TC_C → 中央
**1,346 对 / 17,192 突触**（forward=True，头皮可见偶极子）+ TC_V →
VNC 反射弧 ~83k 对 / 969k 突触（forward=False）+ 本地。

每试次 3.6 s：同侧全体表触觉群 300 pA 触压脉冲 900 ms（@300 ms，
总驱动 ≈ 嗅觉全量标定），类别 = 触压侧别，6 重复。

## 结果（12 试次，17 通道空间模式 LOO + 置换）

| 解码 | 准确率 | perm-p | chance |
|---|---|---|---|
| 触压侧别 L/R | **100%** | **0.0035** | 50%（perm95 83%）|

**解读**：17k 突触的 TC_C + 大群均衡驱动足以把触压侧别印到头皮
（对比：8.4k 的 GRN_C 同样幅度下不可见——结构可见性边界落在
两者之间）。这是第五个被验证的偏侧躯体感觉通道（本体/听觉/触觉
侧别 + 嗅觉/视觉身份）。

**平衡棒注记**：计划中的 haltere 构建器被数据否决——205 个平衡棒
细胞中 201 个已在 `proprioception` 区域内（它们按 class 是
campaniform/chordotonal 感器），再加区域就是重复计数；平衡棒的
陀螺反馈已随 PRO 仿真。

## 用法

```bash
conda activate ffbm
python bci/touch/acquire.py          # 2 类 × 6（与 gustatory 共区域集）
python bci/touch/analyze.py
```
