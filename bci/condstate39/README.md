# bci/condstate39：残留臂 v-trace——三层源竞争图景的第三点实测

## 问题

cs38 建立"头皮谱指数=占优源指纹"三层图景时，naive/hlock 两点有
v-trace 实测，latch 端（+0.06）只有 audit 旧值。本弧补第三点：
20 s cs3resnt + 调度 + 完整 dangate2 旗标（与 resnt62 差
**1.4e-12 V ≈ 逐位**）+ `--v-trace 128`。

## 结果（outputs/）

| v 谱指数 | naive | hlock | **resid** |
|---|---|---|---|
| MID | −0.28 | −0.24 | **−0.24** |
| T45 | −1.10 | −1.05 | **−1.07** |
| OLR | −1.47 | −1.38 | **−1.47** |
| 头皮 | −0.93 | −0.49 | **+0.05**（复现 audit +0.06）|

**第三点落位，图景封口**：残留臂的 v 谱三池全部与 hlock 持平
（视觉通路完全没被闩锁改变——它本来就是背景），而头皮谱爬到
+0.05——**头皮与 v 层在三态间完全解耦**：谱形的爬升（−0.93→
−0.49→+0.05）纯粹是尖峰梳的逐级增强（AL 引擎 20→100 Hz →
KC/MBON 招募），v 层恒定在背景。

三层图景现在是**三点实测**：源竞争的"开关"在尖峰层，不在 v 层。

## 语义

- cs38 的叙事从"两点+推断"升级为**三点全实测**；
- 对 likeness：状态间的谱形差异（人脑相似度的"不像"主项）
  100% 由尖峰源承载——若未来做 likeness 优化，靶点必须是
  尖峰源（同步化/平滑），动 v 层（背景）不会改变状态谱形；
- 压缩件按 pitfall #26 规则抗混叠重制（29 MB）。

## 诚实边界

- 单窗单种子；残留窗 [8,19] s（闩锁弛豫后）；
- ALPN/KC/MBON 的 v 不在 `_mask_offsets`（中枢可塑群无 v 键，
  三层图景的"尖峰层"证词仍来自头皮侧+audit）。

## 复现

```bash
# conda ffbm, repo root
python viz/export_data.py --regions visual_bilateral,ol_rest,central_brain,olfactory \
  --visual-input dark --chem-input cs3resnt --pop-rate ALPN,Kenyon_Cell,MBON,DAN \
  --plastic-mb --plastic-lr=-2e-7 --plastic-w0 0.03 --plastic-kcm-gain 0.015 \
  --mbon-dan-gain 178 --plastic-dan-gate 150,0.02,400 \
  --t-end 20000 --seed 62 --out <trial> --gpu --v-trace 128 \
  --gain-schedule 6000:CEN_C=0.5
```
