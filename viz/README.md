# 3D 可视化 · Interactive 3D demo — 从神经放电到脑电

自包含的 Three.js 页面，回放一整段真实仿真（MaleCNS v1.0 全 CNS 五区域
装配，×202 放大装入人脑四球壳；10.5 s @ 1 kHz）。界面双语，英文为默认
（右上 `中文/EN` 切换）。

> **English in one line**: a self-contained Three.js page replaying a real
> full-CNS simulation — the fly network ×202 inside a human four-sphere
> head, 45-channel scalp EEG, human-video fly-vision input, bilingual UI
> (English default).

## 页面内容

- **3D 场景**（拖动旋转 / 滚轮缩放 / 右键平移，默认顶部俯视：Cz 居中，
  Fpz 朝下）：五层视觉级联 + VPN / 中央脑 / 其余视叶 / VNC 点云、三级
  突触采样线、45 导 10-10 电极与标签、四层头壳、人头形态 ghost 壳；
  左上开关逐层显示/隐藏
- **刺激视图**：人类视频 | 果蝇眼采样双画布同步回放（`demo_bounce`
  演示：灰度视频 → 小网膜采样映射）
- **头皮电极阵列**：热力图 / 经典时序波形两种显示（标题右侧
  "时序波形 / waveform traces" 勾选切换）
- **五层群体放电率** 与刺激亮度同步图表
- **交互标定**（场景左上按钮）：10-20 手柄标定（ni/th/yaw/roll，整帽
  实时重建）、电极位拖拽编辑、视角标定、人头形态校准——均可
  「复制JSON」交给管线重导出（工作流见 docs/USAGE.md）

## 使用

```bash
cd viz
python -m http.server 8613     # 任意本地服务器；file:// 会被浏览器拦截
# 打开 http://localhost:8613
```

文件：`index.html`（页面）、`data/viz_data.json`（回放数据，已入库开箱
即用）、`data/stim_frames.bin` + `data/stimuli/demo_bounce.npy`（演示
刺激）、`vendor/`（Three.js r128 本地副本，离线可用）。

## 重新生成数据

```bash
python viz/export_data.py --visual-input demo_bounce \
    --elec-layout viz/data/elec_layout_1010.json \
    --regions visual_bilateral,vpn_central,ol_rest,central_brain,vnc
```

全 CNS 导出约 40 分钟（16 核；历史基线见 docs/PERFORMANCE.md）。
`--smoke` 为 400 ms 快速冒烟测试；刺激目录见 `data/visual_inputs.json`
（自定义视频用 `make_stimulus_from_video.py` 转换，键说明见
docs/STIMULUS.md §5）。电极配置（45/64/128/EGI-241）见
`ELEC_CONFIGS.md`。

## 已知简化

- 放电以"层平均速率调制点云亮度"呈现；连接线为静态采样，透明度由
  突触前层活动驱动
- 64/128/241 导配置下波形为插值预览（精确数据需按配置重导出）
