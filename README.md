# male-fruit-fly-brain-map

基于官方 **MaleCNS v1.0** 雄性果蝇中枢神经连接组（HHMI Janelia FlyEM，CC-BY 4.0，
166,700 神经元 / 1.25 亿突触）的建模研究仓库。

当前主线：**视觉系统群体放电活动 → 细胞外场电位（眼表面 ERG 类信号）的前向生成建模**。

## 目录结构

```
├── data/                    # 数据（gitignore，可重新下载/生成）
│   ├── raw/                 #   官方原始文件（见 docs/data.md）
│   └── derived/             #   脚本产物（视觉子网络等）
├── docs/                    # 数据说明、背景笔记
├── scripts/                 # 数据管线（下载、子网络抽取）
├── src/ffbm/                # 基础设施包（安装: pip install -e .）
│   ├── data.py              #   数据加载（注释/边表/递质/胞体坐标）
│   ├── simulation.py        #   LIF 神经元群 + 指数突触（向量化）
│   └── forward.py           #   细胞外电位前向核（点源-汇对，准静态）
├── experiments/             # 每个独立实验一个目录
│   └── exp001_erg_forward/  #   闪光→光感受器→板层放电→眼表面电位
└── pyproject.toml
```

## 快速开始

```bash
pip install -e .                          # 安装基础设施包
python scripts/download_data.py           # 下载原始数据 (~1.1 GB)
python scripts/build_visual_subnetwork.py # 生成视觉子网络 (data/derived/)
python experiments/exp001_erg_forward/run.py
```

## 实验索引

| 实验 | 问题 | 状态 |
|---|---|---|
| exp001_erg_forward | 光感受器+板层群体放电能否前向生成 ERG 样眼表面电位？ | 完成 |

## 数据来源与引用

- MaleCNS v1.0: <https://male-cns.janelia.org/> — Berg et al., Cell (2026)
- 视觉回路验证基准: Shinomiya et al. 2019/2022（T4/T5 输入），Lappalainen et al., Nature 2024（连接组约束视觉网络）
