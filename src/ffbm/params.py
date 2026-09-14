"""Hyperparameter registry: every fixed number in the pipeline, with
provenance, uncertainty and sensitivity. Single source of truth for
ffbm.pipeline.CAL and ffbm.vizprep background defaults; regenerates
docs/PARAMS.md so the config and its documentation cannot drift.

Status vocabulary:
  dataset      directly from MaleCNS v1.0 (peer-reviewed, most reliable)
  literature   standard value from published work (fly or human EEG)
  calibrated   fitted by scripts/calibrate_working_point.py (round 4,
               score 0.0 = all metrics in literature windows)
  phenomenol.  hand-built to reproduce a measured waveform feature
  assumed      working assumption, untested / untestable here
  numerical    algorithmic choice, validated (convergence/cross-checks)
  deferred     known-open item; value not yet determined (see notes)

Regenerate the documentation:   python -m ffbm.params
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (kept for downstream imports)

# name: (value, unit, status, note)
SECTIONS: dict = {
    "circuit": {
        "_title": "电路（左叶五层级联）",
        "lobe_split_iters": (20, "iterations", "numerical",
                             "x 轴 2-means 收敛轮数（20 轮内必然收敛）"),
        "rhabd_offset_um": (23.5, "um", "calibrated",
                            "光感受器偶极长度（小网膜方向外推，exp002 对 ERG 标定）"),
    },
    "neuron_lif": {
        "_title": "LIF 神经元（量级取自果蝇文献，未逐细胞拟合）",
        "v_rest": (-70.0, "mV", "literature", "全层统一"),
        "v_th": (-50.0, "mV", "literature", "全层统一（阈值-静息 20 mV）"),
        "v_reset": (-70.0, "mV", "literature", "等于静息"),
        "r_in": (0.1, "GOhm", "literature",
                 "全层统一输入电阻；注意它把 pA 噪声换算成 mV 波动"),
        "_status": "literature",
        "tau_ms": {"R": (10.0, "ms"), "L": (20.0, "ms"),
                   "MID": (10.0, "ms"), "T45": (10.0, "ms")},
        "t_refrac_ms": {"R": (3.0, "ms"), "L": (2.0, "ms"),
                        "MID": (2.0, "ms"), "T45": (2.0, "ms")},
        "_note_tau": "tau/refrac 为文献量级；放电率可信度由工作点校准兜底，"
                     "而非逐细胞参数。",
    },
    "phototransduction": {
        "_title": "光转导（现象学：双级低通 + 慢适应）",
        "tau_stage_ms": (10.0, "ms", "phenomenol.",
                         "每级 10 ms，两级共 ~20 ms 感受器潜伏期"
                         "（exp002 强度系列 / exp004 闪烁融合对形）"),
        "tau_adapt_ms": (800.0, "ms", "phenomenol.", "慢亮度适应时间常数"),
        "adapt_sag": (0.30, "fraction", "phenomenol.",
                      "光平台期 30% 下垂（对 ERG 平台/瞬变比）"),
        "_note": "非对称变体（tau_fall=2.5 ms，exp003）已实现但主管线未启用——"
                 "exp012b 证明它在 exp009 符号下不改变 off 瞬变。",
    },
    "synapses": {
        "_title": "突触（RL/LM 电流基；MT 电导基）",
        "gain_rl_pa_per_syn": (12.0, "pA", "calibrated",
                               "R→L 每突触每脉冲电流峰值（符号来自数据集组胺 −1）"),
        "gain_lm_pa_per_syn": (20.0, "pA", "calibrated",
                               "L→Mi/Tm 每突触每脉冲（符号来自数据集）"),
        "tau_rl_ms": (5.0, "ms", "literature", "R→L 突触电流衰减"),
        "tau_lm_ms": (5.0, "ms", "literature", "L→Mi/Tm"),
        "g_unit_mt_ns": (0.02, "nS", "calibrated",
                         "MT 每突触峰值电导；在 v=−70 mV 时等效 1.4 pA/突触"),
        "e_rev_exc_mv": (0.0, "mV", "literature", "乙酰胆碱类"),
        "e_rev_inh_mv": (-80.0, "mV", "assumed",
                         "抑制性翻转电位（GABA/GluCl 文献范围 −70~−85 的中值；"
                         "影响抑制强度 ~±20%）"),
        "_status": "phenomenol.",
        "mt_tau_s_ms": {"Mi1": (25.0, "ms"), "Tm3": (8.0, "ms"),
                        "Mi4": (10.0, "ms"), "Mi9": (15.0, "ms"),
                        "Tm1": (20.0, "ms"), "Tm2": (8.0, "ms"),
                        "Tm4": (10.0, "ms"), "Tm9": (20.0, "ms")},
        "_note_mt": "MT 分化动力学（exp005 KINETICS.differentiated）："
                    "为方向选择性涌现设计的每类型时间常数；宽为 ±5 ms 时"
                    "DS 显著减弱（exp006）。",
    },
    "working_point": {
        "_title": "工作点（第 4 轮校准，score 0.0）",
        "i_r_base_pa": (150.0, "pA", "calibrated", "R 基线电流（黑暗期由 OU 噪声驱动 ~2.7 Hz）"),
        "i_l_base_pa": (250.0, "pA", "calibrated",
                        "LMC 暗去极化基流（exp009 的 ON/OFF 架构支柱；"
                        "机制正确版 = inverted-Cl 组胺电导，deferred）"),
        "i_mid_base_pa": (90.0, "pA", "calibrated",
                          "Mi/Tm 基流（黑暗期 19 Hz）"),
        "i_t4_base_pa": (175.0, "pA", "calibrated",
                         "T4 未建模输入背景（黑暗期 7 Hz；见 §6 结构发现）"),
        "_status": "calibrated",
        "ou_sigma_pa": {"R": (50.0, "pA"), "L": (40.0, "pA"),
                        "MID": (40.0, "pA"), "T45": (60.0, "pA")},
        "ou_tau_ms": (8.0, "ms", "assumed",
                      "OU 相关时间 ≈ 突触时间尺度；膜波动 2.7–4 mV 的目标"
                      "由此达成，tau 在 5–12 ms 内结果不敏感"),
        "_note": "全部黑暗/闪光期放电率同时落文献窗（R 2.7 / L 12.5 / "
                 "MID 19.0 / T4 7.0 / T5 3.0 Hz）。敏感性：I_*_BASE ±10 pA "
                 "约改变对应层黑暗率数 Hz（平滑梯度，非刀锋）。",
    },
    "delays": {
        "_title": "传导/突触延迟",
        "syn_delay_ms": (1.0, "ms", "literature", "化学突触固定延迟"),
        "v_axon_um_per_ms": (300.0, "um/ms", "literature",
                             "0.3 m/s，小轴突传导速度"),
        "jitter_ms": (0.5, "ms", "assumed", "±均匀抖动；影响微小（实测校准点不变）"),
    },
    "head_model": {
        "_title": "人脑四球壳 + 电极（思想实验几何）",
        "scale": (400.0, "x", "chosen",
                  "网络放大倍数（exp010 定：铺满大脑）"),
        "radii_mm": ([78.0, 80.0, 85.0, 92.0], "mm", "literature",
                     "脑/CSF/颅骨/头皮半径（成人文献值；±1 mm 只动幅值 ±1%）"),
        "sigmas_sm": ([0.33, 1.79, 0.013, 0.33], "S/m", "literature",
                      "各层电导率；±20% 的幅值包络 ×0.82–1.06"
                      "（scripts/head_sensitivity.py）"),
        "elec_rel_radius": (0.985, "of r_scalp", "chosen",
                            "电极深度（贴表面差 1.7%，可忽略）"),
        "occipital_pole_frac": (0.90, "of r_brain", "chosen",
                                "T4/T5 极贴近球壁的程度"),
        "occipital_margin_frac": (0.98, "of r_brain", "chosen",
                                  "安全边距（实际余量 1.5 mm）"),
        "n_elec": (17, "count", "chosen", "头皮电极数（10-20 系统量级）"),
        "kernel_n_terms": (60, "terms", "numerical",
                           "勒让德截断；网络深度处误差 2.5e-10"),
        "_note_deferred": "**DEFERRED（当前最大不确定项）**：回流电流几何——"
                          "三候选（t-bar 邻近 / 神经突跨区 / 均值位置）核范数差 "
                          "~400×，exp012/012b 未能判别（需先做 LMC 动力学/"
                          "inverted-Cl 电导）。绝对幅值因此未标定；相对量"
                          "（波形/拓扑/相位）不受影响。",
    },
    "background_eeg": {
        "_title": "背景人脑 EEG 与检测论",
        "alpha_hz": (10.0, "Hz", "literature", "闭眼 α 峰（后区主导）"),
        "alpha_amp_uv": (30.0, "uV", "literature", "枕区峰值量级（最不利：与信号同侧）"),
        "envelope_period_ms": (700.0, "ms", "chosen", "α 涨落包络半周期"),
        "aperiodic_uv": (3.0, "uV", "literature", "1/f 成分（谱指数 1）"),
        "sensor_uv": (1.5, "uV", "literature", "放大器白噪声"),
        "snr_band_hz": ([2.0, 20.0], "Hz", "chosen",
                        "带限 SNR 分析窗（避开 α 峰的经典 EEG 带）"),
        "dprime_target": (2.0, "sd", "chosen", "检测阈值"),
        "_note": "当前结果：宽带 842 / 带限 3019 次试验（d'=2）。频带选择"
                 "定量重要：果蝇信号大量能量在 >20 Hz。",
    },
    "open_items": {
        "_title": "已知的开放超参数（数值待定）",
        "return_current_geometry": (None, "-", "deferred",
                                    "三候选差 ~10 倍幅值；判别需 LMC 动力学闭合"),
        "lmc_e_cl_mv": (-70.0, "mV", "calibrated",
                        "已由文献反解闭合（exp013）：E_Cl≈−70mV 常规极性 + "
                        "去极化漏 v_K≈−25mV；详见 lamina_mechanistic 节"),
        "r16_phototransduction_heterogeneity": (None, "-", "deferred",
                                                "R1-6 个体差异（低优先）"),
    },
    "lamina_mechanistic": {
        "_title": "机制板层（exp013，文献反解；主管线采纳待末轮校准）",
        "e_cl_mv": (-70.0, "mV", "calibrated",
                    "组胺氯通道翻转电位：由暗 −38.4mV + 光深度 10–25mV + "
                    "释放比 ~4 双电导稳态反解（常规极性；'inverted-Cl' "
                    "说法适用于其他昆虫中间神经元，不适用 L1/2）"),
        "v_k_lmc_mv": (-25.0, "mV", "calibrated",
                       "LMC 漏通道翻转电位（去极化侧），同一反解"),
        "l_dark_target_mv": (-38.4, "mV", "literature",
                             "L1/2 暗静息（Rusanen & Weckström 2016: "
                             "−38.4 ± 3.2 mV）"),
        "light_depth_window_mv": ([10.0, 25.0], "mV", "literature",
                                  "光反应深度窗（Laughlin/Hardie 经典值）"),
        "g_unit_hist_ns": (2.5, "nS", "calibrated",
                           "每突触单位组胺电导峰值（exp013 二维扫描在"
                           "联合约束内选定；见该实验 outputs）"),
        "release_map_mv": ([-62.0, -22.0], "mV", "calibrated",
                           "v_R→释放率线性映射 [lo, hi]；暗紧张释放约 0.1–0.2"),
        "_note": "exp013 已验证机制（暗电位/超极化方向/off 瞬变可达）；"
                 "主管线采纳 = R 分级 + 组胺分级释放 + L 分级 + L→Mi/Tm "
                 "分级释放，替换 exp009 双反号代理与 I_L_BASE hack，"
                 "需第 5 轮校准与 ON/OFF 复验。",
    },
}


def cal() -> dict:
    """Assemble the pipeline CAL dict from the registry (single source)."""
    wp = SECTIONS["working_point"]
    lif = SECTIONS["neuron_lif"]
    syn = SECTIONS["synapses"]
    dly = SECTIONS["delays"]
    return {
        "I_R_BASE": wp["i_r_base_pa"][0],
        "I_L_BASE": wp["i_l_base_pa"][0],
        "I_MID_BASE": wp["i_mid_base_pa"][0],
        "I_T4_BASE": wp["i_t4_base_pa"][0],
        "GAIN_RL": syn["gain_rl_pa_per_syn"][0],
        "GAIN_LM": syn["gain_lm_pa_per_syn"][0],
        "TAU_RL": syn["tau_rl_ms"][0],
        "TAU_LM": syn["tau_lm_ms"][0],
        "G_UNIT_MT": syn["g_unit_mt_ns"][0],
        "E_REV_EXC": syn["e_rev_exc_mv"][0],
        "E_REV_INH": syn["e_rev_inh_mv"][0],
        "OU": {k: v[0] for k, v in wp["ou_sigma_pa"].items()},
        "OU_TAU_MS": wp["ou_tau_ms"][0],
        "SYN_DELAY_MS": dly["syn_delay_ms"][0],
        "V_AXON_UM_PER_MS": dly["v_axon_um_per_ms"][0],
        "DELAY_JITTER_MS": dly["jitter_ms"][0],
        "LIF": {k: (lif["tau_ms"][k][0], lif["t_refrac_ms"][k][0])
                for k in ("R", "L", "MID", "T45")},
        "RIN_GOHM": lif["r_in"][0],
        "MID_TAU_S": {k: v[0] for k, v in syn["mt_tau_s_ms"].items()},
    }


def bg_defaults() -> dict:
    b = SECTIONS["background_eeg"]
    return {"alpha_hz": b["alpha_hz"][0],
            "alpha_amp_uv": b["alpha_amp_uv"][0],
            "aperiodic_uv": b["aperiodic_uv"][0],
            "sensor_uv": b["sensor_uv"][0],
            "envelope_period_ms": b["envelope_period_ms"][0],
            "seed": 2026}


_STATUS_ZH = {"dataset": "数据集", "literature": "文献", "calibrated": "校准",
              "phenomenol.": "现象学", "assumed": "假设", "numerical": "数值",
              "chosen": "选定", "deferred": "待定"}


def to_markdown() -> str:
    lines = ["# 超参数注册表（自动生成，勿手编）", "",
             "来源：`src/ffbm/params.py`——运行 `python -m ffbm.params` 再生成。",
             "状态含义：数据集（MaleCNS 直接给出）｜文献（文献值）｜校准"
             "（第 4 轮扫描拟合）｜现象学（对测得波形特征手工构造）｜假设"
             "（工作假设）｜数值（算法选择，已验证）｜选定（设计选择）｜"
             "待定（开放项）。", ""]
    for key, sec in SECTIONS.items():
        lines.append(f"## {sec.get('_title', key)}")
        lines.append("")
        lines.append("| 参数 | 值 | 单位 | 状态 | 说明 |")
        lines.append("|---|---|---|---|---|")
        for k, v in sec.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    lines.append(f"| {k}.{k2} | {v2[0]} | {v2[1]} | "
                                 f"{_STATUS_ZH.get(sec_status(sec), '')} | |")
            else:
                val, unit, status, note = v
                lines.append(f"| {k} | {val} | {unit} | "
                             f"{_STATUS_ZH.get(status, status)} | {note} |")
        for k, v in sec.items():
            if k.startswith("_note") or k.startswith("_title"):
                if k != "_title":
                    lines.append("")
                    lines.append(f"> {v}")
        lines.append("")
    return "\n".join(lines)


def sec_status(sec) -> str:
    return sec.get("_status", "literature")


def main():
    from pathlib import Path
    out = Path(__file__).resolve().parents[2] / "docs" / "PARAMS.md"
    out.write_text(to_markdown(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
