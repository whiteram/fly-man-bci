# Project Progress Overview (STATUS)

*(English translation — authoritative version: [../../docs/STATUS.md](../../docs/STATUS.md))*

Updated: 2026-09. Builds on HANDOFF.md (handoff/environment/data) and docs/TECHNICAL.md
(methodology details); this document answers "where are we right now".

---

## 1. Current status in one sentence

Population-spiking simulation of the fruit fly visual system (MaleCNS v1.0
connectome, optional per-region assembly) under natural visual stimuli is now
working end to end, and is projected to scalp EEG via a four-shell human-head
forward kernel, with full 3D visualization, an interactive calibration system,
and several electrode configurations; stimuli support both synthetic protocols
and **real video input** (grayscale → ommatidia-sampling mapping,
`visual_inputs.json` catalog + a synchronized dual view on the page); the
extension path for other modalities is in docs/STIMULUS.md §6.

## 2. Completed simulation stages

| Stage | Status | Output | Code/experiments |
|---|---|---|---|
| Data pipeline (connectome download/subnetwork/annotations) | ✅ | visual subnetwork + annotations + neurotransmitters + coordinates | scripts/, docs/data.md |
| ERG forward (phototransduction → R → lamina → eye-surface potential) | ✅ | benchmarked against measured ERG waveforms | exp001–002 |
| ON/OFF asymmetry, moving gratings | ✅ | direction-response baseline | exp003–004 |
| T4/T5 direction selectivity made mechanistic | ✅ | Hex-axis DS circuit (T4/T5 preferred −135°) | exp005–006, exp013 |
| In-head "fly EEG" + conductor model | ✅ | sealed two-sphere kernel SealedHeadPairField | exp007–008 |
| Lamina polarity/current-pattern corrections | ✅ | RL/LM current basis + mechanistic LMCs | exp009, exp013 |
| Human-scale thought experiment (×400 fit into the four shells) | ✅ | FourSpherePairField + adaptive scale | exp010 |
| Stimulus detection theory (signal vs background EEG) | ✅ | T7 d′≈2 (about 46 trials) | exp011 |
| Return-current constraint/conservation checks | ✅ | — | exp012 |
| Bilateral optic lobes + VPN→central brain + full CNS assembly | ✅ | optional region assembly (regions toggles); full-CNS 150,601-neuron simulation (all Traced∩has-soma cells of the three superclass families; coverage criteria in HANDOFF §4.3) | exp014–017, regions.py |
| Scalp electrode-array expansion | ✅ | four configurations: 45/64/128/EGI241 (elec_configs.json) | make_elec_configs.py |
| **Compute acceleration: numba + GPU + staged caches** | ✅ | numba kernels 2.38×; `--gpu` loop 47 min→156 s (bit-identical trajectories); circuit/kernel caches (switching electrode configs rebuilds only the kernels) | docs/ACCELERATION_PLAN.md P2/P3/P5 |
| Interactive calibration system | ✅ | 10-20 handle calibration (ni/th/yaw/roll), view-direction calibration, head-form calibration | viz/index.html |
| Visualization page | ✅ | electrode-name labels, heatmap, epoch color bar, configuration switching, preview waveforms | viz/index.html |
| **Scalp-EEG EDA (distributions / channel differences / stimulus decodability)** | ✅ | clean signal band-passed 0.5-6 Hz decoding CV R2=0.90, cascade lags 23/17 ms, spatial participation rank ~2; background-premise sweep: full human background needs ~18-65 clip-averages, sensor-noise-only reaches single-clip SNR>2 after PC spatial filtering | exp018 |
| **Stimuli: real image input** | ✅ | general visual-input module: `visual_inputs.json` catalog + grayscale video → ommatidia sampling (blur/normalization/per-eye full-frame eye-plane mapping) + synchronized dual view on the page | docs/STIMULUS.md §5 |
| **Stimuli: other modalities (olfaction etc.)** | ⏸ not started | region-toggle mechanism is ready | docs/STIMULUS.md §6 |
| 64/128/241-channel exact waveform re-export | ⏸ pending finalization | ~3-8 min per configuration with GPU+caches (first kernel build dominates; EGI241 slowest), command in USAGE §6 | — |

## 3. Hyperparameters: where they live and how to change them

| Location | Contents | How to modify |
|---|---|---|
| `src/ffbm/params.py` (SECTIONS registry) | all model hyperparameters: LIF, synapses, phototransduction, MT, operating point, VPN/central brain, human four-shell head, **stimulus protocols** (epoch table/contrast/drift speed/seeds etc.) | edit directly; `python -m ffbm.params` regenerates docs/PARAMS.md (a test keeps the doc from going stale) |
| `params.cal()` | runtime CAL dict (key names are the override entry points, resolved in order) | programmatic override: `cal = dict(fp.CAL); cal["I_R_BASE"]=...` |
| `viz/data/elec_configs.json` | electrode-configuration catalog (45/64/128/EGI241 + custom) | regenerate with `viz/make_elec_configs.py`, or manually append `kind:"custom"` entries |
| `viz/data/elec_layout_1010.json` | the 45-channel layout used for current exports | `viz/make_elec_layout.py` (consumes the JSON exported by the calibration panel) |
| Browser localStorage | working state of the page's three calibrations (calib1020 / viewDirs / headFormCalib_v3) + electrode-editing drafts | adjust directly with page sliders/handles; "Copy JSON" exports them |
| `viz/export_data.py` CLI | `--elec-layout`, `--regions` | command line |

Meaning of the status labels (third column of each row in PARAMS.md):
dataset=given directly by the dataset |
literature=value from the literature | calibrated=fitted to measured
waveforms | phenomenol.=phenomenological construction |
assumed=working assumption | numerical=algorithmic choice (validated) |
chosen=design choice | deferred=open item.

## 4. Documentation index

| Document | Contents |
|---|---|
| HANDOFF.md | handoff: environment recovery, data facts, API reference, bug history, domain survey |
| docs/TECHNICAL.md | methodology: all the math from connectome → circuit → simulation → forward kernel → background EEG |
| docs/PARAMS.md | hyperparameter registry (auto-generated, do not hand-edit) |
| docs/STIMULUS.md | stimulus protocols: implementation, parameters, real-image feasibility |
| docs/USAGE.md | this manual: page operation, calibration workflows, export process |
| docs/data.md | data-file documentation |
| viz/ELEC_CONFIGS.md | electrode-configuration file format + description of the four configurations |

## 5. Known limitations

- Stimuli are vision-only: synthetic protocols (1-D texture line scan +
  full-field flicker) and real video input are both implemented
  (visual_inputs.json catalog); before using real video as a formal
  "experimental paradigm", review the mapping-limitation notes in
  docs/STIMULUS.md (normalization-statistics conventions etc.).
- R7/R8 (color vision) are not assembled; color-vision input first requires
  adding the ommatidial R7/R8 circuitry.
- The cervical VNC participates in the dynamics only and does not project to
  the scalp (it lies outside the magnified head sphere — physically correct).
- 64/128/241 channels currently show interpolated preview waveforms; the
  exact export will be run once things are finalized (last row of §2).
- EGI 256 is the vendor's geodesic geometry (E numbering has no anatomical
  meaning) and does not fully coincide with the rule-derived 10-x cap
  (expected behavior).
