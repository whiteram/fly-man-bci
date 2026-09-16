<div align="center">

<img src="assets/banner.svg" alt="Fly-Man BCI banner" width="100%"/>

# 🪰 Fly-Man BCI · 蝇人脑机接口

**A fly's brain network in a human body — and the BCI he urgently needs.**

*This repository builds the foundation: the fly-man's EEG generation model —
external stimulus → brain network activity → EEG recorded at the scalp.*

**English** · [简体中文](README.zh-CN.md)

▶ **[Live demo](https://whiteram.github.io/fly-man-bci/viz/)** — the
interactive 3D replay runs straight from this repository
(enable GitHub Pages → main /(root) if the link is not live yet; or serve
`viz/` locally).

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Data](https://img.shields.io/badge/data-MaleCNS%20v1.0-38bdf8)
![100%25 simulation](https://img.shields.io/badge/no_flies_harmed-100%25_simulation-8b5cf6)

</div>

---

## The setting

A fly-man: in a teleporter accident (you may know the movie), his brain's neural network
was transformed into that of a fruit fly — the only animal whose brain exists as a
complete wiring diagram — while his body stayed human. His senses all work; but a fly's
motor system driving a human body barely does. He urgently needs a brain–computer
interface.

To build him a BCI, we need something first: **a model of his EEG** — what his brain
activity looks like as signals on the scalp. That is this project.

## The model: his EEG generation model

> **external stimulus → fruit-fly brain network → scalp EEG**

A human head model with a fruit-fly brain network inside. Flash his eyes, watch the
network fire, compute the extracellular currents, read the EEG off his scalp.

<img src="assets/pipeline.svg" alt="pipeline" width="100%"/>

| Part | What it is |
|---|---|
| **Brain** | The real fly connectome — [MaleCNS v1.0](https://male-cns.janelia.org/) (HHMI Janelia, CC-BY 4.0; Berg et al., *Cell* 2026), 166,700 neurons / ~125 M synapses — running simulated spiking dynamics: LIF neurons wired by the *actual* synapse counts and neurotransmitter signs, a phototransduction cascade in the eyes, axonal delays, background noise; 0.5 ms steps; ~150k neurons / ~64 M synapses in full-CNS mode (measured). |
| **Eyes** | The stimulus can be a real human video: grayscale + ommatidial point-spread blur → percentile normalization → per-eye full-frame eye-plane sampling (each eye samples one complete, equally oriented copy of the image) → per-receptor luminance into the cascade (synthetic 1/f flicker & drifting-texture protocols also built in). |
| **Head** | Geometry: the **Lee Perry-Smith 3D head scan** (CC-BY 3.0, via the three.js assets). Physics: the classic four-sphere model (brain / CSF / skull / scalp; literature radii & conductivities) — every synaptic current becomes a source–sink dipole, and a 60-term Legendre expansion carries the field to the scalp. |
| **EEG** | Interchangeable electrode caps from the international 10-20 family: **45 ch (10-20) · 64 ch (10-10) · 128 ch (10-5) · EGI HydroCel 256** (241 usable sites), sampled at 1 kHz; optional realistic background EEG (α rhythm, 1/f activity, sensor noise); SNR and d′ detection analysis. |

One geometric footnote: the fly network is magnified (×202) to fill the human head —
pure geometry, assumed not to change the firing.

## Why the human head matters

A fly's own head is a sealed insulator: solve the physics and the field outside is
*strictly zero* — a fly's brain activity can never be measured from outside. A human head
(conductive brain, insulating skull, conductive scalp) is exactly what lets the signals
out. The fly-man's human body is not cosmetic; it is what makes his EEG possible.

## What the simulation already shows

- Direction selectivity emerges in his visual system from the wiring alone — the
  connectome "sees" motion without being taught.
- His scalp topography matches human VEP intuition: strongest at posterior electrodes,
  antipodal electrodes anticorrelated at −0.98.
- His stimulus-locked signal scales with how much of the visual field is driven:
  ~0.84 µV against ~2.8 µV of background for full-field naturalistic flicker —
  **~46 averaged trials** on the best channel (T7); a sparse demo video (one small
  moving object) drops it to ~0.2 µV and ~1,200 trials. Real, quantifiable, and
  honest about the cost.
- The pipeline passes the fly-scale benchmarks first: a flash at the eye yields the
  textbook Drosophila ERG.

## Repository layout

```
src/ffbm/          simulation engine: connectome data access, vectorized spiking
                   cascade, spherical forward kernels, calibration &
                   region-optional assembly, parameter registry
experiments/       research log exp001–exp017 (each with its own README:
                   design, results, limitations)
viz/               interactive 3D demo + the data-export pipeline behind it
                   (bilingual UI, English default)
docs/              methods & usage (Chinese originals)
docs/en/           English translations of the docs
scripts/           data download / assembly / profiling utilities
tests/             unit tests (forward kernels, calibration, assembly, params)
tools/             banner / pipeline figure generators
assets/            README figures
```

Quickstart: Python 3.12 + NumPy/SciPy/pandas/pyarrow; download the connectome
(~14 GB, public, no registration — `docs/en/data.md`); `pip install -e .`;
`pytest tests/`; serve `viz/` locally (`python -m http.server 8613 -d viz`) and
open `index.html`. Full workflow in `docs/en/USAGE.md`.

## Roadmap: the base for EEG experiments

This library is the foundation — the fly-man's EEG generation model. On top of it, any
human EEG paradigm can be run as a stimulus protocol:

**paradigm → simulated brain → simulated 45-channel EEG → analysis / decoding (BCI)**

- **Flash / pattern VEP** — evoked responses, the simplest channel
- **SSVEP** — frequency-tagged selection channels
- **Oddball / P300-style** — rare-deviant responses
- **Motion & direction** — his visual system's directional machinery
- **High-density caps** — shipped: 10-20 (45 ch) · 10-10 (64 ch) · 10-5 (128 ch) · EGI 256 (241 sites)
- **GPU acceleration** — done: the biology loop + forward recording run entirely on the GPU (`--gpu`, CuPy/NVRTC), 47 min → 156 s with bit-identical trajectories; staged build caches make warm re-exports (new stimulus / electrode layout) ~2.5 min end-to-end (docs/en/ACCELERATION_PLAN.md)
- **Closed loop** — decoded output feeds back into the stimulus

## Engine notes

Three acceleration layers, each verified against the CPU baseline: numba-JIT hot
kernels (2.38×), a GPU-resident CuPy engine (`--gpu`) for the biology loop +
forward recording (**47 min → 156 s, ~18×**, trajectories bit-identical), and
staged build caches (circuit keyed by regions/data/code; forward kernels keyed
by electrode layout — switching cap configurations rebuilds only the kernels).
The full-CNS export (150,601 neurons, 45 channels, 10.5 s) runs end-to-end in
~7.5 min cold / ~2.5 min warm on a 16-core desktop + RTX 4060 Ti. Requirements:
Python 3.12, NumPy / SciPy / pandas / pyarrow (MNE-Python and imageio/PyAV for
the electrode & video tooling; numba + cupy-cuda12x for the acceleration
layers); connectome download ~14 GB (public, no registration); 16+ GB RAM for
the full pipeline.

## Honesty notes

- The fly-man is fiction; the connectome, the head physics and the EEG engineering
  standards are real. This is a thought experiment built on real data.
- The ×202 magnification is geometry only — a real neuron scaled up 202× would not work.
- Neuron dynamics are calibrated approximations matched to literature firing-rate
  windows; absolute amplitudes are order-of-magnitude honest.

## Credits & acknowledgements

This project is a small stage built on other people's work. All of it belongs here:

**The brain — data**

- **MaleCNS v1.0**, the male fruit-fly CNS connectome — Berg et al., *Cell* (2026),
  HHMI Janelia FlyEM, https://male-cns.janelia.org/ — **CC-BY 4.0**

**The head — geometry & electrodes**

- **3D Head Scan by Lee Perry-Smith / Infinite-Realities** — **CC-BY 3.0**; the
  human-head mesh our electrode cap sits on, widely known through the three.js
  example assets
- **International 10-20 system** of electrode placement — H. H. Jasper,
  *Electroencephalogr. Clin. Neurophysiol.* (1958)
- **10-10 "five-percent" extension** — Oostenveld & Praamstra,
  *Clin. Neurophysiol.* (2001)

**The physics & the science we lean on**

- Rush & Driscoll (1969); Nunez & Srinivasan, *Electric Fields of the Brain* —
  spherical volume-conduction models behind the 4-sphere forward kernel
- Lappalainen et al., *Nature* (2024) — connectome-constrained fly visual
  networks (`flyvis`)
- Shiu et al., *Nature* (2024) — full-connectome LIF simulation of the fly brain
- Wang-Chen et al., *Nature Methods* (2024) — NeuroMechFly v2, embodied simulation
- Nern et al., *Nature* (2025) — optic-lobe connectome
- Shinomiya et al. (2019, 2022) — visual-circuit connectivity benchmarks (T4/T5 inputs)
- Hardie & Raghu (2001); Rusanen & Weckström (2016) — Drosophila phototransduction
  and lamina electrophysiology

Every constant in the simulator is traced to dataset / literature / calibration in
the parameter registry (`src/ffbm/params.py` → `docs/PARAMS.md`).

**Software**

- three.js (MIT) — the real-time 3D replay
- MNE-Python (BSD) — standard electrode montages (EGI HydroCel, 10-05 nomenclature)
- imageio + PyAV — human-video → fly-vision stimulus tooling
- The open scientific Python stack: NumPy, SciPy, pandas, PyArrow, matplotlib, pytest

**Culture**

- *The Fly* (1986), dir. David Cronenberg — for the fly-man. Original short story:
  George Langelaan (1957).

Spotted something we used without credit? Open an issue and we'll fix it.

## License

Code: MIT. Third-party assets keep their own licenses — the MaleCNS v1.0 data is
CC-BY 4.0 (HHMI Janelia) and the Lee Perry-Smith head scan is CC-BY 3.0
(Infinite-Realities); see their terms when redistributing derived data.
