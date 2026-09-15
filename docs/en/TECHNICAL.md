# Core Technical Document — Forward Simulation from Neural Spikes to Scalp EEG

*(English translation — authoritative version: [../TECHNICAL.md](../TECHNICAL.md))*

Scope: `src/ffbm/`, `experiments/exp001–012`, `scripts/`. The visualization pages are not covered (`viz/index.html` has its own README). Third-party libraries used directly (numpy / scipy / pandas / pyarrow) are described only by their purpose; textbook formulas (LIF, exponential synapses, classical spherical-shell models) are given with their provenance and this project's parameters, while the **in-house parts** (circuit inference, numerical implementation details, calibration methodology, thought-experiment geometry, background EEG model) are elaborated in detail.

---

## 1. System Overview

A one-way data flow, fully self-implemented within the repository (no third-party EEG/neural-simulation libraries):

```
MaleCNS v1.0 connectome (official dataset, §3)
        │  circuit construction (left-lobe five-level cascade, §4)
        ▼
Natural stimulus protocol (dark / 1f flicker / forward-reverse drift, §5.6)
        │
        ▼
0.5 ms time-stepped simulation: phototransduction → LIF spiking → exponential
synapses (RL/LM current-based, MT conductance-based + per-edge delays, OU
background noise) (§5)
        │  per-step output: transmembrane current y or g·y·(E−v) per synaptic edge [pA]
        ▼
Quasi-static forward kernels (dipole pair × volume-conductor Green function) (§7)
        ├─ homogeneous infinite-medium kernel StaticPairField
        ├─ sealed two-sphere kernel SealedHeadPairField (electrodes inside the fly head)
        └─ four-sphere kernel FourSpherePairField (17 electrodes on the human scalp, §8)
        ▼
Scalp potential [μV] ──(optional)── superimposed background human EEG (§9) → SNR / detection theory
```

Core idea: **the scalp potential is a linear functional of the synaptic-current state**. For each connection set, `coef` (per electrode × per edge, V/A) is precomputed; at run time, the potential at each millisecond = one matrix multiplication `coef @ y`. Linearity makes superposition, channel expansion, and after-the-fact kernel replacement cheap.

## 2. Units and Numerical Conventions (uniform across the repository)

| Quantity | Unit | Note |
|---|---|---|
| Voltage / current / time / resistance | mV / pA / ms / GΩ | GΩ×pA=mV, simulation layer |
| Position | μm | ×1e-6 → m inside the forward kernels |
| Conductivity σ | S/m | literature values |
| Synaptic conductance g | nS | nS×mV=pA, §5.3 |
| Time step dt | 0.5 ms | field sampled once every 2 steps (1 kHz) |
| Random seed | fixed (sim=42, background EEG=2026) | fully reproducible |

## 3. Data Layer (official dataset, brief)

MaleCNS v1.0 (male fruit fly whole CNS, 166,700 neurons, Cell 2026, CC-BY). This project uses four derived tables (row counts and remote hashes verified by `scripts/verify_data.py`):

- `visual_nodes` (105,265), `visual_edges` (12,490,360): the optic-lobe subnetwork
- `neuron_sites` (89,411,645 rows, 13GB feather): coordinates and type (PreSyn/PostSyn) of **every synaptic site** — the data source for per-synapse geometry (exp002, §7.1)
- annotation tables: `assignedOlHex1/2` (hexagonal-grid coordinates), neurotransmitter type → sign

Loader: `src/ffbm/data.py`; `site_positions()` returns the per-body, per-type site centroid (used by the viz/exp005 pipeline), while exp002 consumes the raw site cloud directly.

## 4. Circuit Construction (in-house, detailed)

Code: `experiments/exp005_medulla_ds/run.py::build_circuit()`, reused by exp006–011 and the viz export.

**Five-level cascade (left lobe)**: R1-R6 (2,265) → L1/L2/L3 (2,677) → 8 Mi/Tm types (7,195) → T4a-d/T5a-d (6,793); edges 6,351 / 25,669 / 154,038; synapses 1,666,962. Weights = the **synapse counts** given by the connectome.

**4.1 Left-lobe selection — pooled x-axis 2-means midline**. The dataset's `hex1` is not a lobe label but the hexagonal-grid u-axis coordinate (corr(hex1, x)=−0.78); selecting the lobe directly by hex is wrong (an early exp005 bug). Correct procedure: run 2-means on each layer's x coordinates (converged in 20 iterations), obtaining one cluster center per side; **take the side where Mi1 has the most hex1 annotations as the left lobe**; the midline = the midpoint of the boundary between the two clusters; each layer is cut independently by x < midline. This keeps the six layers R/L/Mi/Tm/T4/T5 consistent on the same side.

**4.2 Drift-direction axis — hex regression**. The stimulus drift direction is not chosen arbitrarily: project Mi1's 3D positions onto the plane basis (a1,b1) orthogonal to the eye axis u_eye, and least-squares regress hex1 / hex2 to obtain u_dir / v_dir (the directions, in physical space, of the hexagonal grid's two basis vectors); the preferred direction = the hexagonal composite at −135°. This guarantees that "drift along the preferred direction" is aligned with the anatomical delay-line axis (exp006's wavelength matching depends on this).

**4.3 T4/T5 column inference — dominant-partner voting**. Which hexagonal column each T4/T5 subtype (a-d) corresponds to is not directly annotated in the data. Inference: for each T4 (or T5), take the hex coordinates of the most common partner within its dominant presynaptic partner type (Mi1 → T4, Tm9 → T5) as that cell's column. Column-alignment quality determines whether delay-line DS can emerge (verified in exp006).

**4.4 Sign injection**. Since exp009, each edge's sign comes from the dataset's neurotransmitter annotations: R→L histamine −1; L1→Mi1/Tm3/Mi4 glutamate −1; L2/L3→Tm-family acetylcholine +1; M→T per edge per annotation. The LMC dark-depolarization baseline current (§6) together with the double sign inversion generates the ON/OFF architecture (light → R fires → inhibition of L1 → disinhibition of Mi1 → T4 = ON; L2 active in the dark → drives Tm → T5 = OFF) — with no hand-written polarity anywhere.

## 5. Neuron and Synapse Models

Code: `src/ffbm/simulation.py`. Tests: `tests/test_simulation_extensions.py` etc. (16 tests, all passing).

**5.1 LIF neuron** (Lapicque 1907, textbook model). Parameters by layer:

| Layer | τ_m | Refractory | Note |
|---|---|---|---|
| R1-6 | 10 ms | 3 ms | |
| L1-3 | 20 ms | 2 ms | (in reality graded-potential neurons; user deferred the change) |
| Mi/Tm | 10 ms | 2 ms | |
| T4/T5 | 10 ms | 2 ms | |

V_rest=−70, V_th=−50, V_reset=−70 mV, R_m=0.1 GΩ (all layers). With no g_tot, explicit Euler (old behavior preserved byte-for-byte); with conductance input, semi-implicit (§5.3). Threshold and τ magnitudes are taken from the fly literature, not fit per cell — the credibility of the firing rates is backstopped by the §6 working-point calibration.

**5.2 Exponential synapse, base form** (standard). On a presynaptic spike the edge state y_e is incremented by `gain·weight_e` and decays as exp(−dt/τ_s); a single CSR sparse-matrix matvec aggregates it onto the postsynaptic neurons. Key implementation: y is stored sorted by (post,pre), and the forward kernel's coef matrix is built in **the same order** (`lexsort((pre,post))`) — this is the contract that aligns the simulation state with the field kernel; exp001 once failed because of a wrong ordering.

**5.3 Conductance mode (in-house extension)**. Used only for M→T4/T5 (RL/LM stay current-based, see §11). y_e becomes a dimensionless gating variable (kick = weight); per-edge current

    I_e = g_unit · y_e · (E_rev,e − v_post),  E_rev = 0 (sign>0) / −80 mV (sign<0)

g_unit = 0.02 nS (calibrated, §6). Unit chain: nS·mV = pA.

The integration must be semi-implicit: writing the input as i(v) = i_indep − g_tot·v (i_indep = base + Σ g·E_rev, g_tot = Σ g), then

    v_new = (v + (dt/τ)(V_rest + R·i_indep)) / (1 + (dt/τ)(1 + R·g_tot))

which is unconditionally stable. **Lesson recorded**: explicit Euler was used first; under natural stimuli the Mi/Tm peak firing rates came out several times higher than under the calibrated flash, and once the total conductance drove R·g·dt → τ the numerics diverged (φ→NaN); after two export crashes the cause was located and fixed. The existing tests use an extreme conductance of R·g=50 to verify that v stays bounded and clamped at the reversal potential. Implementation: `to_neuron_drive() → (i_indep, g_tot)` + `LIFPopulation.step(i, g_tot=...)`; the field kernel consumes `edge_currents(v)` (which in conductance mode returns real currents rather than gating variables).

**5.4 Per-edge transmission delay (in-house extension)**. Per-edge delay = 1 ms (chemical synapse) + |pre−post site distance| / 300 μm/ms (0.3 m/s, thin axons) + ±0.5 ms uniform jitter. Implemented as a ring buffer (n_bins × n_edges float32, one row flushed per step); delay=0 preserves the old synchronous delivery. Honest conclusion: at the fly's scale delays are 1.0–1.3 ms per stage and have a minute effect on the waveforms (calibration points re-verified unchanged) — they are kept for future phase analysis and geometric sensitivity.

**5.5 OU background noise (in-house, now population-wide)**. `ColoredCurrentNoise`: an exactly discretized Ornstein-Uhlenbeck process x←a·x+k·w, k=σ√(1−a²), with stationary std = σ independent of dt. τ_n=8 ms; σ: R 50 / L 40 / Mi/Tm 40 / T4/T5 60 pA, corresponding to membrane-voltage fluctuations std_V ≈ σ·R·√(τ_n/(τ_n+τ_m)) ≈ 2.7–4 mV (20 pA of white noise gives only 0.3 mV). **Motivation**: the f-I curve of a low-noise LIF is a knife-edge (5↔50 Hz separated by only 200→250 pA), and T4 mathematically cannot fire during darkness; real spontaneous activity arises from the bombardment by thousands of unsampled synapses (temporally correlated, realistically scaled), and OU is its minimal credible stand-in. After the round-4 calibration all four layers use OU (the R dark rate was lifted from 0 to 2.7 Hz, closer to the real dark tonic discharge; Mi/Tm spontaneity went from "sitting on a baseline-current knife-edge" to noise-driven).

**5.6 Phototransduction cascade (in-house phenomenological model)**. `PhotoCascadeVector`: two cascaded first-order low-pass stages (τ=10 ms ×2, ≈20 ms receptor latency) + slow light adaptation (τ=800 ms, 30% plateau sag). The input is a light-intensity increment (pA semantics) and the output the adapted current increment. The shape parameters were cross-checked against the exp002 intensity series / exp004 flicker fusion; the parameters are shared by all R cells (no heterogeneity, §12).

## 6. Working-Point Calibration Methodology (in-house, three rounds)

Code: `scripts/calibrate_working_point.py`; results in `scripts/outputs/working_point_calibration{,_r3}.json`.

**Method**: short simulations (4 s = 2 s darkness + 2 s 350 pA flash; only firing rates measured, no field kernel built), grid-scanning the free parameters; targets = literature windows for spontaneous firing (Hz): Mi/Tm 5–20, T4 2–12, T5 2–15; penalty = sum of squared distances outside the windows (dark-window weight 1.0, flash-window 0.35) + a 400-point penalty if the T4 flash response is <5 Hz.

**Four rounds of evolution** (each redone after a new mechanism; from round 4 on, calibration and export share the same `ffbm.pipeline` code path instead of maintaining separate loops):

| Round | Mechanism background | Result (dark-period R/MID/T4/T5 Hz) | Score |
|---|---|---|---|
| 1 | white noise | 0 / 22.4 / 0 / 11.7 (T4 dead silent) | 411 |
| 2 | +T4 hand-tuned baseline 195 pA knife-edge | 0 / 22.4 / 2.1 / 11.7 (fragile) | 5.8 |
| 3 | +OU(T45) + conductance MT + delays | 0 / 21.7 / 5.8 / 5.0 | 5.8 |
| 4 | population-wide OU (R50/L40/MID40/T4560) | **2.7 / 19.0 / 7.0 / 3.0** | **0.0** |

Round-4 score 0.0 = all metrics in both the dark and flash periods (including R/L) fall inside the literature windows simultaneously. **Structural finding**: T4's Mi1 input is completely suppressed by L1 during darkness — it stays 0 under any gain — so its real spontaneous activity can only come from **unwired partners**. Handling: acknowledge this explicitly and give the T4 population an "unmodeled input background" (I_T4_BASE=175 pA + OU noise); this is standard practice in connectome simulations (injecting a baseline for unsampled inputs), and the parameters are transparently recorded in meta and HANDOFF.

**Final parameters** (viz export CAL): I_MID_BASE=90, I_T4_BASE=175, OU σ=60 pA τ=8 ms, g_unit=0.02 nS, E_inh=−80 mV; I_L_BASE=250 (LMC dark depolarization, exp009), GAIN_RL=12, GAIN_LM=20 pA/synapse, τ_RL=τ_LM=5 ms; differentiated MT kinetics τ_s (ms): Mi1 25 / Tm3 8 / Mi4 10 / Mi9 15 / Tm1 20 / Tm2 8 / Tm4 10 / Tm9 20 (exp005 KINETICS, designed for DS emergence).

**Honest boundaries**: the calibration matches only mean firing rates, not ISI variability (Fano/CV); the working point is still sensitive to parameters (OU has already reduced that sensitivity from "tens of Hz per 5 pA" to a smooth gradient).

## 7. Extracellular Field Forward Model

Code: `src/ffbm/forward.py`. Audits: `scripts/audit_scalp_forward.py`, `tests/test_*_kernel.py`.

**7.1 Dipole-pair approximation and sign conventions (in-house design decision) — return-current geometry closed (exp013, 2026-09-14)**: the verdict is **neurite** (source on the distal neurite of the L cell), and the absolute-amplitude calibration = pipeline value ×1.7 (neurite/meanpos kernel-norm ratio). Quasi-static approximation (capacitive/inductive effects negligible below 1 kHz, the EEG standard): each synaptic edge is approximated by a current pair of equal magnitude and opposite sign — **source at the presynaptic site, sink at the postsynaptic site**:

    φ(electrode) = Σ_e  y_e · (1/4πσ) · (1/|r−r_pre,e| − 1/|r−r_post,e|)

The signs were validated by exp001–002 and the ERG waveform (cornea-negative). **The return-current problem**: the real return current is distributed over the postsynaptic cell's membrane, and "source at the pre site" is one of the assumptions; exp002 quantified a ~400× spread of kernel norms across the three geometries; exp012 (§10) attempted the discrimination with ERG anchors — not yet closed, so the absolute amplitude was left uncalibrated (relative quantities: waveform, topography, and phase are unaffected). Also: the photoreceptor current is modeled as a rhabdomere→soma dipole (23.5 μm extrapolation, calibrated in exp002).

**7.2 Homogeneous infinite-medium kernel `StaticPairField`** (textbook Green function). coef = (1/|r−pre| − 1/|r−post|)/(4πσ), a direct implementation of the formula above.

**7.3 Sealed two-sphere kernel `SealedHeadPairField`**. Inner sphere (CNS) + shell + zero normal current at the outermost boundary (the insect cuticle insulates). For each order, the Legendre coefficients solve a 3×3 boundary system (V continuous, σ∂V/∂r continuous, sealed). **Physical conclusion** (exp008): the field outside a sealed shell is strictly zero — no EEG whatsoever can be measured outside the fly head, which is what motivated the move to the human-brain thought experiment.

**7.4 Four-sphere kernel `FourSpherePairField`** (a self-implementation of the classical 4-sphere model, in detail). Geometry: brain 78 / CSF 80 / skull 85 / scalp 92 mm; σ = 0.33 / 1.79 / 0.013 / 0.33 S/m (human-literature values). Solution for each order n:

    V1 = a·r^n + src·r^−(n+1)        (inside the brain, source term known)
    V2 = b·r^n + c·r^−(n+1)          (CSF)
    V3 = d·r^n + e·r^−(n+1)          (skull)
    V4 = f·r^n + g·r^−(n+1)          (scalp)

Seven boundary conditions: 2 each at r1/r2/r3 (V continuous + σ∂V/∂r continuous) + sealing at r4 (∂V/∂r=0) → a 7×7 linear system solved order by order for (f_n, g_n); the scalp-electrode potential = Σ_n P_n(cosθ)[f_n r0^n r^n + g_n r0^n r^−(n+1)]/(4πσ1 r1).

Numerical details (in-house, all required for stability): radii normalized by r1 (the matrix in raw dimensions is ill-conditioned); each row equilibrated by its largest element (equilibration) before solving; **n=0 returns 0** (a monopole inside a sealed head is unsolvable; for current pairs it cancels exactly anyway, and this also fixes the gauge — potentials are relative in the "zero spherical mean" sense); Legendre polynomials accumulated order by order with the three-term recurrence. Series to order 60 (export default): truncation error 2.5×10⁻¹⁰ for network depth r0/r1≤0.75.

**7.5 Validation and audit results** (all can be re-run):

| Check | Result | Meaning |
|---|---|---|
| Two-solution cross-validation (four-sphere ≡ three-sphere when σ_CSF=σ_brain; two independent implementations) | agree to 3.9×10⁻⁷ | strongest evidence: two different codebases, same physics |
| Dipole antisymmetry (flip the dipole moment → potential sign flips) | error 0.0 | exact |
| Series convergence S=60 vs 200 | 2.5×10⁻¹⁰ | truncation negligible |
| Skull attenuation (superficial / deep) | 0.29 / 0.87 | literature 0.3–0.5 / 0.7–0.9 ✓ |
| CSF shunting effect (superficial radial dipole) | ×0.7 | the highly conductive CSF makes current recirculate within the layer and reduces the trans-skull fraction (opposite to intuition, consistent with the literature) |

## 8. Human-Brain-Shell Thought-Experiment Setup (in-house geometry, detailed)

Established by exp010/011; implemented in the viz export. Assertion: **the firing pattern is assumed not to change under the upscaling** (a pure geometry/detection-theory experiment; a real neuron at ×400 could not operate per cable theory, see §11).

**8.1 Scaling**: all network sites (p−center)×400; the head-sphere center = the network centroid.

**8.2 Occipital-pole placement** (maximum offset via quadratic constraints, in-house). The real visual cortex lines the inner skull wall at the occipital pole, not the head center. The T4/T5 end (the output/"cortex" end, −u_eye direction) is pushed toward the sphere wall: goal = the extremum on the T4/T5 side reaches 0.90·R_brain; constraint = |p−center| ≤ 0.98·R_brain for all points, each point contributing one univariate quadratic inequality in the offset d, whose smallest positive root is taken as the feasible upper bound. Result: offset 31.1 mm, minimum margin 1.5 mm. **Effect**: the strongest scalp signal moves from 0° (eye axis) to 141–162° (occipital side), topographically consistent with real visual-evoked-potential occipital sites (Oz/O1/O2); the two poles anticorrelate with corr(0°,162°) ≈ −0.98 (a standard signature of dipole topography).

**8.3 Electrode array**: 17 point electrodes, located at 0.985·R_scalp inside the scalp layer. Layout = 1 electrode anchored on the eye axis + a 17-point Fibonacci sphere with the nearest point removed; channels sorted by angle to the eye axis (0°→162°), so the heat map shows a clean spatial gradient.

## 9. Background Human EEG and Detection Theory (in-house model)

**Motivation**: the complete form of the thought experiment must answer "can the signal be read out of a real recording", not merely present a clean signal.

**Generation** (`viz/export_data.py`, seed 2026): three components superposed per electrode — (i) eyes-closed α rhythm: 10 Hz sinusoid × amplitude envelope (0.55+0.45·sin, ~1.4 s fluctuation period) × spatial weight cos²(angular distance from the occipital pole) (the α rhythm is occipitally dominant, on the same side as our network — the worst-case scenario); peak ~30 μV. (ii) 1/f aperiodic activity 3 μV (synthesized by the FFT method, spectral exponent 1). (iii) sensor white noise 1.5 μV. No EMG/drift/electrode-pop artifacts (§12).

**Detection theory**: signal amplitude = the std of the strongest electrode over the flicker window (1.5–4.5 s); background = the std of the same electrode over the same window; after averaging k trials, d′ = √k·(Δμ/σ), and the k required for d′=2 is (2σ_bg/Δμ)². Band-limited version (`vizprep.snr_metrics`): ratio of variances within the 2–20 Hz band (out-of-band noise excluded).

**Results (round-4 full-mechanism stack)**: strongest electrode at 147°, fly signal 0.69 μV vs background 10.0 μV → broadband d′=2 requires averaging **842** trials; **the 2–20 Hz band-limited version instead requires 3019**. Spectral explanation: a substantial fraction of the fly signal's energy lies above 20 Hz (synaptic time constants 5–25 ms → 40–200 Hz), whereas the 2–20 Hz in-band background is dominated by the α peak — if the acquisition system read a high-frequency band (avoiding α, retaining the signal), the SNR would beat both numbers; but high-frequency bands in real scalp recordings are contaminated by muscle activity. This is the quantitative conclusion that "the detectability of the thought-experiment signal depends on band selection". (exp011's old d′≈28/trial is a historical record under the old working point without background and is superseded by these numbers.)

**Head-model parameter sensitivity** (`scripts/head_sensitivity.py`, kernel-norm ratios): under literature spreads of ±20% in conductivity and ±1 mm in layer thickness, the scalp-amplitude envelope for deep sources spans only **×0.82–×1.06** (the biggest lever is the 1/σ scaling of brain/scalp conductivity at −18%; skull σ ±20% moves it only ±6%). Conclusion: head-model parameters are not the main source of amplitude uncertainty — the return-current geometry (order-of-magnitude leverage, §10) is.

## 10. Return-Current Discrimination Experiments exp012 / exp012b (in-house design)

`experiments/exp012_return_constraint/`: re-run the flash ERG under the current sign conventions, with three geometries (t-bar proximity / neurite cross-region / mean position) sharing the same spike sequences (same seed, only the kernel swapped), judged against four classic ERG anchors: on/plateau ∈[1,3], off/on ∈[0.3,1], LMC@on ∈[0.5,1], LMC@plateau ∈[0,0.5].

**First result (honestly inconclusive)**: no variant passes all anchors; off/on is 0 for every variant.

**exp012b (after incorporating exp003's τ_fall=2.5 ms asymmetric phototransduction)**: the results are **identical** to the symmetric condition — the off transient is still zero. Mechanism: at flash offset the R dipole returns to baseline with a positive sign, while the L population recovers from histamine inhibition and its component returns to baseline **negatively**; the two cancel (in exp003, under the old sign conventions, the two pointed the same way, which is what produced the positive overshoot). **The path to closing the discrimination converges on LMC-side dynamics**: either a rebound mechanism in the L population (sag/rectification) or a genuine inverted-chloride histamine conductance (§12 A3) — the latter is the same thing as #A3. The neurite variant shows a persistent pathology (L/R dipole cancellation); absolute amplitudes differ ~10× across the three variants, and calibrating the absolute scalp-EEG amplitude remains stuck here.

## 11. Known Limitations and Assumptions (an honest list)

**Amplitude class**: the return-current geometry is undiscriminated (§10, ~10× leverage); the LIF/phototransduction/synapse parameters are order-of-magnitude-reasonable hand-tuned + calibrated values, not per-cell fits; the per-synapse current gains (GAIN_RL/GAIN_MT) have no electrophysiological calibration.
**Dynamics class**: L1/L2 use spiking LIF in place of graded potentials (real LMCs do not spike); Off-pathway rebound dynamics are not wired into the main pipeline (this is why exp012 failed); no active axonal currents (excluded by the user); white noise is still used in R/L/MID.
**Geometry class**: concentric spherical shells (real heads are not spherical, skull thickness is uneven, white matter is anisotropic — ellipsoid/anisotropy excluded by the user); the network is placed as a centroid-centered spherical cluster (real cortex is a folded thin sheet); point electrodes, no reference definition (the reference-electrode question was excluded by the user).
**Thought-experiment assertion**: ×400 scaling changes geometry only, with firing assumed unchanged (explicitly accepted by the user); temperature 25→37 °C not adjusted (excluded by the user).
**Statistics class**: single seed; the calibration matches mean firing rates only; the SNR uses full-band std (the α band partially overlaps the signal band; band-limited SNR is more rigorous, §12).

## 12. Future Improvements (third review 2026-09-13: A2/A4/A5/B7/B8/B10 completed)

**A level (directly affects credibility)**
1. **Close the return-current discrimination** (the only remaining A-level item): the path has converged — add rebound dynamics to the L population, or upgrade RL to an inverted-chloride conductance (see 3), then re-run exp012; alternative anchor: the intensity series (V-log I slope).
2. ~~Upgrade R/L/MID background noise~~ **Done** (round 4: population-wide OU, score 0.0).
3. **Upgrade RL/LM to a conductance basis — now judged to be the key to the same problem as #1**: exp012b showed the missing off transient is rooted in LMC-side dynamics. The real lamina biophysics (histamine-gated chloride conductance, high [Cl]i in LMCs, dark-state tonic release) can replace the "I_L_BASE baseline-current hack" with a mechanistically correct structure while also restoring the off transient. **Precondition for implementation: literature verification** (Hardie-1989-style lamina electrophysiology: E_Cl values, dark tonic release rates, R1-6 dark graded-potential amplitudes) — implementing from fuzzy memory risks breaking the already-verified architecture, so this round explicitly defers it, to be carried out as exp013 once verified.
4. ~~Single code path for export/calibration~~ **Done** (ffbm.pipeline).
5. ~~Unit tests for the export pipeline~~ **Done** (5 vizprep tests).

**B level**
6. Multi-seed variability: script built (scripts/multiseed_ci.py).
7. ~~Conductivity sensitivity~~ **Done**: envelope ×0.82–×1.06; the head model is not a major source of uncertainty.
8. ~~Band-limited SNR~~ **Done**: broadband 842 vs band-limited 3019 trials; the band-selection conclusion is in §9.
9. exp011 refresh: covered by §9's SNR meta + band-limited analysis.
10. ~~Delay-profile jitter~~ **Done** (±0.5 ms uniform jitter into the pipeline); R1-6 heterogeneity remains (low priority).

**C level (excluded or deferred per user instruction)**: circuit expansion (lobula plate/central brain), L1/L2 graded potentials, reference electrodes/montages, action-potential currents, temperature- and frequency-dependent conductances, ellipsoidal head/anisotropy, the thought-experiment assertion itself.

## 14. Follow-up Extension Plan (brain-region expansion / L1-L2 graded potentials, planned 2026-09-13)

**The registry of all fixed-valued hyperparameters is in `docs/PARAMS.md`** (generated by `src/ffbm/params.py`, regenerated with `python -m ffbm.params`) — every parameter is annotated with its source (dataset/literature/calibration/phenomenology/assumption/chosen) and its sensitivity; open items (return-current geometry, LMC E_Cl, R1-6 heterogeneity) are listed separately as "pending". Both the CAL and the background-EEG defaults are assembled from the registry, and tests keep the document from drifting away from the code.

### 14a. L1/L2 graded potentials (exp013b; recommended to be implemented together with the A3 literature verification)

Status: the real L1/L2 are **non-spiking** analog cells (graded voltage transmission, continuous transmitter release); this pipeline uses a spiking-LIF surrogate (12.5 Hz pseudo-spikes in the dark, as declared in §11 of this document).

Why do it: (1) the lamina is the second-largest dipole source of the ERG, and graded release would turn the L component from "event-quantized" into "smoothly continuous", changing its temporal structure and amplitude; (2) it and the deferred inverted-Cl histamine conductance (§12 A3) are **two sides of the same biology** — both the real LMC input side (histamine chloride conductance) and the output side (graded release) are non-spiking; (3) exp012b showed the missing off transient is rooted in LMC-side dynamics.

Implementation steps:
1. `simulation.py`: `GradedUnit` (leaky integrator, no threshold/reset; release rate r(v) = sigmoid with a calibratable slope parameter) + a graded-drive mode for `ExponentialSynapses` (edge state y driven by a first-order filter of the presynaptic release rate, replacing event delivery).
2. `pipeline`: a `cal["L_GRADED"] = True` switch; round-5 working-point calibration (darkness windows + ON/OFF-architecture re-verification + ERG waveform matching).
3. Criteria: how much the L-component waveform is smoothed; whether the off/on anchor improves (can the exp012 discrimination be closed); the quantitative change in scalp-signal amplitude.
4. Expected impact: on scalp-signal amplitude the direction is uncertain and the magnitude moderate (±30–50%: smooth transmission removes high-frequency quantal components, while tonic release may strengthen low-frequency dipoles); the impact on ERG waveform fidelity is large. **Recommended to check the literature before acting** (same survey as A3).

### 14c. Endgame: close the return-current discrimination + move graded L1/L2 into the main pipeline (impact assessment)

**How** (two steps, the first being the precondition of the second):

1. **Closing the discrimination (exp013 wrap-up: one focused debugging pass + one re-scan)**: the two-dimensional scan already delivers a monotonic discrimination signal (neurite passes off/on in 3/9 feasible combinations; the other geometries 0/9); the remaining debugging point = the light-state release rate is compressed (v_R light-state −39 mV vs analytic −30.5; candidate causes: the phototransduction-adaptation implementation / the actual tetrad-synapse weight of L1/L2). After the fix, re-scan the 3×3 grid; if neurite is robustly unique within the joint feasible region → the return-current geometry is judged neurite and **all absolute amplitudes are ×1.7** (a single factor, immediate effect).
2. **Adopting graded L1/L2 (round-5 calibration)**: graded R (v_th=∞) + graded histamine release (GradedSynapsePool; E_Cl=−70/v_K=−25 already in the parameter registry) + graded L + graded L→Mi/Tm release, replacing the exp009 double-inversion surrogate and the I_L_BASE hack. The mechanism has been verified by exp013; the risk is contained.

**Is the impact large?** — by target:
- Amplitude: closing the discrimination = a single factor ×1.7 (moderate, direction certain);
- Waveform/spectrum: the graded replacement smooths the L component and reduces the high-frequency quantal content — band-limited (2–20 Hz) SNR is expected to improve (the fly signal's >20 Hz content is inherently synaptic quantization), broadband amplitude ±30–50%;
- Topography/phase/DS conclusions: unchanged (determined by geometry);
- Mechanistic correctness: removes two hacks (the I_L_BASE baseline current, the LMC spiking surrogate) — the item with the greatest scientific value.
- VS/LPTC were shown by exp014 to have no signal value; the endgame involves no brain-region expansion.

### 14d. Brain-region expansion (exp014–016, executed 2026-09-13/14; the conclusions rewrote the original plan)

- **exp014 (lobula-plate VS)**: stage 1 inventory → VS (18 cells / 158k synapses) is the cleanest single hop; stage 2 **refutes the amplification hypothesis** — VS contributes <1e-4 μV to the scalp potential (the signal is dominated by synapse count; dipole length does not rescue cell count); unexpected bonus: forward/reverse drift scalp amplitudes differ by 15× (field-level direction selectivity holds). VS does not enter the main pipeline.
- **exp015 (bilateral optic lobes, now in the main pipeline)**: the two lobes' native geometry spans 692 μm, which at ×400 is 277 mm wide — **it does not fit inside the human brain sphere** (the "×400" premise holds only for a single lobe) → scale-adaptive fitting (`vizprep.fit_scale_shift`: elongated spike ×400, wide body centered ×202, native relative geometry untouched); zero-cost transfer of the working point (driven L1/2 dark potentials −35.1/−35.3 mV on the two sides, cross-lobe edges measured at 0); **the signal drops rather than rises**: the best electrode for both lobes at ×202 = 0.69× that of a single lobe at ×400 (source doubling is more than cancelled by the scale shrink), and the topography turns to the 90° lateral electrodes. The main-pipeline export has switched to both lobes.
- **exp016 (visual downstream VPN→central brain)**: inventory (circuit→VPN 1.76M synapses; the main players LC17/LC12/LC10a… all cholinergic large-tangential types; top targets PVLP/AOTU large-fan-in cells) + stage-2 circuit (2,557 VPN + 250 CB, 1.36M synapses, soma positions, compact bodyId remapping) → **verdict: negligible impact** (three-electrode signal ratios 0.91/1.05/1.01). Consistent with exp014: the visual downstream makes no measurable contribution to scalp EEG and does not enter the main pipeline; the pipeline keeps an optional-region mechanism (the circuit is built whenever `vpn_ids`/`cb_ids` are present; parameters in the registry under `vpn_central`).
- Stage 3 (whole central brain / whole brain): kept as a long-term goal; exp015/016 evidence further supports the assessment that "expanding the central brain has low signal value".
- **Optional region-assembly mechanism (2026-09-14)**: the infrastructure for "stepwise expansion to the whole brain" is in place — `ffbm.regions` maintains the region registry and the `DEFAULT_REGIONS` switch (parameter registry `circuit.regions_default`); `build_circuit(regions)` assembles on demand; on the pipeline side there is the generic `extra_pops`/`extra_edges` specification (topological-order stepping, conductance synapses, full delays), with **zero compute and zero memory for disabled regions**; the export supports `--regions visual_bilateral,vpn_central`. A new region (olfactory, taste→MN9, central brain, VNC) needs only a builder plus one registry line.

- **exp017 (whole CNS, completed 2026-09-14)**: the remaining optic lobe 62,305 + central brain 33,480 + ventral cord 15,367 cells wired in (3.64M pairs / 59.91M synapses; w≥5; neurotransmitter sign mapping; the VNC joins the simulation but not the head-shell forward kernel). A whole-CNS simulation of ≈160k cells / 63M synapses runs through; working point g=0.002 nS, I=150/150/120 pA (dark rates 1.9/11.4/3.4 Hz, stable over 6 s); the core cascade's dark rates are unaffected by the downstream (bit-for-bit identical). **Scalp verdict: 2.03/1.08/0.88 (three electrodes) — the increment is non-phase-locked spontaneous-activity variance, not visual signal**. The default configuration remains the two optic lobes; the whole-CNS one-switch option is `--regions visual_bilateral,vpn_central,ol_rest,central_brain,vnc`.

## 13. Tests and Reproducibility

- `tests/`: 16 tests (LIF/synapses/OU/delays/conductance stability/analytic verification of the three kernels), all passing.
- `scripts/audit_scalp_forward.py`: 5 physics audits (§7.5).
- `scripts/verify_data.py`: data integrity (remote hashes + row counts).
- All randomness is fixed-seeded; the grids and result JSONs of the three calibration rounds are committed and traceable.

---
Maintenance convention: this document is updated in sync with the code; whenever a mechanism in §5–§9 is changed, the §6 calibration and the §9 SNR numbers must be recomputed and recorded in the git commit message.

## 15. Recent Extensions (exp013–017 and the visualization system, 2026-09 addendum)

This section records work completed after the §14 planning; details are scattered across the experiment READMEs and docs/STIMULUS.md, docs/STATUS.md.

### 15.1 Optional Region Assembly (exp014–017)

`ffbm.regions`: every brain region (visual_bilateral / vpn_central / ol_rest / central_brain / vnc) can be enabled or disabled independently. A disabled region is never built (no populations/synapses/forward kernel). Assembly order and dependencies live in `_ORDER`; the circuit is attached via the `extra_pops` (CAL key names including i_base/tau, resolved per occurrence) and `extra_edges` specifications, with `pipeline.build_stack` doing the construction. The VNC group has `forward=False` (it joins the dynamics but not the head-shell forward kernel). exp016/017 conclusions: VPN→central brain contributes 0.91/1.05/1.01× to the scalp potential (≈ noise level); the whole-CNS difference relative to the two lobes is non-phase-locked spontaneous activity (2.03/1.08/0.88).

### 15.2 The Scalp Electrode Array System (45/64/128/241)

- 45 channels (current): the 10-20 19 channels + 26 10-10 midline/subdivision channels, derived parametrically by `vp.standard_1020` (anchors Cz/Fz/Oz/A1/A2 + `ni_arc_deg` + `fpz_arc_deg` + `yaw_deg` + `roll_deg`); the coronal ring is scaled by the ear-position angle th (C3/C4=0.4·th, T7/T8=0.8·th).
- 64 channels = 45 + chain midpoints (AF7/8, F5/6, FC1/2, C5/6, CP1/2, P5/6, FT7/8, TP7/8, PO7/8) + Iz; 128 channels = 64 with a further 5% subdivision (new points matched to official 10-05 names in MNE standard_1005); EGI 241 = the MNE 'EGI_256' real geodesic geometry, rigidly aligned with chirality determination via the montage's own nasion/lpa fiducials, plus neck-region filtering. Catalog file `viz/data/elec_configs.json` (viz/ELEC_CONFIGS.md).

### 15.3 Interactive Calibration (page ↔ pipeline closed loop)

10-20 calibration handles (Cz pitch / Fz-Fpz-Oz midline voting / A1-A2 mirrored ear positions) + yaw/roll sliders re-derive the whole cap in real time; waveforms are a spherical-distance interpolation preview, with exact numbers re-exported by `export_data.py --elec-layout`. View calibration (six orthographically unfolded views from two anchors, front/top) and head-form calibration are persisted independently (localStorage). Saved calibrations are applied automatically at startup, eliminating "old layout at startup, jump when entering a mode".

### 15.4 Stimulus-Protocol Parameters Moved into the Registry

The natural-stimulus epoch table/contrast/drift speed/lamina column spacing/seed were taken over from export_data.py constants into `params.SECTIONS["stimulus"]` (automatically included in docs/PARAMS.md; tests keep the document from going stale). For stimulus mechanisms and the real-image input route, see docs/STIMULUS.md.
