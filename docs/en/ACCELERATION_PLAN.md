# Compute Acceleration Roadmap (next phase, on hold)

*(English translation — authoritative version: [../../docs/ACCELERATION_PLAN.md](../../docs/ACCELERATION_PLAN.md))*

Status (updated 2026-09-15): **P0 + P2 (phases 1 and 2) implemented and
verified** — numba 0.67 + CuPy 14.2 installed in the conda `ffbm` env
(GPU compute verified); seven JIT kernels (graded-synapse step /
conductance drive / edge currents / exponential-synapse decay+delivery
(delayed and non-delayed) / OU noise / LIF explicit and semi-implicit)
landed with **bitwise-identical trajectories** (unit parity + smoke
end-to-end A/B, tests/test_numba_parity.py), and the biology loop
measured at **2.38×** (600 ms full-region in-process A/B: 75.7 s →
31.8 s; full export est. 22.1 → 9.3 min). P3 (CuPy) remains. The
prerequisite work (Route A) is in PERFORMANCE.md §7.

---

## 0. Baseline facts (measured 2026-09)

The full export (45 channels / 10.5 s biology time / full CNS 5 regions /
1.23M projected dipoles) currently takes **~55 min** in total, broken down
as:

| Component | Share | Measured hotspots (scripts/profile_export_loop.py) |
|---|---|---|
| Per-step biology (21,000 steps) | ~70% | scipy CSR matvec in `to_neuron_drive` (~38 small matvecs per step, 32%) + exponential synapse decay (15%, gather + 4 passes creating full-array temporaries) |
| Per-millisecond forward recording (10,500 times) | ~28% | `edge_currents` computes currents over 1.23M edges; an f64 v_post **implicitly upcasts** the f32 state, doubling traffic |
| Other (delayed delivery/OU/LIF) | scattered | — |

Each step makes ~80-120 numpy calls, each allocating full-array temporaries.

## 1. Execution environment

**Run all data jobs in the conda `ffbm` environment**
(`C:\Software\Devel\Anaconda3\envs\ffbm\python.exe`,
Python 3.12.14 + numpy 2.5.3 + scipy 1.18.1). Certified: the phi_scalp
checksum from a smoke run in this environment is bit-identical to system
Python (numpy 2.4.4) (L2=4.798226e-05). The environment **does not yet
have** numba/cupy/torch; P0 installs them.

## 2. Phased plan

### P0 environment preparation (half an hour)
```bash
C:\Software\Devel\Anaconda3\envs\ffbm\python.exe -m pip install "numba>=0.65" cupy-cuda12x
```
- numba ≥0.65 is required for numpy 2.5 (0.64 supports 2.4);
- cupy-cuda12x matches the driver's CUDA 12.x (driver 591.86 ✓, RTX 4060 Ti
  16GB);
- after installing, run one smoke pass each to certify that the checksum is
  unchanged.

### P1 CPU micro-optimizations (half a day, low risk)
- `edge_currents`: pre-convert v_post to f32 (removes the implicit upcast),
  pre-allocate buffers to eliminate temporaries;
- merge the two CSR matvecs in `to_neuron_drive` into a single pass
  (hand-written CSR loop);
- expected 55 → ~40 min; verified by bit-exact A/B comparison.

### P2 Numba JIT (phases 1+2 implemented 2026-09-15)
- **Landed (7 kernels)**: `_graded_step` (release-rate smoothing +
  weight, in-place), `_drive_cond` (the two conductance CSR matvecs
  fused into a single-pass f32 accumulation with per-row add order
  matching scipy csr_matvec), `_edge_currents` (f64 output so the
  downstream mixed-dtype dot stays on the exact same BLAS path),
  `_exp_step` / `_exp_step_delayed` (decay+delivery fused; the
  non-delayed path replicates numpy's BUFFERED fancy-add semantics —
  duplicate targets last-write-wins — and the delayed path replicates
  ascending-bin order with buffered bin-0 and add.at bins >= 1),
  `_ou_step` (noise still drawn by the numpy rng), `_lif_step` /
  `_lif_step_g32` (explicit + semi-implicit Euler; the f32-g_tot
  variant keeps the whole denominator rounding in f32 per numpy's
  weak-scalar promotion);
- **RNG stays in numpy throughout**: random sequence unchanged; bitwise
  identity verified (unit parity + smoke end-to-end FFBM_NUMBA=0/1 A/B
  + a first-step per-state probe);
- **Measured**: biology loop **2.38×** (31.8 s vs 75.7 s @600 ms
  full-region, scripts/profile_export_loop.py in-process A/B); full
  export est. 22.1 → 9.3 min (excluding the 68 s kernel build and
  assembly);
- **dtype-semantics traps on record** (what it takes to replicate
  numpy bitwise): (1) e_rev_edge must be a per-edge 1-D array (was a
  scalar / 0-d np.where result in the signless cases); (2) weak-scalar
  promotion keeps `g_unit * y` in f32 while `y *= np.float64 decay`
  multiplies in f64 then stores f32 (verified across all taus); (3)
  **L/MID receive f32 g_tot** (the drive arrays passed straight
  through) while T45/extra regions accumulate into f64 zeros — the
  semi-implicit denominator rounds differently, so the kernel
  dispatches on dtype; (4) an empty delivery selection must be an
  empty int array, never None (numba cannot type None); (5) the fancy
  add `y[t] += k[t]` is buffered (last-write-wins) — different from
  add.at accumulation — and each delivery path must use its own;
- **remaining candidates**: the release-rate sigmoids, PhotoCascade,
  spike-mask overhead — small shares now; head straight to P3.

### P3 CuPy full GPU (2-4 days, second stage)
- Architecture (dual precedent: FastFly + GeNN/Brian2GeNN): **all state f32
  and resident in GPU memory, zero host readback over the whole simulation**;
  ~15 kinds of element-wise operations via `cupy.fuse`/`RawKernel` (NVRTC
  runtime compilation, **no Visual Studio/nvcc needed**); the whole step
  captured as a `cupy.cuda.Graph` stream + replayed 21,000 times; the Y
  buffer and the 45×N GEMM stay on the GPU (cublas); only the 45×10,500
  scalp potentials are transferred back at the end;
- optional borrowings from FastFly: event-driven push for spike-driven
  groups (MT_*/V2C/CEN_*); FP16 weights (long term, requires separate
  calibration validation);
- expected **full export <2 min** (stepping is memory-bound, on the order of
  ~5-10 s);
- **validation criteria become statistical** (see the §3 RNG decision).

### P4 multi-trial parallelism (zero risk, can be added at any time)
- the SNR scenario at d′=2 needs ~1,630 trials → **throughput matters more
  than latency**;
- multiprocessing with 16 seeds in parallel, zero code changes,
  multiplicative with P2/P3 (16 cores).

## 3. ⚠️ Must be decided before B3: RNG stream consistency (a lurking pitfall found by external review)

The red-line text requires "point-by-point A/B comparison of GPU vs CPU
trajectories", but the CPU simulation's OU noise / delivery jitter comes
from a fixed-seed numpy random stream. After GPU-ification, choose one of
two options:

- **Option a (recommended)**: noise is still generated in CPU numpy and
  transferred to the GPU in batches (e.g. one batch per 100 steps)
  (pin memory; 21k steps × ~4 small arrays per batch ≈ ~100 MB of transfer
  in total, ~seconds) → the GPU trajectory is **bit-identical** to CPU, and
  the red line is unchanged;
- **Option b**: noise generated on the GPU with a CuPy counter-based RNG
  (philox) → the sequence differs from numpy's → the trajectories
  decorrelate statistically (the same phenomenon as the "missing --regions
  incident"), so **the red line must be changed to statistical criteria**
  (band-limited power spectra / correlation coefficients / rate trajectories
  + SNR magnitude recomputation), and this must be put on record.

Default is option a; consider b only if option a's transfers become the new
bottleneck.

## 4. Memory notes

- Default exact path: 45 × 1.23M float64 coefficients ≈ **443 MB** (already
  built in batches by group; in non-pool mode the f64 copy is released after
  use, keeping only the f32 application copy);
- Future EGI 256-channel full-length exact waveforms: coefficient memory ×5
  (~2.2 GB f64) — solvable with f32 + grouped batched placement, but it is
  not tolerant of low free runtime memory; run a real measurement first
  when the time comes.

## 5. Explicitly out of scope / rejected

- **B1 GPU kernel build**: the thread pool already achieves 68 s (15×); the
  marginal GPU benefit is ~1 min, not worth it;
- **migrating to Brian2/Brian2GeNN**: a ready-made GPU engine, but it would
  require rewriting custom models such as PhotoCascade, graded synapses,
  and delay loops; the drift risk far outweighs porting ~15 operators
  ourselves;
- **kernel pooling (A1)**: previously had a 24% accuracy problem; keep the
  status quo of explicit `--pool` opt-in + in-run validation.

## 6. First steps for resuming work

1. `git log --oneline -5` to confirm you are on a version after the one
   containing Route A;
2. Run P0 package installation + smoke certification;
3. For P1, edit `src/ffbm/simulation.py` (edge_currents / to_neuron_drive),
   running `scripts/profile_export_loop.py 600` at each step to see the
   gain;
4. For P2, JIT function by function, running tests/ + a smoke A/B comparison
   for each function before starting the next;
5. Full-scale validation command (in the background):
   `python viz/export_data.py --visual-input demo_bounce --elec-layout viz/data/elec_layout_1010.json --regions visual_bilateral,vpn_central,ol_rest,central_brain,vnc`
   (do not omit --regions!)

## 7. References

- FastFly (eonfathom): real-time single-GPU simulation of the fruit fly
  whole-brain connectome; a complete precedent of Python/CuPy + NVRTC
  runtime-compiled kernels
  <https://github.com/eonfathom/FastFly>
- Brian2GeNN (resident-state architecture, 35-400×):
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC6962409/>
- Brian2CUDA: <https://www.frontiersin.org/articles/10.3389/fninf.2022.883700/full>
- cupy.cuda.Graph: <https://docs.cupy.dev/en/latest/reference/generated/cupy.cuda.Graph.html>
- CuPy custom kernels (fuse/RawKernel): <https://docs.cupy.dev/en/latest/user_guide/kernel.html>
- numpy 2.5 requires numba ≥0.65: <https://github.com/numba/numba/releases>
- NVIDIA CUDA Graphs: <https://developer.nvidia.com/blog/cuda-graphs/>
