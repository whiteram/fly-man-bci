# Compute Acceleration Roadmap (next phase, on hold)

*(English translation — authoritative version: [../../docs/ACCELERATION_PLAN.md](../../docs/ACCELERATION_PLAN.md))*

Status (updated 2026-09-16): **P0 + P2 + P3 all landed** —
P2's seven numba kernels are bitwise-identical (biology loop 2.38×);
**the P3 CuPy full-GPU route is complete**: the resident engine in
`src/ffbm/gpu.py` plus the `viz/export_data.py --gpu` record path have
trajectories **bit-identical to CPU** (three-level parity: synthetic
tiny circuit, real visual_bilateral, full CNS), and the full export's
biology loop + forward recording drops **47 min → 156 s (~18×)**,
end-to-end ~52 → **~7.5 min** (the 201 s assembly and 91 s kernel
build stay on CPU); phi differs only at rel 3.4e-7 / corr 1.000000000
(cublas summation order — the statistical criterion, ~3000× inside the
red line). Remaining optional item: P4 multi-trial parallelism (SNR
throughput). The prerequisite work (Route A) is in PERFORMANCE.md §7.

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
- **validation criteria become statistical** (see the §3 RNG decision);
- **P3-1 PoC passed (2026-09-16, scripts/poc_cupy_parity.py, 16/16
  bitwise)**: RawKernels for the three LIF variants / OU / graded /
  delivery scatter match the numba reference bit-for-bit. Key findings:
  (1) NVRTC **must** compile with `-fmad=false` (default fma contraction
  breaks OU by ~1e5, reproduced); (2) the drive must be restructured as a
  **post-major CSR** — numba accumulates in global edge order, GPU atomics
  would reorder; a stable argsort(post) CSR with one serial thread per
  post reproduces the order bitwise (one-time build, like Route A);
  (3) graded's numba typing: the clip literals 0.0/1.0 unify `r` to
  **f64**, so `(r - s)` is an exact f64 subtraction; (4) delivery
  targets are unique (pre rows are disjoint edge ranges) →
  gather-add-scatter is race-free, atomics-free, deterministic;
  (5) NVRTC has no system headers (no stdint.h — use `long long`);
  CuPy 14 calls RawKernels as `kernel(grid, block, args_tuple)`;
  (6) latent note: graded's numba and pure-numpy paths differ at ~1e-7
  on synthetic data (numba's k is f64, numpy weak-scalar keeps f32) —
  not triggered in production (r_pre∈[0,1]); the GPU reference is
  numba (the production path).
- **P3-2 measurements (2026-09-16, scripts/poc_cupy_graph.py)**:
  (1) **CUDA Graphs give no benefit** (eager vs graph <2% — launch
  overhead is negligible at real kernel runtimes) → dropped, simpler
  engineering; (2) at a synthetic random layout the per-post gather
  kernel costs 61.6 ms/step (random gather is a latency floor: 64M
  random reads ~29 ms; unrolling/int32 don't help); (3) **the
  production layout has no such problem** — y is stored post-sorted
  (`lexsort((pre, post))`), so the drive is a streaming segment sum
  (6.8 ms at 64M edges, bitwise-identical order to the numba global
  edge scan), no permute/gather; (4) edge_currents' real scale is the
  1.23M projection edges (the PoC mistakenly used 64M); (5) estimated
  production-layout step ~16 ms → 21k steps ≈ 5.6 min; fusion
  (decay+ring+clear) could halve traffic. Conclusion: build P3-3 with
  the production layout and measure before optimizing further.
- **P3-3 engine landed (2026-09-16, src/ffbm/gpu.py)**: `GPUTrial`
  keeps the whole mech-branch loop state on the GPU — RNG option a is
  implemented (noise pre-drawn on CPU in the exact per-step order
  R→L→MID→T45→extras, uploaded in 512-step batches; the numpy stream is
  unchanged); drives are streaming segment sums on the production
  post-sorted layout; delivery is one thread per presynaptic row
  (disjoint ranges, no atomics), the ring pointer advances BEFORE
  delivery (numba order, bin b lands in row (ptr+b-1)%len). **Verified**:
  (a) synthetic tiny circuit (with an extra region and a dual-source
  group) — 60/60 sampled steps bitwise on all states (v/refrac/s/y/
  buffer/ptr/OU-x, scripts/gpu_loop_parity.py); (b) real
  visual_bilateral circuit — 200 steps bitwise, **2.36 ms/step**
  (scripts/gpu_real_parity.py). One critical bug found and fixed: the
  LIF kernels must consume the post-update OU state x, not the raw
  noise w.
- **P3-4 landed and fully verified (2026-09-16)**:
  `viz/export_data.py --gpu` — the record path is fully on-device
  (k_ecur_f32 computes per-edge currents with the CPU expression's
  dtype path and writes f32 ybuf rows directly, including a
  kernel_keep subsampled variant; flushes use cublas GEMM; rate
  statistics accumulate on device and transfer once, removing 5 host
  syncs per record). **Measured**: full CNS (150,601 neurons / 5
  regions) bare stepping **4.62 ms/step × 21,000 steps**; the full
  export's biology loop + forward recording is **156 s vs the CPU's
  47 min ≈ 18×**, end-to-end ~7.5 min; output vs the CPU baseline:
  scalp rel 3.4e-7, corr 1.000000000, fly-head rel 3.4e-10 (f32/f64
  GEMM summation order — the statistical criterion). At smoke scale
  GPU≈CPU (17 vs 16 s — launch/Python overhead dominates a tiny
  circuit; the GPU advantage appears at real scale). `--pool` and
  `--gpu` are mutually exclusive (explicit error at flush). Fixed
  along the way: a KeyError for the PHOTO group in keep_g; a 7.46 GiB
  `_indices` allocation failure when pytest ran concurrently with the
  full export (memory contention, passes on re-run — not a
  regression).

### P4 multi-trial parallelism (zero risk, can be added at any time)
- the SNR scenario at d′=2 needs ~1,630 trials → **throughput matters more
  than latency**;
- multiprocessing with 16 seeds in parallel, zero code changes,
  multiplicative with P2/P3 (16 cores).

### P5 staged build caches (landed 2026-09-16, src/ffbm/cache.py)
With the GPU loop at 156 s, the fixed costs (assembly ~3 min + kernel
build 91 s) dominate. Two **independently-keyed** caches; `--no-cache`
bypasses both:
- **circuit cache**: key = region set + data/ file stats + builder code
  hashes; a hit (~0.3 s) replaces the ~3 min assembly. npz stores plain
  arrays (NOT pickle — a 4 GB object pickle measured >35 min to write
  on this machine); position dicts are pruned to the bodies the circuit
  actually references (the raw post_pos covers 87.5M witness bodies →
  ~200k, shrinking the cache from 2.9 GB to MBs; semantics preserved —
  a needed-but-missing key raises KeyError rather than silently
  misaligning);
- **forward-kernel cache**: key = circuit key + **electrode layout**
  (--elec-layout file hash or the default-layout identity) + pair cap
  (smoke/full differ) + forward code; a hit (~1 s) replaces the 48-91 s
  build. **Switching electrode configurations rebuilds only this
  stage**;
- the stack stage (build_stack) is not cached: it draws the
  seed-dependent delay jitter and is comparatively cheap;
- **verified**: at smoke and full scale, cache-hit vs from-scratch
  _debug_phi* outputs are **bit-identical**;
- known quirk: the first run after touching files occasionally fails
  with an `llvmlite.dll` load error (a transient Windows
  handle/antivirus scan) — rerun; unrelated to the cache.

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
