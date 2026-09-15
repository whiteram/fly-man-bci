# Performance Analysis and Acceleration Roadmap (simulation and forward pipeline)

*(English translation — authoritative version: [../../docs/PERFORMANCE.md](../../docs/PERFORMANCE.md))*

Benchmark target: full-CNS export (45 channels, 10.5 s of biology time, 21,000
steps, ~111k neurons / ~3M synapse states / ~420k projected edge dipoles).
Measured total runtime **8–10 min**; breakdown below.

Hardware context: this machine has an NVIDIA RTX 4060 Ti (16 GB; FP32 ≈22
TFLOPS, 288 GB/s bandwidth) — the GPU route is realistically viable; requires
`pip install cupy-cuda12x`.

---

## 1. Runtime breakdown (measured anchors)

| Stage | Time | Share | Computational nature |
|---|---|---|---|
| Scalp kernel build (Legendre series, 45 electrodes × ~300k edge dipoles) | 351–361 s | ~55% | element-wise series summation, embarrassingly parallel, numpy single-threaded |
| Biology simulation loop (21,000 steps × 0.5 ms) | ~2–4 min | ~25% | per step: 3M synapse-state decay + mechanistic lamina currents + LIF + OU; numpy small-array step-by-step loop |
| Forward kernel application (one GEMV per 1 ms) | ~1–2 min | ~12% | (45 × 420k) coef @ y, float64 ≈ 152 MB — **memory-bandwidth-bound**, 10,500 calls ≈ 1.6 TB of memory traffic |
| Background EEG / assembly / disk writes | < 1 min | ~8% | — |

## 2. Bottleneck conclusions

1. **Kernel build**: parallelism is inherent (electrodes are independent);
   numpy simply is not using multiple cores/GPU;
2. **Biology loop**: the per-step workload is tiny (a few MFLOPs); the cost
   is Python call overhead and cache-unfriendly access — the time axis is
   inherently serial, so time itself cannot be parallelized, only the single
   step can be accelerated;
3. **Kernel application**: the GEMV is bandwidth-bound — compute power has
   nowhere to shine; the key is to **move less data**.

## 3. Route A: CPU structural optimization (no GPU, no precision risk, half a day of work)

| Item | Approach | Expected |
|---|---|---|
| A1 runtime edge-dipole pooling | cluster the 420k edge dipoles into 3–5k representative dipoles (weight = edge count; the forward kernel is smooth in position space, and the existing 300k subsample + reweight mechanism is already validated to <1% error) → the per-record GEMV becomes 45×5k (1.4 MB) | kernel application 100×↓ (~90 s → ~2 s) |
| A2 kernel build multiprocessing | the 45 electrodes are mutually independent → multiprocessing(8) | 361 s → 60–90 s |
| A3 float32 coefficients | halves the bandwidth | another ~2× on application |

**Key dividend (an extension of A1)**: the biology simulation's y(t) is
**independent** of the electrodes — after pooling, write Y(t) (5k × 10,500
float32 ≈ 210 MB) to disk once, and each electrode configuration
45/64/128/241 = one small GEMM (seconds). Multi-configuration export goes
from "one full run per configuration" to "one simulation + N second-scale
projections".

## 4. Route B: GPU (CuPy; the RTX 4060 Ti is entirely sufficient)

| Item | Content | Expected | Risk |
|---|---|---|---|
| B1 kernel build on CuPy | the code is already fully vectorized and CuPy is nearly drop-in; **must be FP32** (consumer-card FP64 is 1/64 throughput) | 351 s → 10–30 s | low (coefficient errors need an A/B comparison; expected <0.1%) |
| B2 GPU-resident kernel application | coef resident in GPU memory + y transferred per record (3.4 MB) | full export ~15 s | low; but after A1 this stage is no longer the bottleneck |
| B3 biology loop fully on GPU | the state variables (3M synapses ×8B ≈ 24 MB) fit into GPU memory easily; the **launch overhead** of 21k steps × ~10–30 kernels is the main term (compressible with CUDA Graphs / kernel fusion) | simulation loop ~2–3 min → ~10–20 s | **high**: ExponentialSynapses / mechanistic lamina / neurotransmitter gates all get rewritten, and the result must be A/B-compared point-by-point against the CPU calibrated trajectory |
| B4 Numba/Cython hotspots (the compromise replacing B3) | JIT only the synapse decay / mechanistic currents / delivery | 5–20× per step | medium |

## 5. Recommended combination

- **Near term (half a day)**: A1 + A2 + A3 → 45-channel export 12 min →
  **2–3 min**, and the marginal cost of 64/128/241 channels drops to seconds;
- **Mid term (1 day)**: B1 kernel build on GPU → full export ≈ **1 min**;
- **Long term (3–5 days, as needed)**: B3 full-GPU simulation → **seconds**;
  only worthwhile once 256+ channels or large-scale parameter sweeps become
  routine.

## 6. Red lines and validation

- No acceleration may change the calibrated dynamics: run trajectory A/B
  comparisons against the existing 45-channel export as the baseline
  (thresholds: scalp-potential error <0.1 μV or <1%);
- Put the FP32 conversion on record; the SNR numbers (currently 0.15 μV @FC6,
  demo video) are sensitive to amplitude scaling and must be recomputed
  before and after any acceleration;
- CuPy must match CUDA 12.x (driver 591.86 ✓), `pip install cupy-cuda12x`.

---

## 7. Implementation progress (Route A has landed)

Measured at smoke scale (17 electrodes × 352k dipoles, 400 ms protocol, 16
logical cores):

| Item | Before | After | Notes |
|---|---|---|---|
| Kernel build | 29 s | **5–6 s** | `four_sphere_rows`: fg tables cached by head geometry (previously 60 7×7 solutions recomputed per electrode per repetition); a running product R_n=(r0·re)^n replaces per-order pow (~1.3×); 45/17 electrodes parallelized with a thread pool (numpy element-wise ops release the GIL, ~5.8× on 16 cores) |
| Forward application | per-record f64 GEMV | **CHUNK=256 row-major buffer + f32 GEMM** | the per-record, per-group `coef@y` (bandwidth-bound) is merged into one GEMM per flush; records are written contiguously row-by-row into ybuf, and the flush feeds BLAS through a zero-copy transposed view. Smoke 400 records 23 s→17 s; the gain is larger for the full 10,500 records (originally estimated 1–2 min → seconds) |
| Precision A/B comparison | — | **L2 bit-identical, max rel 4.8e-7** | full-array A/B comparison of phi_scalp between the old and new paths; the deviation is exactly of f32-coefficient magnitude (a 20× margin under the <1% red line). Unit tests: CSF merge regression <1e-8, reference series <1e-10, batch/single-electrode bit-identical (tests/test_four_sphere_rows.py) |

Pre-existing bugs fixed along the way (all exposed by small-scale iteration):

1. the `pool_err = ...` assignment inside `record()` turned the closure
   variable into a local → the default (non-`--pool`) export path crashed
   with `UnboundLocalError` on the very first record;
2. `acc_s` was not zeroed on pooling-refinement retries → scalp potentials
   doubled wholesale after a failed retry;
3. the `snr_metrics` window [1500, 4500) exceeded the short-record length →
   smoke crash (now skipped based on n_field; the page already had null-value
   guards).

Measured at full scale (45 channels / 10.5 s / 21,000 steps / full CNS 5
regions / **1.23M projected dipoles**):

| Stage | Old code (estimated) | New code (measured) | Notes |
|---|---|---|---|
| Kernel build | 15–20 min | **68 s** (~15×) | 16 groups × 45 electrodes batched + thread pool |
| Biology loop + forward application | ~60–75 min | **47 min** | redundant edge_currents calls removed (ker groups 4 calls/record → 1) + f32 GEMM; the remaining cost is the once-per-record 1.23M-edge current computation plus the synapse decay — **the target of routes B3/B4** |
| Precision A/B comparison | — | **scalp max 0.001 μV (0.092% of peak), corr 1.000000; phi bit-identical** | vs the old-code full-scale baseline; f32-coefficient magnitude, a 100× margin within the red line (<0.1 μV or <1%) |

**Important lesson (2026-09 incident)**: `regions_default` enables only
`visual_bilateral` (vpn_central etc. are off by default). Omitting
`--regions visual_bilateral,vpn_central,ol_rest,central_brain,vnc`
on a re-export silently produces "visual regions only" data: the page is left
with only the optic-lobe point cloud, and — because the RNG consumption
stream differs — the scalp trajectory decorrelates point-by-point against
the full-CNS baseline (33% peak difference but normal corr structure) — very
easy to misread as the acceleration having changed the physics. Exports now
print the ON/OFF region list at the start and write `meta.regions` into
viz_data.json.

Debug facilities: at the end of an export, `phi_scalp` (after ×1.7) and
fly-head `phi` are saved to the output directory as `_debug_phi_scalp.npy` /
`_debug_phi.npy` to serve as A/B baselines; the console prints per-stage
timings, the dipole count per group, buffer sizes, and L2 checksums.
