# Stimulus Protocol Documentation (Vision) — Implementation, Parameters, and "Real Image" Feasibility

*(English translation — authoritative version: [../../docs/STIMULUS.md](../../docs/STIMULUS.md))*

Scope: `viz/export_data.py` (natural stimulus protocol), `src/ffbm/pipeline.py`
(stimulus injection interface), `src/ffbm/params.py` (parameter registry, `stimulus`
section).

---

## 1. Overview: Vision Is Currently the Only Sensory Input

All sensory input of the current simulation is **vision**: light signal → R1–R6
photoreceptor population → lamina (L1/L3/Mi1/Tm) → T4/T5 → VPN → central brain.
Olfactory/tactile/auditory pathways exist in the connectome but are not assembled
(`ffbm.regions` enables only the visual regions), and no corresponding forward stimulus
models exist.

The stimulus entry point is a **pure function**:

```python
fp.simulate(circuit, cal, input_fn, ...)
# input_fn(t) -> brightness increment waveform sample for each photoreceptor (see §2)
```

The simulation calls `input_fn(t)` once every 0.5 ms to obtain the brightness λᵢ(t) of
each R cell (dimensionless, 1 = dark baseline), which is then converted into
photocurrent by the phototransduction cascade.

## 2. Brightness → Photocurrent Chain

```
λᵢ(t) (brightness, dimensionless)
   │  ×I_LUM = 150 pA        (overall gain, params: stimulus.i_lum)
   ▼
Two first-order low-pass stages (τ = 10 ms ×2 ≈ 20 ms receptor latency)
   │  + slow light adaptation (τ = 800 ms, 30% plateau sag)
   ▼
inc_f(t) (adapted photocurrent increment, pA)
   │  + I_R_BASE = 130 pA    (dark baseline current, params: working_point)
   ▼
i_photo = I_R_BASE + inc_f  →  injected current of the R cells (LIF input)
```

Brightness is by construction **non-negative** (lower bound 0.05), so negative light
intensity never occurs; the parameters of `PhotoCascadeVector` are shared by all R cells
(no heterogeneity), calibrated against the exp002 intensity series / exp004 flicker
fusion.

### 2.1 The "800 ms" Is the Adaptation Gain, Not a Delay

A common misreading needs clarifying: the signal path of `PhotoCascade.step` contains
only the `y1 → y2` two 10 ms low-pass stages — **the response latency to
flicker/transients is ≈ 20 ms**, consistent with measured fruit-fly R1-R6 (on the order
of 20–50 ms). The 800 ms branch is a **multiplicative slow gain**:

```
g(t):  in light (inc>0), slowly approaches 1−30% with τ=800 ms (light adaptation / desensitization)
       slowly recovers to 1 on return to darkness
output = g(t) × y2(t)
```

It only tracks the **mean-brightness envelope on sub-minute, supra-second timescales**
and turns the gain down — for fast 20–100 ms fluctuations g is nearly constant, so fast
signals pass through essentially intact. This is exactly what "light adaptation" means:
stepping into blazing sunlight does not blind the eye for a second; instead, sensitivity
drops over roughly 1 s. The real fly's multi-timescale adaptation (~100 ms fast
component + second-scale slow component + pupil pigment migration) is compressed into
this single 800 ms single-time-constant phenomenological model — the outcome of
calibrating exp002 against the ERG plateau-sag shape, not a response delay.

## 3. Epoch Protocol (Natural Stimulus, 10.5 s)

All protocol parameters live in **`src/ffbm/params.py` → `SECTIONS["stimulus"]`** (edit
them there; `python -m ffbm.params` updates docs/PARAMS.md; re-run the export for
changes to take effect).

| Epoch | Time (ms) | Brightness definition | Page color bar |
|---|---|---|---|
| dark | 0–1500 | λ = 1 (dark baseline) | blue-gray |
| flicker | 1500–4500 | λ = 1 + C·f(t), f is 1/f temporal noise | yellow |
| drift+ | 4500–6000 | 1/f spatial texture drifting forward along the preferred direction | green |
| drift− | 6000–7500 | **time-reversed** replay of the same movie | cyan |
| band+ | 7500–9000 | band-pass texture at the DS scale drifting forward | pink |
| band− | 9000–10500 | its time reversal | purple |

Here C = `stimulus.stim_contrast` = 2.0; the texture/noise generation seed
`stimulus.seed` = 42.

### Mathematical construction of each epoch

- **1/f flicker** `f(t)`: white-noise FFT → amplitude divided by f^(1/2) → inverse FFT
  → normalize. Uniform over the whole field (identical for all R cells), no spatial
  structure.
- **Spatial texture** `tex(x)`: 8192-point 1/f noise covering "eye span + drift
  travel".
- **drift epochs**: each R cell is sampled at **its own projected coordinate on the
  preferred axis**, λᵢ(t) = 1 + C·tex(xᵢ − v·s(t)), v = `drift_speed` = 0.1 μm/ms,
  with s(t) played forward/backward according to the epoch. This is a **spatially
  structured** stimulus: photoreceptors at different positions see different brightness
  time series, and their temporal order determines the direction response of T4/T5.
- **band epochs**: same as drift, but the texture is band-passed at k ∈ [1/45, 1/15]
  μm⁻¹ — set from the lamina cartridge spacing `lam_um` = 22 μm, which is exactly the
  characteristic scale of T4/T5 motion detection.
- **drift−/band−**: time-reversal controls — same power spectrum, reversed correlation
  structure, used to separate "directional response" from "power-spectrum response".

### Origin of the preferred direction e_ds (hexagonal-axis regression)

The preferred direction of T4/T5 is not assigned manually but regressed from connectome
annotations: the `assignedOlHex1/2` of left-lobe Mi1 (hexagonal ommatidial-axis
assignment) and the soma positions are least-squares fit to obtain the two
hexagonal-axis directions u and v, which are then combined as
e_ds = cos(−135°)·u + sin(−135°)·v (−135° carries over the DS calibration direction from
exp005). Only the left lobe is regressed — the hexagonal axes of the two lobes are
mirrored in physical x, so a joint regression would cancel out.

## 4. Overview of Modifiable Hyperparameters (stimulus section)

| Parameter | Default | Description |
|---|---|---|
| `seed` | 42 | random seed for the 1/f noise and texture |
| `t_epochs` | see §3 table | epoch table (name, start, stop); the page color bar follows epoch additions/removals automatically |
| `stim_contrast` | 2.0 | brightness modulation amplitude (× the baseline of 1) |
| `i_lum` | 150 pA | brightness → photocurrent overall gain (calibrated value) |
| `drift_speed` | 0.1 μm/ms | texture drift speed |
| `lam_um` | 22 μm | lamina cartridge spacing (band-pass center scale of the band epochs) |

## 5. Receiving "Human-Perceived Images": The Visual Input Module (implemented)

**Yes — and it is already implemented as a configurable, general-purpose visual input
module.** The catalog file `viz/data/visual_inputs.json` defines a number of "visual
inputs"; at export time one is selected with `--visual-input <id>`:

```json
{"inputs": [
  {"id": "natural_1d", "kind": "line_scan",
   "label": "Natural protocol (1/f flicker + drift)"},
  {"id": "demo_bounce", "kind": "video",
   "label": "Demo: bouncing ball (human video → fly eye)",
   "source": "data/stimuli/demo_bounce.npy",
   "fps": 30.0, "blur_px": 1.2, "lo": 0.05, "hi": 3.0}
]}
```

- `kind: "line_scan"` — the original natural protocol (§3), parameters in params.py;
- `kind: "video"` — human video → ommatidial sampling, processing pipeline:

```
(frames, height, width) grayscale .npy (display-referred, sRGB gamma encoded)
   │ 0) sRGB → linear decoding     — luma weighting and Gaussian blur are
   │                                 linear-domain operations
   │                                 (linearize_srgb: false disables this)
   │ 1) Gaussian blur σ=blur_px    — ommatidial receptive field (~5°),
   │                                 anti-aliasing
   │ 2) whole-clip quantile        — maps to [0,1] (norm_pct, default
   │    normalization                [1,99]; clips with drastic shot
   │                                 changes will saturate — use wider
   │                                 quantiles or per-shot processing)
   │ 3) eye-plane affine mapping   — each photoreceptor's eye-surface
   │                                 coordinate (xᵢ,yᵢ) (the projection
   │                                 of r_pos onto the a1/b1 basis),
   │                                 bilinearly interpolated into image
   │                                 pixels (order=1)
   ▼
λᵢ(frame index) = lo + (hi−lo)·frame[yᵢ, xᵢ]   (lo=0.05, hi=3.0)
   ▼
Afterwards everything is identical to §2 (phototransduction → LIF → synapses → forward kernel, zero changes)
```

Keys available for video entries: `fps`, `blur_px`, `lo`/`hi`, `margin` (edge margin
fraction for the eye-plane mapping), `norm_pct` (normalization quantiles, default
[1,99]), `linearize_srgb` (default true: uint8 frames are sRGB-decoded into the linear
domain), `loop` (default false; when true, playback loops after the simulation outlasts
the video instead of freezing on the last frame).

Adding a new "visual input" = append an entry to the catalog JSON + place a
(frames, height, width) `.npy`. Real-video conversion script:

```bash
pip install imageio imageio-ffmpeg
python viz/make_stimulus_from_video.py clip.mp4 \
       viz/data/stimuli/my_clip.npy --fps 30 --duration 10.5
```

Built-in demo: `viz/make_demo_stimulus.py` generates a human-watchable video of a
"bouncing ball + moving dark bar" (10.5 s × 30 fps), used with
`--visual-input demo_bounce`.

### 5.1 Dimension-by-Dimension Comparison (why the mapping is this way)

| Dimension | Human image | Fly R1-R6 pathway | Processing |
|---|---|---|---|
| Color | RGB | **no color vision** (R1–R6 carry brightness only) | grayscale conversion (luma weighting) |
| Space | vast numbers of pixels | ~2,265+1,112 photoreceptors = equivalent pixels | 5° receptive-field blur + ommatidial sampling |
| Field of view | full rectangular frame | the eye-surface region covered by the population is retained | eye-plane affine mapping (bounding rectangle covering the R population) |
| Brightness | arbitrary/HDR | λ ∈ [0.05, hi] | quantile normalization + lo/hi mapping |
| Time | 24–60 fps | 20 ms latency + adaptation | frame rate configurable; fast motion is low-passed biologically |

### 5.2 Synchronized Page Visualization (implemented)

Exporting a video input also produces `viz/data/stim_frames.bin`: two previews per
frame — the **human video** (downsampled source frames) and the **fly-eye sampling**
(a rasterized brightness map on the eye plane of all photoreceptor sample values). The
page's "stimulus view" panel (located above the stimulus brightness bar) plays the two
canvases frame-synchronously with the simulation time cursor and shares the same time
axis with the channel heatmap, firing-rate curves, and scalp-potential coloring — an
intuitive juxtaposition of "what the human sees / what the fly sees / how neurons and
EEG respond".

## 6. Other Modalities

None yet. To add olfaction etc. in the future, the upstream of the corresponding
sensory region in the connectome needs (a) an input model, (b) a region toggle plus
circuit assembly for that region (`ffbm.regions` already supports per-region
enabling/disabling), and (c) a forward kernel (the current-based functional applies
automatically). The stimulus entry point is likewise an `input_fn(t)`.
