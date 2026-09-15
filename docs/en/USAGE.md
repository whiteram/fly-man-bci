# User Manual (Visualization Page + Data Export)

*(English translation — authoritative version: [../../docs/USAGE.md](../../docs/USAGE.md))*

Page entry: `viz/index.html` (after starting a local server with `python -m http.server 8613`,
visit `http://localhost:8613/index.html`). Data is provided by `viz/data/viz_data.json`;
for regeneration see §6.

---

## 1. Page Layout

- **Left 3D view**: transparent human head + fruit-fly visual-system point cloud inside
  the four-shell layers, scalp electrodes, and scalp-potential-colored spheres;
  left mouse button rotates, wheel zooms, right button pans.
- **Right info panel**: title and simulation description, playback controls, stimulus
  brightness bar, channel heatmap, population firing-rate curves for each layer.
- **Top-left button column**: ⏵ (layer panel, collapsed by default), head calibration,
  electrode editing, 10-20 calibration, view-direction calibration.
- **Time controls**: pause/play, replay, 1×/2×/4× speed, progress bar;
  the colored segments on the stimulus brightness bar correspond one-to-one with the
  current stimulus epochs (color definitions in docs/STIMULUS.md §3).

## 2. Layer Panel (expand ⏵)

| Item | Description |
|---|---|
| Electrode configuration | Dropdown: switch between 10-20×45 / 10-10×64 / 10-5×128 / EGI 241 montages (details in viz/ELEC_CONFIGS.md) |
| R1-R6 photoreceptors | First stage of the visual cascade (3,377) |
| L1-L3 lamina | Lamina neurons (5,327) |
| Mi1/Tm medulla interneurons | Medulla layer (14,358) |
| T4 / T5 | Direction-selective neurons (6,861/6,719) |
| VPN projection neurons | Vision → central brain (2,557) |
| Central brain targets (exp016) | Central brain targets of the VPNs (250) |
| Remaining brain regions (62,305) | Point clouds of brain regions that are not enabled (display only) |
| Midbrain (33,480) / ventral cord VNC (15,367) | exp017 whole-CNS extension; the VNC participates in the dynamics only and does not project to the scalp |
| Connection sampling | Brain↔VNC neck connection bundle (opacity 0.05; automatically hidden when either endpoint is hidden) |
| Scalp electrodes / head form / scalp-potential coloring | Electrode cylinders (with name labels) / LeePerrySmith transparent head model / red-blue potential spheres |
| Brain shell / CSF layer / skull shell / scalp shell | Fresnel outline layers of the four shells |

Labels are shown only for the hemisphere facing the viewer; unchecking "Scalp electrodes"
hides the labels as well.

## 3. Three Calibration Workflows (mutually independent, auto-saved)

### 3.1 Head Calibration (sliders)
Align the transparent head model to the electrode sphere: scale / pitch / yaw / roll /
three-axis translation sliders + six orthogonal view buttons (front / back / left /
right / top / chin / reset). Slider changes are saved immediately (localStorage
`headFormCalib_v3`); "Copy JSON" exports them as a hyperparameter snapshot.
The directions of the view buttons are determined by "view-direction calibration".

### 3.2 10-20 Calibration (electrode cap)
Six constrained handles + two global rotation sliders; **after every drag the whole cap
is rebuilt in real time according to the 10-20/10-10 proportional rules**:

| Handle/slider | DOF | Description |
|---|---|---|
| Gold Cz | 1 | Slides along the ear-plane circle = whole-cap fore-aft pitch |
| Gold Fz / Fpz / Oz | 1×3 | Slide along the sagittal arc; the dragged one re-determines the midline arc ni by the 10-20 percentages (20%/40%/40%), and the midline is re-spaced proportionally |
| Cyan A1 / A2 | 1 | Slide along the ear-position circle, automatically mirrored; the ear angle th drives the coronal ring (C3/C4=0.4·th, T7/T8=0.8·th) and all lateral chains |
| Yaw slider | 1 | Rotates the whole cap about the vertex axis (aligns to the nasal midline) |
| Roll slider | 1 | Tilts the whole cap about the nasal axis (toward the left/right ear) |

The panel displays ni / th / yaw / roll in real time. "Copy JSON" hands the state to
the pipeline to regenerate configurations and layouts (`viz/make_elec_layout.py` /
`make_elec_configs.py` consume it); "Restore standard position" returns to the baked-in
state. The state is stored in localStorage `calib1020` and applied automatically at page
startup (so a stale layout is never shown at startup).

### 3.3 View-Direction Calibration (six orthogonal views)
Drag the gold "front" handle (nasal direction) and the cyan "top" handle (vertex
direction); back / bottom / left / right are then derived automatically as orthogonals
(right = the A2 side). The front / back / left / right / top / chin buttons in the panel
give an instant preview. Stored in localStorage `viewDirs`.

### 3.4 Electrode Editing (free fine-tuning)
In "electrode editing" mode the 45 electrodes can be dragged directly (waveforms are
interpolated previews); red spheres add custom anchor points. Once adjusted, "Copy JSON"
hands the result to the pipeline.

## 4. Electrode Configuration System

Four sets in the dropdown: `std_1020_45` (waveforms already exported), `std_1010_64`,
`std_105_128`, `egi_256` (the latter three are interpolated previews — noted in the
title). Switching instantly rebuilds electrodes / labels / heatmap / scalp coloring.
For custom layouts, the format schema, and regeneration commands see
**viz/ELEC_CONFIGS.md**. Note: only the two rule-based configurations with 45/64
channels support the 10-20 calibration sliders; EGI/custom configurations show a
warning and are blocked.

## 5. Waveform Data: Exact vs Preview

- Page waveforms come from the last export (`viz/data/viz_data.json`) and are exact
  only for the **45 channels at export time**;
- After any calibration/dragging, channels that have moved show **preview waveforms**
  from spherical-distance interpolation (noted in the title/panel);
- Once finalized, re-export through the pipeline (§6) to get exact data. For numerical
  methods and error magnitudes see docs/TECHNICAL.md.

## 6. Data Export (pipeline)

```bash
conda activate ffbm
python viz/export_data.py --elec-layout viz/data/elec_layout_1010.json \
                          --regions visual_bilateral=True,vpn_central=True,...
```

- `--elec-layout`: any channel layout file (name → [x,y,z]; can be extracted from
  elec_configs.json, see ELEC_CONFIGS.md §Export);
- `--regions`: region toggles (`--help` lists the available regions);
- Runtime reference: kernel construction 361 s @45 channels, ≈513 s @64, ≈1030 s @128,
  ≈1930 s @EGI241 (linear; a full export adds simulation and background EEG on top);
- Running in the background is recommended; when it finishes, `viz/data/viz_data.json`
  is the page's new data.
- Stimulus protocol parameters (epochs/contrast/drift speed, etc.) are edited in the
  `stimulus` section of `src/ffbm/params.py`; see docs/STIMULUS.md for details.
- `--visual-input <id>`: select the visual input (`viz/data/visual_inputs.json`).
  Default `natural_1d` (1/f flicker + drift natural protocol); `demo_bounce` is the
  built-in "bouncing ball" grayscale video (human video → ommatidial sampling). Convert
  your own videos with `viz/make_stimulus_from_video.py` (requires imageio). Video
  inputs also produce `stim_frames.bin`; the page's "stimulus view" panel plays two
  synchronized canvases: human video | fly-eye sampled brightness map, sharing the time
  axis with the EEG/firing-rate curves.

## 7. FAQ

- **No electrodes/labels visible when the page opens**: check "Scalp electrodes" in the
  layer panel (expand ⏵).
- **Want to return to the delivered state after adjusting calibration**: click
  "Restore standard position" in the corresponding panel.
- **The six view directions are wrong**: enter "view-direction calibration" and drag the
  "front" and "top" handles.
- **Waveforms look wrong/very flat**: check whether the current configuration is an
  "interpolated preview"; exact waveforms require re-export per §6.
