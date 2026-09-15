# Electrode Configuration File Reference (ELEC_CONFIGS)

*(English translation — authoritative version: [../../viz/ELEC_CONFIGS.md](../../viz/ELEC_CONFIGS.md))*

The scalp electrode layout of this project is described by a single configuration
catalog file: `viz/data/elec_configs.json`. The page loads it automatically at
startup; the "Electrode configuration" dropdown at the top of the layers panel
switches between the various configurations in real time; the visualization
pipeline consumes the same layout via `--elec-layout`.

## 1. Hierarchy of the systems

| System | Electrode spacing | Typical lead count | This project's config |
|---|---|---|---|
| 10-20 (Jasper 1958) | 20% | 19–21 | `std_1020_45` (45 leads: 10-20 + commonly used 10-10 midline/intermediate positions) |
| 10-10 (Nuwer 1998 et al.) | 10% | 64–81 | `std_1010_64` (classic 64-lead 10-10 cap) |
| 10-5 (Oostenveld & Praamstra 2001) | 5% | 128–256+ | `std_105_128` (128-lead class) |
| Commercial Geodesic nets | geodesic distribution | 128/256 | `egi_256` (EGI HydroCel GSN 256, 241 positions after face/neck-cone filtering) |

- **10-10** bisects the 20% spacing of each chain of the 10-20 system (10% spacing).
- **10-5** bisects once more on top of 10-10 (5% spacing). Commercial 128-lead caps
  (EGI/BioSemi, etc.) use their own numbering (E1…E128 / A1…); the academic
  equivalent is the 10-5 system. MNE-Python's `standard_1005` montage contains
  all 343 official 10-05 position names and coordinates.
- This project's `std_105_128` = take the midpoint on each 10% chain of the
  64-lead cap (a 5% subdivision); the new position names are obtained by matching
  against official names in MNE `standard_1005` (including the composite names
  AFF/FFC/CCP/CPP/PPO/TPP/TTP/POO and the half-distance names with the `h` suffix).
- **`egi_256`** is the true geometry of the EGI HydroCel GSN 256 net cap
  (MNE `EGI_256` montage, official numbering E1…E256), transformed into this
  project's calibration frame by a rigid alignment whose handedness is determined
  from the montage's own nasion/lpa fiducials; the skirt sites below the neck
  cone (40°) are removed, leaving 241 scalp positions. Note: EGI's E numbering is
  a vendor-defined order with **no anatomical meaning**; this configuration
  differs geometrically from the rule-derived 10-x caps (geodesic nets are
  naturally sparser along the frontal midline) and supports viewing and
  interpolation preview only — not 10-20 rule-based calibration.

## 2. File format

```json
{
  "_schema": "explanatory text",
  "version": 3,
  "frame": { "ni_arc_deg": 223.6, "yaw_deg": 0.0, "roll_deg": -0.3 },
  "configs": [
    {
      "id": "std_1020_45",          // unique ID
      "label": "10-20 system",      // name shown in the UI
      "kind": "std_1020_45",        // kind: see table below
      "count": 45,                  // channel count (maintained automatically)
      "channels": {                 // channel name -> unit direction vector [x, y, z]
        "Cz": [0.04023, 0.15186, -0.98758],
        "..."
      }
    }
  ]
}
```

### Field descriptions

| Field | Required | Description |
|---|---|---|
| `id` | ✓ | unique identifier, absolutely must not be duplicated; recommended format `custom-name_lead-count` |
| `label` | ✓ | the name shown in the dropdown |
| `kind` | ✓ | `std_1020_45` / `std_1010_64`: derived from the 10-20 rules, the calibration sliders are available; `custom` (or any other value): shown as-is from the coordinates, rule-based calibration is disabled |
| `count` | ✓ | number of channels; must equal the number of entries in `channels` |
| `channels` | ✓ | the 3D unit direction vector of each channel (from the sphere center toward the scalp electrode position); the coordinate order matches the exported data |

In `channels`, the **key order = the data channel order** (heatmap row order,
export row order); changing the order is equivalent to reordering the data
columns — be careful.

## 3. Usage

### Switching in the page
Select from the "Electrode configuration" dropdown at the top of the layers panel
(expand it via ⏵ in the top-left). After a switch, the electrodes, name labels,
scalp potential coloring, and channel heatmap are rebuilt immediately. Only
configurations marked "exported" have exact waveforms; for all other
configurations, the waveforms of the additional channels are spherical-distance
interpolation previews (the title notes this).

### Exact export
Once you have settled on a configuration, export it on its own as a layout file
and hand it to `export_data.py`:

```bash
python -c "import json;from pathlib import Path;c=json.loads(Path('viz/data/elec_configs.json').read_text());k=[x for x in c['configs'] if x['id']=='std_1010_64'][0];Path('viz/data/elec_layout_active.json').write_text(json.dumps(k['channels']))"
conda activate ffbm
python viz/export_data.py --elec-layout viz/data/elec_layout_active.json
```

Timing reference (dominated by kernel construction; scales linearly with lead
count; measured 361 s for 45 leads):

| Config | Kernel-build estimate | Full-export estimate |
|---|---|---|
| 45 leads | ~6 min | ~10 min |
| 64 leads | ~8.5 min | ~13 min |
| 128 leads | ~17 min | ~25 min |
| EGI 241 positions | ~32 min | ~45 min |

### Custom layouts
1. Edit `viz/data/elec_configs.json` and append an entry following the format
   above (use `"custom"` for `kind`);
2. Refresh the page, then simply select it in the dropdown;
3. Note: the direction vectors in `channels` must be normalized (unit length),
   and the key names must not duplicate existing channels.

### Regenerating the built-in configurations
The three built-in configurations are derived from the current calibration frame
(handle/ni/yaw/roll from `calib_1020_export.json`, falling back to the ANCHORS
constant inside the script when absent):

```bash
python viz/make_elec_configs.py
```

After recalibrating (changes to ni/th/yaw/roll), run it once and the three
configurations will be updated in sync to the new frame.
Dependency: naming the 10-5/128 configuration requires MNE-Python
(`pip install mne`); if it is not installed, that configuration is skipped and
the rest are generated as usual.

## 4. Relationship between calibration and configurations

- The 10-20 calibration mode (the "10-20 calibration" button on the page) adjusts
  the **rule-derivation parameters** (ni, th, yaw, roll) and is effective only
  for the two rule-based configurations `std_1020_45` / `std_1010_64`;
- After adjusting, "Copy JSON" hands the result to the pipeline, which regenerates
  this file and the baked layout;
- Viewpoint calibration (the "viewpoint calibration" button) is unrelated to the
  electrode configurations; it only affects the camera directions of the six
  orthographic views.
