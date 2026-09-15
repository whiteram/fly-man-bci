# Data

*(English translation — authoritative version: [../../docs/data.md](../../docs/data.md))*

All raw data comes from the official MaleCNS v1.0 release (HHMI Janelia FlyEM,
CC-BY 4.0) via public Google Cloud Storage links — no registration or token needed.

## Layout

- `data/raw/` — official release files, re-downloadable (gitignored)
- `data/derived/` — products of our scripts, reproducible (gitignored)

## Download

```bash
python scripts/download_data.py
```

## Raw files

| File | Size | Content |
|---|---|---|
| `body-annotations.feather` | 14 MB | 211,577 bodies × 36 annotation columns (type, superclass, somaLocation, dimorphism, ...) |
| `body-neurotransmitters.feather` | 43 MB | per-neuron neurotransmitter predictions (consensus_nt, ground truth for 85k) |
| `connectome-weights.feather` | 1.05 GB | edge list: 151,856,684 directed connections (body_pre, body_post, weight=synapse count) |
| `syn-points.feather` | 12.7 GB | all ~125M synapse sites: pre/post body, 3D position, ROI |

Source URL prefix:

```
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
```

Official site: <https://male-cns.janelia.org/> — cite Berg et al., Cell (2026).

## Derived files

Produced by `python scripts/build_visual_subnetwork.py`:

- `visual_nodes.parquet` — the 105,265 visual-system neurons with type, NT, dimorphism
- `visual_edges.parquet` — 12.49M edges within the visual system, with NT sign
