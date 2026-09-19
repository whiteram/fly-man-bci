"""Staged build caches for the export pipeline.

The export's fixed costs split into two independently-keyed stages so
that switching ONE input rebuilds only its own cache:

  circuit  -- ffbm.regions.build_circuit(regions) output.
              key = (region set, derived/raw data file stats, builder
              code).  Independent of electrodes and stimulus.
  kernels  -- the forward-kernel stage (group pairs, SCALE fit, scalp
              coefficient matrices, fly-head coefs, keep/reweight).
              key = (circuit key, electrode layout, head-model
              constants, KERNEL_PAIR_CAP, forward code).  Switching
              --elec-layout (10-10 -> 64ch -> EGI241 ...) hits only
              this stage.

The stack stage (build_stack, which draws the seed-dependent delay
jitter) is NOT cached: it is comparatively cheap and keyed by the seed.

Format: one .npz per stage (sequential array writes -- NOT pickle; a
4 GB object pickle measured >35 min to write on this machine) plus a
.json manifest.  Circuit dict serialization is schema-walking: numpy
arrays, pandas DataFrames (RangeIndex), nested dicts (int/str keys),
tuples/lists and scalars; anything else fails the write loudly and the
caller falls back to a fresh build.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"

# files whose contents define the circuit (derived tables first)
_CIRCUIT_DATA = [
    "derived/visual_nodes.parquet",
    "derived/visual_edges.parquet",
    "derived/neuron_sites.parquet",
    "raw/body-annotations.feather",
]
_CIRCUIT_CODE = [
    "src/ffbm/regions.py",
    "src/ffbm/data.py",
    "experiments/exp015_bilateral/circuit.py",
    "experiments/exp016_vpn_central/circuit2.py",
    "experiments/exp017_full_cns/circuit3.py",
    "experiments/exp019_chemosense/circuit4.py",
    "experiments/exp020_proprioception/circuit5.py",
    "experiments/exp021_audition/circuit6.py",
    "experiments/exp022_touch_haltere/circuit7.py",
    "experiments/exp023_thermal/circuit8.py",
    "experiments/exp005_medulla_ds/run.py",     # imported by exp015
]
_KERNEL_CODE = [
    "src/ffbm/forward.py",
    "src/ffbm/vizprep.py",
    "viz/export_data.py",                        # head consts live here
]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _stat_key(rel: str) -> str:
    p = ROOT / "data" / rel
    if not p.exists():
        return f"{rel}:missing"
    st = p.stat()
    return f"{rel}:{st.st_size}:{st.st_mtime_ns}"


def circuit_key(regions: dict | None) -> str:
    parts = [json.dumps(regions, sort_keys=True) if regions else "default"]
    parts += [_stat_key(p) for p in _CIRCUIT_DATA]
    parts += [_sha(ROOT / c) for c in _CIRCUIT_CODE if (ROOT / c).exists()]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def kernel_key(ckey: str, elec_layout_path: str | None,
               scalp_dirs: np.ndarray, cap: int) -> str:
    parts = [ckey, str(cap)]
    if elec_layout_path:
        parts.append(_sha(Path(elec_layout_path)))
    else:
        parts.append(hashlib.sha256(
            np.ascontiguousarray(scalp_dirs, np.float64).tobytes()
        ).hexdigest()[:16])
    parts += [_sha(ROOT / c) for c in _KERNEL_CODE if (ROOT / c).exists()]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def _path(stage: str, key: str) -> Path:
    return CACHE_DIR / f"{stage}-{key}.npz"


def have(stage: str, key: str) -> bool:
    return _path(stage, key).exists()


# ---------------------------------------------------------------------
# circuit dict <-> npz

def _walk(prefix: str, obj, arrays: dict, meta: dict):
    if isinstance(obj, np.ndarray):
        arrays[prefix] = obj
        meta[prefix] = {"k": "arr", "dt": str(obj.dtype)}
    elif isinstance(obj, pd.DataFrame):
        arrays[f"{prefix}#__idx"] = obj.index.to_numpy()
        for col in obj.columns:
            arrays[f"{prefix}::{col}"] = obj[col].to_numpy()
        meta[prefix] = {"k": "df", "cols": [str(c) for c in obj.columns]}
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        if keys and all(isinstance(k, int) or hasattr(k, "item")
                        and isinstance(k.item(), int) for k in keys[:1]):
            # int-keyed position dicts (pre_pos/post_pos) -> two arrays
            ks = np.array([int(k) for k in keys], np.int64)
            vs = np.stack([np.asarray(v, np.float64) for v in obj.values()])
            arrays[f"{prefix}#k"] = ks
            arrays[f"{prefix}#v"] = vs
            meta[prefix] = {"k": "idict"}
        else:
            meta[prefix] = {"k": "dict", "keys": keys}
            for k, v in obj.items():
                _walk(f"{prefix}.{k}", v, arrays, meta)
    elif isinstance(obj, (list, tuple)):
        meta[prefix] = {"k": "seq", "n": len(obj),
                        "tuple": isinstance(obj, tuple)}
        for i, v in enumerate(obj):
            _walk(f"{prefix}[{i}]", v, arrays, meta)
    elif isinstance(obj, (bool, int, float, str, np.generic)) \
            or obj is None:
        if isinstance(obj, np.generic):
            obj = obj.item()
        meta[prefix] = {"k": "scalar", "v": obj, "t": type(obj).__name__}
    else:
        raise ValueError(f"{prefix}: unserializable {type(obj)}")


def _rebuild(prefix: str, meta: dict, data) -> object:
    m = meta[prefix]
    k = m["k"]
    if k == "arr":
        return data[prefix]
    if k == "df":
        return pd.DataFrame({c: data[f"{prefix}::{c}"]
                             for c in m["cols"]},
                            index=data[f"{prefix}#__idx"])
    if k == "idict":
        ks = data[f"{prefix}#k"]
        vs = data[f"{prefix}#v"]
        return {int(a): vs[i] for i, a in enumerate(ks)}
    if k == "dict":
        return {key: _rebuild(f"{prefix}.{key}", meta, data)
                for key in m["keys"]}
    if k == "seq":
        items = [_rebuild(f"{prefix}[{i}]", meta, data)
                 for i in range(m["n"])]
        return tuple(items) if m["tuple"] else items
    if k == "scalar":
        v = m["v"]
        return {"bool": lambda: bool(v), "int": lambda: int(v),
                "float": lambda: float(v), "str": lambda: v,
                "NoneType": lambda: None}[m["t"]]()
    raise ValueError(f"{prefix}: bad manifest entry {m}")


def _referenced_ids(circuit: dict) -> set[int]:
    """Every body id the pipeline looks up in pre_pos/post_pos."""
    refs: set[int] = set()
    for k in ("r_ids", "l_ids", "mid_ids", "t45_ids"):
        refs |= set(np.asarray(circuit[k]).tolist())
    for name, spec in (circuit.get("extra_pops") or {}).items():
        refs |= set(np.asarray(spec["ids"]).tolist())
    tables = [circuit["e_rl"], circuit["e_lm"]]
    tables += list(circuit["e_mt"].values())
    tables += [s["table"] for s in
               (circuit.get("extra_edges") or {}).values()]
    for t in tables:
        refs |= set(t["body_pre"].to_numpy(np.int64).tolist())
        refs |= set(t["body_post"].to_numpy(np.int64).tolist())
    return refs


def _prune_positions(circuit: dict) -> None:
    """pre_pos/post_pos cover every raw synapse witness body (87M+ for
    post); the pipeline only ever looks up circuit bodies.  Pruning is
    semantics-preserving (a needed-but-missing key would KeyError) and
    shrinks the cache from ~GBs to ~MBs."""
    refs = _referenced_ids(circuit)
    for key in ("pre_pos", "post_pos"):
        d = circuit[key]
        circuit[key] = {k: v for k, v in d.items() if k in refs}


def save_circuit(circuit: dict, key: str) -> None:
    circuit = dict(circuit)              # shallow copy: prune locally
    _prune_positions(circuit)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {}
    for name, obj in circuit.items():
        _walk(name, obj, arrays, meta)
    t0 = time.perf_counter()
    np.savez(_path("circuit", key), **arrays)
    (_path("circuit", key).with_suffix(".json")).write_text(
        json.dumps(meta), encoding="utf-8")
    print(f"circuit cache: saved {_path('circuit', key).name} "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)


def load_circuit(key: str) -> dict:
    p = _path("circuit", key)
    t0 = time.perf_counter()
    data = np.load(p)
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    circuit = {name: _rebuild(name, meta, data)
               for name in meta if "." not in name and "[" not in name
               and "#" not in name and "::" not in name}
    print(f"circuit cache: HIT {p.name} "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)
    return circuit


# ---------------------------------------------------------------------
# kernel stage payload <-> npz

def save_kernels(payload: dict, key: str) -> None:
    """payload: flat {name: ndarray or scalar}; group pairs under
    'gp.<g>.0'/'gp.<g>.1', coefs under 'coef.<g>', fly coefs 'fly.<g>',
    keeps under 'keep.<g>' (empty array = None), scalars in the json."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays, scalars = {}, {}
    for name, v in payload.items():
        if isinstance(v, np.ndarray):
            arrays[name] = v
        else:
            scalars[name] = v          # json-native (float/str/list)
    t0 = time.perf_counter()
    np.savez(_path("kernels", key), **arrays)
    (_path("kernels", key).with_suffix(".json")).write_text(
        json.dumps(scalars), encoding="utf-8")
    print(f"kernel cache: saved {_path('kernels', key).name} "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)


def load_kernels(key: str) -> tuple[dict, dict]:
    p = _path("kernels", key)
    t0 = time.perf_counter()
    data = np.load(p)
    scalars = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    arrays = {k: data[k] for k in data.files}
    print(f"kernel cache: HIT {p.name} "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)
    return arrays, scalars
