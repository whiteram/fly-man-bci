"""Region-optional circuit assembly (the on/off switch for brain parts).

The thought experiment can include any subset of the fly CNS regions we
have builders for; regions that are OFF are simply absent from the
circuit dict, so the pipeline builds no populations, no synapses and no
forward kernels for them (zero compute and zero memory).

Usage:
    from ffbm.regions import build_circuit, DEFAULT_REGIONS
    circuit = build_circuit()                          # default set
    circuit = build_circuit({"vpn_central": True})     # visual + VPN/CB
    circuit = build_circuit({"visual_bilateral": False,
                             "vpn_central": True})     # (visual stays on
                                                       #  implicitly -- see
                                                       #  note in build)

Region builders register here as they are added (olfaction, taste->MN9,
central brain, VNC ...). A region builder receives the circuit assembled
so far and returns it extended with extra_pops/extra_edges specs (the
generic optional-layer interface of ffbm.pipeline.build_stack).

The experiment-module imports below are deliberate: circuit builders
live with their experiments (validated there), and this module is the
single place that knows how to compose them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# name -> (module file, builder attr); order = assembly order
_BUILDERS = {
    "visual_bilateral": (ROOT / "experiments" / "exp015_bilateral"
                         / "circuit.py", "build_bilateral_circuit"),
    "vpn_central": (ROOT / "experiments" / "exp016_vpn_central"
                    / "circuit2.py", "build_vpn_circuit"),
    "ol_rest": (ROOT / "experiments" / "exp017_full_cns"
                / "circuit3.py", "build_ol_rest"),
    "central_brain": (ROOT / "experiments" / "exp017_full_cns"
                      / "circuit3.py", "build_central_brain"),
    "vnc": (ROOT / "experiments" / "exp017_full_cns"
            / "circuit3.py", "build_vnc"),
    "olfactory": (ROOT / "experiments" / "exp019_chemosense"
                  / "circuit4.py", "build_olfactory"),
    "gustatory": (ROOT / "experiments" / "exp019_chemosense"
                  / "circuit4.py", "build_gustatory"),
    "proprioception": (ROOT / "experiments" / "exp020_proprioception"
                       / "circuit5.py", "build_proprioception"),
    "audition": (ROOT / "experiments" / "exp021_audition"
                 / "circuit6.py", "build_audition"),
    "touch": (ROOT / "experiments" / "exp022_touch_haltere"
              / "circuit7.py", "build_touch"),
    "thermal": (ROOT / "experiments" / "exp023_thermal"
                / "circuit8.py", "build_thermal"),
}

# vnc/central_brain/ol_rest share exp017's caches: assemble them
# together (ol_rest feeds central, central feeds vnc); the chemosensory
# input regions come last -- they drive INTO central (antennal lobe /
# SEZ partners live in central_brain), so central must be built first
_ORDER = ("visual_bilateral", "vpn_central", "ol_rest", "central_brain",
          "vnc", "olfactory", "gustatory", "proprioception", "audition",
          "touch", "thermal")

DEFAULT_REGIONS = {"visual_bilateral": True, "vpn_central": False,
                   "ol_rest": False, "central_brain": False,
                   "vnc": False, "olfactory": False, "gustatory": False,
                   "proprioception": False, "audition": False,
                   "touch": False, "thermal": False}


def _load(module_file: Path, attr: str):
    """Load (and cache in sys.modules) so several region builders from
    the same module share their heavy tables."""
    import sys
    name = f"_{module_file.stem}_{module_file.parent.name}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, module_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return getattr(sys.modules[name], attr)


def known_regions() -> list:
    return sorted(_BUILDERS)


def build_circuit(regions: dict | None = None):
    """Assemble the circuit for the requested region set.

    regions: {name: bool}; missing entries fall back to DEFAULT_REGIONS.
    Returns (circuit, active_region_names). visual_bilateral is the base
    cascade and the driver of every downstream region we currently
    have -- switching it off while a downstream region is on is rejected
    (a lone VPN layer without its input would be meaningless).
    """
    cfg = dict(DEFAULT_REGIONS)
    if regions:
        unknown = set(regions) - set(_BUILDERS)
        if unknown:
            raise ValueError(f"unknown region(s) {sorted(unknown)}; "
                             f"known: {known_regions()}")
        cfg.update({k: bool(v) for k, v in regions.items()})
    if not cfg["visual_bilateral"] \
            and any(on for n, on in cfg.items() if n != "visual_bilateral"):
        raise ValueError("visual_bilateral is the input driver of every "
                         "currently known downstream region; it cannot "
                         "be switched off alone")
    for chem in ("olfactory", "gustatory", "proprioception", "audition",
                "touch", "thermal"):
        if cfg[chem] and not cfg["central_brain"]:
            raise ValueError(f"{chem} drives its central partners "
                             "(antennal lobe / SEZ / ascending "
                             "targets) inside the central_brain region "
                             "-- enable central_brain with it")
    active = [name for name, on in cfg.items() if on]

    circuit = None
    for name in _ORDER:
        if not cfg.get(name):
            continue
        builder = _load(*_BUILDERS[name])
        circuit = builder(circuit) if circuit is not None else builder()
    return circuit, active
