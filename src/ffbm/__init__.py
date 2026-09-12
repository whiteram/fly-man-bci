"""ffbm — MaleCNS v1.0 connectome modeling infrastructure.

Modules:
    data       - loaders for raw MaleCNS v1.0 files and derived tables
    simulation - vectorized LIF populations and exponential synapses
    forward    - extracellular potential kernels (point source-sink pairs)
"""

from . import data, forward, simulation

__all__ = ["data", "forward", "simulation"]
__version__ = "0.1.0"
