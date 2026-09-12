"""Extracellular potential forward kernels (quasistatic point-current model).

A synaptic current enters the postsynaptic membrane at one location and returns
through the presynaptic membrane at another: a source-sink current pair. The
potential at an electrode is the superposition of all pairs

    phi(r) = (1 / 4 pi sigma) * sum_e y_e * (1/|r - r_post| - 1/|r - r_pre|)

with y_e > 0 meaning positive current flows into the membrane at r_post.
Geometry is static, so per-electrode edge coefficients are precomputed once and
each time step is a single vector dot product.

Units: positions in micrometers, currents in pA, sigma in S/m -> volts out.
"""

import numpy as np


class StaticPairField:
    def __init__(
        self,
        pre_pos: np.ndarray,
        post_pos: np.ndarray,
        electrodes: np.ndarray,
        sigma: float = 0.33,
    ):
        pre = np.asarray(pre_pos, dtype=np.float64) * 1e-6   # um -> m
        post = np.asarray(post_pos, dtype=np.float64) * 1e-6
        electrodes = np.atleast_2d(np.asarray(electrodes, dtype=np.float64)) * 1e-6
        coef = np.empty((len(electrodes), len(pre)))
        for k, r in enumerate(electrodes):
            inv_post = 1.0 / np.linalg.norm(post - r, axis=1)
            inv_pre = 1.0 / np.linalg.norm(pre - r, axis=1)
            coef[k] = (inv_post - inv_pre) / (4.0 * np.pi * sigma)
        self.coef = coef
        self.n_electrodes = len(electrodes)

    def field(self, y_pa: np.ndarray) -> np.ndarray:
        """Per-edge currents (pA) -> electrode potentials (volts)."""
        return self.coef @ np.asarray(y_pa, dtype=np.float64) * 1e-12

    def field_timeseries(self, y_series: np.ndarray) -> np.ndarray:
        """(n_t, n_edges) or (n_edges,) stacked states -> (n_t, n_electrodes)."""
        y = np.asarray(y_series, dtype=np.float64) * 1e-12
        return y @ self.coef.T
