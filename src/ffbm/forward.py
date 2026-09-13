"""Extracellular potential forward kernels (quasistatic point-current model).

A synaptic current enters the postsynaptic membrane at one location and returns
through the presynaptic membrane at another: a source-sink current pair. The
potential at an electrode is the superposition of all pairs

    phi(r) = (1 / 4 pi sigma) * sum_e y_e * (1/|r - r_pre| - 1/|r - r_post|)

with y_e > 0 meaning positive current flows into the membrane at r_post (a
medium sink, negative potential) and returns into the medium at r_pre (source).
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
            inv_pre = 1.0 / np.linalg.norm(pre - r, axis=1)
            inv_post = 1.0 / np.linalg.norm(post - r, axis=1)
            # y>0: current enters the membrane at post (extracellular sink, -)
            # and returns into the medium at pre (source, +)
            coef[k] = (inv_pre - inv_post) / (4.0 * np.pi * sigma)
        self.coef = coef
        self.n_electrodes = len(electrodes)

    def field(self, y_pa: np.ndarray) -> np.ndarray:
        """Per-edge currents (pA) -> electrode potentials (volts)."""
        return self.coef @ np.asarray(y_pa, dtype=np.float64) * 1e-12

    def field_timeseries(self, y_series: np.ndarray) -> np.ndarray:
        """(n_t, n_edges) or (n_edges,) stacked states -> (n_t, n_electrodes)."""
        y = np.asarray(y_series, dtype=np.float64) * 1e-12
        return y @ self.coef.T


class SealedHeadPairField:
    """Source-sink pair kernel in a sealed two-layer spherical conductor.

    Concentric geometry: inner sphere (CNS, radius r1, sigma1) holding all
    sources and electrodes; shell (outer radius r2, sigma2); sealed outer
    boundary (no radial current -- insect cuticle). The point-source
    potential in the inner sphere is the Legendre series

        V(r, th) = (I / 4 pi sigma1) sum_n P_n(cos th)
                   [ r_<^n / r_>^{n+1} + a_n(r0) r^n ]

    with r_< = min(r, r0), r_> = max(r, r0); a_n is linear in the source
    amplitude ~ r0^n and solves the 3x3 boundary system (V and sigma dV/dr
    continuous at r1; zero radial current at r2). Same API and sign
    conventions as StaticPairField; converges to it as r2/r1 -> infinity
    with sigma2 = sigma1.
    Units: positions um, sigma S/m -> coefficients as StaticPairField.
    """

    def __init__(self, pre_pos, post_pos, electrodes, center, r1, r2,
                 sigma1=0.33, sigma2=0.033, n_terms=80):
        center = np.asarray(center, dtype=np.float64)
        pre = np.asarray(pre_pos, dtype=np.float64) - center
        post = np.asarray(post_pos, dtype=np.float64) - center
        elec = np.atleast_2d(np.asarray(electrodes, dtype=np.float64)) - center
        for arr, name in ((pre, "pre"), (post, "post"), (elec, "electrode")):
            r = np.linalg.norm(arr, axis=-1)
            if r.max() >= r1:
                raise ValueError(f"{name} outside inner sphere "
                                 f"(max r={r.max():.1f} >= r1={r1:.1f})")
        pre, post, elec = pre * 1e-6, post * 1e-6, elec * 1e-6   # um -> m
        r1_m = r1 * 1e-6
        r2n = r2 / r1                          # all radii normalized by r1

        def solve_a(n):
            """a_n for a unit source pattern (linear in the ~r0^n source
            term; radii normalized so r1 = 1). Row-equilibrated 3x3 solve.
            n = 0 (monopole) is inadmissible in a sealed sphere and cancels
            exactly for source-sink pairs -> returned as 0."""
            if n == 0:
                return 0.0
            src = 1.0
            dsrc = -(n + 1)
            m = np.zeros((3, 3))
            rhs = np.zeros(3)
            m[0] = [1.0, -1.0, -1.0]
            rhs[0] = -src
            m[1] = [sigma1 * n, -sigma2 * n, sigma2 * (n + 1)]
            rhs[1] = -sigma1 * dsrc
            m[2] = [0.0, n * r2n ** (n - 1), -(n + 1) * r2n ** (-(n + 2))]
            for i in range(3):                 # equilibrate rows
                s = np.abs(m[i]).max()
                m[i] /= s
                rhs[i] /= s
            return np.linalg.solve(m, rhs)[0]

        a_ns = np.array([solve_a(n) for n in range(n_terms)])
        n_orders = np.arange(n_terms)

        def series(r0n, cos_th, re_n):
            """Sum_n P_n(cos th) [ r_<^n/r_>^{n+1} + a_n r0^n r^n ], radii
            normalized by r1 (result carries units of 1/r1)."""
            r_less = np.minimum(r0n, re_n)
            r_greater = np.maximum(r0n, re_n)
            total = np.zeros(len(r0n))
            p_nm2 = np.ones(len(r0n))              # P_0
            p_nm1 = cos_th.copy()                  # P_1
            for n in n_orders:
                if n == 0:
                    p_n = p_nm2
                elif n == 1:
                    p_n = p_nm1
                else:
                    p_n = ((2 * n - 1) * cos_th * p_nm1
                           - (n - 1) * p_nm2) / n
                    p_nm2, p_nm1 = p_nm1, p_n
                geom = r_less ** n * r_greater ** (-(n + 1))
                reg = a_ns[n] * (r0n ** n) * (re_n ** n)
                total += p_n * (geom + reg)
            return total

        coef = np.empty((len(elec), len(pre)))
        for k, re in enumerate(elec):
            re_n = np.linalg.norm(re) / r1_m
            r0_pre = np.linalg.norm(pre, axis=1) / r1_m
            r0_post = np.linalg.norm(post, axis=1) / r1_m
            cpre = (pre @ re) / (np.linalg.norm(pre, axis=1)
                                 * np.linalg.norm(re))
            cpost = (post @ re) / (np.linalg.norm(post, axis=1)
                                   * np.linalg.norm(re))
            coef[k] = (series(r0_pre, cpre, re_n)
                       - series(r0_post, cpost, re_n)) \
                / (4.0 * np.pi * sigma1 * r1_m)
        self.coef = coef
        self.n_electrodes = len(elec)

    field = StaticPairField.field
    field_timeseries = StaticPairField.field_timeseries
