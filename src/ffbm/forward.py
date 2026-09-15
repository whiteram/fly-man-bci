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


class ScalpPairField:
    """Source-sink pair kernel in a 3-layer head open to the scalp.

    Brain (r1, sigma1) / skull (r2, sigma2) / scalp (r3, sigma3), sealed by
    air outside r3 (no radial current), sources in the brain, electrodes in
    the scalp layer (r2 < r_e < r3). Per Legendre order n the five unknown
    regular/irregular coefficients solve a 5x5 boundary system (V and sigma
    dV/dr continuous at r1 and r2; zero radial current at r3); all are
    linear in the source term ~ r0^n. Classic human-EEG three-sphere model,
    same API and sign conventions as StaticPairField.
    """

    def __init__(self, pre_pos, post_pos, electrodes, center, r1, r2, r3,
                 sigma1=0.33, sigma2=0.013, sigma3=0.33, n_terms=60):
        center = np.asarray(center, dtype=np.float64)
        pre = np.asarray(pre_pos, dtype=np.float64) - center
        post = np.asarray(post_pos, dtype=np.float64) - center
        elec = np.atleast_2d(np.asarray(electrodes, dtype=np.float64)) - center
        for arr, name in ((pre, "pre"), (post, "post")):
            r = np.linalg.norm(arr, axis=-1)
            if r.max() >= r1:
                raise ValueError(f"{name} outside brain sphere")
        re_all = np.linalg.norm(elec, axis=1)
        if not ((re_all > r2) & (re_all < r3)).all():
            raise ValueError("electrodes must lie in the scalp layer "
                             "(r2 < r < r3)")
        pre, post, elec = pre * 1e-6, post * 1e-6, elec * 1e-6
        r1_m = r1 * 1e-6
        r2n, r3n = r2 / r1, r3 / r1

        def solve_de(n):
            """[a, b, c, d, e] for a unit source pattern (r0^n factored out,
            radii normalized r1 = 1). n = 0 (monopole) is inadmissible in a
            sealed head and cancels exactly for source-sink pairs -> 0."""
            if n == 0:
                return 0.0, 0.0
            src, dsrc = 1.0, -(n + 1)
            m = np.zeros((5, 5))
            rhs = np.zeros(5)
            m[0] = [1.0, -1.0, -1.0, 0.0, 0.0]
            rhs[0] = -src
            m[1] = [sigma1 * n, -sigma2 * n, sigma2 * (n + 1), 0.0, 0.0]
            rhs[1] = -sigma1 * dsrc
            m[2] = [0.0, r2n ** n, r2n ** (-(n + 1)), -r2n ** n,
                    -r2n ** (-(n + 1))]
            m[3] = [0.0, sigma2 * n * r2n ** (n - 1),
                    -sigma2 * (n + 1) * r2n ** (-(n + 2)),
                    -sigma3 * n * r2n ** (n - 1),
                    sigma3 * (n + 1) * r2n ** (-(n + 2))]
            m[4] = [0.0, 0.0, 0.0, n * r3n ** (n - 1),
                    -(n + 1) * r3n ** (-(n + 2))]
            for i in range(5):
                s = np.abs(m[i]).max()
                if s > 0:
                    m[i] /= s
                    rhs[i] /= s
            sol = np.linalg.solve(m, rhs)
            return sol[3], sol[4]          # d_n, e_n

        de = np.array([solve_de(n) for n in range(n_terms)])
        n_orders = np.arange(n_terms)

        def series(r0n, cos_th, re_n):
            """V in the scalp layer: sum_n P_n [d_n r0^n r^n + e_n r0^n
            r^-(n+1)] (normalized radii)."""
            total = np.zeros(len(r0n))
            p_nm2 = np.ones(len(r0n))
            p_nm1 = cos_th.copy()
            for n in n_orders:
                if n == 0:
                    p_n = p_nm2
                elif n == 1:
                    p_n = p_nm1
                else:
                    p_n = ((2 * n - 1) * cos_th * p_nm1
                           - (n - 1) * p_nm2) / n
                    p_nm2, p_nm1 = p_nm1, p_n
                reg = de[n, 0] * (r0n ** n) * (re_n ** n)
                irr = de[n, 1] * (r0n ** n) * (re_n ** (-(n + 1)))
                total += p_n * (reg + irr)
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
                       - series(r0_post, cpost, re_n))                 / (4.0 * np.pi * sigma1 * r1_m)
        self.coef = coef
        self.n_electrodes = len(elec)

    field = StaticPairField.field
    field_timeseries = StaticPairField.field_timeseries


_FG_CACHE = {}


def _four_sphere_fg(r1, r2, r3, r4, sigma1, sigma2, sigma3, sigma4,
                    n_terms):
    """Legendre (f_n, g_n) scalp-layer pair table for a head geometry,
    shared across all dipoles and electrodes (7x7 boundary solves per
    order, independent of source/electrode positions)."""
    key = (float(r1), float(r2), float(r3), float(r4), float(sigma1),
           float(sigma2), float(sigma3), float(sigma4), int(n_terms))
    hit = _FG_CACHE.get(key)
    if hit is not None:
        return hit
    r2n, r3n, r4n = r2 / r1, r3 / r1, r4 / r1
    fg = np.zeros((n_terms, 2))
    for n in range(1, n_terms):
        src, dsrc = 1.0, -(n + 1)
        m = np.zeros((7, 7))
        rhs = np.zeros(7)
        m[0] = [1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        rhs[0] = -src
        m[1] = [sigma1 * n, -sigma2 * n, sigma2 * (n + 1),
                0.0, 0.0, 0.0, 0.0]
        rhs[1] = -sigma1 * dsrc
        m[2] = [0.0, r2n ** n, r2n ** (-(n + 1)), -r2n ** n,
                -r2n ** (-(n + 1)), 0.0, 0.0]
        m[3] = [0.0, sigma2 * n * r2n ** (n - 1),
                -sigma2 * (n + 1) * r2n ** (-(n + 2)),
                -sigma3 * n * r2n ** (n - 1),
                sigma3 * (n + 1) * r2n ** (-(n + 2)), 0.0, 0.0]
        m[4] = [0.0, 0.0, 0.0, r3n ** n, r3n ** (-(n + 1)),
                -r3n ** n, -r3n ** (-(n + 1))]
        m[5] = [0.0, 0.0, 0.0, sigma3 * n * r3n ** (n - 1),
                -sigma3 * (n + 1) * r3n ** (-(n + 2)),
                -sigma4 * n * r3n ** (n - 1),
                sigma4 * (n + 1) * r3n ** (-(n + 2))]
        m[6] = [0.0, 0.0, 0.0, 0.0, 0.0, n * r4n ** (n - 1),
                -(n + 1) * r4n ** (-(n + 2))]
        for i in range(7):
            s = np.abs(m[i]).max()
            if s > 0:
                m[i] /= s
                rhs[i] /= s
        sol = np.linalg.solve(m, rhs)
        fg[n] = sol[5], sol[6]          # f_n, g_n
    _FG_CACHE[key] = fg
    return fg


def four_sphere_rows(pre_pos, post_pos, electrodes, center, r1, r2, r3,
                     r4, sigma1=0.33, sigma2=1.79, sigma3=0.013,
                     sigma4=0.33, n_terms=60, n_threads=None):
    """(n_elec, n_dip) scalp-potential coefficient rows for one
    source-sink pair set and MANY electrodes.

    Mathematically identical to stacking FourSpherePairField rows over
    the same electrodes (same series, same row normalization), but the
    fg table and per-dipole radii are computed once and the
    per-electrode series evaluations run in a thread pool (numpy
    elementwise kernels release the GIL, so threads scale).  The series
    uses running products R_n = (r0*re)^n instead of per-order pow --
    accumulated rounding ~ n * eps (~1e-14), far below model error.
    """
    from concurrent.futures import ThreadPoolExecutor
    center = np.asarray(center, dtype=np.float64)
    pre = np.asarray(pre_pos, dtype=np.float64) - center
    post = np.asarray(post_pos, dtype=np.float64) - center
    elec = np.atleast_2d(np.asarray(electrodes, dtype=np.float64)) - center
    for arr, name in ((pre, "pre"), (post, "post")):
        r = np.linalg.norm(arr, axis=-1)
        if r.max() >= r1:
            raise ValueError(f"{name} outside brain sphere")
    re_all = np.linalg.norm(elec, axis=1)
    if not ((re_all > r3) & (re_all < r4)).all():
        raise ValueError("electrodes must lie in the scalp layer "
                         "(r3 < r < r4)")
    pre, post, elec = pre * 1e-6, post * 1e-6, elec * 1e-6
    r1_m = r1 * 1e-6
    fg = _four_sphere_fg(r1, r2, r3, r4, sigma1, sigma2, sigma3, sigma4,
                         n_terms)
    r0_pre = np.linalg.norm(pre, axis=1) / r1_m
    r0_post = np.linalg.norm(post, axis=1) / r1_m

    def series(r0n, cos_th, re_n):
        # V = sum_n P_n(cos) * R_n * (f_n + g_n * re_n^-(2n+1)),
        # R_n = (r0n*re_n)^n as a running product; n=0 cancels for
        # sealed source-sink pairs (fg[0] = 0).
        total = np.zeros(r0n.size)
        base = r0n * re_n
        R = base.copy()
        p_nm2 = np.ones(r0n.size)
        p_nm1 = cos_th.copy()
        for n in range(1, n_terms):
            if n > 1:
                p_n = ((2 * n - 1) * cos_th * p_nm1
                       - (n - 1) * p_nm2) / n
                p_nm2 = p_nm1
                p_nm1 = p_n
            cn = fg[n, 0] + fg[n, 1] * re_n ** (-(2 * n + 1))
            total += (p_nm1 * R) * cn
            R *= base
        return total

    def one_row(k):
        re = elec[k]
        re_n = np.linalg.norm(re) / r1_m
        cpre = (pre @ re) / (np.linalg.norm(pre, axis=1)
                             * np.linalg.norm(re))
        cpost = (post @ re) / (np.linalg.norm(post, axis=1)
                               * np.linalg.norm(re))
        return (series(r0_pre, cpre, re_n)
                - series(r0_post, cpost, re_n)) \
            / (4.0 * np.pi * sigma1 * r1_m)

    n_elec = len(elec)
    if n_elec == 1 or n_threads == 1:
        coef = np.empty((n_elec, len(pre)))
        for k in range(n_elec):
            coef[k] = one_row(k)
        return coef
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        coef = np.empty((n_elec, len(pre)))
        for k, row in enumerate(ex.map(one_row, range(n_elec))):
            coef[k] = row
    return coef


class FourSpherePairField:
    """Source-sink pair kernel in the standard 4-layer human head.

    Brain (r1, sigma1) / CSF (r2, sigma2) / skull (r3, sigma3) / scalp
    (r4, sigma4), sealed by air outside r4, sources in the brain,
    electrodes in the scalp layer (r3 < r_e < r4). Per Legendre order n
    the seven unknown regular/irregular coefficients solve a 7x7 boundary
    system (V and sigma dV/dr continuous at r1, r2, r3; zero radial
    current at r4); all are linear in the source term ~ r0^n. The
    classical 4-sphere EEG forward model (brain/CSF/skull/scalp), same
    API and sign conventions as ScalpPairField. Setting sigma2 = sigma1
    merges the CSF into the brain and must reproduce ScalpPairField
    exactly (verified in tests).  Coefficient rows are delegated to
    four_sphere_rows (shared fg table + optional threading).
    """

    def __init__(self, pre_pos, post_pos, electrodes, center, r1, r2, r3,
                 r4, sigma1=0.33, sigma2=1.79, sigma3=0.013, sigma4=0.33,
                 n_terms=60):
        self.coef = four_sphere_rows(
            pre_pos, post_pos, electrodes, center, r1, r2, r3, r4,
            sigma1, sigma2, sigma3, sigma4, n_terms)
        self.n_electrodes = self.coef.shape[0]

    field = StaticPairField.field
    field_timeseries = StaticPairField.field_timeseries
