"""four_sphere_rows validation.

The batched kernel-build path must reproduce the reference Legendre
series (per-order pow form, evaluated here inline as the reference)
and the single-electrode FourSpherePairField rows.  The running-product
series reorders floating-point accumulation (~n*eps), so tolerances are
tight but not bitwise.
"""

import numpy as np

from ffbm.forward import FourSpherePairField, four_sphere_rows

C = np.zeros(3)
R_BRAIN, R_CSF, R_SKULL, R_SCALP = 7.8e4, 8.0e4, 8.5e4, 9.2e4
SIG = (0.33, 1.79, 0.013, 0.33)
N_TERMS = 60


def _pairs(n=400, seed=3):
    rng = np.random.default_rng(seed)
    pre = rng.normal(size=(n, 3))
    pre *= (0.7 * R_BRAIN * rng.random(n)
            / np.linalg.norm(pre, axis=1))[:, None]
    post = pre + rng.normal(scale=400.0, size=(n, 3))
    post *= np.minimum(
        1.0, 0.74 * R_BRAIN
        / np.maximum(np.linalg.norm(post, axis=1), 1e-9))[:, None]
    return pre, post


def _elecs(n=6, seed=4):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return C + 0.985 * R_SCALP * d


def _reference_series_row(pre, post, elec, fg):
    """The original per-order pow series (forward.py before the
    running-product optimization), for A/B."""
    pre_m, post_m, elec_m = pre * 1e-6, post * 1e-6, elec * 1e-6
    r1_m = R_BRAIN * 1e-6
    r0_pre = np.linalg.norm(pre_m, axis=1) / r1_m
    r0_post = np.linalg.norm(post_m, axis=1) / r1_m
    re_n = np.linalg.norm(elec_m) / r1_m
    n_orders = np.arange(N_TERMS)

    def series(r0n, cos_th):
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
            reg = fg[n, 0] * (r0n ** n) * (re_n ** n)
            irr = fg[n, 1] * (r0n ** n) * (re_n ** (-(n + 1)))
            total += p_n * (reg + irr)
        return total

    cpre = (pre_m @ elec_m) / (np.linalg.norm(pre_m, axis=1)
                               * np.linalg.norm(elec_m))
    cpost = (post_m @ elec_m) / (np.linalg.norm(post_m, axis=1)
                                 * np.linalg.norm(elec_m))
    return (series(r0_pre, cpre) - series(r0_post, cpost)) \
        / (4.0 * np.pi * SIG[0] * r1_m)


def test_batch_matches_reference_series():
    from ffbm.forward import _four_sphere_fg
    fg = _four_sphere_fg(R_BRAIN, R_CSF, R_SKULL, R_SCALP, *SIG, N_TERMS)
    pre, post = _pairs()
    elecs = _elecs()
    batch = four_sphere_rows(pre, post, elecs, C, R_BRAIN, R_CSF,
                             R_SKULL, R_SCALP, *SIG, N_TERMS)
    for k in range(len(elecs)):
        ref = _reference_series_row(pre, post, elecs[k], fg)
        rel = np.abs(batch[k] - ref).max() / np.abs(ref).max()
        assert rel < 1e-10, f"elec {k}: rel {rel:.2e}"


def test_batch_matches_single_electrode_class():
    pre, post = _pairs(n=150, seed=5)
    elecs = _elecs(n=5, seed=6)
    batch = four_sphere_rows(pre, post, elecs, C, R_BRAIN, R_CSF,
                             R_SKULL, R_SCALP, *SIG, N_TERMS)
    for k in range(len(elecs)):
        single = FourSpherePairField(pre, post, elecs[k][None, :], C,
                                     R_BRAIN, R_CSF, R_SKULL, R_SCALP,
                                     *SIG, N_TERMS).coef[0]
        assert np.array_equal(batch[k], single), f"elec {k} differs"


if __name__ == "__main__":
    test_batch_matches_reference_series()
    print("test_batch_matches_reference_series OK")
    test_batch_matches_single_electrode_class()
    print("test_batch_matches_single_electrode_class OK")
    print("ALL_PASS")
