"""FourSpherePairField validation.

The load-bearing check: with sigma_CSF = sigma_brain the CSF layer merges
into the brain, so the 4-sphere must reproduce ScalpPairField (3-sphere,
brain radius = CSF outer radius) to numerical precision -- two solvers of
different size solving the same physical problem.
"""

import numpy as np
import pytest

from ffbm.forward import FourSpherePairField, ScalpPairField

C = np.zeros(3)
R_BRAIN, R_CSF, R_SKULL, R_SCALP = 7.8e4, 8.0e4, 8.5e4, 9.2e4
SIG_BRAIN, SIG_CSF, SIG_SKULL, SIG_SCALP = 0.33, 1.79, 0.013, 0.33


def _pairs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.normal(size=(n, 3))
    pre *= (0.7 * R_BRAIN * rng.random(n)
            / np.linalg.norm(pre, axis=1))[:, None]
    post = pre + rng.normal(scale=400.0, size=(n, 3))
    post *= np.minimum(
        1.0, 0.74 * R_BRAIN
        / np.maximum(np.linalg.norm(post, axis=1), 1e-9))[:, None]
    return pre, post


def _elecs(n=6, seed=1):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    return 0.985 * R_SCALP * d / np.linalg.norm(d, axis=1)[:, None]


def test_merging_csf_reduces_to_three_sphere():
    pre, post = _pairs()
    elecs = _elecs()
    four = FourSpherePairField(
        pre, post, elecs, center=C, r1=R_BRAIN, r2=R_CSF, r3=R_SKULL,
        r4=R_SCALP, sigma1=SIG_BRAIN, sigma2=SIG_BRAIN,   # CSF -> brain
        sigma3=SIG_SKULL, sigma4=SIG_SCALP, n_terms=120).coef
    three = ScalpPairField(
        pre, post, elecs, center=C, r1=R_CSF, r2=R_SKULL, r3=R_SCALP,
        sigma1=SIG_BRAIN, sigma2=SIG_SKULL, sigma3=SIG_SCALP,
        n_terms=120).coef
    err = np.abs(four - three).max() / np.abs(three).max()
    assert err < 1e-8, f"4-sphere != merged 3-sphere: {err:.2e}"


def test_dipole_antisymmetry():
    pre = np.array([[3.0e4, 0.0, 0.0]])
    post = np.array([[3.4e4, 0.0, 0.0]])
    elecs = _elecs()
    v1 = FourSpherePairField(pre, post, elecs, center=C, r1=R_BRAIN,
                             r2=R_CSF, r3=R_SKULL, r4=R_SCALP,
                             n_terms=120).coef
    v2 = FourSpherePairField(post, pre, elecs, center=C, r1=R_BRAIN,
                             r2=R_CSF, r3=R_SKULL, r4=R_SCALP,
                             n_terms=120).coef
    assert np.abs(v1 + v2).max() == 0.0


def test_series_convergence_at_network_depth():
    """S=60 (export default) vs S=200 for deep sources, r0/r1 up to 0.75."""
    pre = 0.55 * R_BRAIN * np.array([[np.sqrt(0.5), np.sqrt(0.5), 0.0]])
    post = pre + np.array([[1500.0, 0, 0]])
    elec = (0.985 * R_SCALP * np.array([0.2, -0.5, 0.84])
            / np.linalg.norm([0.2, -0.5, 0.84]))[None, :]
    kw = dict(center=C, r1=R_BRAIN, r2=R_CSF, r3=R_SKULL, r4=R_SCALP)
    c60 = FourSpherePairField(pre, post, elec, n_terms=60, **kw).coef
    c200 = FourSpherePairField(pre, post, elec, n_terms=200, **kw).coef
    assert abs(c60 - c200).max() / abs(c200).max() < 1e-6


def test_csf_shunts_superficial_radial_dipole():
    """A high-conductivity CSF shell is a low-impedance return path:
    for a superficial RADIAL dipole it shunts current tangentially and
    REDUCES the scalp potential (classic ~20-30% effect), rather than
    boosting it. Direction depends on source depth/orientation; this
    test pins the behaviour for this configuration."""
    pre = 0.95 * R_BRAIN * np.array([[0.0, 0.0, 1.0]]) - np.array(
        [[0.0, 0.0, 2500.0]])
    post = 0.985 * R_BRAIN * np.array([[0.0, 0.0, 1.0]])
    elec = (0.985 * R_SCALP * np.array([0.0, 0.0, 1.0]))[None, :]
    kw = dict(center=C, r1=R_BRAIN, r2=R_CSF, r3=R_SKULL, r4=R_SCALP,
              sigma3=SIG_SKULL, sigma4=SIG_SCALP, n_terms=120)
    with_csf = FourSpherePairField(pre, post, elec, sigma2=1.79, **kw).coef
    no_csf = FourSpherePairField(pre, post, elec, sigma2=0.33, **kw).coef
    ratio = abs(with_csf[0, 0]) / abs(no_csf[0, 0])
    assert 0.5 < ratio < 0.95, f"CSF shunt ratio {ratio:.3f} out of range"


def test_electrode_placement_enforced():
    pre, post = _pairs(n=2)
    bad = (0.5 * R_SCALP * np.array([0.0, 0.0, 1.0]))[None, :]  # in skull
    with pytest.raises(ValueError):
        FourSpherePairField(pre, post, bad, center=C, r1=R_BRAIN, r2=R_CSF,
                            r3=R_SKULL, r4=R_SCALP)
