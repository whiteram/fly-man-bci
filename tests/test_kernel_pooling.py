"""Fast validation of the runtime kernel pooling approximation
(export_data.py's --pool acceleration): the 4-sphere field is smooth in
dipole position, so per-cluster representative kernels applied to pooled
currents must reproduce the full sum to <1%.

Synthetic dipoles only -- no connectome data, runs in ~10 s.
"""
import numpy as np
import pytest

from scipy.cluster.vq import kmeans2

from ffbm.forward import FourSpherePairField

R_BRAIN, R_CSF, R_SKULL, R_SCALP = 7.8e4, 8.0e4, 8.5e4, 9.2e4
SIGMAS = (0.33, 1.79, 0.013, 0.33)
CENTER = np.zeros(3)


def _dipoles(n, rng, scale=4.747e3):
    """Dipole pairs mimicking the real rhabdomere array: coherently
    oriented (common axis + ~10 deg jitter), inside a cap region of the
    brain sphere (like the occipital-pole placement)."""
    pre = rng.standard_normal((n, 3))
    pre /= np.linalg.norm(pre, axis=1, keepdims=True)
    axis = np.array([0.0, 0.0, 1.0])
    keep = pre @ axis > np.cos(np.radians(30.0))     # cap region
    pre = pre[keep] * (rng.uniform(0.55, 0.85, (keep.sum(), 1))
                       * R_BRAIN)
    d = axis + rng.standard_normal((keep.sum(), 3)) * 0.15
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    post = pre + scale * d
    return pre, post


@pytest.mark.parametrize("elec_dir", [
    [0.0, 0.0, 1.0],
    [0.6, 0.5, 0.62],
])
def test_pooling_matches_exact_sum(elec_dir):
    rng = np.random.default_rng(42)
    pre, post = _dipoles(12_000, rng)
    n = len(pre)
    elec = CENTER + 0.985 * R_SCALP * np.array(elec_dir) / \
        np.linalg.norm(elec_dir)

    y = 50.0 * (1.0 + 0.3 * rng.standard_normal(n))   # correlated, same sign

    coef_full = FourSpherePairField(
        pre, post, elec[None, :], center=CENTER, r1=R_BRAIN, r2=R_CSF,
        r3=R_SKULL, r4=R_SCALP, sigma1=SIGMAS[0], sigma2=SIGMAS[1],
        sigma3=SIGMAS[2], sigma4=SIGMAS[3]).coef[0]
    phi_exact = float(coef_full @ y)

    # the pooling path (mirrors export_data.py): 6D k-means, member-mean
    # representatives, reduceat pooling
    k = int(np.clip(round(np.sqrt(n) * 7.0), 400, 4000))
    C6 = np.hstack([pre, post])
    train = rng.choice(n, min(20_000, n), replace=False)
    cent, _ = kmeans2(C6[train], k, minit="points", iter=10, seed=13)
    labels = np.empty(n, dtype=np.int64)
    c22 = (cent ** 2).sum(1)
    for s0 in range(0, n, 50_000):
        sl = slice(s0, min(s0 + 50_000, n))
        d2 = ((C6[sl] ** 2).sum(1)[:, None] + c22[None, :]
              - 2.0 * C6[sl] @ cent.T)
        labels[sl] = d2.argmin(1)
    counts = np.bincount(labels, minlength=len(cent))
    keepc = np.nonzero(counts)[0]
    remap = np.full(len(cent), -1, dtype=np.int64)
    remap[keepc] = np.arange(len(keepc))
    labels = remap[labels]
    sum_pre = np.zeros((len(keepc), 3))
    sum_post = np.zeros((len(keepc), 3))
    np.add.at(sum_pre, labels, pre)
    np.add.at(sum_post, labels, post)
    rep_pre = sum_pre / counts[keepc, None]
    rep_post = sum_post / counts[keepc, None]
    order = np.argsort(labels, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts[keepc])[:-1]])

    coef_rep = FourSpherePairField(
        rep_pre, rep_post, elec[None, :], center=CENTER, r1=R_BRAIN,
        r2=R_CSF, r3=R_SKULL, r4=R_SCALP, sigma1=SIGMAS[0],
        sigma2=SIGMAS[1], sigma3=SIGMAS[2], sigma4=SIGMAS[3]).coef[0]
    pooled = np.add.reduceat(y[order], starts)
    phi_pool = float(coef_rep @ pooled)

    rel = abs(phi_exact - phi_pool) / max(abs(phi_exact), 1e-30)
    assert rel < 0.01, f"pooling rel err {rel:.4%}"
