"""Numba-kernel parity: JIT paths must reproduce the numpy expressions
BITWISE (same dtype promotion order, same accumulation order per row).

Run with FFBM_NUMBA=0 and FFBM_NUMBA=1 in the same interpreter to
exercise both sides explicitly (the module-level flag is read at
import); simpler: this test compares the numba kernels against the
reference numpy/scipy expressions directly.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffbm.simulation import (_HAVE_NUMBA, ColoredCurrentNoise,
                             ExponentialSynapses, GradedSynapsePool,
                             LIFPopulation, _drive_cond, _edge_currents,
                             _exp_step, _exp_step_delayed, _graded_step,
                             _lif_step, _ou_step)


def _pool(n_pre=50, n_edge=4000, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n_pre, n_edge)
    post = rng.integers(0, 80, n_edge)
    weight = (rng.random(n_edge) + 0.1).astype(np.float32)
    post_index = {}
    post_local = np.array([post_index.setdefault(int(p), len(post_index))
                           for p in post])
    return pre, post, weight, post_local, rng


def test_graded_step_parity():
    if not _HAVE_NUMBA:
        print("numba absent; kernel == reference by construction")
        return
    rng = np.random.default_rng(1)
    n_edge = 5000
    pre_ids = rng.integers(0, 60, n_edge).astype(np.int64)
    s = (rng.random(n_edge) * 0.5).astype(np.float32)
    s2 = s.copy()
    y = np.zeros(n_edge, np.float32)
    y2 = np.zeros(n_edge, np.float32)
    weight = (rng.random(n_edge) + 0.2).astype(np.float32)
    k = 0.03
    r_pre = rng.random(60) * 1.3 - 0.1          # straddles the clip
    _graded_step(r_pre, pre_ids, s, y, weight, k)
    s2 += (np.clip(r_pre[pre_ids], 0.0, 1.0) - s2) * k
    y2 = weight * s2
    assert np.array_equal(s, s2) and np.array_equal(y, y2), \
        f"graded step mismatch: {np.abs(s - s2).max()}"


def test_drive_cond_parity():
    if not _HAVE_NUMBA:
        return
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(2)
    n_edge, n_post = 8000, 90
    post_local = np.sort(rng.integers(0, n_post, n_edge))
    y = (rng.random(n_edge)).astype(np.float32)
    e_rev = (rng.random(n_edge) * 100 - 60).astype(np.float32)
    g_unit = 0.0213
    csr = csr_matrix((np.ones(n_edge, np.float32),
                      (post_local, np.arange(n_edge))),
                     shape=(n_post, n_edge))
    out_i = np.zeros(n_post, np.float32)
    out_g = np.zeros(n_post, np.float32)
    _drive_cond(y, g_unit, e_rev, post_local, out_i, out_g)
    gy = g_unit * y
    ref_i, ref_g = csr @ (gy * e_rev), csr @ gy
    assert np.array_equal(out_i, ref_i), \
        f"drive i mismatch: max {np.abs(out_i - ref_i).max():.3e}"
    assert np.array_equal(out_g, ref_g), \
        f"drive g mismatch: max {np.abs(out_g - ref_g).max():.3e}"


def test_edge_currents_parity():
    if not _HAVE_NUMBA:
        return
    rng = np.random.default_rng(3)
    n_edge = 7000
    post_local = rng.integers(0, 200, n_edge)
    y = rng.random(n_edge).astype(np.float32)
    e_rev = (rng.random(n_edge) * 120 - 70).astype(np.float32)
    v_post = rng.random(200) * 60 - 70            # float64 like pops.v
    g_unit = 0.0187
    out = np.empty(n_edge, np.float64)
    _edge_currents(y, g_unit, e_rev, post_local, v_post, out)
    ref64 = g_unit * y * (e_rev - v_post[post_local])
    assert np.array_equal(out, ref64), \
        f"edge currents mismatch: max {np.abs(out - ref64).max():.3e}"


def test_signless_conductance_pool_parity():
    """sign=None + conductance=True: the constructor used to leave
    e_rev_edge as a 0-d array (kernel crash); now a 1-D array, and the
    kernel must match the numpy path bitwise."""
    rng = np.random.default_rng(4)
    n_pre, n_edge, n_post = 40, 3000, 60
    pre = rng.integers(0, n_pre, n_edge)
    post = rng.integers(0, n_post, n_edge)
    weight = (rng.random(n_edge) + 0.2).astype(np.float32)
    post_index = {}
    post_local = np.array([post_index.setdefault(int(p), len(post_index))
                           for p in post])
    pool = ExponentialSynapses(
        pre, post, weight, post_local, dt=0.5, gain=1.0, tau_s=5.0,
        n_post=n_post, conductance=True, g_unit=0.02,
        e_rev_exc=0.0, e_rev_inh=-75.0)
    assert pool.e_rev_edge.ndim == 1 and pool.e_rev_edge.size == n_edge
    pool.deliver(np.array([1, 3, 7]))
    pool.y *= np.float32(0.7)
    v_post = rng.random(n_post) * 60 - 70
    i1, g1 = pool.to_neuron_drive()
    ec1 = pool.edge_currents(v_post)
    import ffbm.simulation as sim
    sim._HAVE_NUMBA = False
    try:
        i2, g2 = pool.to_neuron_drive()
        ec2 = pool.edge_currents(v_post)
    finally:
        sim._HAVE_NUMBA = True
    assert np.array_equal(i1, i2) and np.array_equal(g1, g2), \
        "signless drive differs numba vs numpy"
    assert np.array_equal(ec1, ec2), "signless edge currents differ"


def _exp_pool(seed, delayed):
    rng = np.random.default_rng(seed)
    n_pre, n_edge, n_post = 30, 4000, 50
    pre = rng.integers(0, n_pre, n_edge)          # duplicates -> one
    post = rng.integers(0, n_post, n_edge)        # pre hits many edges
    weight = (rng.random(n_edge) + 0.2).astype(np.float32)
    post_index = {}
    post_local = np.array([post_index.setdefault(int(p), len(post_index))
                           for p in post])
    delay = (rng.random(n_edge) * 4.0) if delayed else None
    return ExponentialSynapses(
        pre, post, weight, post_local, dt=0.5, gain=1.5, tau_s=5.0,
        n_post=n_post, delay_ms=delay)


def _run_steps(pool, spikes_seq):
    for spiked in spikes_seq:
        pool.step(spiked)
    return pool.y.copy(), pool.buffer.copy() if pool.delayed else None


def test_exp_step_parity():
    """Non-delayed and delayed step kernels vs the numpy path, over a
    spike sequence engineered for duplicate targets and every delay bin
    (incl. bin 0)."""
    if not _HAVE_NUMBA:
        return
    import ffbm.simulation as sim
    rng = np.random.default_rng(7)
    spikes = [np.array(sorted(rng.choice(30, size=rng.integers(1, 12),
                                           replace=False)))
              for _ in range(60)]
    for delayed in (False, True):
        # same construction + same deliveries in both modes
        sim._HAVE_NUMBA = False
        try:
            ref = _run_steps(_exp_pool(11, delayed), spikes)
        finally:
            sim._HAVE_NUMBA = True
        got = _run_steps(_exp_pool(11, delayed), spikes)
        assert np.array_equal(ref[0], got[0]), \
            f"y differs (delayed={delayed}): " \
            f"max {np.abs(ref[0] - got[0]).max():.3e}"
        if delayed:
            assert np.array_equal(ref[1], got[1]), \
                f"ring buffer differs (delayed): " \
                f"max {np.abs(ref[1] - got[1]).max():.3e}"


def test_lif_ou_parity():
    if not _HAVE_NUMBA:
        return
    import ffbm.simulation as sim
    rng = np.random.default_rng(9)
    n = 500
    # LIF with conductance input (f32 drives like the synapse kernels
    # return) and with plain current input
    for g_none in (True, False):
        lif1 = LIFPopulation(n, 0.5, tau_m=10.0, R_m=0.1)
        lif2 = LIFPopulation(n, 0.5, tau_m=10.0, R_m=0.1)
        ou1 = ColoredCurrentNoise(n, 0.5, np.random.default_rng(3), 8.0, 60.0)
        ou2 = ColoredCurrentNoise(n, 0.5, np.random.default_rng(3), 8.0, 60.0)
        for _ in range(80):
            noise1, noise2 = ou1.step(), ou2.step()  # same rng seeds
            i1 = noise1 + rng.random(n) * 90.0
            i2 = i1.copy()
            # g_tot is float64 in the real stack (np.zeros accumulation
            # of the f32 drives); f32 g would follow a different numpy
            # promotion in the semi-implicit denominator
            g1 = None if g_none else rng.random(n) * 0.5
            g2 = None if g_none else g1.copy()
            sim._HAVE_NUMBA = False
            try:
                s1 = lif1.step(i1, g1)
            finally:
                sim._HAVE_NUMBA = True
            s2 = lif2.step(i2, g2)
            assert np.array_equal(lif1.v, lif2.v), \
                f"LIF v differs (g_none={g_none})"
            assert np.array_equal(lif1.refrac, lif2.refrac), \
                f"LIF refrac differs (g_none={g_none})"
            assert np.array_equal(s1, s2), f"spikes differ (g_none={g_none})"
            assert np.array_equal(ou1.x, ou2.x), "OU state differs"

def test_smoke_export_bitwise():
    """End-to-end A/B: the smoke export's debug phi arrays must be
    BITWISE identical with numba on vs off."""
    out = ROOT / "viz" / "data_smoke"
    results = {}
    for flag in ("0", "1"):
        env = {**os.environ, "FFBM_NUMBA": flag,
               "PYTHONPATH": str(ROOT / "src")}
        r = subprocess.run(
            [sys.executable, str(ROOT / "viz" / "export_data.py"),
             "--smoke"],
            capture_output=True, text=True, env=env, cwd=str(ROOT))
        assert r.returncode == 0, f"smoke failed (FFBM_NUMBA={flag}):\n" \
            + r.stdout[-2000:] + r.stderr[-2000:]
        results[flag] = (np.load(out / "_debug_phi_scalp.npy"),
                         np.load(out / "_debug_phi.npy"))
    same = all(np.array_equal(a, b) for a, b in zip(*results.values()))
    assert same, "smoke trajectories differ between numba on/off"
    print("smoke A/B: bitwise identical across FFBM_NUMBA=0/1")


if __name__ == "__main__":
    test_graded_step_parity()
    print("graded_step parity OK")
    test_drive_cond_parity()
    print("drive_cond parity OK")
    test_edge_currents_parity()
    print("edge_currents parity OK")
    test_exp_step_parity()
    print("exp_step parity OK")
    test_lif_ou_parity()
    print("LIF/OU parity OK")
    test_signless_conductance_pool_parity()
    print("signless conductance pool parity OK")
    test_exp_step_parity()
    print("exp_step parity OK")
    test_lif_ou_parity()
    print("LIF/OU parity OK")
    test_smoke_export_bitwise()
    print("ALL_PASS")
