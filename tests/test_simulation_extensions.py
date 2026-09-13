"""ColoredCurrentNoise (OU background) + ExponentialSynapses extensions
(conductance mode, per-edge delays)."""

import numpy as np

from ffbm.simulation import (ColoredCurrentNoise, ExponentialSynapses,
                             GradedSynapsePool, LIFPopulation)


def test_ou_stationary_std_and_corr_time():
    n, dt, tau_n, sigma = 200, 0.5, 8.0, 60.0
    rng = np.random.default_rng(0)
    cn = ColoredCurrentNoise(n, dt, rng, tau_n=tau_n, sigma=sigma)
    xs = [cn.step().copy() for _ in range(20000)]
    xs = np.array(xs[2000:])            # drop transient
    assert abs(xs.std() - sigma) / sigma < 0.02
    # lag-1 autocorrelation should be exp(-dt/tau_n)
    r1 = np.corrcoef(xs[:-1].ravel(), xs[1:].ravel())[0, 1]
    assert abs(r1 - np.exp(-dt / tau_n)) < 0.02


def test_ou_gives_realistic_membrane_fluctuation():
    """Continuum formula std_V = sigma R sqrt(tau_n/(tau_n+tau_m)) predicts
    4.0 mV; the Euler-discretised membrane at dt=0.5 ms runs ~10% hotter.
    The point is the magnitude: ~4 mV where white noise gave ~0.3 mV."""
    n, dt, tau_n, sigma = 100, 0.5, 8.0, 60.0
    rng = np.random.default_rng(1)
    cn = ColoredCurrentNoise(n, dt, rng, tau_n=tau_n, sigma=sigma)
    pop = LIFPopulation(n, dt, tau_m=10.0, t_refrac=1e9)  # never fires
    for _ in range(20000):
        pop.step(cn.step())
    v = pop.v
    expected = sigma * 0.1 * np.sqrt(tau_n / (tau_n + 10.0))
    assert abs(v.std() - expected) / expected < 0.15


def _one_syn(delay_ms=None, conductance=False, sign=None, gain=1.0,
             g_unit=0.02):
    pre = np.array([7])
    post = np.array([3])
    post_index = np.full(4, -1, dtype=np.int64)
    post_index[3] = 0
    return ExponentialSynapses(pre, post, np.array([10.0], np.float32),
                               post_index, dt=0.5, gain=gain, tau_s=5.0,
                               n_post=1, sign=sign, delay_ms=delay_ms,
                               conductance=conductance, g_unit=g_unit)


def test_delay_lands_after_exact_steps():
    syn = _one_syn(delay_ms=np.array([1.0]))   # 2 steps
    y0 = syn.step(np.array([7]))               # spike scheduled
    assert y0[0] == 0.0                        # not yet
    y1 = syn.step(np.array([]))
    assert y1[0] == 0.0                        # still not
    y2 = syn.step(np.array([]))
    assert y2[0] == 10.0                       # lands after 2 steps


def test_delay_zero_is_immediate():
    syn = _one_syn(delay_ms=np.array([0.0]))
    y = syn.step(np.array([7]))
    assert y[0] == 10.0                        # same-step delivery (legacy)


def test_conductance_current_linear_in_v_and_reversal():
    syn = _one_syn(conductance=True, sign=np.array([1.0]), g_unit=0.02)
    syn.step(np.array([7]))                    # y = 10 (unitless gating)
    y = syn.y[0]
    i_exc = syn.to_neuron_current(np.array([-70.0, -70.0, -70.0, -70.0]))[0]
    assert abs(i_exc - 0.02 * y * (0.0 - (-70.0))) < 1e-6
    # at the reversal potential the current vanishes
    i_rev = syn.to_neuron_current(np.array([0.0, 0.0, 0.0, 0.0]))[0]
    assert abs(i_rev) < 1e-9
    # inhibitory edge pulls toward e_rev_inh
    syn_i = _one_syn(conductance=True, sign=np.array([-1.0]), g_unit=0.02)
    syn_i.step(np.array([7]))
    i_inh = syn_i.to_neuron_current(np.full(4, -70.0))[0]
    assert abs(i_inh - 0.02 * 10.0 * (-75.0 - (-70.0))) < 1e-6


def test_current_mode_sign_folding_matches_legacy():
    syn = _one_syn(sign=np.array([-1.0]), gain=2.0)
    y = syn.step(np.array([7]))
    assert y[0] == -20.0                       # gain * weight * sign
    assert syn.to_neuron_current(np.array([0, 0, 0, 0.0]))[0] == -20.0


def test_semi_implicit_conductance_is_stable():
    """With R_m*g*dt comparable to tau_m (strong synaptic conductance),
    explicit Euler diverges; the semi-implicit update must stay bounded
    and clamp v near the reversal potential."""
    n, dt = 50, 0.5
    pop = LIFPopulation(n, dt, tau_m=10.0, t_refrac=1e9)  # never spikes
    g_huge = np.full(n, 500.0)                  # nS: R*g = 50 >> tau/dt
    e_rev = -80.0
    i_drive = g_huge * e_rev                    # pA: base + g*E_rev
    for _ in range(4000):
        pop.step(i_drive, g_tot=g_huge)
    assert np.all(np.isfinite(pop.v))
    assert np.abs(pop.v - e_rev).max() < 1.0    # clamped at reversal


def test_to_neuron_drive_matches_explicit_current():
    """i_indep - g_tot*v must equal the explicit per-edge current sum."""
    syn = _one_syn(conductance=True, sign=np.array([1.0]), g_unit=0.02)
    syn.step(np.array([7]))
    i_indep, g_tot = syn.to_neuron_drive()
    v = np.array([-70.0, -70.0, -70.0, -70.0])
    explicit = syn.to_neuron_current(v)[0]
    assert abs((i_indep[0] - g_tot[0] * (-70.0)) - explicit) < 1e-6


def test_graded_pool_tracks_release_rate():
    """s_e -> r_pre with time constant tau_s; y = weight * s; the drive
    form matches the explicit conductance current."""
    pre = np.array([7, 7])
    post = np.array([3, 3])
    post_index = np.full(4, -1, dtype=np.int64)
    post_index[3] = 0
    pool = GradedSynapsePool(pre, post, np.array([10.0, 5.0], np.float32),
                             post_index, dt=0.5, tau_s=5.0, n_post=1,
                             g_unit=0.02, e_rev=-38.0)
    r = np.zeros(8)
    r[7] = 1.0
    for _ in range(200):                       # ~10 tau -> converged
        y = pool.step(r)
    assert abs(y[0] - 10.0) < 0.1 and abs(y[1] - 5.0) < 0.1
    i_indep, g_tot = pool.to_neuron_drive()
    v = np.full(4, -70.0)
    explicit_neuron = pool.csr @ pool.edge_currents(v)
    assert np.allclose(i_indep - g_tot * (-70.0), explicit_neuron, atol=1e-5)


def test_graded_lamina_dark_clamp():
    """The exp013 core behaviour: with tonic dark release the chloride
    conductance clamps the non-spiking LMC near E_Cl = -38 mV; removing
    the release (light) hyperpolarizes it toward the K-rest."""
    n_r, n_l = 40, 40
    pre = np.repeat(np.arange(n_r), 3)
    post = np.repeat(np.arange(n_l), 3)[:len(pre)]
    post_index = np.arange(n_l, dtype=np.int64)
    pool = GradedSynapsePool(pre, post,
                             np.full(len(pre), 8.0, np.float32), post_index,
                             dt=0.5, tau_s=5.0, n_post=n_l, g_unit=8.0,
                             e_rev=-38.0)
    lmc = LIFPopulation(n_l, 0.5, tau_m=20.0, t_refrac=1e9,
                        v_rest=-58.0, v_th=1e9)     # graded, K-rest -58
    r_dark = np.ones(n_r) * 0.5
    for _ in range(2000):
        pool.step(r_dark)
        i, g = pool.to_neuron_drive()
        lmc.step(i, g)
    assert abs(lmc.v.mean() - (-38.0)) < 2.0       # clamped at E_Cl
    r_light = np.zeros(n_r)                        # release off
    for _ in range(4000):
        pool.step(r_light)
        i, g = pool.to_neuron_drive()
        lmc.step(i, g)
    assert lmc.v.mean() < -50.0                    # hyperpolarized toward K-rest
