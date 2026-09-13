"""ColoredCurrentNoise (OU background) + ExponentialSynapses extensions
(conductance mode, per-edge delays)."""

import numpy as np

from ffbm.simulation import ColoredCurrentNoise, ExponentialSynapses, LIFPopulation


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
