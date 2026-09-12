"""Smoke tests for ffbm core modules. Run: pytest tests/ (or python -m pytest)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffbm.forward import StaticPairField
from ffbm.simulation import ExponentialSynapses, LIFPopulation


def test_lif_fires_above_threshold():
    pop = LIFPopulation(4, dt=0.5, tau_m=20.0, R_m=0.1)
    i = np.array([300.0, 300.0, 0.0, 0.0])  # 30 mV steady depol -> fires
    total = np.zeros(4)
    for _ in range(400):  # 200 ms
        total += pop.step(i)
    assert total[0] > 3 and total[1] > 3
    assert total[2] == 0 and total[3] == 0


def test_synapse_delivery_and_decay():
    dt, tau = 0.5, 5.0
    pre = np.array([10, 10, 11])
    post = np.array([0, 1, 0])
    weight = np.array([2.0, 3.0, 5.0])
    syn = ExponentialSynapses(pre, post, weight,
                              post_index=np.arange(2), dt=dt, gain=1.0, tau_s=tau)
    syn.step(np.array([]))
    syn.step(np.array([10]))
    # y is stored in post-sorted order: check via the public neuron-current API
    i_syn = syn.to_neuron_current()
    assert np.isclose(i_syn[0], 2.0)  # only the pre=10 -> post=0 edge fired
    assert np.isclose(i_syn[1], 3.0)
    syn.step(np.array([11]))
    i_syn = syn.to_neuron_current()
    assert np.isclose(i_syn[0], 2.0 * syn.decay + 5.0)
    assert np.isclose(i_syn[1], 3.0 * syn.decay)


def test_forward_field_sign_and_units():
    # sink (post) closer to the electrode than source (pre) -> negative potential
    pre_pos = np.array([[100.0, 0.0, 0.0]])
    post_pos = np.array([[50.0, 0.0, 0.0]])
    field = StaticPairField(pre_pos, post_pos, electrodes=[[0.0, 0.0, 0.0]])
    phi = field.field(np.array([1.0]))  # 1 pA
    assert phi[0] < 0
    # pA * um / (S/m) -> volts: magnitude sanity (~tens of nV at these distances)
    assert 1e-9 < abs(phi[0]) < 1e-4
