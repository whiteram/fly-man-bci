"""DAN-gated plasticity weight-recovery semantics (bci/condit2).

plast_tau_w_ms adds a slow homeostatic recovery w += (1-w) dt/tau_w
every step (gate-independent).  Default (None) must keep weights frozen
when the reinforcement gate closes -- the cross-trial-memory baseline
behavior of bci/condit.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffbm.simulation import ExponentialSynapses


def make(tau_w=None):
    pre = np.array([10, 10, 11], np.int64)
    post = np.array([1, 2, 1], np.int64)
    post_index = np.array([-1, 0, 1], np.int64)   # id 1 -> row 0, 2 -> 1
    w = np.ones(3, np.float32)
    return ExponentialSynapses(
        pre, post, w, post_index, dt=1.0, tau_s=5.0, n_post=2,
        plast_lr=0.05, plast_tau_ms=400.0, plast_tau_w_ms=tau_w)


def depress(syn, steps=200):
    for _ in range(steps):
        syn.step(np.array([10], np.int64), mod=1.0)


def quiet(syn, steps):
    for _ in range(steps):
        syn.step(np.empty(0, np.int64), mod=0.0)


def test_gate_closed_weights_frozen_by_default():
    syn = make()                      # tau_w = None -> w_rec == 0
    depress(syn)
    w_after = syn.w_scale.copy()
    assert (w_after < 0.99).any(), "gated depression did not fire"
    quiet(syn, 5000)
    assert np.allclose(syn.w_scale, w_after), \
        "weights drifted with recovery OFF (default must be frozen)"


def test_tau_w_recovers_toward_one():
    syn = make(tau_w=2000.0)
    depress(syn)
    assert (syn.w_scale < 0.99).any()
    quiet(syn, 5000)                  # 2.5 tau_w of gate-closed time
    assert (syn.w_scale > 0.9).all(), \
        "weights must recover toward 1 with tau_w on"
    assert (syn.w_scale <= 1.0 + 1e-6).all(), "recovery overshot 1"


def test_recovery_is_monotone_and_capped():
    syn = make(tau_w=2000.0)
    depress(syn, steps=2000)          # drive some edges to the floor
    prev = syn.w_scale.copy()
    quiet(syn, 50)
    assert (syn.w_scale >= prev - 1e-7).all(), \
        "gate-closed steps must never depress further"
