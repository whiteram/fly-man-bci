"""Vectorized point-neuron simulation primitives.

Unit conventions (kept consistent across the package):
    voltage: mV, current: pA, time: ms, resistance: GΩ  (GΩ × pA = mV)
"""

import numpy as np


def _repeat_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenated [start, start+count) ranges for each (start, count) pair."""
    starts = np.asarray(starts, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.arange(total)
    offsets = np.repeat(starts, counts)
    within_run = idx - np.repeat(np.cumsum(counts) - counts, counts)
    return offsets + within_run


class LIFPopulation:
    """Leaky integrate-and-fire population (Euler, dt << tau_m)."""

    def __init__(
        self,
        n: int,
        dt: float,
        tau_m: float = 20.0,
        v_rest: float = -70.0,
        v_th: float = -50.0,
        v_reset: float = -70.0,
        t_refrac: float = 2.0,
        R_m: float = 0.1,
    ):
        self.n = n
        self.dt = dt
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_th = v_th
        self.v_reset = v_reset
        self.t_refrac = t_refrac
        self.R_m = R_m
        self.v = np.full(n, v_rest, dtype=np.float64)
        self.refrac = np.zeros(n)

    def step(self, i_ext: np.ndarray) -> np.ndarray:
        """Advance one dt; i_ext in pA. Returns boolean spike mask."""
        hold = self.refrac > 0
        v = np.where(hold, self.v_reset, self.v)
        v = v + self.dt * (-(v - self.v_rest) + self.R_m * i_ext) / self.tau_m
        spike = v >= self.v_th
        v[spike] = self.v_reset
        self.refrac = np.maximum(self.refrac - self.dt, 0.0)
        self.refrac[spike] = self.t_refrac
        self.v = v
        return spike


class ExponentialSynapses:
    """Edge-resolved exponential synapses driven by presynaptic spikes.

    The per-edge current state y is stored in postsynaptic-sorted order so that
    a scipy CSR matrix maps it onto per-neuron input current in one matvec.
    Delivery uses precomputed index arrays into y for every presynaptic neuron.

    y_e(t) decays as y *= exp(-dt/tau_s); a presynaptic spike adds
    `gain * weight_e` to every edge of that presynaptic neuron.
    """

    def __init__(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        weight: np.ndarray,
        post_index: np.ndarray,
        dt: float,
        gain: float = 1.0,
        tau_s: float = 5.0,
        n_post: int | None = None,
    ):
        """post_index remaps edge post ids to contiguous [0, n_post) rows.

        n_post defaults to max(remapped post)+1; pass the full population size
        when some postsynaptic neurons receive no edge in this synapse set.
        """
        order = np.lexsort((pre, post))  # primary sort: post (for the CSR matvec)
        self.pre = pre[order]
        self.post_local = post_index[post[order]]
        self.weight = weight[order]
        self.kick = gain * self.weight
        self.n_edges = len(order)

        from scipy.sparse import csr_matrix

        n_post = n_post if n_post is not None else int(self.post_local.max()) + 1
        self.csr = csr_matrix(
            (np.ones(self.n_edges, dtype=np.float32),
             (self.post_local, np.arange(self.n_edges))),
            shape=(n_post, self.n_edges),
        )

        # delivery indices into y, grouped by presynaptic id
        by_pre = np.argsort(self.pre, kind="stable")
        unique_pre, starts = np.unique(self.pre[by_pre], return_index=True)
        counts = np.bincount(self.pre)[unique_pre]
        self.unique_pre = unique_pre
        self.pre_row = {int(p): i for i, p in enumerate(unique_pre)}
        self.flat_idx = by_pre.astype(np.int64)
        self.deliv_starts = starts.astype(np.int64)
        self.deliv_counts = counts.astype(np.int64)

        self.decay = np.exp(-dt / tau_s)
        self.y = np.zeros(self.n_edges, dtype=np.float32)

    def deliver(self, spiked_pre: np.ndarray):
        if len(spiked_pre):
            rows = [self.pre_row[int(p)] for p in spiked_pre
                    if int(p) in self.pre_row]
            if rows:
                sel = np.array(rows)
                spans = _repeat_ranges(self.deliv_starts[sel],
                                       self.deliv_counts[sel])
                targets = self.flat_idx[spans]
                self.y[targets] += self.kick[targets]

    def step(self, spiked_pre: np.ndarray) -> np.ndarray:
        """One dt advance; returns per-edge current state (pA)."""
        self.y *= self.decay
        self.deliver(spiked_pre)
        return self.y

    def to_neuron_current(self) -> np.ndarray:
        """Aggregate per-edge current onto postsynaptic neurons (pA)."""
        return self.csr @ self.y
