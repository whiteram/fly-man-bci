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


class ColoredCurrentNoise:
    """Per-neuron Ornstein-Uhlenbeck input current noise.

    The cheap stand-in for background synaptic bombardment: unlike white
    noise (whose membrane-voltage effect is tiny), an OU process with
    correlation time tau_n ~ synaptic tau and std sigma gives a membrane
    fluctuation std_V = sigma * R_m * sqrt(tau_n / (tau_n + tau_m)),
    e.g. sigma=60 pA, tau_n=8 ms, R=0.1 GOhm, tau_m=10 ms -> 4 mV --
    the physiologically realistic range. This smooths the razor-thin
    deterministic f-I transition into graded stochastic firing.
    Exact discrete update: x <- a x + k w with k = sigma sqrt(1 - a^2),
    so the stationary std is sigma for any dt.
    """

    def __init__(self, n: int, dt: float, rng, tau_n: float = 8.0,
                 sigma: float = 60.0):
        self.a = float(np.exp(-dt / tau_n))
        self.k = sigma * np.sqrt(1.0 - self.a * self.a)
        self.x = np.zeros(n, dtype=np.float64)
        self.rng = rng

    def step(self) -> np.ndarray:
        self.x = self.a * self.x + self.k * self.rng.standard_normal(
            self.x.shape)
        return self.x


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

    def step(self, i_ext: np.ndarray, g_tot: np.ndarray | None = None
             ) -> np.ndarray:
        """Advance one dt; i_ext in pA. Returns boolean spike mask.

        g_tot (nS) is an optional summed conductance input whose reversal
        drive is already folded into i_ext (i_ext = base + g·E_rev, so the
        voltage-dependent part is −g·v, units nS·mV = pA). With g_tot the
        update is semi-implicit and unconditionally stable -- explicit
        Euler diverges when R_m * g_tot * dt approaches tau_m (strong
        synaptic conductance)."""
        hold = self.refrac > 0
        v = np.where(hold, self.v_reset, self.v)
        if g_tot is None:
            v = v + self.dt * (-(v - self.v_rest)
                               + self.R_m * i_ext) / self.tau_m
        else:
            a = self.dt / self.tau_m
            v = (v + a * (self.v_rest + self.R_m * i_ext)) \
                / (1.0 + a * (1.0 + self.R_m * g_tot))
        spike = v >= self.v_th
        v[spike] = self.v_reset
        self.refrac = np.maximum(self.refrac - self.dt, 0.0)
        self.refrac[spike] = self.t_refrac
        self.v = v
        return spike


class GradedSynapsePool:
    """Graded-release synapse pool (fly lamina style).

    Presynaptic drive is a continuous release rate r_pre in [0, 1] (one
    value per presynaptic neuron, e.g. a sigmoid of its membrane
    potential), NOT spike events. Each edge's gating s_e follows its
    presynaptic rate with time constant tau_s; the per-edge synaptic
    "current state" consumed by the forward kernels is y_e = weight_e *
    s_e (synapse-count weighted), matching ExponentialSynapses'
    conductance-mode convention: I_e = g_unit * y_e * (e_rev - v_post).

    exp013 uses this for the mechanistic lamina: R1-6 graded release ->
    histamine-gated chloride conductance on the (non-spiking) LMCs with
    E_Cl ~ -38 mV (Rusanen & Weckstrom 2016), replacing the exp009
    base-current proxy.
    """

    def __init__(self, pre: np.ndarray, post: np.ndarray,
                 weight: np.ndarray, post_index: np.ndarray, dt: float,
                 tau_s: float = 5.0, n_post: int | None = None,
                 g_unit: float = 0.02, e_rev: float = -38.0,
                 sign: np.ndarray | None = None,
                 e_rev_inh: float = -80.0):
        order = np.lexsort((pre, post))
        self.pre_local_of_edge = np.argsort(pre[order], kind="stable")
        # per-edge presynaptic index into the caller's r_pre array
        self.pre_ids = pre[order]
        self.weight = weight[order].astype(np.float32)
        self.post_local = post_index[post[order]]
        self.n_edges = len(order)
        self.g_unit = g_unit
        if sign is not None:            # mixed signs -> per-edge E_rev
            sign_sorted = np.asarray(sign)[order]
            self.e_rev_edge = np.where(sign_sorted > 0, e_rev,
                                       e_rev_inh).astype(np.float32)
            self._mixed = True
        else:
            self.e_rev_edge = np.float32(e_rev)
            self._mixed = False

        from scipy.sparse import csr_matrix
        n_post = (n_post if n_post is not None
                  else int(self.post_local.max()) + 1)
        self.csr = csr_matrix(
            (np.ones(self.n_edges, dtype=np.float32),
             (self.post_local, np.arange(self.n_edges))),
            shape=(n_post, self.n_edges))

        self.k = dt / tau_s
        self.s = np.zeros(self.n_edges, dtype=np.float32)
        self.y = np.zeros(self.n_edges, dtype=np.float32)

    def step(self, r_pre: np.ndarray):
        """r_pre: (n_pre_neurons,) release rates in [0, 1]."""
        self.s += (np.clip(r_pre[self.pre_ids], 0.0, 1.0) - self.s) * self.k
        self.y = self.weight * self.s
        return self.y

    def edge_currents(self, v_post: np.ndarray) -> np.ndarray:
        """Per-edge transmembrane current (pA) for the field kernels."""
        return self.g_unit * self.y * (self.e_rev_edge
                                       - v_post[self.post_local])

    def to_neuron_drive(self) -> tuple[np.ndarray, np.ndarray]:
        """(i_indep, g_tot): I(v) = i_indep - g_tot*v, semi-implicit form."""
        gy = self.g_unit * self.y
        return (self.csr @ (gy * self.e_rev_edge), self.csr @ gy)


class ExponentialSynapses:
    """Edge-resolved exponential synapses driven by presynaptic spikes.

    The per-edge current state y is stored in postsynaptic-sorted order so that
    a scipy CSR matrix maps it onto per-neuron input current in one matvec.
    Delivery uses precomputed index arrays into y for every presynaptic neuron.

    y_e(t) decays as y *= exp(-dt/tau_s); a presynaptic spike adds
    `gain * weight_e` (current mode, sign folded into weight or via `sign`)
    to every edge of that presynaptic neuron.

    Extensions:
      sign        per-edge sign (+1/-1). Current mode folds it into the kick
                  unless already folded (backwards compatible: None).
                  Conductance mode uses it to select the reversal potential.
      delay_ms    per-edge transmission delay (synaptic + axonal); spikes are
                  scheduled into a ring buffer and land after
                  round(delay_ms/dt) steps (0 = same step, as before).
      conductance True switches to conductance-based input: per-edge current
                  g_unit * y_e * (E_rev_e - v_post), with E_rev_e = e_rev_exc
                  for sign>0 and e_rev_inh for sign<0. Requires passing the
                  postsynaptic membrane potential to to_neuron_current(); the
                  state y then carries unitless gating (pass gain ~ 1 and
                  scale g_unit instead of gain).
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
        sign: np.ndarray | None = None,
        delay_ms: np.ndarray | None = None,
        conductance: bool = False,
        g_unit: float = 0.02,
        e_rev_exc: float = 0.0,
        e_rev_inh: float = -75.0,
    ):
        """post_index remaps edge post ids to contiguous [0, n_post) rows.

        n_post defaults to max(remapped post)+1; pass the full population size
        when some postsynaptic neurons receive no edge in this synapse set.
        """
        order = np.lexsort((pre, post))  # primary sort: post (for the CSR matvec)
        self.pre = pre[order]
        self.post_local = post_index[post[order]]
        self.weight = weight[order]
        if sign is not None:
            sign = np.asarray(sign)[order]
        self.sign = sign
        self.conductance = conductance
        if conductance:
            self.kick = (gain * self.weight).astype(np.float32)
            self.e_rev_edge = np.where(
                sign > 0 if sign is not None else True,
                e_rev_exc, e_rev_inh).astype(np.float32)
            self.g_unit = g_unit
        else:
            eff_sign = sign if sign is not None else np.ones(len(order))
            self.kick = (gain * self.weight * eff_sign).astype(np.float32)
        self.n_edges = len(order)

        from scipy.sparse import csr_matrix

        n_post = n_post if n_post is not None else int(self.post_local.max()) + 1
        self.csr = csr_matrix(
            (np.ones(self.n_edges, dtype=np.float32),
             (self.post_local, np.arange(self.n_edges))),
            shape=(n_post, self.n_edges),
        )

        # delivery indices into y, grouped by presynaptic id (factorized:
        # body ids can reach 1e9 and a raw bincount would allocate GBs)
        by_pre = np.argsort(self.pre, kind="stable")
        unique_pre, starts = np.unique(self.pre[by_pre], return_index=True)
        counts = np.diff(np.append(starts, len(by_pre)))
        self.unique_pre = unique_pre
        self.pre_row = {int(p): i for i, p in enumerate(unique_pre)}
        self.flat_idx = by_pre.astype(np.int64)
        self.deliv_starts = starts.astype(np.int64)
        self.deliv_counts = counts.astype(np.int64)

        # per-edge transmission delays (ring buffer, in steps); an empty
        # edge group (a region selection with no surviving edges) is
        # legal and stays non-delayed
        self.delayed = False
        if delay_ms is not None:
            bins = np.clip(
                np.round(np.asarray(delay_ms, dtype=np.float64)[order] / dt
                         ).astype(np.int64), 0, None)
            self.delay_bins = bins
            self.buf_len = int(bins.max()) + 1 if len(bins) else 1
            self.buffer = np.zeros((self.buf_len, self.n_edges),
                                   dtype=np.float32)
            self.ptr = 0
            self.delayed = bool(len(bins) and bins.max() > 0)

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

    def _deliver_delayed(self, spiked_pre: np.ndarray):
        if not len(spiked_pre):
            return
        rows = [self.pre_row[int(p)] for p in spiked_pre
                if int(p) in self.pre_row]
        if not rows:
            return
        sel = np.array(rows)
        spans = _repeat_ranges(self.deliv_starts[sel], self.deliv_counts[sel])
        targets = self.flat_idx[spans]
        bins = self.delay_bins[targets]
        kicks = self.kick[targets]
        for b in np.unique(bins):
            m = bins == b
            if b == 0:
                self.y[targets[m]] += kicks[m]
            else:
                row = (self.ptr + int(b) - 1) % self.buf_len
                np.add.at(self.buffer[row], targets[m], kicks[m])

    def step(self, spiked_pre: np.ndarray) -> np.ndarray:
        """One dt advance; returns per-edge current state (pA)."""
        self.y *= self.decay
        if self.delayed:
            self.y += self.buffer[self.ptr]
            self.buffer[self.ptr].fill(0.0)
            self.ptr = (self.ptr + 1) % self.buf_len
            self._deliver_delayed(spiked_pre)
        else:
            self.deliver(spiked_pre)
        return self.y

    def edge_currents(self, v_post: np.ndarray | None = None) -> np.ndarray:
        """Per-edge transmembrane current (pA), the quantity the extracellular
        forward kernels consume. Current mode: the state y itself.
        Conductance mode: g_unit * y * (E_rev - v_post) (needs v_post)."""
        if self.conductance:
            if v_post is None:
                raise ValueError("conductance mode needs v_post")
            return (self.g_unit * self.y
                    * (self.e_rev_edge - v_post[self.post_local]))
        return self.y

    def to_neuron_drive(self) -> tuple[np.ndarray, np.ndarray]:
        """(i_indep, g_tot) form of the input: i_indep collects all
        voltage-independent parts of the per-neuron current (pA) and g_tot
        the summed conductance (nS) whose voltage-dependent part is
        -g_tot * v (nS*mV = pA). Current mode returns (sum y, 0).
        Feeds LIFPopulation.step(i_ext, g_tot) for stable integration."""
        if self.conductance:
            gy = self.g_unit * self.y
            return (self.csr @ (gy * self.e_rev_edge), self.csr @ gy)
        return self.csr @ self.y, np.zeros(1, dtype=np.float32)

    def to_neuron_current(self, v_post: np.ndarray | None = None) -> np.ndarray:
        """Aggregate per-edge current onto postsynaptic neurons (pA).

        Conductance mode requires the current postsynaptic membrane
        potential (mV) and returns g_unit * y * (E_rev - v)."""
        if self.conductance:
            if v_post is None:
                raise ValueError("conductance mode needs v_post")
        return self.csr @ self.edge_currents(v_post)
