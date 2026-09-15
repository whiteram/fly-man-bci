"""P3 PoC step 2: per-step kernel-launch overhead and CUDA Graph benefit.

A representative one-step kernel chain (decay / ring-add / per-pre
delivery / graded / post-major drive / OU / LIF / edge currents) at
FULL scale (64M exp-synapse edges, 6M graded edges, 150k neurons) and
a SMALL scale (1/16 of it), timed two ways:

  eager  -- python-side RawKernel launches in a loop (cupy launch
            overhead ~5-15 us each pays per step);
  graph  -- capture the chain once into a CUDA graph, replay it.

This sizes how much of the 21,000-step loop is launch overhead and
whether cuBLAS-free stepping can hit the "<2 min full export" target.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import cupy as cp

FAMD = ("-fmad=false",)

SRC = r"""
extern "C" __global__
void k_decay(float* y, double d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)(y[i] * d);
}

extern "C" __global__
void k_ringadd(float* y, const float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { y[i] += buf[i]; }
}

// per-PRE-neuron delivery: thread p owns its (disjoint) edge range --
// unique targets, no atomics, fully deterministic, fixed launch size
extern "C" __global__
void k_deliver(const unsigned char* spiked, const long long* starts,
               const long long* counts, const long long* flat_idx,
               float* y, const float* kick, int n_pre)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pre || !spiked[p]) return;
    for (long long j = starts[p]; j < starts[p] + counts[p]; ++j) {
        long long t = flat_idx[j];
        y[t] = y[t] + kick[t];
    }
}

extern "C" __global__
void k_graded(const float* r_pre, const long long* pre_ids, float* s,
              float* y, const float* weight, double k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double r = (double)r_pre[pre_ids[i]];
    if (r < 0.0) r = 0.0;
    else if (r > 1.0) r = 1.0;
    s[i] = (float)((double)s[i] + k * (r - (double)s[i]));
    y[i] = weight[i] * s[i];
}

extern "C" __global__
void k_drive(const float* y, float g_unit, const float* e_rev,
             const long long* edge_of, const long long* indptr,
             float* out_i, float* out_g, int n_post)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_post) return;
    float g32 = g_unit;
    float ai = 0.0f, ag = 0.0f;
    for (long long j = indptr[p]; j < indptr[p + 1]; ++j) {
        long long e = edge_of[j];
        float gy = g32 * y[e];
        ai += gy * e_rev[e];
        ag += gy;
    }
    out_i[p] += ai;
    out_g[p] += ag;
}

extern "C" __global__
void k_ou(double* x, const double* w, double a, double kk, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = a * x[i] + kk * w[i];
}

extern "C" __global__
void k_lif(double* v, double* refrac, const double* i_ext, const double* g,
           unsigned char* spike, int n, double dt, double tau_m,
           double v_rest, double v_th, double v_reset, double t_refrac,
           double R_m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = dt / tau_m;
    if (refrac[i] > 0.0) v[i] = v_reset;
    v[i] = (v[i] + a * (v_rest + R_m * i_ext[i]))
           / (1.0 + a * (1.0 + R_m * g[i]));
    spike[i] = v[i] >= v_th;
    if (spike[i]) v[i] = v_reset;
    double r = refrac[i] - dt;
    refrac[i] = r < 0.0 ? 0.0 : r;
    if (spike[i]) refrac[i] = t_refrac;
}

extern "C" __global__
void k_ecur(const float* y, float g_unit, const float* e_rev,
            const long long* post_local, const double* v_post,
            double* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g32 = g_unit;
    double t1 = (double)(g32 * y[i]);
    double t2 = (double)e_rev[i] - v_post[post_local[i]];
    out[i] = t1 * t2;
}
"""

K = {name: cp.RawKernel(SRC, name, options=FAMD)
     for name in ("k_decay", "k_ringadd", "k_deliver", "k_graded", "k_drive",
                  "k_ou", "k_lif", "k_ecur")}


def launch(name, n, args, block=256):
    K[name](((n + block - 1) // block,), (block,), tuple(args))


def bench(tag, n_edges, n_graded, n_pre, n_post, steps=200):
    rng = np.random.default_rng(0)
    f32 = np.float32
    y = cp.asarray(rng.random(n_edges).astype(f32))
    buf = cp.asarray(rng.random(n_edges).astype(f32))
    kick = cp.asarray(rng.random(n_edges).astype(f32))
    flat = cp.asarray(rng.integers(0, n_edges, n_edges).astype(np.int64))
    # disjoint per-pre edge ranges
    cuts = np.sort(rng.choice(n_edges, n_pre - 1, replace=False))
    counts = np.diff(np.concatenate(([0], cuts, [n_edges])))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    starts = cp.asarray(starts.astype(np.int64))
    counts = cp.asarray(counts.astype(np.int64))
    spiked = cp.asarray((rng.random(n_pre) < 0.02).astype(np.uint8))

    s_ = cp.asarray(rng.random(n_graded).astype(f32))
    ygr = cp.zeros(n_graded, cp.float32)
    pid = cp.asarray(rng.integers(0, n_pre, n_graded).astype(np.int64))
    wgr = cp.asarray(rng.random(n_graded).astype(f32))
    r_pre = cp.asarray(rng.random(n_pre).astype(f32))

    post_local = cp.asarray(rng.integers(0, n_post, n_edges).astype(np.int64))
    e_rev = cp.asarray(rng.random(n_edges).astype(f32))
    order = np.argsort(cp.asnumpy(post_local), kind="stable")
    counts_p = np.bincount(cp.asnumpy(post_local), minlength=n_post)
    indptr = np.zeros(n_post + 1, np.int64)
    np.cumsum(counts_p, out=indptr[1:])
    edge_of = cp.asarray(order.astype(np.int64))
    indptr = cp.asarray(indptr)
    oi = cp.zeros(n_post, cp.float32)
    og = cp.zeros(n_post, cp.float32)

    xou = cp.asarray(rng.standard_normal(n_pre))
    wou = cp.asarray(rng.standard_normal(n_pre))
    v = cp.asarray(rng.standard_normal(n_pre) * 5 - 70)
    refrac = cp.zeros(n_pre, cp.float64)
    g64 = cp.asarray(rng.random(n_pre))
    spike = cp.zeros(n_pre, cp.uint8)
    vp = cp.asarray(rng.standard_normal(n_post))
    ecur = cp.zeros(n_edges, cp.float64)

    decay_d = float(np.exp(-5e-4 / 5.0))
    a_ou, k_ou = 0.99, 1.0

    def chain():
        launch("k_decay", n_edges, (y, np.float64(decay_d), np.int32(n_edges)))
        launch("k_ringadd", n_edges, (y, buf, np.int32(n_edges)))
        launch("k_deliver", n_pre, (spiked, starts, counts, flat, y, kick,
                                    np.int32(n_pre)))
        launch("k_graded", n_graded, (r_pre, pid, s_, ygr, wgr,
                                      np.float64(1e-4), np.int32(n_graded)))
        launch("k_drive", n_post, (y, np.float32(0.02), e_rev, edge_of,
                                   indptr, oi, og, np.int32(n_post)))
        launch("k_ou", n_pre, (xou, wou, np.float64(a_ou), np.float64(k_ou),
                               np.int32(n_pre)))
        launch("k_lif", n_pre, (v, refrac, xou, g64, spike, np.int32(n_pre),
                                *(np.float64(x) for x in
                                  (5e-4, 10.0, -70.0, -50.0, -70.0, 2.0,
                                   0.1))))
        launch("k_ecur", n_edges, (y, np.float32(0.02), e_rev, post_local,
                                   vp, ecur, np.int32(n_edges)))

    chain()                      # warm up / compile
    cp.cuda.runtime.deviceSynchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        chain()
    cp.cuda.runtime.deviceSynchronize()
    t_eager = (time.perf_counter() - t0) / steps * 1e3

    # --- CUDA graph capture + replay ----------------------------------
    stream = cp.cuda.Stream()
    with stream:
        stream.begin_capture()
        try:
            chain()
        finally:
            graph = stream.end_capture()
    gexec = graph            # cupy's captured Graph auto-instantiates and
    t0 = time.perf_counter() # exposes .launch() directly
    for _ in range(steps):
        gexec.launch(stream)
    stream.synchronize()
    t_graph = (time.perf_counter() - t0) / steps * 1e3

    print(f"{tag:>6}: eager {t_eager:8.3f} ms/step | graph {t_graph:8.3f}"
          f" ms/step | speedup {t_eager / t_graph:5.2f}x |"
          f" 21,000 steps -> {t_graph * 21_000 / 1e3:6.1f} s (graph)")


if __name__ == "__main__":
    bench("SMALL", n_edges=4_000_000, n_graded=400_000, n_pre=10_000,
          n_post=10_000)
    bench("FULL", n_edges=64_000_000, n_graded=6_000_000, n_pre=150_000,
          n_post=150_000)
