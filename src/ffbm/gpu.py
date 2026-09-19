"""P3: GPU-resident trial engine (CuPy RawKernels, NVRTC).

Replicates ``pipeline.simulate()``'s mechanistic-lamina loop with all
state resident on the GPU, BITWISE to the CPU/numba path:

- every kernel is compiled with NVRTC ``-fmad=false`` so float
  arithmetic never contracts into fused multiply-adds (validated by
  scripts/poc_cupy_parity.py: LIF variants / OU / graded / delivery all
  match the numba kernels bit-for-bit);
- the OU noise stays a CPU numpy stream: batches are pre-drawn in the
  EXACT per-step draw order (R, L, MID, T45, then extra regions) and
  uploaded, so the RNG red line is untouched;
- conductance drives are streaming per-post segment sums: pool state y
  is stored post-sorted (lexsort((pre, post))), so the numba kernel's
  global-edge-order accumulation equals a serial walk over each post's
  contiguous edge range -- no gather, no permute, no atomics;
- delivery is one thread per presynaptic row: rows partition the edges
  (disjoint ranges -> unique targets), so scatter-add is race-free and
  deterministic.  The ring pointer advances BEFORE delivery, exactly
  like the numba step (bin b lands in ring row (ptr + b - 1) % len);
- scalar folding follows numpy weak-scalar promotion order everywhere
  (see the traps list in docs/ACCELERATION_PLAN.md P3-1).

Only the ``LAMINA_MECHANISTIC`` branch is supported (the production
path).  Record support for viz/export_data.py --gpu is layered on top.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    _HAVE_CUPY = True
except ImportError:                                    # pragma: no cover
    cp = None
    _HAVE_CUPY = False

from .pipeline import PhotoCascade, build_stack

_FAMD = ("-fmad=false",)

_SRC = r"""
extern "C" __global__
void k_release(const double* v, double lo, double hi, double* r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = (v[i] - lo) / (hi - lo);
    if (x < 0.0) x = 0.0;
    else if (x > 1.0) x = 1.0;
    r[i] = x;
}

// numba typing note: the clip literals make r float64, so (r - s) is an
// exact f64 subtraction -- replicate with doubles, not floats
extern "C" __global__
void k_graded(const double* r_pre, const long long* pre_ids, float* s,
              float* y, const float* weight, double k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double r = r_pre[pre_ids[i]];
    if (r < 0.0) r = 0.0;
    else if (r > 1.0) r = 1.0;
    s[i] = (float)((double)s[i] + k * (r - (double)s[i]));
    y[i] = weight[i] * s[i];
}

// streaming per-post segment sum: y/e_rev are the pool's own post-sorted
// arrays, indptr gives each post's contiguous ascending edge range -- the
// accumulation order equals the numba _drive_cond global-edge scan
extern "C" __global__
void k_drive(const float* y, float g_unit, const float* e_rev,
             const int* indptr, float* out_i, float* out_g, int n_post)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_post) return;
    float g32 = g_unit;
    float ai = 0.0f, ag = 0.0f;
    for (int j = indptr[p]; j < indptr[p + 1]; ++j) {
        float gy = g32 * y[j];
        ai += gy * e_rev[j];
        ag += gy;
    }
    out_i[p] += ai;
    out_g[p] += ag;
}

extern "C" __global__
void k_ou(double* x, const double* w, double a, double k, long long stride,
          int step, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = a * x[i] + k * w[(long long)step * stride + i];
}

// R: i_ext = (base + inc) + w      (explicit Euler, g_is_none)
extern "C" __global__
void k_lif_none(double* v, double* refrac, const double* inc,
                const double* x_n, double base,
                unsigned char* spike, int n,
                double dt, double tau_m, double v_rest, double v_th,
                double v_reset, double t_refrac, double R_m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double i_ext = (base + inc[i]) + x_n[i];
    if (refrac[i] > 0.0) v[i] = v_reset;
    v[i] = v[i] + dt * (-(v[i] - v_rest) + R_m * i_ext) / tau_m;
    spike[i] = v[i] >= v_th;
    if (spike[i]) v[i] = v_reset;
    double r = refrac[i] - dt;
    refrac[i] = r < 0.0 ? 0.0 : r;
    if (spike[i]) refrac[i] = t_refrac;
}

// L: i_ext = (double)di + w, g_tot = dg (f32) -- the _lif_step_g32 path
extern "C" __global__
void k_lif_g32(double* v, double* refrac, const float* di, const float* dg,
               const double* x_n, unsigned char* spike, int n,
               double dt, double tau_m, double v_rest, double v_th,
               double v_reset, double t_refrac, double R_m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double i_ext = (double)di[i] + x_n[i];
    double a = dt / tau_m;
    float a32 = (float)a;
    float rm32 = (float)R_m;
    if (refrac[i] > 0.0) v[i] = v_reset;
    double num = v[i] + a * (v_rest + R_m * i_ext);
    float den = 1.0f + a32 * (1.0f + rm32 * dg[i]);
    v[i] = num / (double)den;
    spike[i] = v[i] >= v_th;
    if (spike[i]) v[i] = v_reset;
    double r = refrac[i] - dt;
    refrac[i] = r < 0.0 ? 0.0 : r;
    if (spike[i]) refrac[i] = t_refrac;
}

// MID: i_ext = (base[i] + (double)di[i]) + w, g_tot = dg (f32)
extern "C" __global__
void k_lif_g32_base(double* v, double* refrac, const float* di,
                    const float* dg, const double* base, const double* x_n,
                    unsigned char* spike, int n, double dt, double tau_m, double v_rest,
                    double v_th, double v_reset, double t_refrac,
                    double R_m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double i_ext = (base[i] + (double)di[i]) + x_n[i];
    double a = dt / tau_m;
    float a32 = (float)a;
    float rm32 = (float)R_m;
    if (refrac[i] > 0.0) v[i] = v_reset;
    double num = v[i] + a * (v_rest + R_m * i_ext);
    float den = 1.0f + a32 * (1.0f + rm32 * dg[i]);
    v[i] = num / (double)den;
    spike[i] = v[i] >= v_th;
    if (spike[i]) v[i] = v_reset;
    double r = refrac[i] - dt;
    refrac[i] = r < 0.0 ? 0.0 : r;
    if (spike[i]) refrac[i] = t_refrac;
}

// T45 / extras: f64 i_ext and g_tot accumulators built on device
extern "C" __global__
void k_acc_init(double* i_x, double* g_x, const double* base,
                const double* x_n, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    i_x[i] = base[i] + x_n[i];
    g_x[i] = 0.0;
}

extern "C" __global__
void k_acc_add(double* i_x, double* g_x, const float* di, const float* dg,
               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        i_x[i] += (double)di[i];
        g_x[i] += (double)dg[i];
    }
}

// chemosensory drive (exp019): add a per-neuron stimulus current
extern "C" __global__
void k_chem_add(double* i_x, const double* chem, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) i_x[i] += chem[i];
}

extern "C" __global__
void k_lif_g64(double* v, double* refrac, const double* i_ext,
               const double* g, unsigned char* spike, int n,
               double dt, double tau_m, double v_rest, double v_th,
               double v_reset, double t_refrac, double R_m)
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

// delayed exp step A: decay (f64 mult, f32 store) + ring row add
extern "C" __global__
void k_exp_ring(float* y, const float* buf, int ptr, double decay, int nE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;
    float yy = (float)(y[i] * decay);
    y[i] = yy + buf[(long long)ptr * nE + i];
}

extern "C" __global__
void k_buf_clear(float* buf, int row, int nE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nE) buf[(long long)row * nE + i] = 0.0f;
}

// delivery: one thread per presynaptic row (disjoint edge ranges ->
// unique targets -> no atomics); bin 0 -> y, bins >= 1 -> ring row
// (ptr + b - 1) % len -- ptr is the ALREADY-ADVANCED ring pointer
extern "C" __global__
void k_deliver(float* y, float* buf, int ptr, int nE, int buf_len,
               const unsigned char* spike_row, const long long* starts,
               const long long* counts, const long long* flat_idx,
               const int* bins, const float* kick, int n_rows)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_rows || !spike_row[p]) return;
    for (long long j = starts[p]; j < starts[p] + counts[p]; ++j) {
        long long t = flat_idx[j];
        int b = bins[t];
        if (b == 0) {
            y[t] = y[t] + kick[t];
        } else {
            int row = (ptr + b - 1) % buf_len;
            buf[(long long)row * nE + t] += kick[t];
        }
    }
}

// short-term depression: per-edge recovery toward 1 (each step)
extern "C" __global__
void k_std_rec(float* d, float rec, int nE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nE) d[i] += (1.0f - d[i]) * rec;
}

// delivery with a depression gate (depletion at RELEASE time; the
// gated kick goes to y for bin 0 or into the ring row for later bins)
extern "C" __global__
void k_deliver_std(float* y, float* buf, int ptr, int nE, int buf_len,
                   const unsigned char* spike_row,
                   const long long* starts, const long long* counts,
                   const long long* flat_idx, const int* bins,
                   const float* kick, float* d, float u, int n_rows)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_rows || !spike_row[p]) return;
    for (long long j = starts[p]; j < starts[p] + counts[p]; ++j) {
        long long t = flat_idx[j];
        float k = kick[t] * d[t];
        d[t] *= (1.0f - u);
        int b = bins[t];
        if (b == 0) {
            y[t] += k;
        } else {
            int row = (ptr + b - 1) % buf_len;
            buf[(long long)row * nE + t] += k;
        }
    }
}

// DAN-gated plasticity advance: eligibility decay + gated weight
// depression (mod = reinforcement proxy, scalar this step) + optional
// slow homeostatic recovery of w toward 1 (rec = dt/tau_w, 0 = off)
extern "C" __global__
void k_plast_adv(float* elig, float* w, float decay, float lr,
                 float mod, float floor, float rec, int nE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nE) return;
    float e = elig[i] * decay;
    elig[i] = e;
    if (mod > 0.0f) {
        float wv = w[i] * (1.0f - lr * mod * e);
        w[i] = wv < floor ? floor : wv;
    }
    if (rec > 0.0f) {
        w[i] += (1.0f - w[i]) * rec;
    }
}

// delivery with a plastic weight scale (depletion at RELEASE time,
// delayed groups route the scaled kick through the ring)
extern "C" __global__
void k_deliver_plast(float* y, float* buf, int ptr, int nE, int buf_len,
                     const unsigned char* spike_row,
                     const long long* starts, const long long* counts,
                     const long long* flat_idx, const int* bins,
                     const float* kick, float* w, float* elig,
                     int n_rows)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_rows || !spike_row[p]) return;
    for (long long j = starts[p]; j < starts[p] + counts[p]; ++j) {
        long long t = flat_idx[j];
        float k = kick[t] * w[t];
        elig[t] += 1.0f;
        int b = bins[t];
        if (b == 0) {
            y[t] += k;
        } else {
            int row = (ptr + b - 1) % buf_len;
            buf[(long long)row * nE + t] += k;
        }
    }
}

// gather this step's spike bits for a group's presynaptic rows from the
// concatenated all-population mask buffer
extern "C" __global__
void k_spike_row(unsigned char* spike_row, const int* src_off,
                 const int* local, const unsigned char* mask_all,
                 int n_rows)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n_rows) spike_row[p] = mask_all[src_off[p] + local[p]];
}

// record path: per-edge current with the CPU expression's exact dtype
// path (f32 g*y product widened to f64, f64 (e_rev - v_post), f64
// product) then the same f64 -> f32 store the numpy ybuf row cast does.
// The _keep variant writes only the subsampled edges (kernel_keep),
// like the CPU `ybuf[..] = y[keep]` path.
extern "C" __global__
void k_ecur_f32(const float* y, float g_unit, const float* e_rev,
                const long long* post_local, const double* v_post,
                const long long* keep, float* out_row, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long e = keep == 0 ? i : keep[i];
    float g32 = g_unit;
    double t1 = (double)(g32 * y[e]);
    double t2 = (double)e_rev[e] - v_post[post_local[e]];
    out_row[i] = (float)(t1 * t2);
}
"""

_K = {}


def _kernels():
    if not _K:
        for name in ("k_release", "k_graded", "k_drive", "k_ou",
                     "k_lif_none", "k_lif_g32", "k_lif_g32_base",
                     "k_acc_init", "k_acc_add", "k_chem_add",
                     "k_lif_g64", "k_exp_ring",
                     "k_buf_clear", "k_deliver", "k_spike_row",
                     "k_std_rec", "k_deliver_std",
                     "k_plast_adv", "k_deliver_plast",
                     "k_ecur_f32"):
            _K[name] = cp.RawKernel(_SRC, name, options=_FAMD)
    return _K


def _go(kernel, n, args, block=256):
    kernel(((n + block - 1) // block,), (block,), tuple(args))


def _f64(x):
    return np.float64(x)


def _indptr(post_local, n_post):
    counts = np.bincount(post_local, minlength=n_post)
    indptr = np.zeros(n_post + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr.astype(np.int32)


class _GradedPoolG:
    """GradedSynapsePool on device (RL / LM)."""

    def __init__(self, pool):
        self.n_edges = int(pool.n_edges)
        self.n_post = int(pool.csr.shape[0])
        self.pre_ids = cp.asarray(pool.pre_ids.astype(np.int64))
        self.s = cp.asarray(pool.s)
        self.y = cp.asarray(pool.y)
        self.weight = cp.asarray(pool.weight)
        self.e_rev = cp.asarray(pool.e_rev_edge)
        self.k = _f64(pool.k)
        self.g_unit = np.float32(pool.g_unit)
        self.indptr = cp.asarray(_indptr(pool.post_local, self.n_post))
        self.post_local = cp.asarray(pool.post_local.astype(np.int64))
        self.out_i = cp.zeros(self.n_post, cp.float32)
        self.out_g = cp.zeros(self.n_post, cp.float32)

    def step(self, K, r_pre):
        if self.n_edges:
            _go(K["k_graded"], self.n_edges,
                (r_pre, self.pre_ids, self.s, self.y, self.weight,
                 self.k, np.int32(self.n_edges)))

    def ecur_row(self, K, v_post, out_row, keep=None):
        """edge_currents(v_post) cast f32 into a ybuf row (bitwise the
        CPU edge_currents -> ybuf[..] = y[keep] path)."""
        if self.n_edges:
            _go(K["k_ecur_f32"], self.n_edges if keep is None
                else keep.size,
                (self.y, self.g_unit, self.e_rev, self.post_local,
                 v_post, np.int64(0) if keep is None else keep,
                 out_row,
                 np.int32(self.n_edges if keep is None else keep.size)))

    def drive(self, K):
        if not self.n_edges:
            return self.out_i, self.out_g
        self.out_i.fill(0)
        self.out_g.fill(0)
        _go(K["k_drive"], self.n_post,
            (self.y, self.g_unit, self.e_rev, self.indptr,
             self.out_i, self.out_g, np.int32(self.n_post)))
        return self.out_i, self.out_g


class _ExpPoolG:
    """Conductance ExponentialSynapses on device (MT_* / extra groups)."""

    def __init__(self, pool, mask_all, src_off, local):
        self.n_edges = int(pool.n_edges)
        self.n_post = int(pool.csr.shape[0])
        self.mask_all = mask_all
        self.y = cp.asarray(pool.y)
        self.delayed = bool(pool.delayed)
        if self.delayed:
            self.buffer = cp.asarray(np.ascontiguousarray(pool.buffer))
            self.buf_len = int(pool.buffer.shape[0])
            self.bins = cp.asarray(pool.delay_bins.astype(np.int32))
        else:
            self.buffer = cp.zeros(1, cp.float32)
            self.buf_len = 1
            self.bins = cp.zeros(max(self.n_edges, 1), cp.int32)
        self.ptr = 0
        self.decay = _f64(pool.decay)
        self.kick = cp.asarray(pool.kick)
        self.flat_idx = cp.asarray(pool.flat_idx.astype(np.int64))
        self.starts = cp.asarray(pool.deliv_starts.astype(np.int64))
        self.counts = cp.asarray(pool.deliv_counts.astype(np.int64))
        self.n_rows = int(len(pool.deliv_counts))
        self.spike_row = cp.zeros(max(self.n_rows, 1), cp.uint8)
        self.src_off = cp.asarray(src_off.astype(np.int32))
        self.local = cp.asarray(local.astype(np.int32))
        self.g_unit = np.float32(pool.g_unit)
        self.e_rev = cp.asarray(pool.e_rev_edge)
        self.indptr = cp.asarray(_indptr(pool.post_local, self.n_post))
        self.post_local = cp.asarray(pool.post_local.astype(np.int64))
        self.out_i = cp.zeros(self.n_post, cp.float32)
        self.out_g = cp.zeros(self.n_post, cp.float32)
        self.std = bool(getattr(pool, "std", False))
        if self.std:
            self.std_d = cp.asarray(pool.std_d)
            self.std_u = np.float32(pool.std_u)
            self.std_rec = np.float32(pool.std_rec)
        self.plast = bool(getattr(pool, "plast", False))
        if self.plast:
            self.plast_elig = cp.asarray(pool.elig)
            self.plast_w = cp.asarray(pool.w_scale)
            self.plast_decay = np.float32(pool.elig_decay)
            self.plast_lr = np.float32(pool.plast_lr)
            self.plast_floor = np.float32(pool.w_floor)
            self.plast_rec = np.float32(getattr(pool, "w_rec", 0.0))

    def ecur_row(self, K, v_post, out_row, keep=None):
        """edge_currents(v_post) cast f32 into a ybuf row (bitwise the
        CPU edge_currents -> ybuf[..] = y[keep] path)."""
        if self.n_edges:
            _go(K["k_ecur_f32"], self.n_edges if keep is None
                else keep.size,
                (self.y, self.g_unit, self.e_rev, self.post_local,
                 v_post, np.int64(0) if keep is None else keep,
                 out_row,
                 np.int32(self.n_edges if keep is None else keep.size)))

    def drive(self, K):
        if not self.n_edges:
            return self.out_i, self.out_g
        self.out_i.fill(0)
        self.out_g.fill(0)
        _go(K["k_drive"], self.n_post,
            (self.y, self.g_unit, self.e_rev, self.indptr,
             self.out_i, self.out_g, np.int32(self.n_post)))
        return self.out_i, self.out_g

    def step(self, K, mod=0.0):
        """End-of-step advance: decay + ring + delivery (numba order:
        decay, ring add, ring clear, ptr advance, THEN delivery).  mod
        = reinforcement gate for plastic groups."""
        if not self.n_edges:
            return
        if self.delayed:
            _go(K["k_exp_ring"], self.n_edges,
                (self.y, self.buffer, np.int32(self.ptr), self.decay,
                 np.int32(self.n_edges)))
            _go(K["k_buf_clear"], self.n_edges,
                (self.buffer, np.int32(self.ptr), np.int32(self.n_edges)))
            self.ptr = (self.ptr + 1) % self.buf_len
        _go(K["k_spike_row"], self.n_rows,
            (self.spike_row, self.src_off, self.local, self.mask_all,
             np.int32(self.n_rows)))
        if self.plast:
            _go(K["k_plast_adv"], self.n_edges,
                (self.plast_elig, self.plast_w, self.plast_decay,
                 self.plast_lr, np.float32(mod), self.plast_floor,
                 self.plast_rec, np.int32(self.n_edges)))
            _go(K["k_deliver_plast"], self.n_rows,
                (self.y, self.buffer, np.int32(self.ptr),
                 np.int32(self.n_edges), np.int32(self.buf_len),
                 self.spike_row, self.starts, self.counts,
                 self.flat_idx, self.bins, self.kick,
                 self.plast_w, self.plast_elig,
                 np.int32(self.n_rows)))
            return
        if self.std:
            _go(K["k_std_rec"], self.n_edges,
                (self.std_d, self.std_rec, np.int32(self.n_edges)))
            _go(K["k_deliver_std"], self.n_rows,
                (self.y, self.buffer, np.int32(self.ptr),
                 np.int32(self.n_edges), np.int32(self.buf_len),
                 self.spike_row, self.starts, self.counts,
                 self.flat_idx, self.bins, self.kick,
                 self.std_d, self.std_u, np.int32(self.n_rows)))
            return
        _go(K["k_deliver"], self.n_rows,
            (self.y, self.buffer, np.int32(self.ptr), np.int32(self.n_edges),
             np.int32(self.buf_len), self.spike_row, self.starts,
             self.counts, self.flat_idx, self.bins, self.kick,
             np.int32(self.n_rows)))


class _PopG:
    def __init__(self, pop, mask_off):
        self.n = pop.n
        self.v = cp.asarray(pop.v)
        self.refrac = cp.asarray(pop.refrac)
        self.mask_off = mask_off          # slice into the shared mask buf
        for attr in ("dt", "tau_m", "v_rest", "v_th", "v_reset",
                     "t_refrac", "R_m"):
            setattr(self, attr, _f64(getattr(pop, attr)))


class GPUTrial:
    """One trial's full state on device; run() mirrors simulate()'s loop."""

    NOISE_BATCH = 512

    def __init__(self, st):
        assert st["mech"], "GPU engine supports the mechanistic branch only"
        assert _HAVE_CUPY, "cupy not installed"
        self.st = st
        self.K = _kernels()
        cal = st["cal"]

        # -- concatenated spike-mask buffer: R, L, MID, T45, extras ------
        self.pop_order = ["R", "L", "MID", "T45", *st["extra_pops"]]
        n_all = sum(st["pops"][p].n for p in self.pop_order)
        self.mask_all = cp.zeros(n_all, cp.uint8)
        self.pops = {}
        self._mask_offsets = {}
        off = 0
        for name in self.pop_order:
            self.pops[name] = _PopG(st["pops"][name], off)
            self._mask_offsets[name] = off
            off += self.pops[name].n
        self._id_maps = {name: {int(b): i for i, b in
                                enumerate(np.asarray(ids).tolist())}
                         for name, ids in self._pop_ids().items()}

        # -- release maps / bases --------------------------------------
        self.rrl = (_f64(cal["R_RELEASE_MAP_MV"][0]),
                    _f64(cal["R_RELEASE_MAP_MV"][1]))
        self.rlm = (_f64(cal["L_RELEASE_MAP_MV"][0]),
                    _f64(cal["L_RELEASE_MAP_MV"][1]))
        self.i_r_base = _f64(cal["I_R_BASE"])
        self.mid_base = cp.asarray(st["mid_base"])
        self.t45_base = cp.asarray(st["t45_base"])
        self.extra_base = {n: cp.asarray(b)
                           for n, b in st["extra_base"].items()}

        # -- synapse groups ---------------------------------------------
        self.graded = {"RL": _GradedPoolG(st["syn"]["RL"]),
                       "LM": _GradedPoolG(st["syn"]["LM"])}
        self.exp = {}
        for name, pool in st["syn"].items():
            if not (name.startswith("MT_") or name in st["extra_pre"]):
                continue
            src_pops = (("MID",) if name.startswith("MT_")
                        else tuple(st["extra_pre"][name]))
            src_off, local = self._pre_lookup(pool, src_pops)
            self.exp[name] = _ExpPoolG(pool, self.mask_all, src_off, local)
        self.mt_names = [f"MT_{mt}" for mt in cal["MID_TAU_S"]]

        # -- noise (drawn on CPU in the exact per-step order) ------------
        self.noise_names = ["R", "L", "MID", "T45", *st["extra_pops"]]
        self._noise_rng = st["noises"]["R"].rng
        self._ou = {k: (_f64(st["noises"][k].a), _f64(st["noises"][k].k),
                        cp.asarray(st["noises"][k].x))
                    for k in self.noise_names}
        self._noise = {}
        self._batch_pos = 0
        self._draw_noise_batch()

        self.r_release = cp.zeros(st["n_r"], cp.float64)
        self.l_release = cp.zeros(st["n_l"], cp.float64)
        self._t45_acc = (cp.zeros(st["n_t45"], cp.float64),
                         cp.zeros(st["n_t45"], cp.float64))
        self._extra_acc = {n: (cp.zeros(len(ids), cp.float64),
                               cp.zeros(len(ids), cp.float64))
                           for n, ids in st["extra_ids"].items()}

    # ------------------------------------------------------------------
    def _pop_ids(self):
        st = self.st
        return {"R": st["r_ids"], "L": st["l_ids"], "MID": st["mid_ids"],
                "T45": st["t45_ids"],
                **{n: ids for n, ids in st["extra_ids"].items()}}

    def _pre_lookup(self, pool, src_pops):
        """Per-row (mask offset, local index) for a pool's presynaptic
        body ids, searching the group's source pops in CPU order."""
        src_off = np.zeros(len(pool.unique_pre), np.int64)
        local = np.full(len(pool.unique_pre), -1, np.int64)
        for i, body in enumerate(pool.unique_pre.tolist()):
            for nm in src_pops:
                j = self._id_maps.get(nm, {}).get(body, -1)
                if j >= 0:
                    src_off[i] = self._mask_offsets[nm]
                    local[i] = j
                    break
        return src_off, local

    def _draw_noise_batch(self):
        """Interleaved draws in the per-step pop order (R, L, MID, T45,
        extras) keep the numpy stream bitwise-identical to the CPU loop."""
        B = self.NOISE_BATCH
        n_of = {k: self.st["pops"][k].n for k in self.noise_names}
        rng = self._noise_rng
        buf = {k: np.empty((B, n_of[k]), np.float64)
               for k in self.noise_names}
        for s in range(B):
            for k in self.noise_names:
                buf[k][s] = rng.standard_normal(n_of[k])
        self._noise = {k: cp.asarray(np.ascontiguousarray(v))
                       for k, v in buf.items()}
        self._batch_pos = 0

    def _mask(self, name):
        off = self._mask_offsets[name]
        return self.mask_all[off:off + self.pops[name].n]

    def _x(self, name):
        return self._ou[name][2]

    # ------------------------------------------------------------------
    def _step(self, t, inc_f, mod=0.0):
        st = self.st
        K = self.K
        inc = cp.asarray(np.asarray(inc_f, dtype=np.float64))

        # OU updates (x feeds this step's i_ext, order across pops is
        # value-irrelevant; per-pop values match ColoredCurrentNoise)
        for nm in self.noise_names:
            a, kk, x = self._ou[nm]
            w = self._noise[nm]
            _go(K["k_ou"], x.size,
                (x, w, a, kk, np.int64(w.shape[1]),
                 np.int32(self._batch_pos), np.int32(x.size)))

        # ---- R -> RL -> L -> LM -> MID --------------------------------
        pR = self.pops["R"]
        _go(K["k_lif_none"], pR.n,
            (pR.v, pR.refrac, inc, self._x("R"),
             self.i_r_base, self._mask("R"), np.int32(pR.n),
             pR.dt, pR.tau_m, pR.v_rest, pR.v_th, pR.v_reset,
             pR.t_refrac, pR.R_m))
        _go(K["k_release"], st["n_r"],
            (pR.v, self.rrl[0], self.rrl[1], self.r_release,
             np.int32(st["n_r"])))
        self.graded["RL"].step(K, self.r_release)
        di, dg = self.graded["RL"].drive(K)
        pL = self.pops["L"]
        _go(K["k_lif_g32"], pL.n,
            (pL.v, pL.refrac, di, dg, self._x("L"),
             self._mask("L"), np.int32(pL.n), pL.dt, pL.tau_m, pL.v_rest,
             pL.v_th, pL.v_reset, pL.t_refrac, pL.R_m))
        _go(K["k_release"], st["n_l"],
            (pL.v, self.rlm[0], self.rlm[1], self.l_release,
             np.int32(st["n_l"])))
        self.graded["LM"].step(K, self.l_release)
        di, dg = self.graded["LM"].drive(K)
        pM = self.pops["MID"]
        _go(K["k_lif_g32_base"], pM.n,
            (pM.v, pM.refrac, di, dg, self.mid_base,
             self._x("MID"), self._mask("MID"), np.int32(pM.n),
             pM.dt, pM.tau_m, pM.v_rest, pM.v_th, pM.v_reset, pM.t_refrac,
             pM.R_m))

        # ---- T45: base + noise, then per-group f64 accumulation -------
        pT = self.pops["T45"]
        i_x, g_x = self._t45_acc
        _go(K["k_acc_init"], pT.n,
            (i_x, g_x, self.t45_base, self._x("T45"), np.int32(pT.n)))
        for name in self.mt_names:
            g = self.exp.get(name)
            if g is None:
                continue
            di, dg = g.drive(K)
            if g.n_edges:
                _go(K["k_acc_add"], pT.n, (i_x, g_x, di, dg, np.int32(pT.n)))
        _go(K["k_lif_g64"], pT.n,
            (pT.v, pT.refrac, i_x, g_x, self._mask("T45"), np.int32(pT.n),
             pT.dt, pT.tau_m, pT.v_rest, pT.v_th, pT.v_reset, pT.t_refrac,
             pT.R_m))

        # ---- extras ----------------------------------------------------
        chem = getattr(self, "_chem", None) or {}
        for name in st["extra_pops"]:
            p = self.pops[name]
            i_x, g_x = self._extra_acc[name]
            _go(K["k_acc_init"], p.n,
                (i_x, g_x, self.extra_base[name],
                 self._x(name), np.int32(p.n)))
            c = chem.get(name)
            if c is not None:
                _go(K["k_chem_add"], p.n, (i_x, c, np.int32(p.n)))
            for grp in st["extra_in"][name]:
                g = self.exp.get(grp)
                if g is None:
                    continue
                di, dg = g.drive(K)
                if g.n_edges:
                    _go(K["k_acc_add"], p.n, (i_x, g_x, di, dg, np.int32(p.n)))
            _go(K["k_lif_g64"], p.n,
                (p.v, p.refrac, i_x, g_x, self._mask(name), np.int32(p.n),
                 p.dt, p.tau_m, p.v_rest, p.v_th, p.v_reset, p.t_refrac,
                 p.R_m))

        # ---- end-of-step delivery (MT groups, then extra groups) -------
        for name in self.mt_names:
            if name in self.exp:
                self.exp[name].step(K)
        for grp in st["extra_pre"]:
            if grp in self.exp:
                self.exp[grp].step(K, mod)

        self._batch_pos += 1
        if self._batch_pos >= self.NOISE_BATCH:
            self._draw_noise_batch()

    def run(self, lum_inc_fn, n_steps, hook=None, on_record=None,
            chem_fn=None, mod_fn=None):
        """hook(k, t) fires after each step (parity checking).
        on_record(k, j, t, inc_f) mirrors simulate()'s on_sample cadence
        (every 2 steps); inc_f is the CPU photo increment (numpy f64).
        chem_fn(t) -> {pop: f64 increment array} (exp019 chem drive).
        mod_fn(t) -> float reinforcement gate for plastic groups."""
        photo = PhotoCascade(self.st["n_r"], self.st["pops"]["R"].dt)
        dt = float(self.st["pops"]["R"].dt)
        for k in range(n_steps):
            t = k * dt
            inc_f = photo.step(lum_inc_fn(t))
            if chem_fn is not None:
                self._chem = {n: cp.asarray(v, dtype=cp.float64)
                              for n, v in chem_fn(t).items()}
            _mod = mod_fn(t) if mod_fn is not None else 0.0
            self._step(t, inc_f, _mod)
            if hook is not None:
                hook(k, t)
            if on_record is not None and k % 2 == 0:
                on_record(k, k // 2, t, inc_f)

    # -- record-side helpers ------------------------------------------
    def v_post(self, name):
        """Device membrane potential of a postsynaptic population."""
        return self.pops[name].v

    def spike_count(self, name):
        """Spikes this step (sync; call once per record step)."""
        return int(self._mask(name).sum())

    def release_mean(self, which):
        """Mean release rate of R ('R') or L ('L') this step (sync).
        Statistical only (cupy reduction order), display quantity."""
        r = self.r_release if which == "R" else self.l_release
        return float(r.mean())


def gpu_simulate(circuit, cal, lum_inc_fn, seed, t_end_ms, hook=None,
                 on_record=None, chem_fn=None, mod_fn=None):
    """GPU twin of pipeline.simulate (mech branch).

    Consumes the seed exactly like simulate (build_stack draws the
    delay jitter first), then the OU stream in per-step pop order."""
    rng = np.random.default_rng(seed)
    st = build_stack(circuit, cal, rng)
    trial = GPUTrial(st)
    trial.run(lum_inc_fn, int(t_end_ms / st["pops"]["R"].dt), hook=hook,
              on_record=on_record, chem_fn=chem_fn, mod_fn=mod_fn)
    return st, trial
