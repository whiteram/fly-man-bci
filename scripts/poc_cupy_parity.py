"""P3 PoC step 1: can CuPy RawKernels match the numba hot kernels BITWISE?

Small random data, fast iteration.  The numba kernels are the semantic
spec (they are themselves bitwise-identical to the numpy path), so each
case A/B's a CUDA kernel compiled with NVRTC -fmad=false against the
corresponding numba kernel on identical inputs:

  A. lif_g32      f32-g semi-implicit LIF  (L / MID drive path)
  B. lif_g64      f64-g semi-implicit LIF  (T45 / extra-region path)
  C. lif_none     explicit-Euler LIF       (g_is_none branch)
  D. ou_step      f64 OU noise update
  E. graded_step  release smoothing + y = w * s
  F. drive_postmajor  per-post sequential CSR accumulation -- the
     restructure the GPU port needs: numba scans edges in GLOBAL edge
     order accumulating into out[post]; a post-major CSR (stable) has
     each post's edges in ascending edge order, so per-thread serial
     sums see the identical add sequence.
  G. deliver_unique   exp-synapse scatter (unique targets: rows are
     disjoint edge ranges) -- gather-add-scatter is race-free.

Also records the -fmad=true failure for case D (documents why the flag
is mandatory).  Exit code 0 iff every bitwise check passes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import cupy as cp

import ffbm.simulation as sim

assert sim._HAVE_NUMBA, "this PoC needs the numba kernels as reference"

DEV = cp.cuda.Device(0)
rng = np.random.default_rng(20260916)

COMMON = ""
FAMD_OFF = ("-fmad=false",)


def mk(name, src, options=FAMD_OFF):
    return cp.RawKernel(COMMON + src, name, options=options)


LIF_SRC = r"""
extern "C" __global__
void lif_g32(double* v, double* refrac, const float* i_ext, const float* g,
             unsigned char* spike, int n,
             double dt, double tau_m, double v_rest, double v_th,
             double v_reset, double t_refrac, double R_m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = dt / tau_m;
    float a32 = (float)a;
    float rm32 = (float)R_m;
    if (refrac[i] > 0.0) v[i] = v_reset;
    double num = v[i] + a * (v_rest + R_m * (double)i_ext[i]);
    float den = 1.0f + a32 * (1.0f + rm32 * g[i]);
    v[i] = num / (double)den;
    spike[i] = v[i] >= v_th;
    if (spike[i]) v[i] = v_reset;
    double r = refrac[i] - dt;
    refrac[i] = r < 0.0 ? 0.0 : r;
    if (spike[i]) refrac[i] = t_refrac;
}

extern "C" __global__
void lif_g64(double* v, double* refrac, const double* i_ext, const double* g,
             unsigned char* spike, int n,
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

extern "C" __global__
void lif_none(double* v, double* refrac, const double* i_ext,
              unsigned char* spike, int n,
              double dt, double tau_m, double v_rest, double v_th,
              double v_reset, double t_refrac, double R_m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (refrac[i] > 0.0) v[i] = v_reset;
    v[i] = v[i] + dt * (-(v[i] - v_rest) + R_m * i_ext[i]) / tau_m;
    spike[i] = v[i] >= v_th;
    if (spike[i]) v[i] = v_reset;
    double r = refrac[i] - dt;
    refrac[i] = r < 0.0 ? 0.0 : r;
    if (spike[i]) refrac[i] = t_refrac;
}
"""

OU_SRC = r"""
extern "C" __global__
void ou_step(double* x, const double* w, double a, double k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = a * x[i] + k * w[i];
}
"""

GRADED_SRC = r"""
extern "C" __global__
void graded_step(const float* r_pre, const long long* pre_ids, float* s,
                 float* y, const float* weight, double k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // numba unifies r to float64 (the clip literals 0.0/1.0 are f64), so
    // (r - s) is an EXACT f64 subtraction -- not an f32 one
    double r = (double)r_pre[pre_ids[i]];
    if (r < 0.0) r = 0.0;
    else if (r > 1.0) r = 1.0;
    s[i] = (float)((double)s[i] + k * (r - (double)s[i]));
    y[i] = weight[i] * s[i];
}
"""

DRIVE_SRC = r"""
extern "C" __global__
void drive_postmajor(const float* y, float g_unit, const float* e_rev,
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
"""

DELIVER_SRC = r"""
extern "C" __global__
void decay(float* y, double decay, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (float)(y[i] * decay);
}

// unique targets only (disjoint pre-row edge ranges): race-free
extern "C" __global__
void deliver_unique(float* y, const float* kick, const long long* targets,
                     int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        long long t = targets[i];
        y[t] = y[t] + kick[t];
    }
}
"""


def go(kernel, n, args, block=256):
    """cupy RawKernel call: kernel(grid, block, args_tuple)."""
    kernel(((n + block - 1) // block,), (block,), tuple(args))


RESULTS = []


def check(name, ref, got, expect_fail=False):
    got_np = cp.asnumpy(got) if isinstance(got, cp.ndarray) else np.asarray(got)
    if ref.shape != got_np.shape or ref.dtype != got_np.dtype:
        ok = False
        nbad, relmax = -1, float("nan")
    else:
        diff = ref.view(np.uint8) != got_np.view(np.uint8)
        nbad = int(np.count_nonzero(diff))
        ok = nbad == 0
        with np.errstate(invalid="ignore"):
            relmax = float(np.max(np.abs(
                ref.astype(np.float64) - got_np.astype(np.float64))
                / np.maximum(np.abs(ref.astype(np.float64)), 1e-300))) \
                if ref.size else 0.0
    RESULTS.append((name, ok))
    tag = "PASS" if ok else ("SOFT-FAIL(expected)" if expect_fail else "FAIL")
    extra = "" if ok else \
        f"  [{nbad} differing bytes, max rel diff {relmax:.3e}]"
    print(f"[{tag}] {name}{extra}")
    return ok


def main():
    n = 2_000_000
    dt, tau_m = 5e-4, 10.0
    v_rest, v_th, v_reset, t_refrac, R_m = -70.0, -50.0, -70.0, 2.0, 0.1

    lif_k = mk("lif_g32", LIF_SRC)
    k_g64 = mk("lif_g64", LIF_SRC)
    k_none = mk("lif_none", LIF_SRC)
    ou_k = mk("ou_step", OU_SRC)
    ou_k_fmad = mk("ou_step", OU_SRC, options=())       # default: fmad ON
    graded_k = mk("graded_step", GRADED_SRC)
    drive_k = mk("drive_postmajor", DRIVE_SRC)
    decay_k = mk("decay", DELIVER_SRC)
    deliver_k = mk("deliver_unique", DELIVER_SRC)

    def lif_args(v, refrac, spike, **kw):
        return dict(v=v, refrac=refrac, spike=spike, dt=dt, tau_m=tau_m,
                    v_rest=v_rest, v_th=v_th, v_reset=v_reset,
                    t_refrac=t_refrac, R_m=R_m, **kw)

    # ---- A. lif_g32 --------------------------------------------------
    for seed in (1, 2):
        r = np.random.default_rng(seed)
        i_ext = (r.standard_normal(n) * 200).astype(np.float32)     # pA
        g = (r.random(n) * 2.0).astype(np.float32)                  # nS
        v1 = r.standard_normal(n) * 5 - 70.0
        v2 = v1.copy()
        refr1 = r.random(n) * t_refrac
        refr2 = refr1.copy()
        sp1 = np.empty(n, np.bool_)
        sp2 = cp.zeros(n, cp.uint8)
        sim._lif_step_g32(v1, refr1, i_ext, g, dt, tau_m, v_rest, v_th,
                          v_reset, t_refrac, R_m, sp1)
        refr2_d = cp.asarray(refr2)
        args = (cp.asarray(v2), refr2_d, cp.asarray(i_ext),
                cp.asarray(g), sp2, np.int32(n),
                *(np.float64(x) for x in (dt, tau_m, v_rest, v_th, v_reset,
                                          t_refrac, R_m)))
        go(lif_k, n, args)
        check(f"A lif_g32 seed{seed} v", v1, args[0])
        check(f"A lif_g32 seed{seed} refrac", refr1, refr2_d)
        check(f"A lif_g32 seed{seed} spike", sp1.view(np.uint8), sp2)

    # ---- B. lif_g64 --------------------------------------------------
    r = np.random.default_rng(3)
    i_ext = r.standard_normal(n) * 200
    g = r.random(n) * 2.0
    v1 = r.standard_normal(n) * 5 - 70.0
    v2 = v1.copy()
    refr1 = r.random(n) * t_refrac
    refr2 = refr1.copy()
    sp1 = np.empty(n, np.bool_)
    sp2 = cp.zeros(n, cp.uint8)
    sim._lif_step(v1, refr1, i_ext, g, False, dt, tau_m, v_rest, v_th,
                  v_reset, t_refrac, R_m, sp1)
    args = (cp.asarray(v2), cp.asarray(refr2), cp.asarray(i_ext),
            cp.asarray(g), sp2, np.int32(n),
            *(np.float64(x) for x in (dt, tau_m, v_rest, v_th, v_reset,
                                      t_refrac, R_m)))
    go(k_g64, n, args)
    check("B lif_g64 v", v1, args[0])
    check("B lif_g64 spike", sp1.view(np.uint8), sp2)

    # ---- C. lif_none -------------------------------------------------
    v1 = np.full(n, v_rest)
    v1[::7] += 25.0                    # force some spikes
    v2 = v1.copy()
    refr1 = np.zeros(n)
    refr2 = refr1.copy()
    sp1 = np.empty(n, np.bool_)
    sp2 = cp.zeros(n, cp.uint8)
    sim._lif_step(v1, refr1, i_ext, refr1, True, dt, tau_m, v_rest, v_th,
                  v_reset, t_refrac, R_m, sp1)
    args = (cp.asarray(v2), cp.asarray(refr2), cp.asarray(i_ext), sp2,
            np.int32(n),
            *(np.float64(x) for x in (dt, tau_m, v_rest, v_th, v_reset,
                                      t_refrac, R_m)))
    go(k_none, n, args)
    check("C lif_none v", v1, args[0])
    check("C lif_none spike", sp1.view(np.uint8), sp2)

    # ---- D. ou_step (f64, fma-sensitive) ------------------------------
    a, k = float(np.exp(-dt / 8.0)), 60.0 * float(np.sqrt(1 - np.exp(-2 * dt / 8.0)))
    x1 = np.asarray(rng.standard_normal(n))
    w = np.asarray(rng.standard_normal(n))
    x2 = cp.asarray(x1.copy())
    sim._ou_step(x1, w, a, k)
    go(ou_k, n, (x2, cp.asarray(w), np.float64(a), np.float64(k),
                 np.int32(n)))
    check("D ou_step -fmad=false", x1, x2)
    x3 = cp.asarray(x1.copy())
    go(ou_k_fmad, n, (x3, cp.asarray(w), np.float64(a), np.float64(k),
                     np.int32(n)))
    check("D ou_step default-fmad (diagnostic: does nvrtc contract?)",
          x1, x3, expect_fail=True)

    # ---- E. graded_step ----------------------------------------------
    m = 1_500_000
    n_pre = 12_000
    pre_ids = rng.integers(0, n_pre, m).astype(np.int64)
    r_pre = rng.random(n_pre).astype(np.float32) * 1.3 - 0.15   # spills clip
    weight = (rng.random(m).astype(np.float32) - 0.5)
    s1 = rng.random(m).astype(np.float32)
    s2 = cp.asarray(s1.copy())
    y1 = np.zeros(m, np.float32)
    y2 = cp.zeros(m, cp.float32)
    kk = dt / 5.0
    sim._graded_step(r_pre, pre_ids, s1, y1, weight, kk)
    go(graded_k, m, (cp.asarray(r_pre), cp.asarray(pre_ids), s2, y2,
                   cp.asarray(weight), np.float64(kk), np.int32(m)))
    check("E graded s", s1, s2)
    check("E graded y", y1, y2)

    # ---- F. drive: post-major CSR vs global-edge-order numba ---------
    n_post = 90_000
    n_edges = 1_800_000
    post_local = rng.integers(0, n_post, n_edges).astype(np.int64)
    y = (rng.random(n_edges).astype(np.float32) - 0.5)
    e_rev = (rng.random(n_edges).astype(np.float32) - 0.5) * 80
    g_unit = 0.02
    oi1 = np.zeros(n_post, np.float32)
    og1 = np.zeros(n_post, np.float32)
    sim._drive_cond(y, g_unit, e_rev, post_local, oi1, og1)
    # post-major CSR with per-post ascending edge order == numba scan order
    order = np.argsort(post_local, kind="stable")
    counts = np.bincount(post_local, minlength=n_post)
    indptr = np.zeros(n_post + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    oi2 = cp.zeros(n_post, cp.float32)
    og2 = cp.zeros(n_post, cp.float32)
    go(drive_k, n_post, (cp.asarray(y), np.float32(g_unit),
                     cp.asarray(e_rev), cp.asarray(order.astype(np.int64)),
                     cp.asarray(indptr), oi2, og2, np.int32(n_post)))
    check("F drive out_i (post-major CSR)", oi1, oi2)
    check("F drive out_g (post-major CSR)", og1, og2)

    # ---- G. exp decay + unique-target delivery ------------------------
    q = 1_000_000
    y1 = (rng.random(q).astype(np.float32) - 0.5) * 1e-3
    y2 = cp.asarray(y1.copy())
    kick = (rng.random(q).astype(np.float32) - 0.5) * 1e-2
    dcy = float(np.exp(-dt / 5.0))
    # unique targets: sample without replacement
    tgt = rng.choice(q, size=q // 3, replace=False).astype(np.int64)
    t_t = np.empty(len(tgt) * 2, np.int64)          # numba two-pass form
    t_v = np.empty(len(tgt) * 2, np.float32)
    sim._exp_step(y1, dcy, np.array([0]), np.array([0, len(tgt)]),
                  np.array([len(tgt)]), tgt, kick, t_t, t_v)
    # numba's _exp_step with sel=[0], spans=[0,len) -> delivers tgt
    go(decay_k, q, (y2, np.float64(dcy), np.int32(q)))
    go(deliver_k, len(tgt), (y2, cp.asarray(kick), cp.asarray(tgt),
                             np.int32(len(tgt))))
    check("G exp decay+deliver y", y1, y2)

    print()
    hard = [(nm, ok) for nm, ok in RESULTS if "diagnostic" not in nm]
    n_ok = sum(ok for _, ok in hard)
    print(f"{n_ok}/{len(hard)} bitwise checks passed")
    fails = [nm for nm, ok in hard if not ok]
    if fails:
        print("FAILURES:", fails)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
