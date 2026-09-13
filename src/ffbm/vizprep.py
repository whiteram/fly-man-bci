"""Pure helpers for the viz data export (unit-testable, no I/O).

scalp electrode array layout, occipital network shift, background human
EEG generation, and SNR metrics (broadband + band-limited). Extracted
from viz/export_data.py after an axis-slice NaN bug survived a 12-minute
run -- every block here now has a test with dummy data.
"""

from __future__ import annotations

import numpy as np

BG_DEFAULTS = {"alpha_hz": 10.0, "alpha_amp_uv": 30.0,
               "aperiodic_uv": 3.0, "sensor_uv": 1.5,
               "envelope_period_ms": 700.0, "seed": 2026}


def scalp_electrode_dirs(n_elec: int, anchor: np.ndarray) -> np.ndarray:
    """Quasi-uniform scalp-electrode directions (unit vectors, rows sorted
    by angle from `anchor`; row 0 = the anchor itself).

    n_elec Fibonacci-sphere points, the one nearest the anchor dropped,
    anchor prepended -> n_elec total. The angle sort makes the channel
    heatmap read as a spatial gradient."""
    anchor = np.asarray(anchor, dtype=np.float64)
    i = np.arange(n_elec) + 0.5
    az = np.pi * (1.0 + 5.0 ** 0.5) * i
    z = 1.0 - 2.0 * i / n_elec
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fib = np.stack([r * np.cos(az), r * np.sin(az), z], axis=1)
    fib = np.delete(fib, int(np.argmax(fib @ anchor)), axis=0)
    dirs = np.vstack([anchor[None, :], fib])
    ang = np.degrees(np.arccos(np.clip(dirs @ anchor, -1, 1)))
    return dirs[np.argsort(ang, kind="stable")]


def occipital_shift(points: np.ndarray, u_eye: np.ndarray, r_brain: float,
                    pole_frac: float = 0.90,
                    margin_frac: float = 0.98) -> np.ndarray:
    """Max shift along -u_eye putting the farthest point (T4/T5 pole) at
    pole_frac*r_brain while keeping every point inside margin_frac*r_brain.

    points are already scaled, head-centered. Per-point constraint
    |p - d*u| <= r_safe expands to a quadratic in d; the positive root
    upper-bounds d for that point."""
    proj = points @ u_eye
    r_safe = margin_frac * r_brain
    # |p - d*u|^2 = d^2 - 2 d proj + |p|^2 <= r_safe^2. Discriminant
    # r_safe^2 - |p_perp|^2 clipped at 0: a point whose perpendicular
    # offset exceeds r_safe can never fit at any d (root -> proj <= 0
    # forces a zero shift through the final max(d, 0)).
    disc = np.maximum(r_safe ** 2
                      - (np.sum(points ** 2, axis=1) - proj ** 2), 0.0)
    root = proj + np.sqrt(disc)
    d = float(min(pole_frac * r_brain - (-proj.min()), root.min()))
    return -np.asarray(u_eye, dtype=np.float64) * max(d, 0.0)


def _one_over_f(n: int, rng) -> np.ndarray:
    x = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)
    f[0] = 1.0
    y = np.fft.irfft(np.fft.rfft(x) / f ** 0.5, n)
    return y / y.std()


def generate_background_eeg(dirs: np.ndarray, u_eye: np.ndarray,
                            n_t: int, params: dict | None = None
                            ) -> np.ndarray:
    """Per-electrode background human EEG (uV): eyes-closed alpha
    (posterior-dominant, waxing envelope), 1/f aperiodic activity, sensor
    white noise. dirs rows are unit electrode directions (row 0 on the
    eye axis); alpha weight = cos^2(angle to the occipital -u_eye)."""
    p = {**BG_DEFAULTS, **(params or {})}
    rng = np.random.default_rng(p["seed"])
    t = np.arange(n_t, dtype=np.float64)
    cos_occ = dirs @ (-np.asarray(u_eye, dtype=np.float64))
    w_occ = np.clip(cos_occ, 0.0, None) ** 2
    bg = np.zeros((len(dirs), n_t))
    for e in range(len(dirs)):
        env = 0.55 + 0.45 * np.sin(
            2 * np.pi * t / p["envelope_period_ms"] + 0.7 * e)
        alpha = p["alpha_amp_uv"] * w_occ[e] * env * np.sin(
            2 * np.pi * p["alpha_hz"] * t / 1000.0 + 0.15 * e)
        bg[e] = alpha + p["aperiodic_uv"] * _one_over_f(n_t, rng) \
            + rng.normal(0, p["sensor_uv"], n_t)
    return bg


def _band_var(x: np.ndarray, dt_ms: float, band_hz: tuple) -> float:
    f = np.fft.rfftfreq(len(x), d=dt_ms / 1000.0)
    X = np.fft.rfft(x - x.mean())
    m = (f >= band_hz[0]) & (f <= band_hz[1])
    return float(np.sum(np.abs(X[m]) ** 2) / len(x))


def snr_metrics(clean_uv: np.ndarray, bg_uv: np.ndarray, i0: int, i1: int,
                dt_ms: float = 1.0,
                band_hz: tuple = (2.0, 20.0)) -> dict:
    """SNR of the fly signal against the background, per channel.

    clean/bg: (n_elec, n_t) uV, time-major slices [i0, i1). Broadband:
    std ratio. Band-limited: variance within `band_hz` (the honest
    measure where the alpha peak overlaps the broadband signal band).
    Returns metrics at the best (broadband) channel."""
    clean_w = clean_uv[:, i0:i1]
    bg_w = bg_uv[:, i0:i1]
    sig_std = clean_w.std(axis=1)
    bg_std = bg_w.std(axis=1)
    best = int(np.argmax(sig_std))
    sig_b = np.array([_band_var(c, dt_ms, band_hz) for c in clean_w])
    bg_b = np.array([_band_var(b, dt_ms, band_hz) for b in bg_w])
    ratio = float(sig_std[best] / bg_std[best])
    ratio_band = float(np.sqrt(sig_b[best] / bg_b[best]))
    return {
        "best_elec": best,
        "sig_uv": round(float(sig_std[best]), 3),
        "bg_uv": round(float(bg_std[best]), 3),
        "amp_ratio": round(ratio, 4),
        "k_for_dprime2": int(np.ceil((2.0 / ratio) ** 2)),
        "band_hz": list(band_hz),
        "amp_ratio_band": round(ratio_band, 4),
        "k_for_dprime2_band": int(np.ceil((2.0 / ratio_band) ** 2)),
    }
