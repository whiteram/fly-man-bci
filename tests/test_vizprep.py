"""Unit tests for the viz-prep pure helpers (the axis-slice NaN class of
bug must never survive a 12-minute run again)."""

import numpy as np

from ffbm.vizprep import (fit_scale_shift, generate_background_eeg,
                          occipital_shift, scalp_electrode_dirs,
                          snr_metrics)


def test_electrode_dirs_layout():
    d = scalp_electrode_dirs(17, np.array([0.3, -0.5, 0.81]
                                          ) / np.linalg.norm([0.3, -0.5, 0.81]))
    assert d.shape == (17, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    ang = np.degrees(np.arccos(np.clip(d @ d[0], -1, 1)))
    assert ang[0] == 0.0 and np.all(np.diff(ang) >= -1e-9)  # sorted
    dd = d @ d.T
    np.fill_diagonal(dd, -2)
    assert dd.max() < 0.99            # no coincident electrodes


def test_occipital_shift_constraints():
    rng = np.random.default_rng(0)
    u = np.array([0.0, 0.0, 1.0])
    # compact elongated blob, |z| <= ~42 mm, lateral <= ~16 mm
    pts = rng.normal(scale=8e3, size=(500, 3))
    pts[:, 2] -= pts[:, 2].max() + 2e3
    r_brain = 78e3
    s = occipital_shift(pts, u, r_brain)
    moved = pts + s
    assert np.all(np.linalg.norm(moved, axis=1) <= 0.98 * r_brain + 1e-6)
    # the pole point should be pushed out toward -u (T4/T5 side)
    assert -moved[:, 2].min() >= 0.5 * r_brain


def test_occipital_shift_zero_when_already_at_pole():
    u = np.array([0.0, 0.0, 1.0])
    pts = np.array([[0, 0, -70e3], [0, 0, -20e3]])
    s = occipital_shift(pts, u, 78e3, pole_frac=0.85, margin_frac=0.98)
    assert np.linalg.norm(s) == 0.0     # pole already beyond 0.85*r


def test_fit_scale_shift_small_cloud_keeps_nominal():
    rng = np.random.default_rng(1)
    u_occ = np.array([0.0, 0.0, 1.0])
    pts = rng.normal(scale=40.0, size=(400, 3))    # native um, small
    scale, shift, r_after = fit_scale_shift(pts, pts.mean(0), u_occ, 78e3)
    assert scale == 400.0
    assert r_after <= 0.98 * 78e3 + 1e-6
    # elongated along u_occ: the u_occ-side extreme is pushed to the wall
    moved = (pts - pts.mean(0)) * scale + shift
    assert abs(moved[:, 2].max() - 0.90 * 78e3) < 5e3


def test_fit_scale_shift_huge_cloud_scales_down():
    # wide V-ish cloud: two arms extending toward -x -> cannot fit x400
    rng = np.random.default_rng(2)
    u_occ = np.array([1.0, 0.0, 0.0])
    arm = rng.normal(scale=30.0, size=(300, 3))
    arm[:, 0] = -np.abs(arm[:, 0]) - 100.0        # extends toward -x
    arm2 = arm.copy()
    arm2[:, 1] += 250.0                            # second arm offset
    pts = np.vstack([arm, arm2])
    center = pts.mean(0)
    scale, shift, r_after = fit_scale_shift(pts, center, u_occ, 78e3)
    assert scale < 400.0
    assert r_after <= 0.98 * 78e3 + 1e-6
    assert scale > 50.0                             # sane magnitude


def test_background_eeg_components():
    dirs = np.eye(3)
    u_eye = np.array([0.0, 0.0, -1.0])   # occipital = +z
    bg = generate_background_eeg(dirs, u_eye, 5000,
                                 {"seed": 3, "sensor_uv": 0.0,
                                  "aperiodic_uv": 0.0})
    # electrode 2 sits on the occipital pole: alpha present
    assert bg[2].std() > 5.0
    # electrode on the equator (idx 0): weight ~ 0 -> near-silent
    assert bg[0].std() < 0.5


def test_snr_metrics_no_nans_and_band_consistency():
    rng = np.random.default_rng(1)
    t = np.arange(6000)
    # fly "signal": 8 Hz oscillation, 1 uV; background: 10 Hz at 8 uV
    clean = np.stack([0.5 * np.sin(2 * np.pi * 8 * t / 1000)] * 4)
    bg = np.stack([8 * np.sin(2 * np.pi * 10 * t / 1000)
                   + rng.normal(0, 1, 6000) for _ in range(4)])
    m = snr_metrics(clean, bg, 100, 5900)
    assert np.isfinite(m["k_for_dprime2"])
    assert np.isfinite(m["k_for_dprime2_band"])
    # the 8 Hz signal lives inside the 2-20 Hz band, noise mostly out:
    assert m["amp_ratio_band"] > m["amp_ratio"]
    assert 0.05 < m["amp_ratio"] < 0.2
