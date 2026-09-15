"""Unit tests for the viz-prep pure helpers (the axis-slice NaN class of
bug must never survive a 12-minute run again)."""

import numpy as np

from ffbm.vizprep import (ELEC_DEFAULTS, fit_scale_shift,
                          generate_background_eeg, occipital_shift,
                          scalp_electrode_dirs, scalp_electrode_dirs_capped,
                          snr_metrics, standard_1020)


def test_standard_1020_from_canonical_anchors():
    # canonical anchors: head frame with up = +y, face = +z
    # Cz at the pole would break the sag-circle fit, so tilt the frame:
    # place Cz at (0, 1, 0) rotated 30 deg toward +z (anterior = +z)
    cz = np.array([0.0, np.cos(np.radians(30.0)), np.sin(np.radians(30.0))])
    ant = np.array([0.0, np.sin(np.radians(30.0)), -np.cos(np.radians(30.0))])
    left = np.cross(cz, ant)
    layout = standard_1020({
        "CZ": cz,
        "FZ": cz * np.cos(np.radians(36.0))
              + ant * np.sin(np.radians(36.0)),
        "OZ": cz * np.cos(np.radians(36.0))
              - ant * np.sin(np.radians(36.0)),
        "A1": np.cos(np.radians(90.0)) * cz - left * np.sin(np.radians(90.0)),
        "A2": np.cos(np.radians(90.0)) * cz + left * np.sin(np.radians(90.0)),
    }, system="1010")
    deg = lambda a, b: np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
    assert abs(deg(layout["Fz"], layout["Cz"]) - 36.0) < 1e-6
    assert abs(deg(layout["Cz"], layout["Pz"]) - 36.0) < 1e-6
    assert abs(deg(layout["Pz"], layout["Oz"]) - 36.0) < 1e-6
    assert abs(deg(layout["Cz"], layout["A1"]) - 90.0) < 1e-6
    assert abs(deg(layout["Cz"], layout["A2"]) - 90.0) < 1e-6
    # nasion is 90 deg from Cz toward the face: positive on the
    # anterior axis of the test frame
    assert layout["Nasion"] @ ant > 0.9
    # mirror symmetry: C3/C4 equidistant from Cz and mirrored about the
    # sagittal plane (spanned by Cz/ant -- check via equal arc + planarity)
    assert abs(deg(layout["C3"], layout["Cz"])
               - deg(layout["C4"], layout["Cz"])) < 1e-6
    sag_normal = np.cross(layout["Cz"], ant)
    assert abs(layout["C3"] @ sag_normal + layout["C4"] @ sag_normal) < 1e-9
    # 10-10 extras present
    for k in ("FCz", "CPz", "F1", "C1", "P1", "FC3", "CP3", "FC5", "CP5"):
        assert k in layout


def test_standard_1020_fpz_override_rebuilds_chain():
    # fpz_arc_deg re-pins Fpz only; the frontal chain (Fp1, AFz, ...)
    # must follow the moved Fpz, not stay at the rule-based spot
    cz = np.array([0.0, np.cos(np.radians(30.0)), np.sin(np.radians(30.0))])
    ant = np.array([0.0, np.sin(np.radians(30.0)), -np.cos(np.radians(30.0))])
    left = np.cross(cz, ant)
    anchors = {
        "CZ": cz,
        "FZ": cz * np.cos(np.radians(36.0))
              + ant * np.sin(np.radians(36.0)),
        "OZ": cz * np.cos(np.radians(36.0))
              - ant * np.sin(np.radians(36.0)),
        "A1": -left, "A2": left,
    }
    base = standard_1020(anchors, system="1010", ni_arc_deg=260.0)
    lifted = standard_1020(anchors, system="1010", ni_arc_deg=260.0,
                           fpz_arc_deg=94.0)
    deg = lambda a, b: np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
    # Fpz sits exactly at the requested arc; unchanged electrodes stay
    assert abs(deg(lifted["Fpz"], cz) - 94.0) < 1e-6
    assert abs(deg(base["Oz"], lifted["Oz"]) - 0.0) < 1e-12
    assert abs(deg(base["Cz"], lifted["Cz"]) - 0.0) < 1e-12
    # the whole frontal chain moved WITH Fpz (still anchored to it)
    assert deg(base["Fp1"], lifted["Fp1"]) > 5.0
    assert abs(deg(lifted["Fpz"], lifted["Fp1"])
               - deg(base["Fpz"], base["Fp1"])) < 1e-6
    # AFz remains the arc midpoint of Fpz-Fz
    afz_mid = lifted["Fpz"] + lifted["Fz"]
    afz_mid /= np.linalg.norm(afz_mid)
    assert deg(afz_mid, lifted["AFz"]) < 1e-6


def test_standard_1020_transverse_ring_scales_with_ear_elevation():
    # the coronal chain follows the measured Cz-ear elevation th:
    # C3/C4 at 0.4*th, T7/T8 at 0.8*th (th=90 deg -> classical 36/72)
    cz = np.array([0.0, np.cos(np.radians(30.0)), np.sin(np.radians(30.0))])
    ant = np.array([0.0, np.sin(np.radians(30.0)), -np.cos(np.radians(30.0))])
    left = np.cross(cz, ant)
    anchors = {
        "CZ": cz,
        "FZ": cz * np.cos(np.radians(48.46))
              + ant * np.sin(np.radians(48.46)),
        "OZ": cz * np.cos(np.radians(48.46))
              - ant * np.sin(np.radians(48.46)),
        "A1": cz * np.cos(np.radians(130.8)) + left * np.sin(np.radians(130.8)),
        "A2": cz * np.cos(np.radians(130.8)) - left * np.sin(np.radians(130.8)),
    }
    lay = standard_1020(anchors, system="1010", ni_arc_deg=242.3)
    deg = lambda a, b: np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
    assert abs(deg(lay["C3"], cz) - 0.40 * 130.8) < 1e-6
    assert abs(deg(lay["C4"], cz) - 0.40 * 130.8) < 1e-6
    assert abs(deg(lay["T7"], cz) - 0.80 * 130.8) < 1e-6
    assert abs(deg(lay["T8"], cz) - 0.80 * 130.8) < 1e-6
    assert abs(deg(lay["A1"], cz) - 130.8) < 1e-6
    # and the idealized case reproduces the classical fixed placements
    anchors90 = dict(anchors)
    anchors90["A1"] = -left
    anchors90["A2"] = left
    lay90 = standard_1020(anchors90, system="1010", ni_arc_deg=180.0)
    assert abs(deg(lay90["C3"], cz) - 36.0) < 1e-6
    assert abs(deg(lay90["T7"], cz) - 72.0) < 1e-6


def test_standard_1020_yaw_rotates_whole_cap_about_cz():
    cz = np.array([0.0, np.cos(np.radians(30.0)), np.sin(np.radians(30.0))])
    ant = np.array([0.0, np.sin(np.radians(30.0)), -np.cos(np.radians(30.0))])
    left = np.cross(cz, ant)
    anchors = {
        "CZ": cz,
        "FZ": cz * np.cos(np.radians(48.46))
              + ant * np.sin(np.radians(48.46)),
        "OZ": cz * np.cos(np.radians(48.46))
              - ant * np.sin(np.radians(48.46)),
        "A1": cz * np.cos(np.radians(130.8)) + left * np.sin(np.radians(130.8)),
        "A2": cz * np.cos(np.radians(130.8)) - left * np.sin(np.radians(130.8)),
    }
    base = standard_1020(anchors, system="1010", ni_arc_deg=242.3)
    yawed = standard_1020(anchors, system="1010", ni_arc_deg=242.3,
                          yaw_deg=5.0)
    deg = lambda a, b: np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
    # every electrode equals an independent Rodrigues rotation of the
    # base placement about Cz (note: the DISPLACEMENT angle is < 5 deg
    # away from the equator -- the azimuthal rotation is exactly 5 deg)
    def rodr(v, th):
        c, s = np.cos(th), np.sin(th)
        return v * c + np.cross(cz, v) * s + cz * (cz @ v) * (1 - c)
    for k in ("Fpz", "Fz", "Oz", "A1", "A2", "T7", "C3"):
        assert np.linalg.norm(
            yawed[k] - rodr(base[k], np.radians(5.0))) < 1e-9
    # arcs to Cz are invariant (Cz sits on the rotation axis)
    for k in ("Fpz", "Oz", "A1", "T7"):
        assert abs(deg(yawed[k], yawed["Cz"])
                   - deg(base[k], base["Cz"])) < 1e-9


def test_electrode_dirs_capped_excludes_face_and_neck():
    from ffbm.vizprep import ELEC_DEFAULTS
    face = np.array([0.0, 0.0, 1.0])
    neck = np.array([0.0, -1.0, 0.0])
    d = scalp_electrode_dirs_capped(17, face, neck)
    assert d.shape == (17, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    ang_face = np.degrees(np.arccos(np.clip(d @ face, -1, 1)))
    ang_neck = np.degrees(np.arccos(np.clip(d @ neck, -1, 1)))
    assert ang_face.min() >= ELEC_DEFAULTS["face_excl_deg"] - 1e-9
    assert ang_neck.min() >= ELEC_DEFAULTS["neck_excl_deg"] - 1e-9
    assert np.all(np.diff(ang_face) >= -1e-9)     # sorted frontal -> back
    dd = d @ d.T
    np.fill_diagonal(dd, -2)
    assert dd.max() < 0.99


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
