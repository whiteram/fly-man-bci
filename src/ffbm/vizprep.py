"""Pure helpers for the viz data export (unit-testable, no I/O).

scalp electrode array layout, occipital network shift, background human
EEG generation, and SNR metrics (broadband + band-limited). Extracted
from viz/export_data.py after an axis-slice NaN bug survived a 12-minute
run -- every block here now has a test with dummy data.
"""

from __future__ import annotations

import numpy as np

from .params import bg_defaults as _params_bg
from .params import elec_defaults as _params_elec

BG_DEFAULTS = _params_bg()
ELEC_DEFAULTS = _params_elec()


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


def _sph_rotate(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation of unit vector v around axis by deg."""
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return (v * c + np.cross(axis, v) * s
            + axis * np.dot(axis, v) * (1.0 - c))


def _sph_mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint direction of the great-circle arc a -> b."""
    return a + b


def standard_1020(anchors: dict, system: str = "1020",
                  ni_arc_deg: float = 180.0,
                  fpz_arc_deg: float | None = None) -> dict:
    """Standard 10-20 / 10-10 electrode directions from hand-placed
    anchors (Cz, Fz, Oz, A1, A2 unit vectors, any consistent head frame).

    Regularization (10-20 construction, Jasper 1958):
      - the sagittal great circle is fit through Cz with the anterior
        tangent from Fz - Oz; the midline electrodes are placed at the
        10-20 PERCENTAGES of the nasion-inion arc: Fz at 30%, Cz at 50%,
        Pz at 70%, Oz at 90%, Fpz at 10%; ni_arc_deg is that arc in
        degrees -- 180 is the spherical idealization (nasion/inion on
        the ear-line plane); a real head/ghost-head mesh measures
        ~250-260 deg (both landmarks below the ear-line plane), which
        stretches the midline so Oz lands ON the occiput as seen;
        fpz_arc_deg optionally re-pins Fpz directly (angle from Cz,
        overriding the 10% rule) for visual calibration against the
        head mesh -- the whole frontal chain (Fp1/2, F7/8, F3/4, AFz,
        AF3/4, F1/2, FC5/6) is rebuilt from the moved Fpz;
      - the coronal great circle through Cz (perpendicular to the
        sagittal one) carries C3/C4 at 20% of the LPA-RPA arc on either
        side of Cz (=36 deg at ni_arc=180) and T7/T8 at 40% (=72 deg);
      - A1/A2 keep the USER'S elevation (earlobes sit below the
        N-I plane), mirrored exactly about the sagittal plane;
      - the lateral chains are built by the arc rules scaled to the
        same ni_arc: Fp1 at 10% of ni_arc from Fpz toward T7, O1 at
        10% from Oz toward T7, F7/P7 as arc midpoints (Fp1-T7 /
        O1-T7), F3/P3 as midpoints of Fz-F7 / Pz-P7 (mirrored right);
      - system="1010" adds the 10-10 midpoint subdivisions (FCz, CPz,
        F1/F2, C1/C2, P1/P2, FC3-6, CP3-6, AF3/AF4/AFz, PO3/PO4/POz).

    Returns dict name -> unit vector in the SAME frame as the anchors.
    """
    get = {k.upper(): np.asarray(v, dtype=np.float64)
           for k, v in anchors.items()}
    for k in ("CZ", "FZ", "OZ", "A1", "A2"):
        if k not in get:
            raise ValueError(f"missing anchor {k}")
    cz = get["CZ"] / np.linalg.norm(get["CZ"])
    a_raw = get["FZ"] - get["OZ"]
    a_raw /= np.linalg.norm(a_raw)
    e_ant = a_raw - (a_raw @ cz) * cz
    e_ant /= np.linalg.norm(e_ant)          # anterior tangent at Cz

    # left axis: normal of the sagittal plane, signed toward A1
    L = np.cross(cz, e_ant)
    L /= np.linalg.norm(L)
    if L @ get["A1"] < L @ get["A2"]:
        L = -L
    # sagittal rotations around +L move Cz toward the ANTERIOR?
    probe = _sph_rotate(cz, L, 10.0)
    L_sag = L if probe @ e_ant > 0 else -L
    # coronal rotations around the anterior axis toward A2?
    probe = _sph_rotate(cz, e_ant, 10.0)
    L_cor = e_ant if probe @ get["A2"] > probe @ get["A1"] else -e_ant

    sag = lambda phi: _sph_rotate(cz, L_sag, phi)    # +phi = anterior
    cor = lambda phi: _sph_rotate(cz, L_cor, phi)    # +phi = A2 side
    # midline arc from Cz; percentages of ni_arc_deg. +phi = anterior.
    # Cz is the 50% mark; the nasion sits at +ni_arc/2, the inion at
    # -ni_arc/2 (ni_arc > 180 when both landmarks sit below the
    # ear-line plane, as on a real head / the ghost-head mesh)
    ni = ni_arc_deg
    out = {
        "Cz": cz,
        "Fz": sag(0.20 * ni), "Pz": sag(-0.20 * ni),
        "Fpz": sag(fpz_arc_deg if fpz_arc_deg is not None
                   else 0.40 * ni),
        "Oz": sag(-0.40 * ni),
        "Nasion": sag(0.50 * ni), "Inion": sag(-0.50 * ni),
        "C3": cor(-36.0), "C4": cor(36.0),
        "T7": cor(-72.0), "T8": cor(72.0),
    }
    # A1/A2: keep the USER'S elevation (real earlobes sit below the
    # nasion-inion plane -- snapping them onto the coronal great circle
    # at 90 deg was over-correcting) but enforce exact mirror symmetry
    # about the sagittal plane: the left-right axis IS the sagittal
    # normal, so A1/A2 = Cz rotated toward +/-n_sag by the mean
    # Cz-ear angle
    n_sag = np.cross(cz, e_ant)
    n_sag /= np.linalg.norm(n_sag)
    L_s = n_sag if n_sag @ get["A1"] > n_sag @ get["A2"] else -n_sag
    th = 0.5 * (np.arccos(np.clip(get["A1"] @ cz, -1, 1))
                + np.arccos(np.clip(get["A2"] @ cz, -1, 1)))
    out["A1"] = cz * np.cos(th) + L_s * np.sin(th)
    out["A2"] = cz * np.cos(th) - L_s * np.sin(th)
    # lateral chains: Fp1 at 10% of ni_arc from Fpz toward T7, O1 at
    # 10% from Oz toward T7, then F7/P7 as arc midpoints, F3/P3 as
    # midpoints of the Fz/Pz spokes
    fp_off = 0.10 * ni

    def slerp_deg(a, b, deg):
        total = np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
        t = deg / max(total, 1e-9)
        return (a * np.sin((1 - t) * np.radians(total))
                + b * np.sin(t * np.radians(total))) \
            / np.sin(np.radians(total))

    def mid(a, b):
        m = a + b
        return m / np.linalg.norm(m)

    fp1 = slerp_deg(out["Fpz"], out["T7"], fp_off)
    fp2 = slerp_deg(out["Fpz"], out["T8"], fp_off)
    o1 = slerp_deg(out["Oz"], out["T7"], fp_off)
    o2 = slerp_deg(out["Oz"], out["T8"], fp_off)
    out.update({
        "Fp1": fp1, "Fp2": fp2, "O1": o1, "O2": o2,
        "F7": mid(fp1, out["T7"]), "F8": mid(fp2, out["T8"]),
        "P7": mid(o1, out["T7"]), "P8": mid(o2, out["T8"]),
        "F3": mid(out["Fz"], mid(fp1, out["T7"])),
        "F4": mid(out["Fz"], mid(fp2, out["T8"])),
        "P3": mid(out["Pz"], mid(o1, out["T7"])),
        "P4": mid(out["Pz"], mid(o2, out["T8"])),
    })
    if system == "1010":
        out.update({
            "FCz": mid(out["Fz"], out["Cz"]),
            "CPz": mid(out["Cz"], out["Pz"]),
            "AFz": mid(out["Fpz"], out["Fz"]),
            "POz": mid(out["Pz"], out["Oz"]),
            "F1": mid(out["Fz"], out["F3"]),
            "F2": mid(out["Fz"], out["F4"]),
            "C1": mid(out["Cz"], out["C3"]),
            "C2": mid(out["Cz"], out["C4"]),
            "P1": mid(out["Pz"], out["P3"]),
            "P2": mid(out["Pz"], out["P4"]),
            "FC3": mid(out["F3"], out["C3"]),
            "FC4": mid(out["F4"], out["C4"]),
            "CP3": mid(out["C3"], out["P3"]),
            "CP4": mid(out["C4"], out["P4"]),
            "FC5": mid(out["F7"], out["C3"]),
            "FC6": mid(out["F8"], out["C4"]),
            "CP5": mid(out["C3"], out["P7"]),
            "CP6": mid(out["C4"], out["P8"]),
            "AF3": mid(out["Fpz"], out["F3"]),
            "AF4": mid(out["Fpz"], out["F4"]),
            "PO3": mid(out["P3"], out["O1"]),
            "PO4": mid(out["P4"], out["O2"]),
        })
    return {k: v / np.linalg.norm(v) for k, v in out.items()}


def regularize_report(anchors: dict) -> dict:
    """Diagnostics for hand-placed anchors: raw arc distances and the
    corrections standard_1020 applies (degrees moved per anchor)."""
    std = standard_1020(anchors)
    get = {k.upper(): np.asarray(v, dtype=np.float64)
           for k, v in anchors.items()}
    for k in get:
        get[k] = get[k] / np.linalg.norm(get[k])
    def ang(a, b):
        return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
    report = {"raw_arcs_deg": {
        "Fz_Cz": round(ang(get["FZ"], get["CZ"]), 1),
        "Cz_Oz": round(ang(get["CZ"], get["OZ"]), 1),
        "A1_Cz": round(ang(get["A1"], get["CZ"]), 1),
        "Cz_A2": round(ang(get["CZ"], get["A2"]), 1),
        "A1_A2": round(ang(get["A1"], get["A2"]), 1)},
        "corrections_deg": {}}
    for name, ukey in (("Cz", "CZ"), ("Fz", "FZ"), ("Oz", "OZ"),
                       ("A1", "A1"), ("A2", "A2")):
        report["corrections_deg"][name] = round(
            ang(get[ukey], std[name]), 1)
    return report


def scalp_electrode_dirs_capped(n_elec: int, face_axis: np.ndarray,
                                neck_axis: np.ndarray | None = None,
                                face_excl_deg: float | None = None,
                                neck_excl_deg: float | None = None
                                ) -> np.ndarray:
    """Quasi-uniform scalp electrode directions on the CAP region only.

    Candidates come from a long Fibonacci stream; directions inside the
    FACE cone (angle to face_axis < face_excl_deg) or the NECK cone
    (angle to neck_axis < neck_excl_deg) are rejected -- nothing lands
    on the eyes/mouth or below the ears. Rows sorted by angle from the
    face axis (row 0 = most frontal allowed electrode). Defaults come
    from the params registry (head_model.elec_*_excl_deg)."""
    p = ELEC_DEFAULTS
    if face_excl_deg is None:
        face_excl_deg = p["face_excl_deg"]
    if neck_excl_deg is None:
        neck_excl_deg = p["neck_excl_deg"]
    face = np.asarray(face_axis, dtype=np.float64)
    neck = (None if neck_axis is None
            else np.asarray(neck_axis, dtype=np.float64))
    n_cand = max(600, 60 * n_elec)
    i = np.arange(n_cand) + 0.5
    az = np.pi * (1.0 + 5.0 ** 0.5) * i
    z = 1.0 - 2.0 * i / n_cand
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    cand = np.stack([r * np.cos(az), r * np.sin(az), z], axis=1)
    ok = np.degrees(np.arccos(np.clip(cand @ face, -1, 1))) >= face_excl_deg
    if neck is not None:
        ok &= (np.degrees(np.arccos(np.clip(cand @ neck, -1, 1)))
               >= neck_excl_deg)
    dirs = cand[ok][:n_elec]
    if len(dirs) < n_elec:
        raise ValueError("exclusion cones too wide for the electrode count")
    ang = np.degrees(np.arccos(np.clip(dirs @ face, -1, 1)))
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


def fit_scale_shift(points_native: np.ndarray, center: np.ndarray,
                    u_occ: np.ndarray, r_brain: float, nominal: float = 400.0,
                    pole_frac: float = 0.90,
                    margin_frac: float = 0.98) -> tuple:
    """Largest scale <= nominal whose occipitally-shifted cloud fits.

    exp015: the BOTH-lobe cloud in native geometry does not fit at x400
    (692 um span -> 277 mm vs the 156 mm brain diameter), so the
    thought-experiment scale auto-fits. u_occ points from the cloud
    centroid toward the pole end (the T4/T5 side); occipital_shift does
    the placement -- a pole-anchored shift for elongated clouds (single
    lobe: the T4/T5 extreme lands at pole_frac*r_brain) and a zero/
    centered shift for clouds wider than long along u_occ (bilateral V:
    the long inter-eye axis rides a diameter through the head center).
    Bisection maximizes the scale under the margin_frac*r_brain bound.
    Returns (scale, shift, r_after_um); scale == nominal whenever it
    fits (single-lobe case)."""
    rel = np.asarray(points_native, dtype=np.float64) - center
    u_occ = np.asarray(u_occ, dtype=np.float64)
    r_safe = margin_frac * r_brain

    def place(s):
        scaled = rel * s
        shift = occipital_shift(scaled, -u_occ, r_brain, pole_frac,
                                margin_frac)
        return shift, float(np.linalg.norm(scaled + shift, axis=1).max())

    shift, r_after = place(nominal)
    if r_after <= r_safe + 1e-6:
        return nominal, shift, r_after
    lo, hi = 1.0, nominal      # lo assumed feasible (tiny clouds)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        _, r = place(mid)
        if r <= r_safe + 1e-6:
            lo = mid
        else:
            hi = mid
    shift, r_after = place(lo)
    return lo, shift, r_after


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
