"""2026-10-wk2 Apollo 13: the Moon pass and the way home, by vis-viva.

1. Moon leg. Twenty Moon-relative (altitude, speed) calls from the
   NASA Apollo Flight Journal, 75:06 GET to the PC+2 ignition, are
   fitted with ONE Moon-centred hyperbola (one v_inf). The fit also
   decides that the calls are altitudes, not centre distances.
   Vis-viva then predicts the speed at the real pericynthion
   (Mission Report: 136.2 nmi, 8,053.4 ft/s), and eq. 09 gives e
   and the turn angle.
2. The hyperbola is flown with the integrator (DOP853, Moon only)
   for slide 1's animation; the calls' clock times are checked
   against it.
3. Earth leg. Eighteen Earth-relative calls after the last course
   correction fix the return conic's energy; vis-viva gives the
   speed at the entry interface (400,000 ft), compared with the
   Mission Report (36,210.9 ft/s) and with escape speed there.

Writes outputs/2026-10-wk2/: moon_pass.csv, moon_readings.csv,
earth_readings.csv, visviva_curves.csv, free_return.npz and the
free_return.json sidecar (the contract with the visualizer).

    python examples/2026-10-wk2_free_return.py [--check]

--check asserts the numbers quoted in
posts/2026-10-wk2-apollo-13/code-results.md (exit 1 on a mismatch).
Not in scope: the figure-eight / three-body picture (week 3).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from orbital_sim import apollo13_data as D
from orbital_sim import fly, no_thrust
from orbital_sim.apollo13 import (burn_angle, earth_energy_change,
                                  entry_speed, fit_energy, fly_pass,
                                  get_seconds, sample_pass,
                                  speed_misfit, time_at_radius,
                                  to_radius_speed, trailing_offset)
from orbital_sim.constants import (FT, H_ENTRY, MU_EARTH, MU_MOON,
                                   NMI, R_EARTH_EQ, R_MOON)
from orbital_sim.dynamics import specific_energy
from orbital_sim.flyby import turn_angle
from orbital_sim.visviva import (escape_speed, hyperbola_speed,
                                 v_inf_squared)

OUT = Path(__file__).resolve().parents[1] / "outputs" / "2026-10-wk2"
DT = 20.0                    # s, animation time step
SPAN = 3.0 * 3600.0          # s, integrate +-3 h about pericynthion
MPH = 0.44704                # m/s per mph (exact)
KMH = 1 / 3.6                # m/s per km/h

# Values quoted in code-results.md, checked by --check: (value, abs tol)
EXPECTED = {
    "moon.n_calls": (20, 0),
    "moon.v_inf_kms": (1.046, 5e-4),
    "moon.a_km": (-4482, 1.0),
    "moon.misfit_max_pct": (0.21, 0.005),
    "moon.misfit_rms_pct": (0.08, 0.005),
    "moon.centre_reading_negative_vinf2": (5, 0),
    "moon.peri_mr.r_km": (1989.6, 0.05),
    "moon.peri_mr.v_model_fps": (8051.3, 0.05),
    "moon.peri_mr.diff_pct": (-0.026, 5e-4),
    "moon.peri_mr.v_model_kms": (2.454, 5e-4),
    "moon.peri_mr.v_model_mph": (5490, 5),
    "moon.peri_mr.e": (1.444, 5e-4),
    "moon.peri_mr.delta_deg": (87.66, 0.005),
    "moon.peri_mr.v_esc_kms": (2.220, 5e-4),
    "moon.peri_afj.v_model_fps": (8048.9, 0.05),
    "moon.peri_afj.delta_deg": (87.64, 0.005),
    "moon.film_line.v_model_fps": (7047, 0.5),
    "moon.pc2_mr.v_model_fps": (4547.0, 0.05),
    "moon.timeline.peri_get": (77.350, 5e-4),
    "moon.timeline.rms_s": (21, 0.5),
    "moon.integrator.r_min_km": (1989.6, 0.05),
    "earth.n_calls": (18, 0),
    "earth.r_entry_km": (6500.06, 0.005),
    "earth.v_entry_model_fps": (36194, 0.5),
    "earth.v_entry_model_kms": (11.032, 5e-4),
    "earth.v_entry_model_mph": (24680, 5),
    "earth.v_entry_mr_kms": (11.037, 5e-4),
    "earth.diff_pct": (-0.046, 5e-4),
    "earth.v_esc_kms": (11.075, 5e-4),
    "earth.v_esc_fps": (36334, 1),
    "earth.below_escape_mr_pct": (0.34, 0.005),
    "earth.below_escape_model_pct": (0.38, 0.005),
    "earth.misfit_max_pct": (0.26, 0.005),
    "earth_frame.x_deg": (0.23, 1e-6),
    "earth_frame.d_eps_pass_km2s2": (0.0056, 5e-5),
    "earth_frame.d_eps_pass_per_deg_km2s2": (0.0244, 5e-5),
    "earth_frame.d_eps_max_km2s2": (1.40, 0.005),
    "earth_frame.v_earth_in_ms": (755, 0.5),
    "earth_frame.v_earth_out_ms": (755, 0.5),
    "earth_frame.eps_pre_patched_km2s2": (-0.701, 5e-4),
    "earth_frame.eps_call_59_51_km2s2": (-0.688, 5e-4),
    "earth_frame.eps_call_70_12_km2s2": (-0.646, 5e-4),
    "earth_frame.eps_call_86_01_km2s2": (-0.419, 5e-4),
    "earth_frame.eps_call_104_37_km2s2": (-0.459, 5e-4),
    "earth_frame.v_inf_post_pc2_kms": (1.242, 5e-4),
    "earth_frame.burn_off_velocity_deg": (58.5, 0.05),
    "earth_frame.eps_post_patched_km2s2": (-0.477, 5e-4),
    "earth_frame.eps_entry_fit_km2s2": (-0.470, 5e-4),
    "earth_frame.d_eps_burn_km2s2": (0.218, 5e-4),
    "earth_frame.d_eps_calls_km2s2": (0.217, 5e-4),
    "earth_frame.v_earth_after_pc2_ms": (1008, 0.5),
    "earth.misfit_rms_pct": (0.08, 0.005),
    "earth.v_entry_mr_mph": (24689, 0.5),
    "moon.v_inf_fps": (3431, 0.5),
    "moon.integrator.turn_in_window_deg": (83.0, 0.05),
}


def rows(table, idx):
    return np.array([row[idx] for row in table])


def moon_calls():
    alt, fps = rows(D.MOON_PASS, 1), rows(D.MOON_PASS, 2)
    r, v = to_radius_speed(alt, fps, R_MOON)
    r_c, _ = to_radius_speed(alt, fps, 0.0)   # centre reading
    return alt, fps, r, v, r_c


def peri_numbers(eps, r_p, v_ref_fps):
    v_inf = np.sqrt(2 * eps)
    v_p = hyperbola_speed(r_p, v_inf, MU_MOON)
    e = 1 + r_p * v_inf ** 2 / MU_MOON
    v_esc = escape_speed(r_p, MU_MOON)
    out = {
        "alt_km": (r_p - R_MOON) / 1e3, "r_km": r_p / 1e3,
        "v_model_kms": v_p / 1e3, "v_model_fps": v_p / FT,
        "v_model_mph": v_p / MPH, "v_model_kmh": v_p / KMH,
        "e": e, "delta_deg": np.degrees(
            turn_angle(v_inf, r_p, MU_MOON)),
        "v_esc_kms": v_esc / 1e3, "v_esc_fps": v_esc / FT,
        "v_over_v_esc": v_p / v_esc,
        # dv_p/dr_p = -mu / (r_p^2 v_p): datum sensitivity
        "dv_dr_fps_per_km": -MU_MOON / (r_p ** 2 * v_p) / FT * 1e3,
    }
    if v_ref_fps is not None:
        out["v_ref_fps"] = v_ref_fps
        out["diff_fps"] = out["v_model_fps"] - v_ref_fps
        out["diff_pct"] = 100 * (out["v_model_fps"] / v_ref_fps - 1)
    return out


def moon_leg():
    alt, fps, r, v, r_c = moon_calls()
    eps = fit_energy(r, v, MU_MOON)
    v_inf = np.sqrt(2 * eps)
    mis = speed_misfit(r, v, eps, MU_MOON)
    vinf2 = v_inf_squared(r, v, MU_MOON)
    vinf2_c = v_inf_squared(r_c, v, MU_MOON)
    r_mr = R_MOON + D.PERI_ALT_NMI * NMI
    r_afj = R_MOON + D.PERI_ALT_AFJ_KM * 1e3
    i_film = [g for g, *_ in D.MOON_PASS].index(D.FILM_LINE_GET)
    r_pc2 = R_MOON + D.PC2_ALT_NMI * NMI
    v_pc2 = hyperbola_speed(r_pc2, v_inf, MU_MOON)
    return {
        "n_calls": len(D.MOON_PASS),
        "altitude_reading": "altitude above the mean radius "
                            f"{R_MOON / 1e3} km",
        "eps_J_kg": eps, "v_inf_kms": v_inf / 1e3,
        "v_inf_fps": v_inf / FT, "a_km": -MU_MOON / v_inf ** 2 / 1e3,
        "misfit_pct": list(100 * mis),
        "misfit_max_pct": float(100 * np.max(np.abs(mis))),
        "misfit_rms_pct": float(100 * np.sqrt(np.mean(mis ** 2))),
        "vinf2_km2s2": list(vinf2 / 1e6),
        "vinf2_min_max_vs_fit_pct": [
            float(100 * (vinf2.min() / v_inf ** 2 - 1)),
            float(100 * (vinf2.max() / v_inf ** 2 - 1))],
        "centre_reading_vinf2_km2s2": list(vinf2_c / 1e6),
        "centre_reading_negative_vinf2": int(np.sum(vinf2_c < 0)),
        "peri_mr": peri_numbers(eps, r_mr, D.PERI_SPEED_FPS),
        "peri_afj": peri_numbers(eps, r_afj, D.PERI_SPEED_FPS),
        "film_line": {
            "get": D.FILM_LINE_GET, "alt_nmi": float(alt[i_film]),
            "v_said_fps": float(fps[i_film]),
            "v_model_fps": float(hyperbola_speed(
                r[i_film], v_inf, MU_MOON) / FT)},
        "pc2_mr": {"get": D.PC2_GET, "alt_nmi": D.PC2_ALT_NMI,
                   "v_mr_fps": D.PC2_SPEED_FPS,
                   "v_model_fps": v_pc2 / FT},
    }, (r, v, v_inf, r_mr)


def timeline(inbound, outbound, r):
    """Model time of each call vs its GET; fit the offset."""
    legs = rows(D.MOON_PASS, 3)
    t = np.array([time_at_radius(
        inbound if leg == "in" else outbound, ri)
        for leg, ri in zip(legs, r)])
    get = np.array([get_seconds(g) for g, *_ in D.MOON_PASS])
    t_peri = np.mean(get - t)
    res = get - t - t_peri
    return t, t_peri, res


def moon_pass(res_moon, state):
    r, v, v_inf, r_p = state
    inbound, outbound = fly_pass(v_inf, r_p, SPAN, SPAN, MU_MOON)
    t_call, t_peri, res = timeline(inbound, outbound, r)
    t_start = get_seconds(D.MOON_PASS[0][0]) - t_peri
    t_end = get_seconds(D.PC2_GET) - t_peri
    t, Y = sample_pass(inbound, outbound, DT)
    keep = (t >= np.floor(t_start / DT) * DT) & (t <= t_end)
    t, Y = t[keep], Y[:, keep]
    E = specific_energy(Y, MU_MOON)
    speed = np.hypot(Y[2], Y[3])
    rr = np.hypot(Y[0], Y[1])
    res_moon["timeline"] = {
        "peri_get": t_peri / 3600,
        "peri_get_hms": hms(t_peri),
        "peri_get_mr_table_4_III": D.PERI_GET,
        "rms_s": float(np.sqrt(np.mean(res ** 2))),
        "max_abs_s": float(np.max(np.abs(res))),
        "residual_s": list(res),
        "pc2_after_peri_hms": hms(get_seconds(D.PC2_GET) - t_peri),
    }
    res_moon["integrator"] = {
        "energy_rel_drift": float(np.ptp(E) / abs(E.mean())),
        "r_min_km": rr.min() / 1e3,
        "v_max_kms": speed.max() / 1e3,
        "window_get": [hms(t[0] + t_peri), hms(t[-1] + t_peri)],
        "turn_in_window_deg": float(np.degrees(
            turn_of(Y[2:4, 0], Y[2:4, -1]))),
    }
    calls = call_points(inbound, outbound, t_call, t_peri, res)
    return t, t_peri, Y, calls


def turn_of(a, b):
    return np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)


def hms(sec):
    h, rem = divmod(round(sec), 3600)
    return f"{h:03d}:{rem // 60:02d}:{rem % 60:02d}"


def call_points(inbound, outbound, t_call, t_peri, res):
    """Each call placed on the model hyperbola (same r, leg)."""
    pts = []
    for (g, alt, fps, leg, note), tc, dt in zip(
            D.MOON_PASS, t_call, res):
        y = (inbound if leg == "in" else outbound).sol(tc)
        pts.append({
            "get": g, "alt_nmi": alt, "speed_fps": fps,
            "leg": leg, "t_model_s": tc, "time_residual_s": dt,
            "x_km": y[0] / 1e3, "y_km": y[1] / 1e3,
            "model_speed_fps": np.hypot(y[2], y[3]) / FT,
            "film_line": g == D.FILM_LINE_GET, "source": note})
    return pts


def earth_leg():
    alt, fps = rows(D.EARTH_RETURN, 1), rows(D.EARTH_RETURN, 2)
    r, v = to_radius_speed(alt, fps, R_EARTH_EQ)
    r_c, _ = to_radius_speed(alt, fps, 0.0)
    eps = fit_energy(r, v, MU_EARTH)
    eps_c = fit_energy(r_c, v, MU_EARTH)
    mis = speed_misfit(r, v, eps, MU_EARTH)
    mis_c = speed_misfit(r_c, v, eps_c, MU_EARTH)
    r_ei = R_EARTH_EQ + H_ENTRY
    v_ei = entry_speed(eps, r_ei, MU_EARTH)
    v_mr = D.ENTRY_SPEED_FPS * FT
    v_esc = escape_speed(r_ei, MU_EARTH)
    return {
        "n_calls": len(D.EARTH_RETURN),
        "altitude_reading": "altitude above the equatorial "
                            f"radius {R_EARTH_EQ / 1e3} km",
        "eps_J_kg": eps, "a_km": -MU_EARTH / (2 * eps) / 1e3,
        "misfit_pct": list(100 * mis),
        "misfit_max_pct": float(100 * np.max(np.abs(mis))),
        "misfit_rms_pct": float(100 * np.sqrt(np.mean(mis ** 2))),
        "centre_reading_misfit_max_pct": float(
            100 * np.max(np.abs(mis_c))),
        "h_entry_km": H_ENTRY / 1e3, "r_entry_km": r_ei / 1e3,
        "v_entry_model_kms": v_ei / 1e3,
        "v_entry_model_fps": v_ei / FT,
        "v_entry_model_mph": v_ei / MPH,
        "v_entry_model_kmh": v_ei / KMH,
        "v_entry_mr_fps": D.ENTRY_SPEED_FPS,
        "v_entry_mr_kms": v_mr / 1e3, "v_entry_mr_mph": v_mr / MPH,
        "diff_fps": (v_ei - v_mr) / FT,
        "diff_pct": 100 * (v_ei / v_mr - 1),
        "v_esc_kms": v_esc / 1e3, "v_esc_fps": v_esc / FT,
        "below_escape_mr_pct": 100 * (1 - v_mr / v_esc),
        "below_escape_model_pct": 100 * (1 - v_ei / v_esc),
        # dv/dr = -mu/(r^2 v): reference-radius sensitivity
        "dv_dr_fps_per_km": -MU_EARTH / (r_ei ** 2 * v_ei) / FT * 1e3,
    }, (alt, fps, r, v, mis, eps)


def eps_calls(table):
    """Earth-relative energy at each call (J/kg)."""
    alt, fps = rows(table, 1), rows(table, 2)
    r, v = to_radius_speed(alt, fps, R_EARTH_EQ)
    return 0.5 * v ** 2 - MU_EARTH / r


def earth_frame(state, eps_entry):
    """Does the Moon pass alone raise the Earth-frame energy?

    Patched conic in the perifocal frame of moon_pass: the
    Earth lies along -x (pericynthion is the anti-Earth
    point to 0.5 deg), the Moon moves along -y (the only
    choice that matches the pre-pass Earth calls).
    """
    _, _, v_inf, r_p = state
    e = 1 + r_p * v_inf ** 2 / MU_MOON
    f_inf = np.arccos(-1 / e)
    V = np.array([0.0, -D.V_MOON_HORIZONS])
    u_in = np.array([-np.cos(f_inf), np.sin(f_inf)])
    u_out = np.array([np.cos(f_inf), np.sin(f_inf)])
    mu_d = MU_EARTH / D.D_MOON_HORIZONS
    x = trailing_offset(D.PERI_LON_EAST_DEG,
                        D.SUB_EARTH_LON_EAST_DEG)
    d_pass = earth_energy_change(D.V_MOON_HORIZONS, v_inf, e, x)
    one_deg = earth_energy_change(D.V_MOON_HORIZONS, v_inf, e,
                                  np.radians(1.0))
    eps_pre = 0.5 * np.sum((V + v_inf * u_in) ** 2) - mu_d
    # PC+2: burn direction from the speeds before/after it
    _, outb = fly_pass(v_inf, r_p, 10.0, 3 * 3600.0, MU_MOON)
    t_b = get_seconds(D.PC2_GET) - get_seconds(D.PERI_GET)
    y = outb.sol(t_b)
    r_m, v_m = y[:2], y[2:4]
    r_post, v_post = to_radius_speed(
        rows(D.MOON_POST_PC2, 1), rows(D.MOON_POST_PC2, 2), R_MOON)
    vinf_post = np.sqrt(2 * fit_energy(r_post, v_post, MU_MOON))
    rb = np.hypot(*r_m)
    s_after = np.sqrt(vinf_post ** 2 + 2 * MU_MOON / rb)
    dv = D.PC2_DV_FPS * FT
    alpha = burn_angle(np.hypot(*v_m), dv, s_after)
    best = None
    for side in (1, -1):
        u = v_m / np.hypot(*v_m)
        n = np.array([-u[1], u[0]])
        d = dv * (np.cos(alpha) * u + side * np.sin(alpha) * n)
        sol = fly(np.r_[r_m, v_m + d, 0.0], (0.0, 60 * 86400.0),
                  no_thrust, MU_MOON)
        va = sol.y[2:4, -1]
        va = vinf_post * va / np.hypot(*va)
        eps_post = 0.5 * np.sum((V + va) ** 2) - mu_d
        if best is None or abs(eps_post - eps_entry) < \
                abs(best[0] - eps_entry):
            best = (eps_post, side, np.hypot(*(V + va)))
    pre, post = eps_calls(D.EARTH_PRE_PASS), eps_calls(D.EARTH_POST_PC2)
    k = 1e-6
    return {
        "x_deg": float(np.degrees(x)),
        "d_eps_pass_km2s2": d_pass * k,
        "d_eps_pass_per_deg_km2s2": one_deg * k,
        "d_eps_max_km2s2": D.V_MOON_HORIZONS * 2 * v_inf / e * k,
        "dv_inf_turn_kms": 2 * v_inf / e / 1e3,
        "v_moon_kms": D.V_MOON_HORIZONS / 1e3,
        "v_earth_in_ms": float(np.hypot(*(V + v_inf * u_in))),
        "v_earth_out_ms": float(np.hypot(*(V + v_inf * u_out))),
        "eps_pre_patched_km2s2": eps_pre * k,
        "eps_calls_pre_km2s2": list(pre * k),
        "eps_calls_post_pc2_km2s2": list(post * k),
        "eps_call_59_51_km2s2": pre[0] * k,
        "eps_call_70_12_km2s2": pre[-1] * k,
        "eps_call_86_01_km2s2": post[0] * k,
        "eps_call_104_37_km2s2": post[-1] * k,
        "v_inf_post_pc2_kms": vinf_post / 1e3,
        "burn_off_velocity_deg": float(np.degrees(alpha)),
        "burn_side": best[1],
        "eps_post_patched_km2s2": best[0] * k,
        "eps_entry_fit_km2s2": eps_entry * k,
        "d_eps_burn_km2s2": (best[0] - eps_pre - d_pass) * k,
        "d_eps_calls_km2s2": (eps_entry - pre[0]) * k,
        "v_earth_after_pc2_ms": float(best[2]),
    }


def curves(eps_moon, eps_earth):
    """Vis-viva speed-vs-altitude curves for the plots."""
    h_m = np.linspace(0, 12_000e3, 601)
    h_e = np.linspace(0, 70_000e3, 701)
    v_m = np.sqrt(2 * (eps_moon + MU_MOON / (h_m + R_MOON)))
    v_e = np.sqrt(2 * (eps_earth + MU_EARTH / (h_e + R_EARTH_EQ)))
    return h_m, v_m, h_e, v_e


def write_csv(path, header, data):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


PASS_COLS = [
    ("t_s", "s", "time from pericynthion (fitted 77:20:59 GET)"),
    ("get_h", "h", "ground elapsed time = t_s/3600 + peri GET"),
    ("x_km", "km", "Moon-centred perifocal x (+x: pericynthion)"),
    ("y_km", "km", "Moon-centred perifocal y (motion CCW)"),
    ("vx_kms", "km/s", "Moon-relative velocity x"),
    ("vy_kms", "km/s", "Moon-relative velocity y"),
    ("r_km", "km", "distance from the Moon's centre"),
    ("alt_km", "km", "altitude above mean radius 1737.4 km"),
    ("speed_kms", "km/s", "Moon-relative speed |v|"),
    ("speed_fps", "ft/s", "Moon-relative speed (NASA units)"),
    ("speed_mph", "mph", "Moon-relative speed"),
]


def outputs(res, t, t_peri, Y, calls, earth_arr):
    OUT.mkdir(parents=True, exist_ok=True)
    r = np.hypot(Y[0], Y[1])
    sp = np.hypot(Y[2], Y[3])
    tab = np.column_stack([
        t, (t + t_peri) / 3600, Y[0] / 1e3, Y[1] / 1e3,
        Y[2] / 1e3, Y[3] / 1e3, r / 1e3, (r - R_MOON) / 1e3,
        sp / 1e3, sp / FT, sp / MPH])
    write_csv(OUT / "moon_pass.csv", [c[0] for c in PASS_COLS],
              np.round(tab, 6).tolist())
    keys = ["get", "alt_nmi", "speed_fps", "leg", "t_model_s",
            "time_residual_s", "x_km", "y_km", "model_speed_fps",
            "film_line"]
    write_csv(OUT / "moon_readings.csv", keys,
              [[c[k] for k in keys] for c in calls])
    alt, fps, r_e, v_e, mis, eps_e = earth_arr
    write_csv(OUT / "earth_readings.csv",
              ["get", "alt_nmi", "speed_fps", "r_km",
               "model_speed_fps", "misfit_pct"],
              [[g, a, s, ri / 1e3, s * (1 + m), 100 * m]
               for (g, *_), a, s, ri, m
               in zip(D.EARTH_RETURN, alt, fps, r_e, mis)])
    h_m, v_m, h_e, v_ee = curves(res["moon"]["eps_J_kg"], eps_e)
    np.savez(OUT / "free_return.npz",
             moon_pass=tab, h_curve_moon_km=h_m / 1e3,
             v_curve_moon_fps=v_m / FT, h_curve_earth_km=h_e / 1e3,
             v_curve_earth_fps=v_ee / FT,
             moon_alt_nmi=np.array([c["alt_nmi"] for c in calls]),
             moon_speed_fps=np.array([c["speed_fps"] for c in calls]),
             moon_xy_km=np.array([[c["x_km"], c["y_km"]]
                                  for c in calls]),
             earth_alt_nmi=alt, earth_speed_fps=fps)
    write_csv(OUT / "visviva_curves.csv",
              ["body", "alt_km", "speed_fps"],
              [["moon", a, s] for a, s in zip(h_m / 1e3, v_m / FT)]
              + [["earth", a, s]
                 for a, s in zip(h_e / 1e3, v_ee / FT)])
    side = sidecar(res, tab)
    (OUT / "free_return.json").write_text(
        json.dumps(side, indent=1, default=float))
    return side


def sidecar(res, tab):
    n = tab.shape[0]
    return {
        "post": "2026-10-wk2-apollo-13",
        "script": "examples/2026-10-wk2_free_return.py",
        "model": "patched conics: Moon-centred two-body hyperbola "
                 "(Moon leg), Earth-centred two-body conic "
                 "(return leg); r'' = -mu r / r^3",
        "integrator": "scipy solve_ivp DOP853, rtol 1e-12 "
                      "(propagate.fly), from pericynthion both ways",
        "frame": "Moon-centred, inertial, perifocal and planar: "
                 "Moon at (0,0), +x through pericynthion, motion "
                 "counter-clockwise. Pericynthion was the "
                 "anti-Earth point to 0.5 deg (Mission Report + "
                 "JPL Horizons), so the EARTH lies along -x and "
                 "the Moon's own motion (0.967 km/s) is along -y "
                 "(the only orientation that fits the Earth-"
                 "relative calls). The Earth's real distance and "
                 "the orbit plane's tilt are not modelled.",
        "time_base": f"t_s from pericynthion, uniform {DT} s; "
                     "pericynthion = 77:20:59 GET (fitted to the "
                     "calls' clock times). Window: the first call "
                     "(75:06 GET) to PC+2 ignition (79:27:39 GET).",
        "parameters": {
            "mu_moon_m3_s2": MU_MOON, "r_moon_m": R_MOON,
            "mu_earth_m3_s2": MU_EARTH, "r_earth_eq_m": R_EARTH_EQ,
            "h_entry_m": H_ENTRY, "ft_m": FT, "nmi_m": NMI,
            "mph_m_s": MPH,
            "r_p_m": res["moon"]["peri_mr"]["r_km"] * 1e3,
            "v_inf_m_s": res["moon"]["v_inf_kms"] * 1e3,
        },
        "csv": {
            "moon_pass.csv": {
                "rows": n,
                "columns": [{"name": a, "unit": u, "meaning": m}
                            for a, u, m in PASS_COLS]},
            "moon_readings.csv": "the 20 Flight Journal calls: "
                "GET, altitude (nmi), speed (ft/s), leg in/out, "
                "model time from pericynthion (s), clock residual "
                "(s), position on the model hyperbola (km, same "
                "frame), model speed (ft/s), film_line = the "
                "CAPCOM 7,062 ft/s call at 77:38",
            "earth_readings.csv": "the 18 Earth-relative calls "
                "after the last course correction: GET, altitude "
                "(nmi), speed (ft/s), r (km), vis-viva speed "
                "(ft/s), misfit (%)",
            "visviva_curves.csv": "vis-viva speed (ft/s) vs "
                "altitude (km) for the fitted Moon hyperbola "
                "(0-12,000 km) and Earth return conic "
                "(0-70,000 km)",
        },
        "npz_arrays": {
            "moon_pass": f"({n}, 11) float, columns as moon_pass.csv",
            "h_curve_moon_km / v_curve_moon_fps": "(601,) curve",
            "h_curve_earth_km / v_curve_earth_fps": "(701,) curve",
            "moon_alt_nmi, moon_speed_fps": "(20,) calls",
            "moon_xy_km": "(20, 2) calls placed on the hyperbola",
            "earth_alt_nmi, earth_speed_fps": "(18,) calls",
        },
        "caveats": [
            "Two-body Moon only on the Moon leg: Earth's tide is "
            "ignored; the fit shows it is below the calls' scatter "
            "over this window.",
            "Flight Journal times are spoken to the minute; two "
            "near-ignition calls have inferred times.",
            "Lunar altitudes use the mean radius 1737.4 km; the "
            "report references the landing-site radius. Each km "
            "of datum moves pericynthion speed by ~1.7 ft/s.",
            "No figure-eight, no Earth-Moon geometry (week 3).",
        ],
        "results": res,
    }


def lookup(res, key):
    node = res
    for part in key.split("."):
        node = node[part]
    return node


def check(res):
    bad = []
    for key, (want, tol) in EXPECTED.items():
        got = lookup(res, key)
        if not abs(got - want) <= tol:
            bad.append(f"{key}: got {got:.6g}, expected {want}")
    if res["moon"]["integrator"]["energy_rel_drift"] > 1e-10:
        bad.append("integrator energy drift > 1e-10")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    moon, state = moon_leg()
    earth, earth_arr = earth_leg()
    res = {"moon": moon, "earth": earth}
    t, t_peri, Y, calls = moon_pass(moon, state)
    res["earth_frame"] = earth_frame(state, earth["eps_J_kg"])
    outputs(res, t, t_peri, Y, calls, earth_arr)
    m, e = moon, earth
    print(f"Moon: v_inf {m['v_inf_kms']:.4f} km/s, "
          f"a {m['a_km']:.0f} km, misfit max "
          f"{m['misfit_max_pct']:.3f}% over {m['n_calls']} calls")
    p = m["peri_mr"]
    print(f"  pericynthion {p['alt_km']:.1f} km: "
          f"{p['v_model_fps']:.1f} ft/s vs 8053.4 "
          f"({p['diff_pct']:+.3f}%), e {p['e']:.4f}, "
          f"delta {p['delta_deg']:.2f} deg")
    print(f"  peri GET {m['timeline']['peri_get_hms']}, clock rms "
          f"{m['timeline']['rms_s']:.1f} s")
    print(f"Earth: v_entry {e['v_entry_model_fps']:.0f} ft/s vs "
          f"{e['v_entry_mr_fps']} ({e['diff_pct']:+.3f}%), "
          f"escape {e['v_esc_fps']:.0f} ft/s")
    f = res["earth_frame"]
    print(f"Earth frame: pass alone {f['d_eps_pass_km2s2']:+.4f} "
          f"km2/s2 (+-{f['d_eps_pass_per_deg_km2s2']:.3f}/deg), "
          f"PC+2 {f['d_eps_burn_km2s2']:+.3f}, calls "
          f"{f['d_eps_calls_km2s2']:+.3f}")
    print(f"wrote {OUT}")
    if args.check:
        bad = check(res)
        for b in bad:
            print("FAIL", b)
        print("CHECK FAILED" if bad else "CHECK OK")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
