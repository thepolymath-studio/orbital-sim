"""2026-10-wk2 Apollo 13 Sim post: alternate histories.

Same model and same fitted Moon hyperbola as the concept post
(2026-10-wk2_free_return.py: one v_inf from 20 Flight Journal calls).
Three what-ifs:

1. A different Moon pass. Keep v_inf, move the closest approach
   (graze, 50, 150, 252 = actual, 500, 1,000, 3,000 km). Eq. 09
   gives e and the turn delta, vis-viva the closest-approach speed;
   each pass is flown with the integrator in one Moon-centred frame
   (all arrive along +x) so the paths overlay.
2. No PC+2 burn. The final PAD's burn direction (LVLH) on the
   fitted hyperbola predicts the v_inf after the burn; six later
   calls check it. Without the burn v_inf stays 1.046 km/s. Entry
   speed: vis-viva bounds it for ANY bound fall from the Moon's
   distance. The ~9 h is the Mission Report's, NOT computed here
   (it needs the Earth and the Moon together).
3. No free-return burn. Not simulated (two-body would mislead).
   Only: the burn against the Earth-relative speed, and a labelled
   TOY straight-line drift.

Writes outputs/2026-10-wk2-sim/: passes.csv, passes_summary.csv,
pc2.csv, post_pc2_calls.csv, alt_histories.npz and the
alt_histories.json sidecar (the contract with the visualizer).

    python examples/2026-10-wk2_alt_histories.py [--check]

--check asserts the numbers quoted in
posts/2026-10-wk2-apollo-13-sim/code-results.md (exit 1 on mismatch).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from orbital_sim import apollo13_data as D
from orbital_sim.apollo13 import (fit_energy, get_seconds,
                                  sample_pass, speed_misfit,
                                  time_at_radius, to_radius_speed,
                                  entry_speed)
from orbital_sim.constants import (D_MOON, FT, H_ENTRY, MU_EARTH,
                                   MU_MOON, NMI, R_EARTH_EQ, R_MOON)
from orbital_sim.dynamics import specific_energy

from orbital_sim.whatif import (along_velocity, burn_v_inf,
                                drift_without_gravity,
                                entry_speed_band, flight_path_angle,
                                fly_aimed_pass, fly_with_and_without,
                                what_if_pass)

OUT = (Path(__file__).resolve().parents[1] / "outputs"
       / "2026-10-wk2-sim")
DT = 20.0                     # s, sample step for every path
R_OUT = 15_000e3              # m, passes drawn out to this radius
ALTS_KM = [0.0, 50.0, 150.0, None, 500.0, 1000.0, 3000.0]
END_GET = "082:00"            # end of the PC+2 window
MPH = 0.44704                 # m/s per mph (exact)
MILE = 1609.344               # m, statute mile (exact)

# Values quoted in code-results.md, checked by --check: (value, tol)
EXPECTED = {
    "v_inf_kms": (1.046, 5e-4),
    "pass.actual.alt_km": (252.2, 0.05),
    "pass.actual.delta_deg": (87.66, 0.005),
    "pass.graze.delta_deg": (92.21, 0.005),
    "pass.graze.e": (1.388, 5e-4),
    "pass.graze.v_p_fps": (8516, 0.5),
    "pass.graze.b_km": (4312, 0.5),
    "pass.actual.b_km": (4668, 0.5),
    "pass.a50.delta_deg": (91.27, 0.005),
    "pass.a150.delta_deg": (89.44, 0.005),
    "pass.a500.delta_deg": (83.67, 0.005),
    "pass.a1000.delta_deg": (76.75, 0.005),
    "pass.a3000.delta_deg": (58.17, 0.005),
    "pass.a3000.e": (2.057, 5e-4),
    "pass.a50.v_p_fps": (8416, 0.5),
    "pass.a3000.v_p_fps": (5836, 0.5),
    "pass.a50.d_delta_deg": (3.60, 0.005),
    "pass.a3000.d_delta_deg": (-29.49, 0.005),
    "pass.graze.d_delta_deg": (4.55, 0.005),
    "sens.ddelta_deg_per_100km": (-1.70, 0.005),
    "pc2.gamma_deg": (72.94, 0.005),
    "pc2.burn_angle_deg": (58.6, 0.05),
    "pc2.speed_gain_ms": (153, 0.5),
    "pc2.v_inf_after_model_kms": (1.241, 5e-4),
    "pc2.v_inf_after_calls_kms": (1.242, 5e-4),
    "pc2.model_vs_calls_pct": (-0.06, 0.005),
    "pc2.calls_misfit_max_pct": (0.03, 0.005),
    "pc2.v_inf_gain_pct": (18.7, 0.05),
    "pc2.clock_max_abs_s": (30, 0.5),
    "pc2.at_8145.r_with_km": (23201, 1),
    "pc2.at_8145.r_without_km": (22283, 1),
    "pc2.at_8145.gap_km": (2108, 1),
    "entry.v_slow_kms": (10.980, 5e-4),
    "entry.v_fast_kms": (11.075, 5e-4),
    "entry.band_pct": (0.85, 0.005),
    "entry.v_mr_kms": (11.037, 5e-4),
    "sourced.hours_saved_mr": (9.12, 0.005),
    "toy.dv_ms": (11.52, 0.005),
    "toy.v_earth_ms": (940, 0.5),
    "toy.dv_over_v_pct": (1.23, 0.005),
    "toy.dv_mph": (25.8, 0.05),
    "toy.to_peri_h": (15.85, 0.005),
    "toy.drift_to_peri_km": (658, 0.5),
    "toy.to_entry_h": (81.18, 0.005),
    "toy.drift_to_entry_km": (3367, 0.5),
    "toy.drift_to_entry_mi": (2092, 0.5),
}


def rows(table, idx):
    return np.array([row[idx] for row in table], float)


def moon_fit():
    """One v_inf from the 20 pre-PC+2 calls (as the concept)."""
    r, v = to_radius_speed(rows(D.MOON_PASS, 1),
                           rows(D.MOON_PASS, 2), R_MOON)
    eps = fit_energy(r, v, MU_MOON)
    return np.sqrt(2 * eps), r


def pass_key(alt):
    if alt is None:
        return "actual"
    return "graze" if alt == 0 else f"a{alt:g}"


def what_if_1(v_inf):
    """The pass sweep: numbers, and each pass flown."""
    alt_mr = D.PERI_ALT_NMI * NMI
    ref = what_if_pass(v_inf, alt_mr, MU_MOON, R_MOON)
    res, paths = {}, {}
    for alt in ALTS_KM:
        h = alt_mr if alt is None else alt * 1e3
        p = what_if_pass(v_inf, h, MU_MOON, R_MOON)
        inb, outb = fly_aimed_pass(v_inf, p["r_p"], MU_MOON,
                                   R_OUT)
        t, Y = sample_pass(inb, outb, DT)
        res[pass_key(alt)] = pass_numbers(p, ref, h, t, Y)
        paths[pass_key(alt)] = (h, t, Y)
    d = [what_if_pass(v_inf, alt_mr + s, MU_MOON, R_MOON)
         for s in (-1e3, 1e3)]
    sens = np.degrees(d[1]["delta"] - d[0]["delta"]) / 2 * 100
    return res, paths, {"ddelta_deg_per_100km": sens}


def pass_numbers(p, ref, h, t, Y):
    r = np.hypot(Y[0], Y[1])
    E = specific_energy(Y, MU_MOON)
    v_end = Y[2:4, -1]
    return {
        "alt_km": h / 1e3, "r_p_km": p["r_p"] / 1e3,
        "e": p["e"], "delta_deg": np.degrees(p["delta"]),
        "d_delta_deg": np.degrees(p["delta"] - ref["delta"]),
        "heading_out_deg": np.degrees(p["delta"]),
        "v_p_kms": p["v_p"] / 1e3, "v_p_fps": p["v_p"] / FT,
        "v_p_mph": p["v_p"] / MPH,
        "b_km": p["b"] / 1e3,
        "d_b_km": (p["b"] - ref["b"]) / 1e3,
        "clears_surface": bool(p["r_p"] >= R_MOON),
        "integrated": {
            "r_min_km": r.min() / 1e3,
            "energy_rel_drift": float(np.ptp(E) / abs(E.mean())),
            "heading_at_r_out_deg": float(np.degrees(
                np.arctan2(v_end[1], v_end[0]))),
            "t_span_s": [float(t[0]), float(t[-1])],
        },
    }


def peri_get(inb, outb, r_calls):
    """Pericynthion GET from the calls' clock times."""
    legs = [row[3] for row in D.MOON_PASS]
    t = np.array([time_at_radius(inb if g == "in" else outb, r)
                  for g, r in zip(legs, r_calls)])
    get = np.array([get_seconds(row[0]) for row in D.MOON_PASS])
    return float(np.mean(get - t))


def what_if_2(v_inf, r_calls):
    """No PC+2: the burn's v_inf gain, checked by later calls."""
    r_p = R_MOON + D.PERI_ALT_NMI * NMI
    inb, outb = fly_aimed_pass(v_inf, r_p, MU_MOON, R_OUT)
    t_peri = peri_get(inb, outb, r_calls)
    t_mid = get_seconds(D.PC2_GET) + D.PC2_BURN_S / 2
    y_b = outb.sol(t_mid - t_peri)
    r_b, v_b = np.hypot(*y_b[0:2]), np.hypot(*y_b[2:4])
    v_p = np.hypot(*outb.y[2:4, 0])
    gamma = flight_path_angle(r_b, v_b, r_p, v_p)
    pad = np.array(D.PC2_PAD_LVLH_FPS) * FT
    dv = pad * D.PC2_DV_FPS / np.linalg.norm(D.PC2_PAD_LVLH_FPS)
    v_after = burn_v_inf(r_b, v_b, gamma, dv, MU_MOON)
    along = along_velocity(gamma, dv)
    v_new = np.sqrt(v_b ** 2 + 2 * v_b * along + dv @ dv)
    span = get_seconds(END_GET) - t_mid
    w, wo = fly_with_and_without(y_b, dv, span, MU_MOON)
    calls = post_calls(w, t_mid)
    res = {
        "fitted_peri_get_h": t_peri / 3600,
        "burn_mid_get_h": t_mid / 3600,
        "r_burn_km": r_b / 1e3, "v_before_kms": v_b / 1e3,
        "gamma_deg": np.degrees(gamma),
        "dv_ms": float(np.linalg.norm(dv)),
        "burn_angle_deg": np.degrees(np.arccos(
            along / np.linalg.norm(dv))),
        "along_ms": along,
        "speed_gain_ms": v_new - v_b,
        "v_inf_before_kms": v_inf / 1e3,
        "v_inf_after_model_kms": v_after / 1e3,
        "v_inf_gain_pct": 100 * (v_after / v_inf - 1),
    }
    res.update(calls["summary"])
    res["model_vs_calls_pct"] = 100 * (
        v_after / (res["v_inf_after_calls_kms"] * 1e3) - 1)
    res["at_8145"] = gap_at(w, wo, t_mid, "081:45")
    return res, (inb, outb, t_peri, t_mid, w, wo), calls["rows"]


def post_calls(w, t_mid):
    """Fit the six post-burn calls; clock-check the model."""
    r, v = to_radius_speed(rows(D.MOON_AFTER_PC2, 1),
                           rows(D.MOON_AFTER_PC2, 2), R_MOON)
    eps = fit_energy(r, v, MU_MOON)
    mis = speed_misfit(r, v, eps, MU_MOON)
    t = np.array([time_at_radius(w, ri) for ri in r])
    get = np.array([get_seconds(row[0])
                    for row in D.MOON_AFTER_PC2])
    res = get - (t_mid + t)
    out = []
    for row, ti, dt, ri in zip(D.MOON_AFTER_PC2, t, res, r):
        y = w.sol(ti)
        out.append({"get": row[0], "alt_nmi": row[1],
                    "speed_fps": row[2], "r_km": ri / 1e3,
                    "t_after_burn_s": ti, "clock_residual_s": dt,
                    "x_km": y[0] / 1e3, "y_km": y[1] / 1e3,
                    "model_speed_fps": np.hypot(*y[2:4]) / FT})
    return {"rows": out, "summary": {
        "n_calls_after": len(r),
        "v_inf_after_calls_kms": np.sqrt(2 * eps) / 1e3,
        "calls_misfit_max_pct": float(100 * np.abs(mis).max()),
        "clock_max_abs_s": float(np.abs(res).max()),
        "clock_rms_s": float(np.sqrt(np.mean(res ** 2)))}}


def gap_at(w, wo, t_mid, get):
    t = get_seconds(get) - t_mid
    a, b = w.sol(t), wo.sol(t)
    return {"get": get,
            "r_with_km": np.hypot(*a[0:2]) / 1e3,
            "r_without_km": np.hypot(*b[0:2]) / 1e3,
            "gap_km": np.hypot(*(a[0:2] - b[0:2])) / 1e3,
            "speed_with_fps": np.hypot(*a[2:4]) / FT,
            "speed_without_fps": np.hypot(*b[2:4]) / FT}


def entry_numbers():
    """Entry speed: any bound fall from the Moon's distance."""
    r_ei = R_EARTH_EQ + H_ENTRY
    slow, fast = entry_speed_band(D_MOON, r_ei, MU_EARTH)
    r, v = to_radius_speed(rows(D.EARTH_RETURN, 1),
                           rows(D.EARTH_RETURN, 2), R_EARTH_EQ)
    v_fit = entry_speed(fit_energy(r, v, MU_EARTH), r_ei,
                        MU_EARTH)
    v_mr = D.ENTRY_SPEED_FPS * FT
    return {
        "r_far_km": D_MOON / 1e3, "r_entry_km": r_ei / 1e3,
        "v_slow_kms": slow / 1e3, "v_fast_kms": fast / 1e3,
        "v_slow_fps": slow / FT, "v_fast_fps": fast / FT,
        "v_slow_mph": slow / MPH, "v_fast_mph": fast / MPH,
        "band_pct": 100 * (1 - slow / fast),
        "v_mr_kms": v_mr / 1e3, "v_fit_kms": v_fit / 1e3,
        "mr_inside": bool(slow <= v_mr < fast),
        "fit_inside": bool(slow <= v_fit < fast),
    }


def sourced_hours():
    """Sourced arithmetic only: the model does not time it."""
    pred = get_seconds(D.PC2_PRED_LANDING_GET) / 3600
    ei_afj = get_seconds(D.NO_PC2_EI_GET_AFJ) / 3600
    ei_mr = get_seconds(D.ENTRY_GET) / 3600
    return {
        "no_pc2_landing_h_mr": D.NO_PC2_LANDING_H,
        "pc2_predicted_landing_h_mr": pred,
        "hours_saved_mr": D.NO_PC2_LANDING_H - pred,
        "hours_saved_mr_text": D.MR_HOURS_SAVED,
        "no_pc2_entry_h_afj": ei_afj, "entry_h_mr": ei_mr,
        "entry_shift_h_afj_vs_mr": ei_afj - ei_mr,
        "splashdown_get": D.SPLASHDOWN_GET,
        "note": "Sourced arithmetic, NOT computed by the model: "
                "timing the return needs the Earth and the Moon "
                "together (week 3).",
    }


def toy():
    """What-if 3: sizes, and a straight-line TOY drift."""
    dv = D.FR_BURN_DV_FPS * FT
    v_e = D.EARTH_AROUND_FR_BURN[1][2] * FT
    t0 = get_seconds(D.FR_BURN_GET)
    t_p = get_seconds(D.PERI_GET) - t0
    t_e = get_seconds(D.ENTRY_GET) - t0
    x_p = drift_without_gravity(dv, t_p)
    x_e = drift_without_gravity(dv, t_e)
    return {
        "label": "TOY: straight-line drift, gravity ignored. "
                 "Not a prediction of the miss.",
        "dv_ms": dv, "dv_mph": dv / MPH, "dv_kmh": dv * 3.6,
        "dv_pad_ms": D.FR_BURN_PAD_DV_FPS * FT,
        "v_earth_ms": v_e,
        "v_earth_before_ms": D.EARTH_AROUND_FR_BURN[0][2] * FT,
        "dv_over_v_pct": 100 * dv / v_e,
        "to_peri_h": t_p / 3600, "drift_to_peri_km": x_p / 1e3,
        "drift_to_peri_over_pass_alt": x_p / (
            D.PERI_ALT_NMI * NMI),
        "to_entry_h": t_e / 3600,
        "drift_to_entry_km": x_e / 1e3,
        "drift_to_entry_mi": x_e / MILE,
        "drift_to_entry_nmi": x_e / NMI,
        "drift_to_entry_over_r_earth": x_e / R_EARTH_EQ,
    }


def write_csv(path, header, data):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


def pc2_table(state):
    """GET grid 75:06 -> 82:00; one path, then two."""
    inb, outb, t_peri, t_mid, w, wo = state
    t0 = get_seconds(D.MOON_PASS[0][0])
    get = np.arange(t0, get_seconds(END_GET) + 1, DT)
    rows_ = []
    for g in get:
        tp = g - t_peri
        if g < t_mid:
            leg = inb if tp < 0 else outb
            a = b = leg.sol(tp)
        else:
            a, b = w.sol(g - t_mid), wo.sol(g - t_mid)
        rows_.append([g / 3600, tp, int(g >= t_mid)]
                     + list(a[0:4] / 1e3) + list(b[0:4] / 1e3)
                     + [np.hypot(*a[2:4]) / FT,
                        np.hypot(*b[2:4]) / FT])
    return np.array(rows_)


PC2_COLS = [
    ("get_h", "h", "ground elapsed time"),
    ("t_from_peri_s", "s", "time from fitted pericynthion"),
    ("after_burn", "0/1", "1 once the burn (impulse at its "
                          "midpoint) has happened"),
    ("x_with_km", "km", "Moon-centred x, real history (PC+2)"),
    ("y_with_km", "km", "Moon-centred y, real history"),
    ("vx_with_kms", "km/s", "Moon-relative velocity x"),
    ("vy_with_kms", "km/s", "Moon-relative velocity y"),
    ("x_without_km", "km", "Moon-centred x, no PC+2"),
    ("y_without_km", "km", "Moon-centred y, no PC+2"),
    ("vx_without_kms", "km/s", "velocity x, no PC+2"),
    ("vy_without_kms", "km/s", "velocity y, no PC+2"),
    ("speed_with_fps", "ft/s", "Moon-relative speed, real"),
    ("speed_without_fps", "ft/s", "Moon-relative speed, no PC+2"),
]

PASS_COLS = [
    ("pass", "-", "graze | a50 | a150 | actual | a500 | a1000 "
                  "| a3000"),
    ("alt_km", "km", "closest-approach altitude (mean radius)"),
    ("t_s", "s", "time from this pass's closest approach"),
    ("x_km", "km", "Moon-centred x (+x = arrival direction)"),
    ("y_km", "km", "Moon-centred y"),
    ("vx_kms", "km/s", "Moon-relative velocity x"),
    ("vy_kms", "km/s", "Moon-relative velocity y"),
    ("r_km", "km", "distance from the Moon's centre"),
    ("speed_kms", "km/s", "Moon-relative speed"),
    ("speed_fps", "ft/s", "Moon-relative speed"),
]


def outputs(res, paths, state, call_rows):
    OUT.mkdir(parents=True, exist_ok=True)
    long, arrays = [], {}
    for key, (h, t, Y) in paths.items():
        r, sp = np.hypot(Y[0], Y[1]), np.hypot(Y[2], Y[3])
        tab = np.column_stack([
            np.full(t.size, h / 1e3), t, Y[0] / 1e3, Y[1] / 1e3,
            Y[2] / 1e3, Y[3] / 1e3, r / 1e3, sp / 1e3, sp / FT])
        arrays[f"pass_{key}"] = tab
        long += [[key] + list(np.round(row, 6)) for row in tab]
    write_csv(OUT / "passes.csv", [c[0] for c in PASS_COLS], long)
    keys = ["alt_km", "r_p_km", "e", "delta_deg", "d_delta_deg",
            "heading_out_deg", "v_p_kms", "v_p_fps", "b_km",
            "d_b_km", "clears_surface"]
    write_csv(OUT / "passes_summary.csv", ["pass"] + keys,
              [[k] + [p[c] for c in keys]
               for k, p in res["pass"].items()])
    pc2 = pc2_table(state)
    write_csv(OUT / "pc2.csv", [c[0] for c in PC2_COLS],
              np.round(pc2, 6).tolist())
    ck = list(call_rows[0].keys())
    write_csv(OUT / "post_pc2_calls.csv", ck,
              [[c[k] for k in ck] for c in call_rows])
    h_far = np.linspace(100e3, 2e6, 381) * 1e3
    v_slow = [entry_speed_band(r, R_EARTH_EQ + H_ENTRY,
                               MU_EARTH)[0] for r in h_far]
    np.savez(OUT / "alt_histories.npz", pc2=pc2,
             entry_r_far_km=h_far / 1e3,
             entry_v_slow_kms=np.array(v_slow) / 1e3, **arrays)
    side = sidecar(res, arrays, pc2)
    (OUT / "alt_histories.json").write_text(
        json.dumps(side, indent=1, default=float))


def sidecar(res, arrays, pc2):
    return {
        "post": "2026-10-wk2-apollo-13-sim",
        "script": "examples/2026-10-wk2_alt_histories.py",
        "model": "patched conics, planar: Moon-centred two-body "
                 "hyperbola (r'' = -mu r / r^3) near the Moon; "
                 "vis-viva for the Earth leg. v_inf = 1.046 km/s "
                 "fitted to 20 Flight Journal calls (concept post).",
        "integrator": "scipy solve_ivp DOP853, rtol 1e-12 "
                      "(propagate.fly); passes flown from closest "
                      "approach both ways to r = 15,000 km.",
        "frame": "Moon-centred, inertial, planar. +x = the "
                 "direction every pass ARRIVES from far away "
                 "(motion +x, counter-clockwise turn, so the "
                 "craft passes below the Moon, y < 0, and leaves "
                 "at heading delta from +x). The Earth's direction "
                 "and the real orbit plane are NOT modelled.",
        "time_base": f"uniform {DT} s. passes.csv: t from each "
                     "pass's own closest approach. pc2.csv: "
                     "ground elapsed time (GET), 75:06 to 82:00; "
                     "PC+2 as an impulse at the burn's midpoint "
                     f"({res['pc2']['burn_mid_get_h']:.4f} h GET).",
        "csv": {
            "passes.csv": {"columns": [
                {"name": a, "unit": u, "meaning": m}
                for a, u, m in PASS_COLS],
                "rows_per_pass": {k[5:]: v.shape[0]
                                  for k, v in arrays.items()}},
            "passes_summary.csv": "one row per pass: altitude, "
                "r_p, e, turn delta (deg), change vs the real "
                "pass (deg), exit heading (deg from +x, = delta), "
                "closest-approach speed (km/s, ft/s), aim offset "
                "b (km: the arrival line is y = -b), change in b "
                "vs the real pass (km), clears_surface",
            "pc2.csv": {"rows": int(pc2.shape[0]), "columns": [
                {"name": a, "unit": u, "meaning": m}
                for a, u, m in PC2_COLS]},
            "post_pc2_calls.csv": "the six Flight Journal calls "
                "after PC+2: GET, altitude (nmi), speed (ft/s), r "
                "(km), model time after the burn (s), clock "
                "residual (s), position on the with-burn path "
                "(km), model speed (ft/s)",
        },
        "npz_arrays": {
            "pass_<key>": "(n, 9): alt_km, t_s, x_km, y_km, "
                          "vx_kms, vy_kms, r_km, speed_kms, "
                          "speed_fps",
            "pc2": f"({pc2.shape[0]}, 13), columns as pc2.csv",
            "entry_r_far_km, entry_v_slow_kms": "(381,) slowest "
                "entry speed vs the fall's far point; the fastest "
                "is escape speed, results.entry.v_fast_kms",
        },
        "do_not_claim": [
            "Where (or whether) a different Moon pass, a missing "
            "PC+2 or a missing free-return burn would meet the "
            "Earth: not computed (needs Earth + Moon, week 3).",
            "The ~9 h: the Mission Report's number, not the "
            "model's (results.sourced).",
            "The toy drift is a straight line with gravity "
            "ignored. It is a size, not the miss distance.",
            "The exit heading is in the Moon's frame; it is not "
            "a direction toward the Earth.",
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
    for k, p in res["pass"].items():
        i = p["integrated"]
        if abs(i["r_min_km"] - p["r_p_km"]) > 1e-3:
            bad.append(f"{k}: integrated r_min != r_p")
        if i["energy_rel_drift"] > 1e-10:
            bad.append(f"{k}: energy drift > 1e-10")
    if not (res["entry"]["mr_inside"] and res["entry"]["fit_inside"]):
        bad.append("entry speed outside the vis-viva band")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    v_inf, r_calls = moon_fit()
    passes, paths, sens = what_if_1(v_inf)
    pc2, state, call_rows = what_if_2(v_inf, r_calls)
    res = {"v_inf_kms": v_inf / 1e3, "pass": passes, "sens": sens,
           "pc2": pc2, "entry": entry_numbers(),
           "sourced": sourced_hours(), "toy": toy()}
    outputs(res, paths, state, call_rows)
    print(f"v_inf {v_inf / 1e3:.4f} km/s")
    for k, p in passes.items():
        print(f"  {k:>6}: alt {p['alt_km']:7.1f} km  e "
              f"{p['e']:.4f}  delta {p['delta_deg']:6.2f} deg "
              f"({p['d_delta_deg']:+6.2f})  v_p "
              f"{p['v_p_fps']:5.0f} ft/s  b {p['b_km']:5.0f} km")
    q = pc2
    print(f"PC+2: angle to velocity {q['burn_angle_deg']:.1f} deg,"
          f" v_inf {q['v_inf_before_kms']:.3f} -> "
          f"{q['v_inf_after_model_kms']:.4f} (calls "
          f"{q['v_inf_after_calls_kms']:.4f}), clock max "
          f"{q['clock_max_abs_s']:.0f} s")
    e = res["entry"]
    print(f"entry band {e['v_slow_kms']:.3f}-{e['v_fast_kms']:.3f}"
          f" km/s; NASA {e['v_mr_kms']:.3f}")
    t = res["toy"]
    print(f"toy: {t['dv_ms']:.2f} m/s = {t['dv_over_v_pct']:.2f}% "
          f"of {t['v_earth_ms']:.0f} m/s; drift "
          f"{t['drift_to_peri_km']:.0f} km to the Moon, "
          f"{t['drift_to_entry_km']:.0f} km to entry")
    print(f"wrote {OUT}")
    if args.check:
        bad = check(res)
        for b in bad:
            print("FAIL", b)
        print("CHECK FAILED" if bad else "CHECK OK")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
