"""2026-10 wk4 (Gravity), concept post: the film's cloud, then what happens next.

One system: a satellite on a circular orbit 600 km up (the film's
title card) breaks up into 500 pieces that stay in orbit (NASA
Standard Breakup Model), flown under Earth's gravity + J2
(orbital_sim.debris). Assumed tilt 51.6 deg (the ISS's; the film
does not say).

1. Why it returns every ~90 minutes: the period at 600 km.
2. Why it smears into a ring within a day: different kicks give
   different periods (Kepler), not J2.
3. Earth's bulge (J2) turns the ring into a shell over months.
4. Kessler: collisions grow with the square of the junk (a
   counting statement, Kessler & Cour-Palais 1978).
Plus the closing speeds of crossings and the film's spoken speeds.

Writes outputs/2026-10-wk4/: cloud_loop.npz + cloud_loop.json (the
Slide-1 loop: clump -> ring -> shell) and results.json.

    python examples/2026-10-wk4_debris_cloud.py [--check]

--check asserts the numbers in posts/2026-10-wk4-gravity/
code-results.md (exit 1 on a mismatch). About 2.5 minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from orbital_sim import debris as D
from orbital_sim.breakup import L_MAX, L_MIN
from orbital_sim.cloud import middle_range, read_elements, unwrap
from orbital_sim.constants import (DAY, F_EARTH, MU_EARTH,
                                   OMEGA_EARTH, R_EARTH_EQ)
from orbital_sim.elements import orbit_paths
from orbital_sim.j2 import j2_accel, kepler_accel
from orbital_sim.kessler import (MPH, closing_speed,
                                 crossing_angle, pair_count,
                                 pair_growth)

OUT = (Path(__file__).resolve().parents[1] / "outputs"
       / "2026-10-wk4")
LOOP_STEP = 2                    # day frames: every 2 min
SNAP_DAYS = (0, 1, 30, 90, 180, 250, 365)
IRIDIUM = {"alt_km": 789.0, "v_rel_kms": 11.7}   # research S6

# Values quoted in code-results.md, checked by --check: (value, tol)
EXPECTED = {
    "orbit.period_min": (96.69, 0.005),
    "orbit.speed_kms": (7.558, 0.0005),
    "orbit.speed_kmh": (27208, 0.5),
    "orbit.orbits_per_day": (14.89, 0.005),
    "speeds.head_on_kms": (15.12, 0.005),
    "speeds.right_angle_kms": (10.69, 0.005),
    "speeds.head_on_kmh": (54417, 0.5),
    "speeds.film_20k_mph_kms": (8.94, 0.005),
    "speeds.film_50k_mph_kms": (22.35, 0.005),
    "speeds.max_bound_head_on_kms": (21.38, 0.005),
    "speeds.iridium_angle_deg": (103.3, 0.05),
    "bulge.equator_minus_pole_km": (21.38, 0.005),
    "bulge.j2_share_equator": (0.00136, 5e-6),
    "bulge.j2_share_pole": (0.00271, 5e-6),
    "breakup.n_pieces": (500, 0),
    "breakup.n_drawn": (590, 0),
    "breakup.n_fell": (90, 0),
    "breakup.mass_kg": (695, 0.5),
    "breakup.kick_median_ms": (88, 0.5),
    "breakup.kick_p05_ms": (6.4, 0.05),
    "breakup.kick_p95_ms": (459, 0.5),
    "ring.stream90_km_after_1_orbit": (7338, 5),
    "ring.stream50_km_after_1_orbit": (1414, 5),
    "ring.stream90_deg_after_1_orbit": (60.2, 0.05),
    "ring.stream_t_min": (97.0, 1e-9),
    "ring.ring_h": (10.6, 0.05),
    "ring.ring_orbits": (6.6, 0.05),
    "ring.kepler_ring_h": (10.6, 0.05),
    "ring.no_j2_ring_h": (10.7, 0.05),
    "shell.parent_node_rate_deg_day": (-4.52, 0.005),
    "shell.parent_full_turn_days": (79.7, 0.05),
    "shell.spread90_deg_day": (1.44, 0.005),
    "shell.half_shell_days": (125, 0),
    "shell.shell_days": (250, 0),
    "shell.shell_months": (8.2, 0.05),
    "shell.spread50_deg_at_365": (116, 0.5),
    "tilts.28.5.shell_days": (172, 0.5),
    "tilts.82.5.shell_days": (838, 0.5),
    "kessler.pairs_1000": (499500, 0),
    "kessler.pairs_2000": (1999000, 0),
    "kessler.growth": (4.0, 0.005),
}


def orbit_and_speeds():
    o = D.orbit_numbers()
    v = o["speed_kms"]
    v_ir = np.sqrt(MU_EARTH / (R_EARTH_EQ + IRIDIUM["alt_km"]
                               * 1e3)) / 1e3
    speeds = {
        "head_on_kms": closing_speed(v, np.pi),
        "right_angle_kms": closing_speed(v, np.pi / 2),
        "head_on_kmh": closing_speed(v, np.pi) * 3600,
        "film_20k_mph_kms": 20_000 * MPH / 1e3,
        "film_50k_mph_kms": 50_000 * MPH / 1e3,
        "escape_kms": np.sqrt(2) * v,
        "max_bound_head_on_kms": 2 * np.sqrt(2) * v,
        "iridium_v_kms": v_ir,
        "iridium_angle_deg": np.degrees(crossing_angle(
            IRIDIUM["v_rel_kms"], v_ir))}
    return o, speeds


def bulge():
    r = R_EARTH_EQ + D.ALT
    eq, pole = np.array([r, 0, 0.0]), np.array([0, 0, r])

    def share(p):
        return (np.linalg.norm(j2_accel(p))
                / np.linalg.norm(kepler_accel(p)))
    return {"equator_minus_pole_km": R_EARTH_EQ * F_EARTH / 1e3,
            "j2_share_equator": share(eq),
            "j2_share_pole": share(pole)}


def breakup_numbers(c):
    dv = np.linalg.norm(c.dv, axis=1)
    return {"n_pieces": len(c.lc), "n_drawn": c.n_drawn,
            "n_fell": c.n_fell, "n_escaped": c.n_escaped,
            "share_fell": c.n_fell / c.n_drawn,
            "mass_kg": D.cloud_mass(c),
            "size_range_m": [L_MIN, L_MAX],
            "kick_median_ms": np.median(dv),
            "kick_p05_ms": np.percentile(dv, 5),
            "kick_p95_ms": np.percentile(dv, 95),
            "kick_max_ms": dv.max(),
            "tilt_deg": np.degrees(D.INC), "seed": D.SEED}


def ring_and_shell(run, el_d, elp_d, el_m):
    T = D.orbit_numbers()["period_min"] * 60
    r0 = (R_EARTH_EQ + D.ALT) / 1e3
    rn = D.ring_numbers(run.t_day, el_d, elp_d)
    rn0 = D.ring_numbers(run.t_day, read_elements(run.day_no_j2),
                         read_elements(run.day_parent_no_j2))
    # stream length in space: angle along the orbit from the
    # node (u, the position), ahead of the intact orbit, at the
    # first 60 s sample after one Kepler period (t = 97 min)
    d = unwrap(el_d["u"]) - unwrap(elp_d["u"])
    j = np.searchsorted(run.t_day, T)
    ring = {"ring_h": rn["ring_h"],
            "ring_orbits": rn["ring_h"] * 3600 / T,
            "head_meets_tail_h": rn["head_meets_tail_h"],
            "kepler_ring_h": D.kepler_ring_h(run.cloud),
            "no_j2_ring_h": rn0["ring_h"],
            "stream90_km_after_1_orbit":
                middle_range(d[:, j], 0.9) * r0,
            "stream50_km_after_1_orbit":
                middle_range(d[:, j], 0.5) * r0,
            "stream90_deg_after_1_orbit":
                np.degrees(middle_range(d[:, j], 0.9))}
    o = D.orbit_numbers()
    sh = D.shell_numbers(run.t_months, el_m)
    sim, cf = D.precession_check(run.t_months, el_m)
    shell = {"parent_node_rate_deg_day": o["node_rate_deg_day"],
             "parent_full_turn_days": o["full_turn_days"],
             "spread90_deg_day":
                 np.degrees(middle_range(sim)) * DAY,
             "spread90_deg_day_closed_form":
                 np.degrees(middle_range(cf)) * DAY,
             "half_shell_days": sh["half_shell_days"],
             "shell_days": sh["shell_days"],
             "shell_months": sh["shell_days"] / (365.25 / 12),
             "spread50_deg_at_365":
                 np.degrees(sh["spread50_rad"][-1]),
             "max_rel_error_vs_closed_form":
                 np.max(np.abs(sim / cf - 1))}
    ring["stream_t_min"] = run.t_day[j] / 60
    curves = {"one_orbit_along": d[:, j],
              "along90": rn["spread90_rad"],
              "nodes90": sh["spread90_rad"],
              "nodes50": sh["spread50_rad"]}
    return ring, shell, curves


def kessler():
    return {"pairs_1000": pair_count(1000),
            "pairs_2000": pair_count(2000),
            "growth": pair_growth(1000)}


def results():
    run = D.run_cloud(control=True)
    el_d = read_elements(run.day)
    elp_d = read_elements(run.day_parent)
    el_m = read_elements(run.months)
    o, sp = orbit_and_speeds()
    ring, shell, curves = ring_and_shell(run, el_d, elp_d, el_m)
    res = {"orbit": o, "speeds": sp, "bulge": bulge(),
           "breakup": breakup_numbers(run.cloud), "ring": ring,
           "shell": shell, "tilts": D.tilt_sensitivity(),
           "kessler": kessler()}
    return res, dict(run=run, el_m=el_m, curves=curves)


def km32(x):
    return (np.asarray(x) / 1e3).astype(np.float32)


def write_data(res, data):
    run, el_m, cv = data["run"], data["el_m"], data["curves"]
    k = slice(0, int(DAY / D.DAY_STEP) + 1, LOOP_STEP)
    snaps = [int(np.searchsorted(run.t_months / DAY, s))
             for s in SNAP_DAYS]
    paths = np.stack([orbit_paths(run.months[:, j, :3],
                                  run.months[:, j, 3:])
                      for j in snaps])
    raan = np.degrees(np.unwrap(el_m["raan"], axis=1))
    arrays = {
        "t_day_h": run.t_day[k] / 3600,
        "day_xyz_km": km32(run.day[:, k, :3]),
        "day_intact_xyz_km": km32(run.day_parent[0, k, :3]),
        "day_along_spread90_deg": np.degrees(cv["along90"][k]),
        "t_months_d": run.t_months / DAY,
        "months_xyz_km": km32(run.months[..., :3]),
        "months_node_deg": raan - raan[:, :1],
        "months_node_spread90_deg": np.degrees(cv["nodes90"]),
        "months_node_spread50_deg": np.degrees(cv["nodes50"]),
        "snap_days": np.array(SNAP_DAYS, float),
        "snap_orbits_km": km32(paths),
        "breakup_xyz_km": km32(run.day[0, 0, :3]),
        "one_orbit_t_min": np.array(res["ring"]["stream_t_min"]),
        "one_orbit_along_deg": np.degrees(cv["one_orbit_along"]),
        "piece_size_m": run.cloud.lc,
        "piece_kick_ms": np.linalg.norm(run.cloud.dv, axis=1),
    }
    np.savez_compressed(OUT / "cloud_loop.npz", **arrays)
    (OUT / "cloud_loop.json").write_text(json.dumps(clean(
        sidecar(arrays, res)), indent=1))


def sidecar(arrays, res):
    desc = {
        "t_day_h": ("h", "time axis of day_*: 0..24 h, step 2 "
                    "min (t = 0 is the breakup)"),
        "day_xyz_km": ("km", "every piece [piece, t_day_h, xyz]"),
        "day_intact_xyz_km": ("km", "where the satellite would "
                              "be had it not broken up "
                              "[t_day_h, xyz]"),
        "day_along_spread90_deg": ("deg", "spread along the "
                                   "orbit of the middle 90 % of "
                                   "pieces; 360 = full ring"),
        "t_months_d": ("days", "time axis of months_*: 0..365, "
                       "daily"),
        "months_xyz_km": ("km", "every piece once a day [piece, "
                          "t_months_d, xyz]; phases look random "
                          "(daily samples of 15 orbits/day)"),
        "months_node_deg": ("deg", "how far each orbit has "
                            "turned about the spin axis since "
                            "the breakup (unwrapped, negative = "
                            "west) [piece, t_months_d]"),
        "months_node_spread90_deg": ("deg", "spread of that turn"
                                     " over the middle 90 %; 360 "
                                     "= shell"),
        "months_node_spread50_deg": ("deg", "same, middle 50 %"),
        "snap_days": ("days", "days of the snap_orbits_km "
                      "snapshots"),
        "snap_orbits_km": ("km", "each piece's current orbit "
                           "(two-body ellipse through its state)"
                           " at each snapshot [snap, piece, 96 "
                           "points, xyz]: draw as rings"),
        "breakup_xyz_km": ("km", "the breakup point (over the "
                           "equator, on +x)"),
        "one_orbit_t_min": ("min", "time of the C14 stream "
                            "measure: first 60 s sample after one "
                            "Kepler period (97 min; not a 2-min "
                            "loop frame)"),
        "one_orbit_along_deg": ("deg", "C14: each piece's "
                                "position angle along its orbit "
                                "(from its node, u) minus the "
                                "intact orbit's, at one_orbit_t_min"
                                " [piece]; middle 90 % spans 60.2 "
                                "deg = 7,338 km at 6,978 km radius"),
        "piece_size_m": ("m", "each piece's size (11 cm to 1 m)"),
        "piece_kick_ms": ("m/s", "each piece's kick speed"),
    }
    return {
        "frame": "Earth-centred inertial (not rotating with "
                 "Earth): z = Earth's spin axis (north), x = "
                 "towards the breakup point, which is over the "
                 "equator; the orbit heads north-east from there",
        "units": "km for positions, hours (day_*) or days "
                 "(months_*) for time",
        "time_base": "t = 0 is the breakup for every array. "
                     "day_* and months_* are two windows of the "
                     "same cloud: stitch the loop as day "
                     "(clump -> ring) then months (ring -> shell)",
        "model": "Earth gravity + J2, scipy DOP853 rtol 1e-11; "
                 "NASA Standard Breakup Model pieces (11 cm to 1 m)"
                 "; no air drag, Sun, Moon or collisions",
        "assumptions": "600 km circular (film title card); tilt "
                       "51.6 deg (the ISS's, assumed); 90 of the "
                       "590 pieces dip below 200 km and fall "
                       "back, not shown",
        "earth": {"equatorial_radius_km": R_EARTH_EQ / 1e3,
                  "polar_radius_km": R_EARTH_EQ * (1 - F_EARTH)
                  / 1e3,
                  "spin_deg_per_day": np.degrees(OMEGA_EARTH)
                  * DAY,
                  "note": "to spin the globe under the inertial "
                          "frame, rotate it by spin_deg_per_day "
                          "about z; its angle at t = 0 is free"},
        "markers": {"ring_h": res["ring"]["ring_h"],
                    "half_shell_days":
                        res["shell"]["half_shell_days"],
                    "shell_days": res["shell"]["shell_days"]},
        "caption_hint": "time sped up: a ring in about half a "
                        "day, a shell in about 8 months",
        "arrays": {k: {"shape": list(np.shape(v)),
                       "units": desc[k][0], "what": desc[k][1]}
                   for k, v in arrays.items()},
    }


def lookup(res, key):
    for part in _parts(key):
        res = res[part]
    return res


def _parts(key):
    """Split a key path; numeric parts like 28.5 stay."""
    out = []
    for p in key.split("."):
        if out and out[-1].isdigit() and p.isdigit():
            out[-1] += "." + p
        else:
            out.append(p)
    return out


def check(res):
    bad = []
    for key, (want, tol) in EXPECTED.items():
        got = lookup(res, key)
        ok = got == want if tol is None else abs(got - want) <= tol
        if not ok:
            bad.append(f"{key}: got {got!r}, want {want!r} +- {tol}")
    return bad


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def report(res):
    o, s, r, h = (res[k] for k in ("orbit", "speeds", "ring",
                                   "shell"))
    print(f"  600 km: {o['period_min']:.2f} min, "
          f"{o['speed_kms']:.3f} km/s; head-on "
          f"{s['head_on_kms']:.2f} km/s")
    b = res["breakup"]
    print(f"  breakup: {b['n_drawn']} pieces, {b['n_fell']} fall,"
          f" {b['n_pieces']} stay; median kick "
          f"{b['kick_median_ms']:.0f} m/s")
    print(f"  ring after {r['ring_h']:.1f} h (Kepler "
          f"{r['kepler_ring_h']:.1f} h, no J2 "
          f"{r['no_j2_ring_h']:.1f} h)")
    print(f"  nodes fan {h['spread90_deg_day']:.2f} deg/day; "
          f"half-shell {h['half_shell_days']:.0f} d, shell "
          f"{h['shell_days']:.0f} d")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res, data = results()
    res = clean(res)
    write_data(res, data)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    report(res)
    print(f"wrote {OUT}")
    if args.check:
        bad = check(res)
        for b in bad:
            print("FAIL", b)
        print("CHECK FAILED" if bad else "CHECK OK")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
