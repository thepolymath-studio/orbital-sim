"""2026-10-wk1 The Martian: option 02, continuous thrust (eq. 06, 07).

Hermes-like constant thrust a_T = 2 mm/s^2 (Weir's novel), prograde,
from Earth's circular orbit (1 AU) until r = Mars's orbital radius
(1.5237 AU). Two variants:

  thrust_to_mars : engine on the whole way
  burn_coast     : 40 d burn, 40 d coast, burn to Mars's radius.
                   ILLUSTRATIVE: not Burke's or Weir's schedule.

Writes outputs/2026-10-wk1/{thrust_to_mars,burn_coast}.csv, plus
continuous_thrust.npz and the continuous_thrust.json sidecar.

    python examples/2026-10-wk1_continuous_thrust.py [--check]

--check also asserts the derived numbers against the values quoted
in posts/2026-10-wk1-the-martian/code-results.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from orbital_sim import (A_HERMES, AU, DAY, HOUR, MU_SUN,
                         R_EARTH_ORBIT, R_MARS_ORBIT, hohmann, sample)
from orbital_sim.scenarios import BURN_COAST, THRUST_TO_MARS, to_mars

OUT = Path(__file__).resolve().parents[1] / "outputs" / "2026-10-wk1"
DT = 0.25 * DAY
COLUMNS = [
    ("t_s", "s", "time since departure"),
    ("t_day", "d", "time since departure"),
    ("x_au", "AU", "heliocentric x (x axis through departure)"),
    ("y_au", "AU", "heliocentric y (direction of motion at t=0)"),
    ("vx_kms", "km/s", "heliocentric velocity x"),
    ("vy_kms", "km/s", "heliocentric velocity y"),
    ("r_au", "AU", "distance from the Sun"),
    ("thrust_on", "0/1", "1 = engine on (a_T along v), 0 = coast"),
    ("dv_kms", "km/s", "accumulated thrust dv = a_T * burn time"),
]
VARIANTS = {"thrust_to_mars": THRUST_TO_MARS, "burn_coast": BURN_COAST}

# Values quoted in code-results.md (C#), checked by --check.
EXPECTED = {
    "rate_m_s_per_hour": (7.2, 1e-12),
    "hohmann_dv1_kms": (2.945, 5e-4),
    "hohmann_dv2_kms": (2.649, 5e-4),
    "hohmann_sum_kms": (5.594, 5e-4),
    "g_sun_1au_mm_s2": (5.930, 5e-4),
    "a_over_g_1au": (0.337, 5e-4),
    "thrust_to_mars.dv_kms": (17.42, 5e-3),
    "thrust_to_mars.sweep_deg": (100.9, 0.05),
    "thrust_to_mars.v_arrive_kms": (39.06, 5e-3),
    "thrust_to_mars.v_circ_mars_kms": (24.13, 5e-3),
    "thrust_to_mars.v_escape_sun_at_mars_kms": (34.12, 5e-3),
    "burn_coast.dv_kms": (13.14, 5e-3),
    "burn_coast.sweep_deg": (107.7, 0.05),
    "burn_coast.v_arrive_kms": (33.78, 5e-3),
}


def table(segs):
    """Uniform samples as a (n, 9) array in COLUMNS order."""
    t, Y, on = sample(segs, DT)
    r = np.hypot(Y[0], Y[1])
    return np.column_stack([
        t, t / DAY, Y[0] / AU, Y[1] / AU, Y[2] / 1e3, Y[3] / 1e3,
        r / AU, on.astype(int), Y[4] / 1e3,
    ])


def polar(y):
    """(r, angle deg, speed, v_radial, v_tangential), SI."""
    r = np.hypot(y[0], y[1])
    ang = np.degrees(np.arctan2(y[1], y[0])) % 360
    v_r = (y[0] * y[2] + y[1] * y[3]) / r
    v_t = (y[0] * y[3] - y[1] * y[2]) / r
    return r, ang, np.hypot(y[2], y[3]), v_r, v_t


def arcs_summary(segs):
    """Per-arc switch times and positions."""
    out = []
    for s in segs:
        r0, a0, *_ = polar(s.sol.y[:, 0])
        r1, a1, *_ = polar(s.sol.y[:, -1])
        out.append({
            "thrust_on": s.thrust_on,
            "t0_day": s.t0 / DAY, "t1_day": s.t1 / DAY,
            "r0_au": r0 / AU, "r1_au": r1 / AU,
            "angle0_deg": a0, "angle1_deg": a1,
        })
    return out


def derived(segs):
    """Numbers derived from one variant."""
    y = segs[-1].sol.y[:, -1]
    r, ang, v, v_r, v_t = polar(y)
    burn = sum(s.t1 - s.t0 for s in segs if s.thrust_on)
    return {
        "dv_kms": y[4] / 1e3,
        "dv_equals_aT_t": bool(np.isclose(y[4], A_HERMES * burn,
                                          rtol=1e-10)),
        "sweep_deg": ang,
        "r_arrive_au": r / AU,
        "v_arrive_kms": v / 1e3,
        "v_radial_arrive_kms": v_r / 1e3,
        "v_tangential_arrive_kms": v_t / 1e3,
        "v_circ_mars_kms": np.sqrt(MU_SUN / r) / 1e3,
        "v_escape_sun_at_mars_kms": np.sqrt(2 * MU_SUN / r) / 1e3,
        "bound_to_sun_at_arrival": bool(v ** 2 < 2 * MU_SUN / r),
        "SIDECAR_ONLY_not_for_slides": {
            "t_arrive_day": segs[-1].t1 / DAY,
            "burn_time_day": burn / DAY,
        },
        "arcs": arcs_summary(segs),
    }


def global_numbers():
    """Scenario-independent numbers the slides may quote."""
    dv1, dv2, _ = hohmann(R_EARTH_ORBIT, R_MARS_ORBIT)
    g_1au = MU_SUN / R_EARTH_ORBIT ** 2
    return {
        "rate_m_s_per_hour": A_HERMES * HOUR,
        "g_sun_1au_mm_s2": g_1au * 1e3,
        "a_over_g_1au": A_HERMES / g_1au,
        "hohmann_dv1_kms": dv1 / 1e3,
        "hohmann_dv2_kms": dv2 / 1e3,
        "hohmann_sum_kms": (dv1 + dv2) / 1e3,
    }


def sidecar(results, arrays):
    """The data contract with the visualizer."""
    return {
        "post": "2026-10-wk1-the-martian",
        "script": "examples/2026-10-wk1_continuous_thrust.py",
        "model": "planar two-body, r'' = -mu r/r^3 + a_T (eq. 06)",
        "integrator": "scipy solve_ivp DOP853, rtol 1e-12, "
                      "piecewise per burn/coast arc",
        "frame": "heliocentric inertial, planar (ecliptic "
                 "idealisation). Sun at origin. +x through the "
                 "departure point, motion counter-clockwise.",
        "time_base": f"t from departure, uniform step "
                     f"{DT / DAY} d; last row is the exact stop",
        "stop": "r reaches Mars's orbital radius; no Mars "
                "phasing, capture or speed matching",
        "parameters": {
            "mu_sun_m3_s2": MU_SUN, "au_m": AU,
            "a_T_m_s2": A_HERMES, "thrust_direction": "prograde",
            "r_earth_orbit_au": R_EARTH_ORBIT / AU,
            "r_mars_orbit_au": R_MARS_ORBIT / AU,
            "burn_coast_schedule": "40 d burn, 40 d coast, burn "
                                   "to Mars radius (ILLUSTRATIVE, "
                                   "not Burke's/Weir's schedule)",
        },
        "csv_columns": [
            {"name": n, "unit": u, "meaning": m}
            for n, u, m in COLUMNS
        ],
        "npz_arrays": arrays,
        "caveats": [
            "Constant a_T: mass loss and engine limits ignored "
            "(S7).",
            "Earth's and Mars's gravity ignored; circular "
            "coplanar planet orbits.",
            "Prograde thrust all the way arrives fast and "
            "unbound from the Sun; the novel's Hermes flips and "
            "brakes. The runs show the path shape, not the "
            "mission.",
            "Transit and burn times are banned on slides "
            "(editorial/banned.yaml); they are in this sidecar "
            "for the Sim post and the verifier only.",
        ],
        "globals": global_numbers(),
        "variants": results,
    }


def lookup(side, key):
    node = side["globals"] if "." not in key else side["variants"]
    for part in key.split("."):
        node = node[part]
    return node


def check(side):
    """Assert quoted values; return a list of failures."""
    bad = []
    for key, (want, tol) in EXPECTED.items():
        got = lookup(side, key)
        if abs(got - want) > tol:
            bad.append(f"{key}: got {got:.6g}, expected {want}")
    for name, res in side["variants"].items():
        if not res["dv_equals_aT_t"]:
            bad.append(f"{name}: dv != a_T t")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results, npz, arrays = {}, {}, {}
    for name, arcs in VARIANTS.items():
        segs = to_mars(arcs)
        tab = table(segs)
        head = ",".join(c[0] for c in COLUMNS)
        np.savetxt(OUT / f"{name}.csv", tab, delimiter=",",
                   header=head, comments="", fmt="%.10g")
        npz[name] = tab
        arrays[name] = {"shape": list(tab.shape),
                        "columns": "same as csv_columns"}
        results[name] = derived(segs)
    np.savez(OUT / "continuous_thrust.npz", **npz)
    side = sidecar(results, arrays)
    text = json.dumps(side, indent=1, default=float)
    (OUT / "continuous_thrust.json").write_text(text)
    g = side["globals"]
    for k, v in g.items():
        print(f"{k:24s} {v:.6g}")
    for name, res in results.items():
        print(f"{name}: dv {res['dv_kms']:.3f} km/s, "
              f"sweep {res['sweep_deg']:.2f} deg, "
              f"v_arr {res['v_arrive_kms']:.3f} km/s")
    if args.check:
        bad = check(side)
        print("\n".join(bad) if bad else "CHECK PASSED")
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
