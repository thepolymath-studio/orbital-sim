"""2026-10-wk1 The Martian, Sim post: the race back to Mars.

Hermes is inbound past Earth (t = 0, the Earth encounter) and must
get back to Mars. Three strategies, one integrator, one clock:

  01 Hohmann : stop at Earth, wait for the window (eq. 04), two burns
  02 Thrust  : ion drive only (2 mm/s^2), primer-vector steering
  03 Purnell : Earth flyby (eq. 08, 09) plus the same ion drive

Every arrival is a rendezvous with Mars (position and velocity).
The nominal Mars phasing is the planets' mean longitudes on the
novel's Earth-flyby date; the sweep repeats the race for 24 start
phases, 15 deg apart, to show how robust the ranking is.

    python examples/2026-10-wk1_race.py [--check] [--workers N]
    python examples/2026-10-wk1_race.py --extras [--check]

The first writes outputs/2026-10-wk1-sim/: per-strategy CSVs on one
shared time base, planets.csv, flyby_earth_frame.csv, sweep.csv,
race.npz and the race.json sidecar (the visualizer's contract).
--extras writes race_extras.json: the perihelion-floor sensitivity
and the direct-law optimality gap at the nominal phase. It needs the
outputs of the first command.
--check asserts the numbers quoted in
posts/2026-10-wk1-the-martian-sim/code-results.md.

Day counts are banned on slides (editorial/banned.yaml); they are
only in the sidecar, under SIDECAR_ONLY_not_for_slides.
"""
from __future__ import annotations

import os

# One BLAS thread per worker process: the work is parallel across
# processes, and this keeps linear algebra bit-for-bit repeatable.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from orbital_sim import AU, DAY, fly, no_thrust
from orbital_sim.constants import MU_EARTH, R_EARTH
from orbital_sim.flyby import hyperbola_state, true_anomaly_at
from orbital_sim.constants import (MU_SUN, R_EARTH_ORBIT,
                                   R_MARS_ORBIT)
from orbital_sim.ephemeris import mean_motion
from orbital_sim.hohmann import hohmann
from orbital_sim import race, search
from orbital_sim.lowthrust import direct_time
from orbital_sim.primer import T_UNIT

OUT = Path(__file__).resolve().parents[1] / "outputs" / "2026-10-wk1-sim"
STEP_DEG = 15.0
N_PHASES = 24
DT = 1.0 * DAY                  # shared time base step
T_BEFORE = 30.0 * DAY           # approach shown before t = 0
T_AFTER = 20.0 * DAY            # all ships riding with Mars
NAMES = ("01", "02", "03")
LABELS = {"01": "01 Hohmann", "02": "02 Continuous thrust",
          "03": "03 Purnell (Earth flyby + ion drive)"}
PHASES = {0: "approach (shared inbound coast, t < 0)",
          1: "waiting on Earth's orbit for the window (01)",
          2: "coasting on the Hohmann ellipse (01)",
          3: "thrusting, ion drive on (02, 03)",
          4: "arrived: riding with Mars"}

# Values quoted in code-results.md (C#), checked by --check.
# key: (expected, absolute tolerance)
EXPECTED = {
    "nominal.mars_lead_deg": (219.63, 0.01),
    "nominal.hohmann_alpha_deg": (44.35, 0.01),
    "nominal.ratio_01_over_03": (2.72, 0.01),
    "nominal.ratio_02_over_03": (1.04, 0.01),
    "nominal.ratio_01_over_02": (2.61, 0.01),
    "nominal.dv_kms.01": (11.59, 0.01),
    "nominal.dv_kms.02": (42.25, 0.05),
    "nominal.dv_kms.03": (40.52, 0.15),
    "nominal.flyby.delta_max_deg": (77.2, 0.05),
    "nominal.flyby.e_at_300km": (1.60, 0.005),
    "robustness.rank_03_02_01_count": (24, 0),
    "robustness.ratio_01_over_03_min": (1.50, 0.01),
    "robustness.ratio_02_over_03_max": (1.62, 0.01),
}
EXPECTED_EXTRAS = {
    "direct_gap_pct.8": (0.93, 0.1),
    "phases_floor_matters": (8, 0),
    "ratio_01_over_03_min_floor_free": (1.86, 0.01),
    "ratio_02_over_03_max_floor_free": (1.62, 0.01),
    "nominal_no_floor.ratio_02_over_03": (1.12, 0.01),
}
DIRECT_SEGMENTS = (8,)


# ---------------------------------------------------------------- race
def phase_leads(sc0):
    """Start phases: nominal + k * 15 deg, k = 0 .. 23."""
    k = np.arange(N_PHASES)
    return (sc0.mars_lead + np.radians(STEP_DEG) * k) % (2 * np.pi)


def summarise(task):
    """Fly all three strategies at one phase; plain numbers only
    (picklable, for the process pool)."""
    sc, X02, psi03, X03 = task
    res = {"01": race.race_hohmann(sc)}
    if X02 is not None:
        res["02"] = race.race_lowthrust(sc, X02, 0.0, LABELS["02"])
    if X03 is not None and psi03 is not None:
        res["03"] = race.race_lowthrust(sc, X03, psi03, LABELS["03"])
    out = {}
    for n, r in res.items():
        out[n] = {"t_arrive": r.t_arrive, "dv": r.dv,
                  "miss_r": r.miss_r, "miss_v": r.miss_v,
                  "info": r.info}
    return out


def run_race(sc0, pmap):
    """The sweep over start phases; returns per-phase rows and the
    solutions at the nominal phase."""
    leads = phase_leads(sc0)
    best, psis = search.sweep(sc0, leads, search.PSI_FRACS, pmap,
                              search.SEED_FRACS)
    tasks = [(sc0, l, psis, [best[search._key((l, p))]
                             for p in psis]) for l in leads]
    refined = list(pmap(search.refine_psi, tasks))
    jobs, sols = [], []
    for l, (psi3, X3) in zip(leads, refined):
        X2 = best[search._key((l, 0.0))]
        sc = replace(sc0, mars_lead=float(l))
        jobs.append((sc, X2, psi3, X3))
        sols.append((X2, psi3, X3))
    rows = list(pmap(summarise, jobs))
    return leads, rows, sols


NO_FLOOR = 0.1 * AU    # reference floor: only a sun-grazer is cut


def nofloor_ring(sc_nf, leads, X02, pmap):
    """02 with the floor lowered to 0.1 AU, continued from the
    nominal phase around the circle (both ways)."""
    ring = np.unwrap(np.append(leads, leads[0]))
    path = [(l, 0.0) for l in ring]
    best = {search._key(path[0]): X02}
    for found in pmap(search.chain, [(sc_nf, path, X02),
                                     (sc_nf, path[::-1], X02)]):
        for k, X in found.items():
            best[k] = search.better(best.get(k), X)
    jobs = [(replace(sc_nf, mars_lead=float(l)),
             best.get(search._key((l, 0.0))), None, None)
            for l in leads]
    return [r.get("02") for r in pmap(summarise, jobs)]


def ranking(row):
    """Strategy names ordered by arrival time."""
    return sorted(row, key=lambda n: row[n]["t_arrive"])


def robustness(leads, rows):
    """How the ranking and ratios vary with the start phase."""
    ranks = ["-".join(ranking(r)) for r in rows]
    T = {n: np.array([r[n]["t_arrive"] for r in rows])
         for n in NAMES}
    r13, r23 = T["01"] / T["03"], T["02"] / T["03"]
    gain = 100 * (1 - 1 / r23)
    floor = [r["02"]["info"]["floor_binds"] or
             r["03"]["info"]["floor_binds"] for r in rows]
    return {
        "n_phases": len(rows), "step_deg": STEP_DEG,
        "rank_counts": {k: ranks.count(k) for k in set(ranks)},
        "rank_03_02_01_count": ranks.count("03-02-01"),
        "ratio_01_over_03_min": float(r13.min()),
        "ratio_01_over_03_max": float(r13.max()),
        "ratio_01_over_03_median": float(np.median(r13)),
        "ratio_02_over_03_min": float(r23.min()),
        "ratio_02_over_03_max": float(r23.max()),
        "ratio_02_over_03_median": float(np.median(r23)),
        "flyby_gain_pct_min": float(gain.min()),
        "flyby_gain_pct_max": float(gain.max()),
        "phases_flyby_gain_over_1pct": int((gain > 1).sum()),
        "phases_floor_binds": int(sum(floor)),
        "phases_floor_binds_leads_deg": [
            round(float(np.degrees(l)), 2)
            for l, f in zip(leads, floor) if f],
        "phase_of_min_ratio_01_over_03_deg": float(
            np.degrees(leads[int(np.argmin(r13))])),
        "phase_of_max_ratio_02_over_03_deg": float(
            np.degrees(leads[int(np.argmax(r23))])),
    }


def nominal_numbers(sc, row):
    """Quotable numbers at the nominal phase (no times)."""
    T = {n: row[n]["t_arrive"] for n in NAMES}
    _, _, t_H = hohmann(R_EARTH_ORBIT, R_MARS_ORBIT)
    alpha = np.pi - mean_motion(R_MARS_ORBIT) * t_H
    f3 = row["03"]["info"]
    return {
        "mars_lead_deg": float(np.degrees(sc.mars_lead)),
        "hohmann_alpha_deg": float(np.degrees(alpha)),
        "ranking": ranking(row),
        "ratio_01_over_03": T["01"] / T["03"],
        "ratio_02_over_03": T["02"] / T["03"],
        "ratio_01_over_02": T["01"] / T["02"],
        "dv_kms": {n: row[n]["dv"] / 1e3 for n in NAMES},
        "dv_ratio_02_over_01": row["02"]["dv"] / row["01"]["dv"],
        "miss_km": {n: row[n]["miss_r"] / 1e3 for n in NAMES},
        "miss_mm_s": {n: row[n]["miss_v"] * 1e3 for n in NAMES},
        "flyby": {
            "v_inf_kms": sc.v_inf / 1e3,
            "delta_max_deg": float(np.degrees(race.turn_limit(sc))),
            "e_at_300km": 1 + (R_EARTH + sc.alt_min) * sc.v_inf ** 2
            / MU_EARTH,
            "psi_deg": float(np.degrees(f3["psi"])),
            "periapsis_alt_km": f3.get("alt_p", np.nan) / 1e3,
            "e": f3.get("e", np.nan),
            "dv_helio_kms": f3.get("dv_flyby", np.nan) / 1e3,
            "transversality_sin": f3["transversality"],
        },
        "perihelion_au": {n: row[n]["info"]["r_min"] / AU
                          for n in ("02", "03")},
        "floor_binds": {n: row[n]["info"]["floor_binds"]
                        for n in ("02", "03")},
    }


# --------------------------------------------------- shared time base
def approach(sc):
    """The shared inbound coast, t in [-T_BEFORE, 0]."""
    y0 = np.append(race.inbound_state(sc.v_inf), 0.0)
    return fly(y0, (0.0, -T_BEFORE), no_thrust).sol


def ship_table(res, t, appr, sc):
    """One strategy on the shared clock, columns = SHIP_COLUMNS."""
    rows = []
    kicks = sorted(res.impulses)
    for ti in t:
        rows.append(ship_row(res, ti, appr, sc, kicks))
    return np.array(rows)


def ship_row(res, ti, appr, sc, kicks):
    """[t_day, x, y, vx, vy, phase, thrust_on, ux, uy, dv]."""
    u = (0.0, 0.0)
    if ti < 0:
        y, ph, dv = appr(ti), 0, 0.0
    elif ti >= res.t_arrive:
        r, v = race.mars(sc.mars_lead)(ti)
        y, ph, dv = np.r_[r, v], 4, res.dv
    else:
        leg = next(lg for lg in res.legs if lg.t0 <= ti < lg.t1)
        y = leg.sol(ti)
        ph = {"wait": 1, "transfer": 2, "thrust": 3}[leg.phase]
        if leg.thrust is not None:
            a = leg.thrust(ti, y[:2], y[2:4])
            u = a / np.linalg.norm(a)
            dv = y[4]
        else:
            dv = sum(k[1] for k in kicks if k[0] <= ti)
    return [ti / DAY, y[0] / AU, y[1] / AU, y[2] / 1e3, y[3] / 1e3,
            ph, int(ph == 3), u[0], u[1], dv / 1e3]


SHIP_COLUMNS = [
    ("t_day", "d", "shared clock; t = 0 is the Earth encounter"),
    ("x_au", "AU", "heliocentric x"),
    ("y_au", "AU", "heliocentric y"),
    ("vx_kms", "km/s", "heliocentric velocity x"),
    ("vy_kms", "km/s", "heliocentric velocity y"),
    ("phase", "code", "see phase_codes"),
    ("thrust_on", "0/1", "1 = ion drive on"),
    ("thrust_ux", "-", "thrust direction, unit x (0 when off)"),
    ("thrust_uy", "-", "thrust direction, unit y (0 when off)"),
    ("dv_kms", "km/s", "delta-v used so far (impulses + a_th t)"),
]
PLANET_COLUMNS = [
    ("t_day", "d", "shared clock"),
    ("earth_x_au", "AU", ""), ("earth_y_au", "AU", ""),
    ("mars_x_au", "AU", ""), ("mars_y_au", "AU", ""),
]
FLYBY_COLUMNS = [
    ("x_re", "Earth radii", "Earth-centred x (axes parallel to "
     "the heliocentric frame)"),
    ("y_re", "Earth radii", "Earth-centred y"),
    ("f_deg", "deg", "true anomaly on the hyperbola"),
]
SWEEP_COLUMNS = [
    ("mars_lead_deg", "deg", "Mars ahead of Earth at t = 0"),
    ("rank_01", "1-3", ""), ("rank_02", "1-3", ""),
    ("rank_03", "1-3", ""),
    ("ratio_01_over_03", "-", "arrival-time ratio"),
    ("ratio_02_over_03", "-", "arrival-time ratio"),
    ("dv01_kms", "km/s", ""), ("dv02_kms", "km/s", ""),
    ("dv03_kms", "km/s", ""),
    ("psi03_deg", "deg", "flyby turn chosen by 03"),
    ("floor_binds", "0/1", "02 or 03 touches the 0.387 AU floor: "
     "their times are upper bounds there"),
]


def planets_table(t, sc):
    """Earth and Mars on the shared clock."""
    re, _ = race.earth(t)
    rm, _ = race.mars(sc.mars_lead)(t)
    return np.column_stack([t / DAY, re[0] / AU, re[1] / AU,
                            rm[0] / AU, rm[1] / AU])


def flyby_table(sc, psi, r_max=30 * R_EARTH, n=241):
    """The 03 flyby hyperbola in Earth's frame (inset)."""
    v_in = race.inbound_state(sc.v_inf)[2:] - race.earth(0.0)[1]
    info = race.flyby_info(sc, psi)
    f1 = true_anomaly_at(r_max, info["e"], info["r_p"])
    rows = []
    for f in np.linspace(-f1, f1, n):
        r, _ = hyperbola_state(v_in, psi, info["r_p"], MU_EARTH, f)
        rows.append([r[0] / R_EARTH, r[1] / R_EARTH, np.degrees(f)])
    return np.array(rows)


def sweep_table(leads, rows):
    out = []
    for l, r in zip(leads, rows):
        order = ranking(r)
        T = {n: r[n]["t_arrive"] for n in NAMES}
        fb = r["02"]["info"]["floor_binds"] or \
            r["03"]["info"]["floor_binds"]
        out.append([np.degrees(l)] + [order.index(n) + 1
                                      for n in NAMES]
                   + [T["01"] / T["03"], T["02"] / T["03"]]
                   + [r[n]["dv"] / 1e3 for n in NAMES]
                   + [np.degrees(r["03"]["info"]["psi"]), int(fb)])
    return np.array(out)


def save_csv(name, arr, cols):
    np.savetxt(OUT / f"{name}.csv", arr, delimiter=",",
               header=",".join(c[0] for c in cols), comments="",
               fmt="%.10g")


def col_doc(cols):
    return [{"name": n, "unit": u, "meaning": m} for n, u, m in cols]


# ------------------------------------------------------------- sidecar
def sidecar(sc, nom, rob, rows, leads, arrays, results):
    T = {n: results[n].t_arrive for n in NAMES}
    return {
        "post": "2026-10-wk1-the-martian-sim",
        "script": "examples/2026-10-wk1_race.py",
        "model": "planar heliocentric two-body + thrust, "
                 "r'' = -mu r/r^3 + a_th (eq. 06); patched-conic "
                 "Earth flyby (eq. 08, 09); circular coplanar "
                 "Earth (1 AU) and Mars (1.5237 AU)",
        "integrator": "scipy solve_ivp DOP853, rtol 1e-12, for "
                      "every strategy (propagate.fly)",
        "frame": "heliocentric inertial, planar. Sun at origin. "
                 "Earth on +x at t = 0, planets move "
                 "counter-clockwise.",
        "time_base": f"shared clock t_day, uniform {DT / DAY:g} d "
                     f"steps from -{T_BEFORE / DAY:g} d to the last "
                     f"arrival + {T_AFTER / DAY:g} d. t = 0 is the "
                     "Earth encounter. All CSVs use the same rows. "
                     "Do NOT print t_day or any day count on a "
                     "slide (banned.yaml martian-transit-times); "
                     "show the clock without units, or as ranks.",
        "assumptions": ASSUMPTIONS,
        "parameters": {
            "mu_sun_m3_s2": MU_SUN, "mu_earth_m3_s2": MU_EARTH,
            "r_earth_m": R_EARTH, "au_m": AU,
            "a_th_m_s2": sc.a_th, "v_inf_m_s": sc.v_inf,
            "flyby_alt_min_m": sc.alt_min,
            "perihelion_floor_au": sc.r_min / AU,
            "nominal_mars_lead_deg": float(np.degrees(sc.mars_lead)),
            "rendezvous_tolerance": {"r_m": race.TOL_R,
                                     "v_m_s": race.TOL_V},
        },
        "files": {
            "ship_csvs": {n: f"ship_{n}.csv" for n in NAMES},
            "ship_columns": col_doc(SHIP_COLUMNS),
            "phase_codes": PHASES,
            "planets_csv": "planets.csv",
            "planet_columns": col_doc(PLANET_COLUMNS),
            "flyby_csv": "flyby_earth_frame.csv",
            "flyby_columns": col_doc(FLYBY_COLUMNS),
            "sweep_csv": "sweep.csv",
            "sweep_columns": col_doc(SWEEP_COLUMNS),
            "npz": "race.npz", "npz_arrays": arrays,
        },
        "strategies": LABELS,
        "events": {
            "01": [{"label": lab, "dv_kms": k[1] / 1e3,
                    "time": "see SIDECAR_ONLY_not_for_slides"}
                   for lab, k in zip(("stop", "burn 1", "burn 2"),
                                     results["01"].impulses)],
            "03_flyby": {"at": "t = 0, Earth",
                         **{k: v for k, v in nom["flyby"].items()}},
        },
        "nominal": nom,
        "robustness": rob,
        "caveats": CAVEATS,
        "SIDECAR_ONLY_not_for_slides": {
            "t_arrive_day": {n: T[n] / DAY for n in NAMES},
            "t_wait_01_day": results["01"].info["t_wait"] / DAY,
            "t_transfer_01_day": results["01"].info["t_H"] / DAY,
            "impulse_times_01_day": [k[0] / DAY for k in
                                     results["01"].impulses],
            "sweep_t_arrive_day": [
                {"mars_lead_deg": float(np.degrees(l)),
                 **{n: r[n]["t_arrive"] / DAY for n in NAMES}}
                for l, r in zip(leads, rows)],
            "novel_reference": "Burke (F21): Earth flyby 6 Jul "
                               "2036 to Mars flyby 24 May 2037 "
                               "(a flyby, not a rendezvous).",
        },
    }


ASSUMPTIONS = [
    "One start epoch t = 0 for all three: Hermes at Earth.",
    "Hermes arrives coasting in from Mars's orbit (aphelion "
    "1.5237 AU) with v_inf = 6 km/s relative to Earth, NASA "
    "Glenn's upper limit for the Earth encounter (S15): inbound, "
    "faster than Earth.",
    "Mars phasing: mean longitudes (JPL approximate elements, "
    "Standish) on the novel's Earth-flyby date, 6 Jul 2036 (F21): "
    "Mars leads Earth by 219.6 deg (it trails by 140 deg).",
    "01: impulsive stop (dv = v_inf) onto Earth's orbit, wait for "
    "alpha (eq. 04), impulsive Hohmann burns. Impulsive burns are "
    "generous to 01.",
    "02, 03: a_th = 2 mm/s^2 constant (novel, F5), always on, "
    "steered along Lawden's primer vector: Pontryagin's minimum-"
    "time law. Mass loss ignored (S7).",
    "03: instantaneous patched-conic flyby at t = 0, turn |psi| <= "
    "delta at 300 km periapsis altitude; psi chosen for the "
    "earliest arrival. No thrust during the flyby.",
    "Perihelion floor 0.387 AU (Mercury's mean distance) for "
    "thermal sanity; the Purnell route dips well inside Venus's "
    "orbit (F12).",
    "Planetary gravity ignored outside the flyby; no Earth "
    "departure or Mars capture spirals (heliocentric rendezvous).",
]
CAVEATS = [
    "The low-thrust times are the best Pontryagin extremals the "
    "search found (scan + continuation), not a global-optimality "
    "proof. A direct piecewise-linear law never beats them.",
    "Where the perihelion floor binds (sweep.csv floor_binds) or "
    "matters (race_extras.json floor_matters), 02/03 times are "
    "upper bounds: a route riding or diving below the floor would "
    "be faster. There the 01/03 ratio is a lower bound. The "
    "ranking still holds: 03 <= 02 by construction (psi = 0 is "
    "allowed), and 02 is already faster than 01.",
    "At the nominal phase the floor binds for 03: the heat limit, "
    "not the flyby, limits how far the turn can be used.",
    "03 never loses to 02 by construction; how much the flyby "
    "helps depends strongly on Mars's phase.",
    "Real Hermes thrusts with coasts, and Burke's Purnell route "
    "ends in a Mars flyby, not a rendezvous.",
]


def lookup(d, key):
    for part in key.split("."):
        d = d[part]
    return d


def check(table, expected):
    bad = []
    for key, (want, tol) in expected.items():
        got = lookup(table, key)
        if abs(got - want) > tol + 1e-12:
            bad.append(f"{key}: got {got:.6g}, expected {want}")
    return bad


def structural(nom, rows, results):
    """Model checks that must hold whatever the numbers."""
    bad = []
    for n, r in results.items():
        if r.miss_r > race.TOL_R or r.miss_v > race.TOL_V:
            bad.append(f"nominal {n}: miss {r.miss_r:.3g} m, "
                       f"{r.miss_v:.3g} m/s")
    for k, row in enumerate(rows):
        for n in NAMES:
            if n not in row:
                bad.append(f"phase {k}: no solution for {n}")
                continue
            m = row[n]
            if m["miss_r"] > race.TOL_R or m["miss_v"] > race.TOL_V:
                bad.append(f"phase {k} {n}: rendezvous missed")
        if all(n in row for n in NAMES):
            if row["03"]["t_arrive"] > row["02"]["t_arrive"] + 1:
                bad.append(f"phase {k}: 03 slower than 02")
            for n in ("02", "03"):
                if row[n]["info"]["r_min"] < 0.999 * \
                        race.nominal().r_min:
                    bad.append(f"phase {k} {n}: below the floor")
    return bad


# ---------------------------------------------------------------- main
def main_race(args, pmap):
    sc0 = race.nominal()
    leads, rows, sols = run_race(sc0, pmap)
    X2, psi3, X3 = sols[0]
    results = {"01": race.race_hohmann(sc0),
               "02": race.race_lowthrust(sc0, X2, 0.0, LABELS["02"]),
               "03": race.race_lowthrust(sc0, X3, psi3,
                                         LABELS["03"])}
    t_end = max(r.t_arrive for r in results.values()) + T_AFTER
    t = np.arange(-T_BEFORE, t_end + DT / 2, DT)
    appr = approach(sc0)
    npz, arrays = {}, {}
    for n in NAMES:
        tab = ship_table(results[n], t, appr, sc0)
        save_csv(f"ship_{n}", tab, SHIP_COLUMNS)
        npz[f"ship_{n}"] = tab
    npz["planets"] = planets_table(t, sc0)
    save_csv("planets", npz["planets"], PLANET_COLUMNS)
    npz["flyby_earth_frame"] = flyby_table(sc0, psi3)
    save_csv("flyby_earth_frame", npz["flyby_earth_frame"],
             FLYBY_COLUMNS)
    npz["sweep"] = sweep_table(leads, rows)
    save_csv("sweep", npz["sweep"], SWEEP_COLUMNS)
    for k, v in npz.items():
        arrays[k] = {"shape": list(v.shape),
                     "columns": f"same as {k}.csv"}
    np.savez(OUT / "race.npz", **npz)
    nom = nominal_numbers(sc0, rows[0])
    rob = robustness(leads, rows)
    side = sidecar(sc0, nom, rob, rows, leads, arrays, results)
    (OUT / "race.json").write_text(
        json.dumps(side, indent=1, default=_json))
    report(nom, rob)
    if args.check:
        bad = structural(nom, rows, results)
        bad += check({"nominal": nom, "robustness": rob}, EXPECTED)
        print("\n".join(bad) if bad else "CHECK PASSED")
        raise SystemExit(1 if bad else 0)


def main_extras(args, pmap):
    """Sensitivity to the perihelion floor, and the direct-law gap.
    Needs the main run's outputs (race.npz, race.json)."""
    side = json.loads((OUT / "race.json").read_text())
    sweep = np.load(OUT / "race.npz")["sweep"]
    T = side["SIDECAR_ONLY_not_for_slides"]["sweep_t_arrive_day"]
    sc0 = race.nominal()
    sc_nf = replace(sc0, r_min=NO_FLOOR)
    leads = phase_leads(sc0)
    X2f, (psi3f, X3f) = search.fastest_at(sc_nf, pmap)
    refs = nofloor_ring(sc_nf, leads, X2f, pmap)
    out = {"no_floor_au": NO_FLOOR / AU, "floor_matters": [],
           "SIDECAR_ONLY_not_for_slides": {"no_floor_02_t_day": [
               None if f is None else f["t_arrive"] / DAY
               for f in refs]}}
    # "floor free": no route touches the floor (sweep.csv) and the
    # floor-free re-run of 02 is not faster. 03 is not re-run, so a
    # hidden floor effect on 03 could only raise these 01/03 ratios.
    free13, free23 = [], []
    for k, (row, ref) in enumerate(zip(T, refs)):
        fm = ref is not None and \
            ref["t_arrive"] / DAY < 0.99 * row["02"]
        if fm:
            out["floor_matters"].append({
                "mars_lead_deg": round(row["mars_lead_deg"], 2),
                "no_floor_perihelion_au": round(
                    ref["info"]["r_min"] / AU, 3)})
        elif not sweep[k, -1]:
            free13.append(row["01"] / row["03"])
            free23.append(row["02"] / row["03"])
    out["phases_floor_matters"] = len(out["floor_matters"])
    out["phases_floor_free"] = len(free13)
    out["ratio_01_over_03_min_floor_free"] = min(free13)
    out["ratio_02_over_03_max_floor_free"] = max(free23)
    r2 = race.race_lowthrust(sc_nf, X2f, 0.0, "02")
    r3 = race.race_lowthrust(sc_nf, X3f, psi3f, "03")
    out["nominal_no_floor"] = {
        "ratio_02_over_03": r2.t_arrive / r3.t_arrive,
        "psi03_deg": float(np.degrees(psi3f)),
        "perihelion_au": {"02": r2.info["r_min"] / AU,
                          "03": r3.info["r_min"] / AU},
        "alt_p_km": r3.info.get("alt_p", np.nan) / 1e3}
    out["direct_gap_pct"] = {}
    X2 = search.seeded((sc0, sc0.mars_lead, 0.0))
    ship, target = race.problem(sc0, sc0.mars_lead, 0.0)
    T2 = X2[3] * T_UNIT
    for n in DIRECT_SEGMENTS:
        d = direct_time(ship, target, X2[:3], T2, n)
        out["direct_gap_pct"][str(n)] = 100 * (d.T / T2 - 1)
        out[f"direct_feasible_{n}"] = d.feasible
        out[f"direct_miss_{n}"] = d.miss
    (OUT / "race_extras.json").write_text(
        json.dumps(out, indent=1, default=_json))
    print(json.dumps(out, indent=1, default=_json))
    if args.check:
        bad = check(out, EXPECTED_EXTRAS)
        bad += [f"direct law {n} not feasible"
                for n in DIRECT_SEGMENTS
                if not out[f"direct_feasible_{n}"]]
        bad += [f"direct law {n} beat the primer optimum"
                for n in DIRECT_SEGMENTS
                if out["direct_gap_pct"][str(n)] < -1e-3]
        print("\n".join(bad) if bad else "CHECK PASSED")
        raise SystemExit(1 if bad else 0)


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def report(nom, rob):
    print(f"Mars lead {nom['mars_lead_deg']:.2f} deg, ranking "
          f"{' < '.join(nom['ranking'])}")
    print(f"ratios 01/03 {nom['ratio_01_over_03']:.3f}  02/03 "
          f"{nom['ratio_02_over_03']:.3f}  01/02 "
          f"{nom['ratio_01_over_02']:.3f}")
    print("dv km/s", {k: round(v, 3)
                      for k, v in nom["dv_kms"].items()})
    print("miss km", nom["miss_km"], "mm/s", nom["miss_mm_s"])
    print("flyby", nom["flyby"])
    print("robustness", json.dumps(rob, default=_json))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--extras", action="store_true")
    ap.add_argument("--workers", type=int,
                    default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        if args.extras:
            main_extras(args, ex.map)
        else:
            main_race(args, ex.map)


if __name__ == "__main__":
    main()
