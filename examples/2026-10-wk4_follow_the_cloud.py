"""2026-10 wk4 Sim post (Gravity): follow the film's cloud.

The concept post's cloud (orbital_sim.debris): a satellite on a
circular orbit 600 km up breaks into 500 pieces that stay in orbit
(NASA Standard Breakup Model), flown under Earth's gravity + J2.
Assumed tilt 51.6 deg (the ISS's; the film does not say).

Step 1. Does it come back every 90 minutes, like in the film? Each
   piece's return to the breakup spot (the point on its orbit where
   it started) on passes 1, 2, 5, 10 and 15 (15 = about a day):
   when, and how tightly bunched.
Step 2. How fast does it smear into a ring, then a shell? The
   spread along the orbit over two days (with and without J2), then
   the turning of the orbits over a year, checked piece by piece
   against the closed form dOmega/dt = -(3/2) n J2 (R/p)^2 cos i.

Writes outputs/2026-10-wk4-sim/: follow_cloud.npz +
follow_cloud.json (sidecar) and results.json.

    python examples/2026-10-wk4_follow_the_cloud.py [--check]

--check asserts the numbers in posts/2026-10-wk4-gravity-sim/
code-results.md (exit 1 on a mismatch). About 2.5 minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from orbital_sim import debris as D
from orbital_sim.cloud import middle_range, read_elements
from orbital_sim.constants import DAY, R_EARTH_EQ

OUT = (Path(__file__).resolve().parents[1] / "outputs"
       / "2026-10-wk4-sim")
ALL_PASSES = tuple(range(1, 16))
FRAME_STEP = 2                   # day frames: every 2 min

# Values quoted in code-results.md, checked by --check: (value, tol)
EXPECTED = {
    "cloud.n_pieces": (500, 0),
    "cloud.n_drawn": (590, 0),
    "cloud.n_fell": (90, 0),
    "cloud.kick_median_ms": (88, 0.5),
    "orbit.kepler_period_min": (96.69, 0.005),
    "orbit.intact_return_min": (96.49, 0.006),
    "passes.1.median_return_min": (96.67, 0.005),
    "passes.1.window50_min": (3.1, 0.05),
    "passes.1.window90_min": (16.2, 0.05),
    "passes.1.share_within_1min": (0.432, 0.0005),
    "passes.1.miss_median_km": (37, 0.5),
    "passes.2.window50_min": (6.2, 0.05),
    "passes.2.window90_min": (32.4, 0.05),
    "passes.2.share_within_1min": (0.292, 0.0005),
    "passes.5.window50_min": (15.5, 0.05),
    "passes.5.window90_min": (80.9, 0.05),
    "passes.5.share_within_1min": (0.154, 0.0005),
    "passes.10.window50_min": (31.0, 0.05),
    "passes.10.window90_min": (161.8, 0.05),
    "passes.10.share_within_1min": (0.066, 0.0005),
    "passes.15.intact_return_h": (24.12, 0.005),
    "passes.15.window50_min": (46.5, 0.05),
    "passes.15.window90_min": (242.7, 0.05),
    "passes.15.share_within_1min": (0.048, 0.0005),
    "passes.15.miss_median_km": (554, 0.5),
    "ring.ring_h": (10.6, 0.05),
    "ring.no_j2_ring_h": (10.7, 0.05),
    "ring.kepler_ring_h": (10.6, 0.05),
    "shell.half_shell_days": (125, 0),
    "shell.shell_days": (250, 0),
    "shell.spread50_deg_at_365": (116, 0.5),
    "precession.intact_sim_deg_day": (-4.537, 0.0005),
    "precession.intact_closed_form_deg_day": (-4.533, 0.0005),
    "precession.cloud_median_deg_day": (-4.52, 0.005),
    "precession.spread90_sim_deg_day": (1.444, 0.0005),
    "precession.spread90_closed_form_deg_day": (1.443, 0.0005),
    "precession.max_rel_error": (0.0018, 0.00005),
    "precession.median_rel_error": (0.0008, 0.00005),
}


def cloud_numbers(c):
    dv = np.linalg.norm(c.dv, axis=1)
    return {"n_pieces": len(c.lc), "n_drawn": c.n_drawn,
            "n_fell": c.n_fell, "kick_median_ms": np.median(dv),
            "tilt_deg": np.degrees(D.INC), "alt_km": D.ALT / 1e3,
            "seed": D.SEED}


def step1(run, el, elp):
    """Returns to the breakup spot, passes 1..15."""
    pn = D.pass_numbers(run.t_day, el, elp, run.day, ALL_PASSES)
    for k, p in pn.items():
        p["intact_return_h"] = p["parent_return_min"] / 60
    T = D.orbit_numbers()["period_min"]
    orbit = {"kepler_period_min": T,
             "intact_return_min": pn[1]["parent_return_min"]}
    return orbit, pn


def step2(run, el, elp, el_m):
    rn = D.ring_numbers(run.t_day, el, elp)
    rn0 = D.ring_numbers(run.t_day,
                         read_elements(run.day_no_j2),
                         read_elements(run.day_parent_no_j2))
    ring = {"ring_h": rn["ring_h"], "no_j2_ring_h": rn0["ring_h"],
            "kepler_ring_h": D.kepler_ring_h(run.cloud),
            "head_meets_tail_h": rn["head_meets_tail_h"]}
    sh = D.shell_numbers(run.t_months, el_m)
    shell = {"half_shell_days": sh["half_shell_days"],
             "shell_days": sh["shell_days"],
             "spread50_deg_at_365":
                 np.degrees(sh["spread50_rad"][-1])}
    sim, cf = D.precession_check(run.t_months, el_m)
    isim, icf = D.precession_check(
        run.t_months, read_elements(run.months_parent))
    deg = np.degrees(1.0) * DAY
    rel = np.abs(sim / cf - 1)
    prec = {"intact_sim_deg_day": isim[0] * deg,
            "intact_closed_form_deg_day": icf[0] * deg,
            "cloud_median_deg_day": np.median(sim) * deg,
            "spread90_sim_deg_day": middle_range(sim) * deg,
            "spread90_closed_form_deg_day": middle_range(cf) * deg,
            "max_rel_error": rel.max(),
            "median_rel_error": np.median(rel)}
    curves = {"along": rn["spread90_rad"],
              "along_no_j2": rn0["spread90_rad"],
              "nodes90": sh["spread90_rad"],
              "nodes50": sh["spread50_rad"],
              "rate_sim": sim * deg, "rate_cf": cf * deg}
    return ring, shell, prec, curves


def results():
    run = D.run_cloud(control=True)
    el, elp = read_elements(run.day), read_elements(run.day_parent)
    el_m = read_elements(run.months)
    orbit, pn = step1(run, el, elp)
    ring, shell, prec, curves = step2(run, el, elp, el_m)
    offsets = np.stack([pn[k].pop("_offsets_min")
                        for k in ALL_PASSES], axis=1)
    res = {"cloud": cloud_numbers(run.cloud), "orbit": orbit,
           "passes": {str(k): v for k, v in pn.items()},
           "ring": ring, "shell": shell, "precession": prec}
    return res, dict(run=run, el_m=el_m, offsets=offsets,
                     curves=curves)


def km32(x):
    return (np.asarray(x) / 1e3).astype(np.float32)


def write_data(res, data):
    run, el_m, cv = data["run"], data["el_m"], data["curves"]
    k = slice(0, None, FRAME_STEP)
    m = {"a": el_m["a"].mean(axis=1), "i": el_m["i"].mean(axis=1)}
    raan = np.degrees(np.unwrap(el_m["raan"], axis=1))
    per = [res["passes"][str(p)] for p in ALL_PASSES]
    arrays = {
        "t_day_min": run.t_day[k] / 60,
        "day_xyz_km": km32(run.day[:, k, :3]),
        "day_intact_xyz_km": km32(run.day_parent[0, k, :3]),
        "day_along_spread90_deg": np.degrees(cv["along"][k]),
        "day_along_spread90_deg_no_j2":
            np.degrees(cv["along_no_j2"][k]),
        "pass_k": np.array(ALL_PASSES),
        "pass_intact_min": np.array([p["parent_return_min"]
                                     for p in per]),
        "pass_offset_min": data["offsets"],
        "pass_window50_min": np.array([p["window50_min"]
                                       for p in per]),
        "pass_window90_min": np.array([p["window90_min"]
                                       for p in per]),
        "pass_share_1min": np.array([p["share_within_1min"]
                                     for p in per]),
        "pass_miss_median_km": np.array([p["miss_median_km"]
                                         for p in per]),
        "t_months_d": run.t_months / DAY,
        "months_xyz_km": km32(run.months[..., :3]),
        "months_node_deg": raan - raan[:, :1],
        "months_node_spread90_deg": np.degrees(cv["nodes90"]),
        "months_node_spread50_deg": np.degrees(cv["nodes50"]),
        "rate_sim_deg_day": cv["rate_sim"],
        "rate_closed_form_deg_day": cv["rate_cf"],
        "piece_mean_alt_km": (m["a"] - R_EARTH_EQ) / 1e3,
        "piece_mean_tilt_deg": np.degrees(m["i"]),
        "piece_kick_ms": np.linalg.norm(run.cloud.dv, axis=1),
        "breakup_xyz_km": km32(run.day[0, 0, :3]),
    }
    np.savez_compressed(OUT / "follow_cloud.npz", **arrays)
    (OUT / "follow_cloud.json").write_text(json.dumps(clean(
        sidecar(arrays, res)), indent=1))


DESC = {
    "t_day_min": ("min", "time axis of day_*: 0..2880 min (2 "
                  "days), step 2 min; t = 0 is the breakup"),
    "day_xyz_km": ("km", "every piece [piece, t_day_min, xyz]"),
    "day_intact_xyz_km": ("km", "where the satellite would be "
                          "had it not broken up [t_day_min, xyz]"),
    "day_along_spread90_deg": ("deg", "spread along the orbit "
                               "of the middle 90 % of pieces; 360"
                               " = full ring"),
    "day_along_spread90_deg_no_j2": ("deg", "the same cloud "
                                     "flown without J2 (control:"
                                     " the ring is Kepler's)"),
    "pass_k": ("-", "pass number 1..15 (15 = about one day)"),
    "pass_intact_min": ("min", "when the intact orbit is back at"
                        " the breakup spot on pass k"),
    "pass_offset_min": ("min", "each piece's arrival at the spot"
                        " on pass k minus pass_intact_min "
                        "[piece, pass]; negative = early"),
    "pass_window50_min": ("min", "time for the middle 50 % of "
                          "pieces to go past, per pass"),
    "pass_window90_min": ("min", "same for the middle 90 %; above"
                          " one orbit (96.7) = a ring"),
    "pass_share_1min": ("-", "share of pieces arriving within 1 "
                        "min of the intact orbit, per pass"),
    "pass_miss_median_km": ("km", "median distance of a piece's "
                            "return point from the original "
                            "breakup point (the orbit turns)"),
    "t_months_d": ("days", "time axis of months_*: 0..365, daily"),
    "months_xyz_km": ("km", "every piece once a day [piece, "
                      "t_months_d, xyz]"),
    "months_node_deg": ("deg", "how far each orbit has turned "
                        "about the spin axis (unwrapped, negative"
                        " = west) [piece, t_months_d]"),
    "months_node_spread90_deg": ("deg", "spread of that turn over"
                                 " the middle 90 %; 360 = shell"),
    "months_node_spread50_deg": ("deg", "same, middle 50 %"),
    "rate_sim_deg_day": ("deg/day", "each piece's turning rate, "
                         "fitted to the simulation [piece]"),
    "rate_closed_form_deg_day": ("deg/day", "the closed form "
                                 "-(3/2) n J2 (R/p)^2 cos i for "
                                 "the same piece [piece]"),
    "piece_mean_alt_km": ("km", "each piece's average height "
                          "(semi-major axis minus Earth's "
                          "equatorial radius)"),
    "piece_mean_tilt_deg": ("deg", "each piece's average tilt"),
    "piece_kick_ms": ("m/s", "each piece's kick speed"),
    "breakup_xyz_km": ("km", "the breakup point (over the "
                       "equator, on +x)"),
}


def sidecar(arrays, res):
    return {
        "frame": "Earth-centred inertial (not rotating with "
                 "Earth): z = spin axis (north), x = towards the "
                 "breakup point over the equator",
        "units": "km, minutes (day_*, pass_*), days (months_*)",
        "time_base": "t = 0 is the breakup for every array; day_*"
                     " and months_* are two windows of one run",
        "model": "Earth gravity + J2, scipy DOP853 rtol 1e-11; "
                 "NASA Standard Breakup Model pieces (11 cm to 1 m)"
                 "; no air drag, Sun, Moon or collisions",
        "assumptions": "600 km circular (film title card); tilt "
                       "51.6 deg (the ISS's, assumed); 90 of 590 "
                       "pieces dip below 200 km and are not "
                       "followed",
        "step1_hint": "show the breakup spot and a counter of "
                      "passes; on pass 1 the swarm goes by in "
                      "minutes, by pass 15 it is a ring "
                      "(pass_window90_min > 96.7)",
        "step2_hint": "ring: day_along_spread90_deg reaches 360 at"
                      f" {res['ring']['ring_h']:.1f} h (J2 off: "
                      "same); shell: months_node_spread90_deg "
                      f"reaches 360 at {res['shell']['shell_days']:.0f}"
                      " days; scatter rate_sim vs rate_closed_form "
                      "lies on the diagonal",
        "arrays": {k: {"shape": list(np.shape(v)),
                       "units": DESC[k][0], "what": DESC[k][1]}
                   for k, v in arrays.items()},
    }


def lookup(res, key):
    for part in key.split("."):
        res = res[part]
    return res


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
        return {str(k): clean(v) for k, v in o.items()}
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
    print(f"  intact orbit back after "
          f"{res['orbit']['intact_return_min']:.2f} min")
    for k in D.PASSES:
        p = res["passes"][str(k)]
        print(f"  pass {k:2d}: 50 % in {p['window50_min']:6.1f} "
              f"min, 90 % in {p['window90_min']:6.1f} min, "
              f"{100 * p['share_within_1min']:4.1f} % within 1 "
              f"min, spot moved {p['miss_median_km']:.0f} km")
    r, s, p = res["ring"], res["shell"], res["precession"]
    print(f"  ring {r['ring_h']:.1f} h (no J2 "
          f"{r['no_j2_ring_h']:.1f} h); shell {s['shell_days']:.0f}"
          f" d; precession error max {p['max_rel_error']:.2%}")


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
