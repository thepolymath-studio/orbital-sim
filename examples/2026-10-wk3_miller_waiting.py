"""2026-10 wk3 Sim post (Interstellar): where could Endurance wait near Miller's planet?

While the Ranger visits Miller's planet, Endurance waits in orbit
(Romilly's 23 years). Could it have waited at a balance point near
the planet? Gargantua + Miller's planet "treated as if it were Earth
and Moon" (orbital_sim.miller: Earth-Moon mass ratio and sizes),
quoted in the pair's own units: distance = share of the planet's
distance from Gargantua (a), time = orbits of the planet.

1. The waiting test (orbital_sim.waiting): at each of L1-L5 three
   ships, exactly on the point, a millionth of a off it, and
   drifting at a millionth of the orbital speed. 10 orbits; L4/L5
   also for 300 orbits.
2. A tiny error at L1 (between the planet and Gargantua): 100 ships
   at rest, all within a millionth of a of each other. The saddle
   splits them.
3. The real-numbers check: with Thorne's numbers L1 and L2 lie
   inside the planet, and every balance point sits at the planet's
   own distance from Gargantua.

Writes outputs/2026-10-wk3-sim/: miller.npz + miller.json (sidecar)
and results.json (every quotable number).

    python examples/2026-10-wk3_miller_waiting.py [--check]

--check asserts the numbers in
posts/2026-10-wk3-interstellar-sim/code-results.md (exit 1 on a
mismatch).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from orbital_sim.balance import fly_cloud
from orbital_sim.constants import RHO_EARTH
from orbital_sim.cr3bp import distances, jacobi
from orbital_sim.lagrange import (NAMES, lagrange_points,
                                  landscape_shape, routh_mu,
                                  routh_ratio)
from orbital_sim.miller import (ORBIT, gaps_in_radii, miller_real,
                                stand_in_mu, stand_in_radii)
from orbital_sim.waiting import (departure, drift, efold_orbits,
                                 first_time_over, hill_radius,
                                 point_cloud, saddle_side, spread,
                                 waiting_starts, where_now)

OUT = (Path(__file__).resolve().parents[1] / "outputs"
       / "2026-10-wk3-sim")

ORBITS, STEP = 10.0, 0.01      # shared window and step, orbits
DR, DV = 1e-6, 1e-6            # "slightly off", "slowly drifting"
LONG_ORBITS, LONG_STEP = 300.0, 0.05   # L4/L5 long check
N_CLOUD, CLOUD = 100, 1e-6     # 100 ships within a millionth of a
GONE = 0.01                    # "has left": 1 % of a from its point
TRIES = ("exact", "off", "drift")
AT = (1, 5, 10)                # waiting snapshots, orbits
WIDTH_AT = (0.25, 0.5, 0.75, 1, 2, 5, 10)
FIT = (0.2, 0.5)               # cloud growth fit window, orbits
NEAR_PLANET = {"near the small body": "near the planet",
               "inside the orbit": "inside the planet's orbit",
               "outside the orbit": "outside the planet's orbit",
               "along the orbit": "along the planet's orbit"}

# Values quoted in code-results.md, checked by --check: (value, tol)
EXPECTED = {
    "stand_in.mu": (0.0121506, 5e-8),
    "stand_in.big_to_small": (81.30, 0.005),
    "stand_in.planet_radius_a": (0.004520, 5e-7),
    "points.L1.from_small_a": (0.1509, 5e-5),
    "points.L2.from_small_a": (0.1678, 5e-5),
    "points.L1.from_small_radii": (33.4, 0.05),
    "points.L2.from_small_radii": (37.1, 0.05),
    "points.L3.from_big_a": (0.9929, 5e-5),
    "points.L4.angle_deg": (60.0, 1e-9),
    "points.L5.angle_deg": (-60.0, 1e-9),
    "points.L1.shape": ("saddle", None),
    "points.L4.shape": ("hilltop", None),
    "points.L1.stable": (False, None),
    "points.L2.stable": (False, None),
    "points.L3.stable": (False, None),
    "points.L4.stable": (True, None),
    "points.L5.stable": (True, None),
    "points.L1.efold_orbits": (0.054, 5e-4),
    "points.L2.efold_orbits": (0.074, 5e-4),
    "points.L3.efold_orbits": (0.895, 5e-4),
    "routh.ratio": (24.96, 0.005),
    # waiting test
    "waiting.L1.off.orbits_to_gone": (0.53, 0),
    "waiting.L2.off.orbits_to_gone": (0.70, 0),
    "waiting.L3.off.orbits_to_gone": (5.75, 0),
    "waiting.L1.drift.orbits_to_gone": (0.64, 0),
    "waiting.L2.drift.orbits_to_gone": (0.83, 0),
    "waiting.L3.drift.orbits_to_gone": (6.38, 0),
    "waiting.L1.exact.orbits_to_gone": (2.01, 0),
    "waiting.L2.exact.orbits_to_gone": (2.40, 0),
    "waiting.L1.exact.at_1_a": (0.0, 1e-9),
    "waiting.L1.off.end": ("near the planet", None),
    "waiting.L1.drift.end": ("near the planet", None),
    "waiting.L2.off.end": ("outside the planet's orbit", None),
    "waiting.L2.drift.end": ("outside the planet's orbit", None),
    "waiting.L4.off.max_a": (1.58e-5, 5e-8),
    "waiting.L5.off.max_a": (1.57e-5, 5e-8),
    "waiting.L4.drift.max_a": (9.8e-6, 5e-8),
    "waiting.L5.drift.max_a": (9.7e-6, 5e-8),
    "waiting.L4.off.max_long_a": (1.58e-5, 5e-8),
    "waiting.L4.drift.max_long_a": (9.8e-6, 5e-8),
    "waiting.L5.off.max_long_a": (1.58e-5, 5e-8),
    "waiting.L5.drift.max_long_a": (9.8e-6, 5e-8),
    "waiting.L4.off.long_half_ratio": (1.0, 0.01),
    "waiting.L5.off.long_half_ratio": (1.0, 0.01),
    "waiting.L4.exact.max_long_a": (0.0, 1e-9),
    "waiting.n_left": (6, 0),
    "waiting.n_stayed": (4, 0),
    "waiting.L3.exact.max_a": (0.0, 1e-9),
    # the L1 cloud
    "cloud.n": (100, 0),
    "cloud.max_start_gap_a": (9.8e-7, 5e-9),
    "cloud.width_a.0.5": (0.0065, 5e-5),
    "cloud.width_a.1": (1.70, 0.005),
    "cloud.orbits_to_width.0.01": (0.53, 0),
    "cloud.orbits_to_width.0.1": (0.65, 0),
    "cloud.efold_fit_orbits": (0.054, 5e-4),
    "cloud.n_planet_side": (50, 0),
    "cloud.n_gargantua_side": (50, 0),
    "cloud.split_matches_saddle": (True, None),
    "cloud.n_crossed_back": (0, 0),
    "max_jacobi_drift": (0.0, 1e-10),
    # real numbers (Earth-like density assumed)
    "real.l1_in_planet_radii": (0.50, 0.005),
    "real.l2_in_planet_radii": (0.50, 0.005),
    "real.points_from_gargantua_a.max_dev": (2.8e-5, 5e-7),
    "real.l4_stable": (True, None),
}


def point_table(mu, r_small):
    """L1..L5 in units of a, planet radii and orbits."""
    out = {}
    for name, (x, y) in zip(NAMES, lagrange_points(mu)):
        ef = efold_orbits(mu, name)
        d = np.hypot(x - 1 + mu, y)
        out[name] = {
            "x_a": x, "y_a": y,
            "from_big_a": np.hypot(x + mu, y),
            "from_small_a": d, "from_small_radii": d / r_small,
            "angle_deg": np.degrees(np.arctan2(y, x + mu)),
            "shape": landscape_shape(x, y, mu),
            "stable": bool(np.isinf(ef)),
            "efold_orbits": None if np.isinf(ef) else ef,
        }
    return out


def label(state, mu):
    """Where a ship is, in the post's words."""
    if not np.isfinite(state[0]):
        return "hit a body"
    return NEAR_PLANET[where_now(state, mu)]


def run_waiting(mu, t_orb):
    starts = waiting_starts(mu, DR, DV)
    tracks, hit = fly_cloud(starts, t_orb * ORBIT, mu,
                            stand_in_radii())
    return starts, tracks, hit


def run_long(mu):
    t_orb = np.round(np.arange(0.0, LONG_ORBITS + LONG_STEP
                               / 2, LONG_STEP), 6)
    pick = list(range(9, 15))            # L4, L5 x three tries
    starts = waiting_starts(mu, DR, DV)[pick]
    tracks, _ = fly_cloud(starts, t_orb * ORBIT, mu,
                          stand_in_radii())
    return t_orb, tracks, pick


def waiting_numbers(mu, t_orb, tracks, hit, long):
    pts = lagrange_points(mu)
    out = {}
    for k, tr in enumerate(tracks):
        name, how = NAMES[k // 3], TRIES[k % 3]
        d = drift(tr, pts[k // 3])
        t1 = first_time_over(t_orb, d, GONE)
        row = {f"at_{o}_a": d[np.searchsorted(t_orb, o)]
               for o in AT}
        row.update({
            "max_a": np.nanmax(d),
            "orbits_to_gone": None if np.isnan(t1) else t1,
            "end": label(tr[:, -1], mu),
            "hit_body": bool(np.isfinite(hit[k]))})
        out.setdefault(name, {})[how] = row
    lt, ltr, pick = long
    half = lt.size // 2
    for tr, k in zip(ltr, pick):
        d = drift(tr, pts[k // 3])
        row = out[NAMES[k // 3]][TRIES[k % 3]]
        row["max_long_a"] = np.nanmax(d)
        row["long_half_ratio"] = (np.nanmax(d[half:])
                                  / max(np.nanmax(d[:half]),
                                        1e-300))
    left = [out[n][h]["orbits_to_gone"] is not None
            for n in NAMES for h in ("off", "drift")]
    out["n_left"], out["n_stayed"] = sum(left), 10 - sum(left)
    out["long_orbits"] = LONG_ORBITS
    return out


def body_hit(track, mu, radii):
    """Which body a stopped track ended on."""
    last = track[:, np.isfinite(track[0])][:, -1]
    r1, r2 = distances(last[0], last[1], mu)
    return "planet" if r2 - radii[1] < r1 - radii[0] \
        else "Gargantua"


def cloud_numbers(mu, t_orb, starts, tracks, hit):
    x_l1 = lagrange_points(mu)[0]
    side = saddle_side(mu, starts, "L1")
    went = np.array([departure(tr, x_l1, GONE)
                     for tr in tracks])
    xs = np.sign(tracks[:, 0, :] - x_l1[0])
    after = t_orb > 1.0          # once clear of the saddle
    crossed = [bool(np.any(np.abs(np.diff(s[after][
        np.isfinite(s[after])])) > 0)) for s in xs]
    wid = spread(tracks)
    i, j = np.searchsorted(t_orb, FIT)
    rate = np.log(wid[j] / wid[i]) / (t_orb[j] - t_orb[i])
    radii = stand_in_radii()
    hits = [body_hit(tr, mu, radii) for tr, h in
            zip(tracks, hit) if np.isfinite(h)]
    ends = [label(tr[:, -1], mu) for tr in tracks]
    return {
        "n": N_CLOUD, "diameter_a": CLOUD,
        "max_start_gap_a": spread(starts[:, :, None])[0],
        "width_a": {str(o): wid[np.searchsorted(t_orb, o)]
                    for o in WIDTH_AT},
        "orbits_to_width": {
            "0.01": first_time_over(t_orb, wid, 0.01),
            "0.1": first_time_over(t_orb, wid, 0.1)},
        "efold_fit_orbits": 1 / rate,
        "n_planet_side": int((side > 0).sum()),
        "n_gargantua_side": int((side < 0).sum()),
        "split_matches_saddle": bool(np.array_equal(side, went)),
        "planet_side": (side > 0).tolist(),
        "n_crossed_back": int(sum(crossed)),
        "n_impacts": len(hits),
        "impacts": {b: hits.count(b) for b in set(hits)},
        "end": {e: ends.count(e) for e in sorted(set(ends))},
        "ends": ends,
        "cross_pair": cross_pair(t_orb, tracks, side > 0),
        "_width": wid,
    }


def cross_pair(t_orb, tracks, planet):
    """The closest-starting pair with different fates."""
    a, b = np.flatnonzero(planet), np.flatnonzero(~planet)
    xy = tracks[:, :2, :]
    d0 = np.hypot(*(xy[a, :, 0][:, None] - xy[b, :, 0][None]).T)
    j, i = np.unravel_index(np.argmin(d0), d0.shape)
    d = np.hypot(*(xy[a[i]] - xy[b[j]]))
    return {"ships": [int(a[i]), int(b[j])],
            "start_a": d[0],
            "a_at": {str(o): d[np.searchsorted(t_orb, o)]
                     for o in (0.5, 1, 2)}}


def real_numbers():
    """The real-numbers check (Earth-like density)."""
    S = miller_real(RHO_EARTH)
    pts = lagrange_points(S.mu)
    r_big = np.hypot(pts[:, 0] + S.mu, pts[:, 1])
    g1, g2 = gaps_in_radii(RHO_EARTH)
    return {"density_assumed_kgm3": RHO_EARTH, "mu": S.mu,
            "big_to_small": S.big_to_small,
            "l1_in_planet_radii": g1, "l2_in_planet_radii": g2,
            "points_from_gargantua_a": {
                "values": r_big.tolist(),
                "max_dev": float(np.abs(r_big - 1).max())},
            "l4_stable": bool(S.mu < routh_mu()),
            "hill_radius_a": hill_radius(S.mu)}


def jacobi_drift(*clouds, mu):
    """Largest change of C_J along any track."""
    worst = 0.0
    for c in clouds:
        for tr in c:
            cj = jacobi(tr, mu)
            worst = max(worst, np.nanmax(np.abs(cj - cj[0])))
    return float(worst)


def results():
    mu = stand_in_mu()
    r_small = stand_in_radii()[1]
    t_orb = np.round(np.arange(0.0, ORBITS + STEP / 2, STEP), 6)
    ws, wt, wh = run_waiting(mu, t_orb)
    long = run_long(mu)
    cs = point_cloud(mu, "L1", N_CLOUD, CLOUD)
    ct, ch = fly_cloud(cs, t_orb * ORBIT, mu, stand_in_radii())
    res = {"stand_in": {"mu": mu, "big_to_small": (1 - mu) / mu,
                        "planet_radius_a": r_small,
                        "gargantua_radius_a": stand_in_radii()[0],
                        "hill_radius_a": hill_radius(mu)},
           "points": point_table(mu, r_small),
           "routh": {"ratio": routh_ratio()}}
    res["waiting"] = waiting_numbers(mu, t_orb, wt, wh, long)
    res["cloud"] = cloud_numbers(mu, t_orb, cs, ct, ch)
    res["real"] = real_numbers()
    res["max_jacobi_drift"] = jacobi_drift(wt, long[1], ct, mu=mu)
    data = dict(t_orb=t_orb, wt=wt, ct=ct, long=long,
                wid=res["cloud"].pop("_width"))
    return res, data


def xy(tracks):
    """(n, 4, t) -> (n, t, 2) positions."""
    return tracks[:, :2, :].transpose(0, 2, 1)


def write_data(res, data):
    mu = stand_in_mu()
    pts = lagrange_points(mu)
    lt, ltr, pick = data["long"]
    wdist = np.array([drift(tr, pts[k // 3])
                      for k, tr in enumerate(data["wt"])])
    arrays = {
        "t_orbits": data["t_orb"],
        "waiting_xy_a": xy(data["wt"]),
        "waiting_dist_a": wdist,
        "cloud_xy_a": xy(data["ct"]),
        "cloud_width_a": data["wid"],
        "cloud_planet_side": np.array(
            res["cloud"]["planet_side"]),
        "t_long_orbits": lt,
        "long_xy_a": xy(ltr),
        "lpoints_xy_a": pts,
        "gargantua_xy_a": np.array([-mu, 0.0]),
        "planet_xy_a": np.array([1 - mu, 0.0]),
        "radii_a": np.array(stand_in_radii()),
    }
    np.savez_compressed(OUT / "miller.npz", **arrays)
    labels = [f"{NAMES[k // 3]} {TRIES[k % 3]}" for k in range(15)]
    desc = {
        "t_orbits": ("orbits", "shared time axis of waiting_* and "
                     "cloud_*: 0..10 orbits of the planet, step "
                     "0.01"),
        "waiting_xy_a": ("a", "waiting ships [ship, time, (x, y)];"
                         " ship order = waiting_labels; NaN after "
                         "an impact"),
        "waiting_dist_a": ("a", "each waiting ship's distance "
                           "from its own point [ship, time]"),
        "cloud_xy_a": ("a", "L1 cloud [ship, time, (x, y)]; NaN "
                       "after an impact"),
        "cloud_width_a": ("a", "widest gap between any two cloud "
                          "ships [time]"),
        "cloud_planet_side": ("bool", "per cloud ship: starts on "
                              "the planet's side of L1's dividing "
                              "line (and leaves that way)"),
        "t_long_orbits": ("orbits", "time axis of long_xy_a: "
                          f"0..{LONG_ORBITS:.0f} orbits, step "
                          f"{LONG_STEP}"),
        "long_xy_a": ("a", "L4 exact/off/drift, L5 exact/off/drift"
                      " [ship, time, (x, y)]"),
        "lpoints_xy_a": ("a", "L1..L5 [point, (x, y)]"),
        "gargantua_xy_a": ("a", "Gargantua's centre (fixed)"),
        "planet_xy_a": ("a", "Miller's planet's centre (fixed)"),
        "radii_a": ("a", "drawn radii (Gargantua, planet): "
                    "Earth-Moon proportions"),
    }
    side = {
        "stand_in": "Gargantua + Miller's planet with Earth-Moon "
                    "proportions (mass ratio 81.30, sizes); label "
                    "it as a stand-in on screen",
        "frame": "rotating frame, origin at the barycentre, +x "
                 "from Gargantua to the planet, +y along the "
                 "planet's motion; both bodies fixed",
        "units": "length a = planet's distance from Gargantua; "
                 "time in orbits of the planet (1 orbit = 2 pi "
                 "model units)",
        "time_base": "t_orbits for every waiting_* and cloud_* "
                     "array; continuous integration (no wrap). "
                     "long_xy_a uses t_long_orbits",
        "model": "planar circular restricted three-body problem, "
                 "scipy DOP853 rtol 1e-12 atol 1e-13; Newtonian "
                 "stand-in, no relativity, no other bodies",
        "scale_hint": "planet-centred zoom of about +-0.4 a for "
                      "L1, L2 and the cloud; full view (+-1.3 a) "
                      "for L3, L4, L5. L4/L5 wobbles are ~1e-5 a: "
                      "show them as distance-vs-time "
                      "(waiting_dist_a), not as tracks",
        "waiting_labels": labels,
        "waiting_starts": "exact = on the point; off = 1e-6 a "
                          "along +x, at rest; drift = on the "
                          "point, moving at 1e-6 (n a) along +y",
        "cloud_start": f"{N_CLOUD} ships at rest in a sunflower "
                       f"disc of diameter {CLOUD} a centred on L1",
        "cloud_ends": res["cloud"]["ends"],
        "arrays": {k: {"shape": list(np.shape(v)),
                       "units": desc[k][0], "what": desc[k][1]}
                   for k, v in arrays.items()},
        "points": {n: {"x_a": p["x_a"], "y_a": p["y_a"],
                       "shape": p["shape"], "stable": p["stable"]}
                   for n, p in res["points"].items()},
        "real_numbers_inset": {
            "l1_l2_in_planet_radii": [
                res["real"]["l1_in_planet_radii"],
                res["real"]["l2_in_planet_radii"]],
            "note": "with Thorne's numbers (Earth-like density "
                    "assumed) L1/L2 lie inside the planet at half "
                    "its radius; draw them inside the disc"},
    }
    (OUT / "miller.json").write_text(json.dumps(clean(side),
                                                indent=1))


def lookup(res, key):
    for part in _parts(key):
        res = res[part]
    return res


def _parts(key):
    """Split a key path; numeric leaves like 0.5 stay."""
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
    for n, p in res["points"].items():
        ef = p["efold_orbits"]
        print(f"  {n}: {p['from_small_radii']:6.1f} R from the "
              f"planet, {p['shape']:7s} " + (
                  "stable" if p["stable"] else
                  f"e-fold {ef:.3f} orbits"))
    for n in NAMES:
        for how, w in res["waiting"][n].items():
            print(f"  wait {n} {how:5s}: max {w['max_a']:.2e} a, "
                  f"gone at {w['orbits_to_gone']} orbits, "
                  f"{w['end']}")
    c = res["cloud"]
    print("  cloud width:", {k: f"{v:.2e}" for k, v in
                             c["width_a"].items()})
    print(f"  planet side {c['n_planet_side']}, Gargantua side "
          f"{c['n_gargantua_side']}, ends {c['end']}, impacts "
          f"{c['impacts']}")
    print(f"  real: L1 at {res['real']['l1_in_planet_radii']:.3f}"
          f" R; Jacobi drift {res['max_jacobi_drift']:.1e}")


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
