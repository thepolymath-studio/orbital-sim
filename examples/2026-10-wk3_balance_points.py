"""2026-10 wk3 (Interstellar): balance points and chaos, Gargantua + Miller.

The pair Gargantua + Miller's planet, "treated as if it were Earth
and Moon": the planar circular restricted three-body problem
(orbital_sim.cr3bp) with Earth-Moon proportions (mass ratio 81.30,
body sizes), quoted in the pair's own units (orbital_sim.miller):

* distance: share of the planet's distance from Gargantua (a), or
  planet radii; time: orbits of the planet around Gargantua;
* L1-L5, their landscape shape and stability (orbital_sim.lagrange),
  Routh's 24.96 against the stand-in's 81.30;
* the hook (orbital_sim.balance): ten craft a millionth of a apart
  at rest around L1, six on L4's slow swing; 7 orbits, DOP853;
* U_eff on a grid for the landscape figure;
* the real numbers (miller_real): mass ratio, L1/L2 inside the
  planet, the density they would need to clear the surface, and
  Newton's against the Kerr period at that distance;
* validation only (not for slides): the same code on real
  Earth-Moon (km against the sourced values), Sun-Earth and
  Sun-Jupiter.

Writes outputs/2026-10-wk3/: hook.npz + hook.json, landscape.npz +
landscape.json (sidecars: names, shapes, units, frame, time base)
and results.json (every quotable number).

    python examples/2026-10-wk3_balance_points.py [--check]

--check asserts the numbers in
posts/2026-10-wk3-interstellar/code-results.md (exit 1 on mismatch).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from orbital_sim.balance import (fly_cloud, l1_cloud, l4_ring,
                                 ueff_grid)
from orbital_sim.constants import (AU, C_LIGHT, DAY, G_MILLER,
                                   HOUR, M_GARGANTUA_SUNS,
                                   MU_EARTH, MU_GARGANTUA,
                                   RHO_EARTH)
from orbital_sim.cr3bp import (earth_moon, grad_omega, jacobi,
                               sun_earth, sun_jupiter, u_eff)
from orbital_sim.lagrange import (NAMES, growth_rate,
                                  lagrange_points, landscape_shape,
                                  linearise, routh_mu, routh_ratio)
from orbital_sim.miller import (ORBIT, clearing_density,
                                gaps_in_radii, kerr_period,
                                miller_real, newton_period,
                                orbit_radius, planet_gm,
                                planet_radius, stand_in_mu,
                                stand_in_radii)
from orbital_sim.waiting import hill_radius

OUT = Path(__file__).resolve().parents[1] / "outputs" / "2026-10-wk3"

# Sourced Earth-Moon distances from Earth's centre, km (research S7,
# Wikipedia "Lagrange point", rounded to 100 km there). Validation.
SOURCED_EM_KM = {"L1": 326_400, "L2": 448_900, "L3": 381_700}

ORBITS, STEP = 7.0, 0.01     # hook window and sample step, orbits
N_L1, GAP = 10, 1e-6         # L1: ten craft, a millionth of a apart
N_L4, AMP = 6, 0.05          # L4 ring: six phases, 5 % of a in x
LONG_ORBITS = 27.0           # L4 "stays" check (about 2 Earth yr)
SPLIT_AT = {"tenth_pc": 0.01, "tenth": 0.1}   # gap thresholds, a
GRID_HALF, GRID_N = 1.5, 601
DENSITIES = (1_000.0, RHO_EARTH, 2 * RHO_EARTH, 3 * RHO_EARTH)
HOURS_PER_7_YEARS = 7 * 365.25 * 24      # film: 1 h = 7 years

# Values quoted in code-results.md, checked by --check: (value, tol)
EXPECTED = {
    # stand-in (Earth-Moon proportions), units of a and orbits
    "stand_in.mu": (0.0121506, 5e-8),
    "stand_in.big_to_small": (81.30, 0.005),
    "stand_in.planet_radius_a": (0.004520, 5e-7),
    "stand_in.gargantua_radius_a": (0.01657, 5e-6),
    "stand_in.points.L1.from_big_a": (0.8491, 5e-5),
    "stand_in.points.L2.from_big_a": (1.1678, 5e-5),
    "stand_in.points.L3.from_big_a": (0.9929, 5e-5),
    "stand_in.points.L1.from_small_a": (0.1509, 5e-5),
    "stand_in.points.L2.from_small_a": (0.1678, 5e-5),
    "stand_in.points.L1.from_small_radii": (33.4, 0.05),
    "stand_in.points.L2.from_small_radii": (37.1, 0.05),
    "stand_in.points.L4.from_big_a": (1.0, 1e-12),
    "stand_in.points.L4.from_small_a": (1.0, 1e-12),
    "stand_in.points.L3.from_small_radii": (440.9, 0.05),
    "stand_in.points.L4.from_small_radii": (221.3, 0.05),
    "stand_in.points.L4.angle_deg": (60.0, 1e-9),
    "stand_in.points.L5.angle_deg": (-60.0, 1e-9),
    "stand_in.points.L1.shape": ("saddle", None),
    "stand_in.points.L2.shape": ("saddle", None),
    "stand_in.points.L3.shape": ("saddle", None),
    "stand_in.points.L4.shape": ("hilltop", None),
    "stand_in.points.L5.shape": ("hilltop", None),
    "stand_in.points.L1.stable": (False, None),
    "stand_in.points.L2.stable": (False, None),
    "stand_in.points.L3.stable": (False, None),
    "stand_in.points.L4.stable": (True, None),
    "stand_in.points.L5.stable": (True, None),
    "stand_in.points.L1.efold_orbits": (0.054, 5e-4),
    "stand_in.points.L2.efold_orbits": (0.074, 5e-4),
    "stand_in.points.L3.efold_orbits": (0.895, 5e-4),
    "stand_in.points.L1.jacobi": (3.18834, 5e-6),
    "stand_in.points.L2.jacobi": (3.17216, 5e-6),
    "stand_in.points.L3.jacobi": (3.01215, 5e-6),
    "stand_in.points.L4.jacobi": (2.98800, 5e-6),
    "stand_in.max_grad": (0.0, 1e-13),
    "stand_in.l4_periods_orbits.short": (1.05, 0.005),
    "stand_in.l4_periods_orbits.long": (3.35, 0.005),
    "routh.mu": (0.038521, 5e-7),
    "routh.ratio": (24.96, 0.005),
    "routh.ratio_closed_form": (24.9599358, 5e-8),
    # hook
    "hook.l1.gap_a": (1e-6, 0),
    "hook.l1.n_planet_side": (5, 0),
    "hook.l1.n_gargantua_side": (5, 0),
    "hook.l1.n_impacts": (0, 0),
    "hook.l1.n_crossings": (0, 0),
    "hook.l1.orbits_to_gap.tenth_pc": (0.52, 0),
    "hook.l1.orbits_to_gap.tenth": (0.63, 0),
    "hook.l4.amp_a": (0.05, 0),
    "hook.l4.n_impacts": (0, 0),
    "hook.l4.max_from_l4_a": (0.0864, 5e-5),
    "hook.l4.max_from_l4_long_a": (0.0871, 5e-5),
    "hook.max_jacobi_drift": (0.0, 1e-10),
    # real numbers (Earth-like density assumed)
    "real.gargantua_suns": (1e8, 0),
    "real.surface_gravity_ms2": (12.75, 0.005),
    "real.sep_au": (0.987, 5e-4),
    "real.planet_radius_km": (8_271, 0.5),
    "real.planet_mass_earths": (2.19, 0.005),
    "real.mu": (6.57e-14, 5e-17),
    "real.big_to_small": (1.52e13, 5e10),
    "real.l1_in_planet_radii": (0.50, 0.005),
    "real.l2_in_planet_radii": (0.50, 0.005),
    "real.hill_in_planet_radii": (0.50, 0.005),
    "real.l1_gap_km": (4_132, 0.5),
    "real.clearing_density_kgm3": (44_231, 0.5),
    "real.clearing_vs_earth": (8.0, 0.05),
    "real.scan.1000.l1_in_planet_radii": (0.28, 0.005),
    "real.scan.11026.l1_in_planet_radii": (0.63, 0.005),
    "real.scan.16539.l1_in_planet_radii": (0.72, 0.005),
    "real.points_from_gargantua_a.max_dev": (2.8e-5, 5e-7),
    "real.mu_over_routh": (1.7e-12, 5e-14),
    "real.newton_speed_over_c": (1.0, 1e-12),
    "real.newton_period_h": (0.86, 0.005),
    "real.kerr_period_h": (1.72, 0.005),
    "real.hours_per_7_years": (61_362, 0),
    # validation only: same code, real Earth-Moon / Sun-Earth / Jupiter
    "validation.em.L1.from_big_km": (326_381, 0.5),
    "validation.em.L2.from_big_km": (448_915, 0.5),
    "validation.em.L3.from_big_km": (381_675, 0.5),
    "validation.em.L1.diff_vs_sourced_km": (-19, 0.5),
    "validation.em.L2.diff_vs_sourced_km": (15, 0.5),
    "validation.em.L3.diff_vs_sourced_km": (-25, 0.5),
    "validation.se.L1.from_small_km": (1_497_621, 0.5),
    "validation.se.L2.from_small_km": (1_507_683, 0.5),
    "validation.se.L1.efold_days": (23.0, 0.05),
    "validation.se.L2.efold_days": (23.4, 0.05),
    "validation.sj.big_to_small": (1047.35, 0.005),
    "validation.sj.L4.stable": (True, None),
}


def point_table(mu, r_small):
    """L1..L5 in units of a, planet radii and orbits."""
    out = {}
    for name, (x, y) in zip(NAMES, lagrange_points(mu)):
        lam = growth_rate(x, y, mu)
        stable = bool(lam < 1e-9)
        d_small = np.hypot(x - 1 + mu, y)
        out[name] = {
            "x_a": x, "y_a": y,
            "from_big_a": np.hypot(x + mu, y),
            "from_small_a": d_small,
            "from_small_radii": d_small / r_small,
            "angle_deg": np.degrees(np.arctan2(y, x + mu)),
            "jacobi": 2 * -u_eff(x, y, mu),
            "u_eff": u_eff(x, y, mu),
            "shape": landscape_shape(x, y, mu), "stable": stable,
            "efold_orbits": None if stable
            else 1 / lam / ORBIT,
            "grad": float(np.hypot(*grad_omega(x, y, mu))),
        }
    return out


def l4_periods(mu):
    """Short and long libration periods at L4, orbits."""
    lam = np.linalg.eigvals(linearise(*lagrange_points(mu)[3],
                                      mu))
    w = np.sort(np.abs(lam.imag))[::2]      # each pair once
    p = 2 * np.pi / w / ORBIT
    return {"short": float(p.min()), "long": float(p.max())}


def stand_in_numbers():
    mu = stand_in_mu()
    r_big, r_small = stand_in_radii()
    pts = point_table(mu, r_small)
    return {"mu": mu, "big_to_small": (1 - mu) / mu,
            "gargantua_radius_a": r_big,
            "planet_radius_a": r_small,
            "hill_radius_a": hill_radius(mu),
            "max_grad": max(p["grad"] for p in pts.values()),
            "l4_periods_orbits": l4_periods(mu),
            "points": pts}


def hook(mu):
    """Fly the L1 cloud and the L4 ring (stand-in)."""
    t_orb = np.arange(0.0, ORBITS + STEP / 2, STEP)
    radii = stand_in_radii()
    s1 = l1_cloud(mu, N_L1, GAP)
    s4 = l4_ring(mu, N_L4, AMP)
    l1, hit1 = fly_cloud(s1, t_orb * ORBIT, mu, radii)
    l4, hit4 = fly_cloud(s4, t_orb * ORBIT, mu, radii)
    t2 = np.linspace(0.0, LONG_ORBITS, 2701) * ORBIT
    l4_long, _ = fly_cloud(s4, t2, mu, radii)
    return dict(t_orbits=t_orb, s1=s1, s4=s4, l1=l1, l4=l4,
                hit1=hit1, hit4=hit4, l4_long=l4_long)


def hook_numbers(mu, h):
    pts = lagrange_points(mu)
    x_l1, l4 = pts[0, 0], pts[3]
    l1, t = h["l1"], h["t_orbits"]
    side = np.sign(l1[:, 0, :] - x_l1)
    flips = np.abs(np.diff(side[:, 1:], axis=1)) > 0
    planet = side[:, -1] > 0
    gaps = np.hypot(*np.diff(l1[:, :2, :], axis=0).transpose(
        1, 0, 2))                           # neighbours
    widest = np.nanmax(gaps, axis=0)
    d4 = np.hypot(h["l4"][:, 0] - l4[0], h["l4"][:, 1] - l4[1])
    d4l = np.hypot(h["l4_long"][:, 0] - l4[0],
                   h["l4_long"][:, 1] - l4[1])
    drift = [np.nanmax(np.abs(jacobi(c[i], mu) - jacobi(c[i][:, 0],
             mu))) for c in (h["l1"], h["l4"], h["l4_long"])
             for i in range(len(c))]
    return {
        "l1": {"n": N_L1, "gap_a": GAP,
               "start_offsets_a": (h["s1"][:, 0] - x_l1).tolist(),
               "n_planet_side": int(planet.sum()),
               "n_gargantua_side": int((~planet).sum()),
               "planet_side": planet.tolist(),
               "n_impacts": int(np.isfinite(h["hit1"]).sum()),
               "n_crossings": int(flips.sum()),
               "orbits_to_gap": {k: float(
                   t[np.argmax(widest > g)])
                   for k, g in SPLIT_AT.items()},
               "jacobi_start": float(jacobi(h["s1"][0], mu))},
        "l4": {"n": N_L4, "amp_a": AMP,
               "n_impacts": int(np.isfinite(h["hit4"]).sum()),
               "max_from_l4_a": float(np.nanmax(d4)),
               "max_from_l4_long_a": float(np.nanmax(d4l)),
               "long_orbits": LONG_ORBITS},
        "max_jacobi_drift": float(max(drift)),
    }


def real_numbers():
    """The pair with real numbers (Earth-like density)."""
    S = miller_real(RHO_EARTH)
    g1, g2 = gaps_in_radii(RHO_EARTH)
    R = planet_radius(G_MILLER, RHO_EARTH)
    pts = lagrange_points(S.mu)
    r_big = np.hypot(pts[:, 0] + S.mu, pts[:, 1])
    a = orbit_radius()
    rho_c = clearing_density()
    return {
        "gargantua_suns": M_GARGANTUA_SUNS,
        "surface_gravity_ms2": G_MILLER,
        "density_assumed_kgm3": RHO_EARTH,
        "sep_km": a / 1e3, "sep_au": a / AU,
        "planet_radius_km": R / 1e3,
        "planet_mass_earths": planet_gm(G_MILLER, RHO_EARTH)
        / MU_EARTH,
        "mu": S.mu, "big_to_small": S.big_to_small,
        "l1_in_planet_radii": g1, "l2_in_planet_radii": g2,
        "hill_in_planet_radii": hill_radius(S.mu) * a / R,
        "l1_gap_km": g1 * R / 1e3,
        "hill_radius_km": hill_radius(S.mu) * a / 1e3,
        "clearing_density_kgm3": rho_c,
        "clearing_vs_earth": rho_c / RHO_EARTH,
        "scan": {str(round(r)): {
            "planet_radius_km": planet_radius(G_MILLER, r) / 1e3,
            "l1_in_planet_radii": gaps_in_radii(r)[0]}
            for r in DENSITIES},
        "points_from_gargantua_a": {
            "values": r_big.tolist(),
            "max_dev": float(np.abs(r_big - 1).max())},
        "mu_over_routh": S.mu / routh_mu(),
        "newton_speed_over_c": np.sqrt(MU_GARGANTUA / a)
        / C_LIGHT,
        "newton_period_h": newton_period(MU_GARGANTUA, a) / HOUR,
        "kerr_period_h": kerr_period(MU_GARGANTUA, a, 1.0) / HOUR,
        "hours_per_7_years": HOURS_PER_7_YEARS,
    }


def km_table(S):
    """Validation: L1..L5 of a real system in km/days."""
    mu, km, day = S.mu, S.sep / 1e3, S.time_unit / DAY
    out = {}
    for name, (x, y) in zip(NAMES, lagrange_points(mu)):
        lam = growth_rate(x, y, mu)
        out[name] = {
            "from_big_km": np.hypot(x + mu, y) * km,
            "from_small_km": np.hypot(x - 1 + mu, y) * km,
            "stable": bool(lam < 1e-9),
            "efold_days": None if lam < 1e-9 else day / lam}
    return out


def validation():
    em = km_table(earth_moon())
    for name, km in SOURCED_EM_KM.items():
        em[name]["sourced_km"] = km
        em[name]["diff_vs_sourced_km"] = em[name]["from_big_km"] - km
    sj = km_table(sun_jupiter())
    sj["big_to_small"] = sun_jupiter().big_to_small
    return {"em": em, "se": km_table(sun_earth()), "sj": sj}


def results():
    mu = stand_in_mu()
    res = {"stand_in": stand_in_numbers()}
    res["routh"] = {"mu": routh_mu(), "ratio": routh_ratio(),
                    "ratio_closed_form": (25 + 3 * 69**0.5) / 2}
    h = hook(mu)
    res["hook"] = hook_numbers(mu, h)
    res["real"] = real_numbers()
    res["validation"] = validation()
    return res, h


def write_hook(mu, h, res):
    pts = lagrange_points(mu)
    tr = lambda a: a[:, :2, :].transpose(0, 2, 1)
    vel = lambda a: a[:, 2:, :].transpose(0, 2, 1)
    arrays = {
        "t_orbits": h["t_orbits"],
        "l1_xy_a": tr(h["l1"]), "l1_v": vel(h["l1"]),
        "l4_xy_a": tr(h["l4"]), "l4_v": vel(h["l4"]),
        "l1_planet_side": np.array(res["hook"]["l1"]["planet_side"]),
        "lpoints_xy_a": pts,
        "gargantua_xy_a": np.array([-mu, 0.0]),
        "planet_xy_a": np.array([1 - mu, 0.0]),
        "radii_a": np.array(stand_in_radii()),
    }
    np.savez_compressed(OUT / "hook.npz", **arrays)
    desc = {
        "t_orbits": ("orbits", "time since release, in orbits of "
                     "the planet; uniform step 0.01"),
        "l1_xy_a": ("a", "L1 craft positions [craft, time, (x, y)];"
                    " NaN after an impact (none in this run)"),
        "l1_v": ("n a", "L1 craft velocities, rotating frame"),
        "l4_xy_a": ("a", "L4 ring positions [craft, time, (x, y)]"),
        "l4_v": ("n a", "L4 ring velocities, rotating frame"),
        "l1_planet_side": ("bool", "per L1 craft: ends on the "
                           "planet's side of L1 (x > x_L1)"),
        "lpoints_xy_a": ("a", "L1..L5 positions [point, (x, y)]"),
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
                 "model units); velocity in n a (the pair's "
                 "orbital speed)",
        "time_base": "t_orbits, shared by every time axis; the "
                     "integration is continuous (no wrap)",
        "model": "planar circular restricted three-body problem, "
                 "scipy DOP853 rtol 1e-12 atol 1e-13; Newtonian "
                 "stand-in (no relativity), no other bodies",
        "arrays": {k: {"shape": list(np.shape(v)),
                       "units": desc[k][0], "what": desc[k][1]}
                   for k, v in arrays.items()},
        "scenes": {
            "l1_cloud": "10 craft at rest in the rotating frame, x "
                        "offsets -4.5e-6..+4.5e-6 a (1e-6 a apart) "
                        "about L1; craft index = offset order",
            "l4_ring": "6 craft on L4's slow swing, x amplitude "
                       "0.05 a, phases 60 deg apart; slow period "
                       f"{res['stand_in']['l4_periods_orbits']['long']:.2f}"
                       " orbits (not exactly periodic: nonlinear)",
        },
        "labels": {n: f"{p['from_big_a']:.3f} a from Gargantua"
                   for n, p in res["stand_in"]["points"].items()},
        "summary": res["hook"],
    }
    (OUT / "hook.json").write_text(json.dumps(clean(side),
                                              indent=1))


def write_landscape(mu, res):
    x, y, U = ueff_grid(mu, GRID_HALF, GRID_N)
    pt = res["stand_in"]["points"]
    arrays = {
        "x_a": x, "y_a": y, "u_eff": U,
        "lpoints_xy_a": lagrange_points(mu),
        "lpoints_u_eff": np.array([pt[n]["u_eff"] for n in NAMES]),
        "gargantua_xy_a": np.array([-mu, 0.0]),
        "planet_xy_a": np.array([1 - mu, 0.0]),
    }
    np.savez_compressed(OUT / "landscape.npz", **arrays)
    side = {
        "stand_in": "Gargantua + Miller's planet, Earth-Moon "
                    "proportions",
        "frame": "rotating frame, origin at the barycentre, +x "
                 "from Gargantua to the planet",
        "time_base": "static (no time axis)",
        "grid": f"u_eff[i, j] is at (x_a[j], y_a[i]), {GRID_N} x "
                f"{GRID_N}",
        "u_eff_def": "U_eff = -[(x^2+y^2)/2 + (1-mu)/r1 + mu/r2] "
                     "in units of (n a)^2; -> -inf at the bodies: "
                     "clip for display (e.g. below the L4 level "
                     "minus 1.5)",
        "contour_levels": "lpoints_u_eff: contours through L1..L5 "
                          "show the saddles and hilltops",
        "arrays": {k: {"shape": list(np.shape(v)),
                       "units": "(n a)^2" if "u_eff" in k
                       else "a"} for k, v in arrays.items()},
        "points": {n: {"x_a": pt[n]["x_a"], "y_a": pt[n]["y_a"],
                       "shape": pt[n]["shape"],
                       "stable": pt[n]["stable"]} for n in NAMES},
    }
    (OUT / "landscape.json").write_text(json.dumps(clean(side),
                                                   indent=1))


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
    s = res["stand_in"]
    print(f"stand-in: mu {s['mu']:.7f}, ratio "
          f"{s['big_to_small']:.2f}, planet radius "
          f"{s['planet_radius_a']:.6f} a")
    for n, p in s["points"].items():
        ef = p["efold_orbits"]
        print(f"  {n}: {p['from_big_a']:.4f} a from Gargantua, "
              f"{p['from_small_a']:.4f} a = "
              f"{p['from_small_radii']:.1f} R from the planet, "
              f"{p['shape']:7s} " + ("stable" if p["stable"] else
                                     f"e-fold {ef:.3f} orbits"))
    h = res["hook"]
    print(f"hook: {h['l1']['n_planet_side']} planet side, "
          f"{h['l1']['n_gargantua_side']} Gargantua side; gap > "
          f"0.1 a after {h['l1']['orbits_to_gap']['tenth']} orbits;"
          f" L4 max {h['l4']['max_from_l4_a']:.4f} a")
    r = res["real"]
    print(f"real: mu {r['mu']:.3e}, ratio {r['big_to_small']:.3e},"
          f" L1 at {r['l1_in_planet_radii']:.3f} R, clears above "
          f"{r['clearing_density_kgm3']:,.0f} kg/m^3; P Newton "
          f"{r['newton_period_h']:.3f} h, Kerr "
          f"{r['kerr_period_h']:.3f} h")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res, h = results()
    res = clean(res)
    mu = stand_in_mu()
    write_hook(mu, h, res)
    write_landscape(mu, res)
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
