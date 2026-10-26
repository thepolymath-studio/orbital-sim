"""2026-10 wk4 (Gravity): the film's breakup cloud at 600 km.

One system for both posts: a satellite on a circular orbit 600 km
up (the film's title card, research F2) breaks up; the NASA
Standard Breakup Model (``breakup``) gives 500 pieces that stay in
orbit; ``cloud`` flies them under Earth's gravity + J2.

Assumptions (say them where the numbers are used):
- tilt 51.6 deg (the ISS's; the film does not give the Russian
  satellite's orbit). ``tilt_sensitivity`` shows other tilts;
- the breakup happens over the equator, heading north (ascending
  node on +x), at t = 0;
- pieces from 11 cm to 1 m (the trackable range); pieces whose
  lowest point is below 200 km fall back and are not followed;
- no drag, no Sun/Moon, no collisions between pieces.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .breakup import break_up, mass_for_count
from .cloud import (first_time_over, fitted_rate, mean_elements,
                    middle_range, pass_times, propagate_cloud,
                    read_elements, unwrap)
from .constants import DAY, MU_EARTH, R_EARTH_EQ
from .elements import circular_state, elements, kepler_period
from .j2 import node_rate

ALT = 600e3                    # m, film title card (F2)
INC = np.radians(51.6)         # assumption: the ISS's tilt
N_PIECES = 500
SEED = 2013                    # the film's year
DAY_SPAN, DAY_STEP = 2 * DAY, 60.0        # s
MONTHS, MONTH_STEP = 365, 1.0             # days
PASSES = (1, 2, 5, 10, 15)
SHARE = 0.9                    # "nine in ten pieces"


def parent(inc=INC):
    """The intact satellite at the breakup moment."""
    return circular_state(ALT, inc)


def film_breakup(seed=SEED, n=N_PIECES, inc=INC):
    """The cloud: n pieces that stay in orbit."""
    r0, v0 = parent(inc)
    return break_up(np.random.default_rng(seed), r0, v0, n)


def fly(cloud, t, j2=True, inc=INC):
    """Pieces plus the intact orbit (last row)."""
    r0, v0 = parent(inc)
    r = np.vstack([cloud.r, r0])
    v = np.vstack([cloud.v, v0])
    s = propagate_cloud(r, v, t, j2)
    return s[:-1], s[-1:]


def orbit_numbers():
    """Speed and period of the circular 600 km orbit."""
    r = R_EARTH_EQ + ALT
    v = np.sqrt(MU_EARTH / r)
    T = kepler_period(r)
    rate = node_rate(r, 0.0, INC)
    return {"alt_km": ALT / 1e3, "radius_km": r / 1e3,
            "speed_kms": v / 1e3, "speed_kmh": v * 3.6,
            "period_min": T / 60, "orbits_per_day": DAY / T,
            "node_rate_deg_day": np.degrees(rate) * DAY,
            "full_turn_days": 2 * np.pi / abs(rate) / DAY}


def along_track(el, el_parent):
    """How far each piece is ahead of the intact orbit
    (rad, unwrapped; pieces x times)."""
    return unwrap(el["lam"]) - unwrap(el_parent["lam"])


def ring_numbers(t, el, el_parent):
    """Clump -> ring: spread along the orbit over time."""
    d = along_track(el, el_parent)
    w90 = middle_range(d, SHARE)
    full = d.max(axis=0) - d.min(axis=0)
    return {"t": t, "spread90_rad": w90, "spread_full_rad": full,
            "ring_h": first_time_over(t, w90, 2 * np.pi) / 3600,
            "head_meets_tail_h":
                first_time_over(t, full, 2 * np.pi) / 3600}


def kepler_ring_h(cloud):
    """Ring time from period differences alone (Kepler):
    the middle 90 % of mean motions lap once."""
    el = elements(cloud.r, cloud.v)
    n = np.sqrt(MU_EARTH / el["a"] ** 3)
    return 2 * np.pi / middle_range(n, SHARE) / 3600


def pass_numbers(t, el, el_p, s, ks=PASSES):
    """Each return to the breakup spot: when, how spread.

    The spot: the ascending node, where every piece's
    orbit started. A return = the piece's angle along its
    orbit (u, from its node) reaches 2 pi k.
    """
    u, up = unwrap(el["u"]), unwrap(el_p["u"])
    raan = unwrap(el["raan"])
    r = np.linalg.norm(s[..., :3], axis=-1)
    r0 = R_EARTH_EQ + ALT
    out = {}
    for k in ks:
        tk = pass_times(t, u, k)
        tp = pass_times(t, up, k)[0]
        off = (tk - tp) / 60                       # min
        dO = np.array([np.interp(a, t, o) for a, o in
                       zip(tk, raan)])
        rc = np.array([np.interp(a, t, x) for a, x in
                       zip(tk, r)])
        miss = np.sqrt(rc**2 + r0**2 - 2 * rc * r0 * np.cos(dO))
        out[k] = {
            "parent_return_min": tp / 60,
            "median_return_min": np.nanmedian(tk) / 60,
            "window50_min": middle_range(off, 0.5),
            "window90_min": middle_range(off, SHARE),
            "share_within_1min": np.mean(np.abs(off) <= 1.0),
            "miss_median_km": np.nanmedian(miss) / 1e3,
            "_offsets_min": off}
    return out


def shell_numbers(t, el):
    """Ring -> shell: how far the orbits' nodes fan out."""
    raan = unwrap(el["raan"])
    w90 = middle_range(raan, SHARE)
    w50 = middle_range(raan, 0.5)
    return {"t": t, "spread90_rad": w90, "spread50_rad": w50,
            "shell_days": first_time_over(t, w90, 2 * np.pi)
            / DAY,
            "half_shell_days": first_time_over(t, w90, np.pi)
            / DAY}


def precession_check(t, el):
    """Simulated node drift of each piece vs the closed
    form evaluated at its time-averaged a, e, i."""
    sim = fitted_rate(t, unwrap(el["raan"]))
    m = mean_elements(el)
    cf = node_rate(m["a"], m["e"], m["i"])
    return sim, cf


def tilt_sensitivity(tilts_deg=(28.5, 51.6, 82.5)):
    """Closed-form shell time at other tilts, same seed."""
    out = {}
    for d in tilts_deg:
        c = film_breakup(inc=np.radians(d))
        el = elements(c.r, c.v)
        rate = node_rate(el["a"], el["e"], el["i"])
        w = middle_range(rate, SHARE)
        out[str(d)] = {"shell_days": 2 * np.pi / w / DAY,
                       "spread90_deg_day":
                           np.degrees(w) * DAY}
    return out


def cloud_mass(cloud):
    """Breakup mass (kg) the model needs for this count."""
    return mass_for_count(cloud.n_drawn)


@dataclass
class Run:
    """Both time windows of one cloud (states: pieces x
    times x 6, SI; the intact orbit separately)."""
    cloud: object
    t_day: np.ndarray          # s, 0..2 days, 60 s
    day: np.ndarray
    day_parent: np.ndarray
    t_months: np.ndarray       # s, 0..365 days, 1 day
    months: np.ndarray
    months_parent: np.ndarray
    day_no_j2: np.ndarray | None = None
    day_parent_no_j2: np.ndarray | None = None


def run_cloud(control=False):
    """Fly the film's cloud: two days finely, a year daily.

    control=True also flies the two days without J2.
    """
    c = film_breakup()
    td = np.arange(0.0, DAY_SPAN + 1, DAY_STEP)
    tm = np.arange(0.0, MONTHS + 0.5, MONTH_STEP) * DAY
    run = Run(c, td, *fly(c, td), tm, *fly(c, tm))
    if control:
        run.day_no_j2, run.day_parent_no_j2 = fly(c, td, False)
    return run
