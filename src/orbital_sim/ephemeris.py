"""Circular, coplanar planet ephemerides (the race's idealisation).

Each planet moves on a circle at constant angular rate n = sqrt(mu/r^3).
The angle origin is Earth at t = 0 (on +x), so a planet's state is set
by its radius and its lead angle over Earth at t = 0.

Mean longitudes for the nominal phasing: JPL "Approximate Positions of
the Planets" (E. M. Standish), Table 1 (1800-2050 AD), J2000 elements:
    Earth-Moon barycentre  L = 100.46457166 + 35999.37244981 T  (deg)
    Mars                   L =  -4.55343205 + 19140.30268499 T  (deg)
with T in Julian centuries of TDB from J2000.0. In a circular model the
mean longitude is the consistent angle to use (no equation of centre).
"""
from __future__ import annotations

from datetime import date

import numpy as np

from .constants import MU_SUN

L0_DEG = {"earth": 100.46457166, "mars": -4.55343205}
LDOT_DEG_PER_CENTURY = {"earth": 35999.37244981,
                        "mars": 19140.30268499}
J2000 = date(2000, 1, 1)     # J2000.0 is 2000-01-01 12:00 TT


def mean_motion(radius, mu=MU_SUN):
    """Angular rate of a circular orbit, rad/s."""
    return np.sqrt(mu / radius ** 3)


def planet_state(t, radius, lead, mu=MU_SUN):
    """Position and velocity (m, m/s) on a circle at time t (s).

    lead : angle ahead of Earth's t = 0 position (rad).
    Works on scalar or array t; returns arrays shaped (2, ...).
    """
    n = mean_motion(radius, mu)
    th = lead + n * np.asarray(t, float)
    r = radius * np.array([np.cos(th), np.sin(th)])
    v = radius * n * np.array([-np.sin(th), np.cos(th)])
    return r, v


def mean_longitude(body, day):
    """Mean longitude (deg, 0-360) at 00:00 TT on a date."""
    d = (day - J2000).days - 0.5
    T = d / 36525.0
    L = L0_DEG[body] + LDOT_DEG_PER_CENTURY[body] * T
    return L % 360.0


def mars_lead(day):
    """Mars's mean-longitude lead over Earth on a date (rad)."""
    lead = mean_longitude("mars", day) - mean_longitude("earth", day)
    return np.radians(lead % 360.0)
