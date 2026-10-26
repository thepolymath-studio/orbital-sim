"""Osculating orbital elements from Cartesian states (vectorised).

Frame: inertial, z along Earth's spin axis; SI units, angles in
radians. Inputs r, v have shape (..., 3); every element comes back
with shape (...). Used to read the debris cloud: its node (where
each orbit crosses the equator going north, Omega), its tilt (i)
and how far along its orbit each piece is (u, lam).

Standard textbook relations (e.g. Vallado, Fundamentals of
Astrodynamics, algorithm 9 "RV2COE"); the tests check the round
trip against ``state_from_elements``.
"""
from __future__ import annotations

import numpy as np

from .constants import MU_EARTH, R_EARTH_EQ


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def elements(r, v, mu=MU_EARTH):
    """Dict of a, e, i, raan, argp, nu, u, M, lam.

    u = argp + nu (argument of latitude: angle from the
    node along the orbit); lam = argp + M (its mean
    version, which grows steadily). Angles in [0, 2pi).
    """
    rn = np.linalg.norm(r, axis=-1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    a = 1.0 / (2.0 / rn - _dot(v, v) / mu)
    ev = np.cross(v, h) / mu - r / rn[..., None]
    e = np.linalg.norm(ev, axis=-1)
    i = np.arccos(np.clip(h[..., 2] / hn, -1, 1))
    raan = np.arctan2(h[..., 0], -h[..., 1])
    node = np.stack([np.cos(raan), np.sin(raan),
                     np.zeros_like(raan)], axis=-1)
    w = np.cross(h, node) / hn[..., None]   # in-plane, 90 deg on
    u = np.arctan2(_dot(r, w), _dot(r, node))
    argp = np.arctan2(_dot(ev, w), _dot(ev, node))
    nu = u - argp
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e))
                      * np.tan(nu / 2))
    M = E - e * np.sin(E)
    tau = 2 * np.pi
    return {"a": a, "e": e, "i": i, "raan": raan % tau,
            "argp": argp % tau, "nu": nu % tau, "u": u % tau,
            "M": M % tau, "lam": (argp + M) % tau}


def state_from_elements(a, e, i, raan, argp, nu, mu=MU_EARTH):
    """Position and velocity (m, m/s) from elements."""
    p = a * (1 - e**2)
    rn = p / (1 + e * np.cos(nu))
    r_pf = rn * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array(
        [-np.sin(nu), e + np.cos(nu), 0.0])
    Q = _rot3(-raan) @ _rot1(-i) @ _rot3(-argp)
    return Q @ r_pf, Q @ v_pf


def _rot1(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _rot3(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def circular_state(alt, inc, mu=MU_EARTH, R=R_EARTH_EQ):
    """Circular orbit at altitude alt (m), tilt inc (rad).

    Starts at the ascending node on +x (over the equator,
    heading north-east), so raan = 0 and u = 0.
    """
    r = R + alt
    v = np.sqrt(mu / r)
    return (np.array([r, 0.0, 0.0]),
            v * np.array([0.0, np.cos(inc), np.sin(inc)]))


def kepler_period(a, mu=MU_EARTH):
    """Two-body period 2 pi sqrt(a^3/mu) (s)."""
    return 2 * np.pi * np.sqrt(a**3 / mu)


def perigee_radius(r, v, mu=MU_EARTH):
    """Closest distance to Earth's centre, a (1 - e) (m)."""
    el = elements(r, v, mu)
    return el["a"] * (1 - el["e"])


def orbit_paths(r, v, n=96, mu=MU_EARTH):
    """Points on each piece's current (osculating) ellipse.

    r, v: (pieces, 3). Returns (pieces, n, 3) in m: the
    two-body orbit through the state, sampled evenly in
    the angle from the node, for drawing rings.
    """
    el = elements(r, v, mu)
    raan, i = el["raan"][:, None], el["i"][:, None]
    p = (el["a"] * (1 - el["e"] ** 2))[:, None]
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)[None]
    nu = u - el["argp"][:, None]
    rad = p / (1 + el["e"][:, None] * np.cos(nu))
    cu, su = np.cos(u), np.sin(u)
    cO, sO, ci = np.cos(raan), np.sin(raan), np.cos(i)
    x = rad * (cO * cu - sO * su * ci)
    y = rad * (sO * cu + cO * su * ci)
    z = rad * (su * np.sin(i))
    return np.stack([x, y, z], axis=-1)
