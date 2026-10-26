"""Propagate a debris cloud under Earth's gravity + J2, and read it.

All pieces fly in one scipy DOP853 integration (the state stacks
every piece's position and velocity), with the force from
``j2.earth_accel``. No drag, no Moon or Sun, no collisions between
pieces. The tests check this batch run against one-piece runs at
the repo's tightest tolerance (propagate.RTOL), and the conserved
energy and polar angular momentum of every piece.

Shapes: states are (pieces, times, 6) = (x, y, z, vx, vy, vz) in
m and m/s, ECI frame, z along Earth's spin axis.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .elements import elements
from .j2 import earth_accel

RTOL = 1e-11
ATOL = 1e-6          # m and m/s


def cloud_rhs(t, y, j2):
    """d/dt of every piece's (r, v): (v, a)."""
    s = y.reshape(-1, 6)
    a = earth_accel(s[:, :3], j2)
    return np.hstack([s[:, 3:], a]).ravel()


def propagate_cloud(r, v, t_eval, j2=True, rtol=RTOL):
    """States (pieces, times, 6) at the times t_eval (s)."""
    y0 = np.hstack([r, v]).ravel()
    sol = solve_ivp(cloud_rhs, (t_eval[0], t_eval[-1]), y0,
                    method="DOP853", t_eval=t_eval, rtol=rtol,
                    atol=ATOL, args=(j2,))
    if not sol.success:
        raise RuntimeError(sol.message)
    n = len(r)
    return sol.y.reshape(n, 6, -1).transpose(0, 2, 1)


def read_elements(states):
    """Elements of every piece at every time, (pieces, times)."""
    return elements(states[..., :3], states[..., 3:])


def unwrap(angle):
    """Continuous angle along time (axis 1), from 0 at t=0."""
    a = np.unwrap(angle, axis=1)
    return a - a[:, :1]


def pass_times(t, u, k):
    """Time of each piece's k-th return to the start.

    u: unwrapped angle along the orbit from the breakup
    point (rad, pieces x times). The k-th return is where
    u reaches 2 pi k (linear interpolation; NaN if later
    than t[-1]).
    """
    goal = 2 * np.pi * k
    out = np.full(len(u), np.nan)
    for p, up in enumerate(u):
        j = np.searchsorted(up, goal)
        if 0 < j < up.size:
            f = (goal - up[j - 1]) / (up[j] - up[j - 1])
            out[p] = t[j - 1] + f * (t[j] - t[j - 1])
    return out


def middle_range(x, share=0.9, axis=0):
    """Width of the middle `share` of x (e.g. 90 %)."""
    lo, hi = 50 * (1 - share), 50 * (1 + share)
    p = np.nanpercentile(x, [lo, hi], axis=axis)
    return p[1] - p[0]


def fitted_rate(t, angle):
    """Least-squares slope of each row of angle(t)."""
    tc = t - t.mean()
    ac = angle - angle.mean(axis=1, keepdims=True)
    return ac @ tc / (tc @ tc)


def mean_elements(el):
    """Time-averaged a, e, i of each piece (pieces,)."""
    return {k: el[k].mean(axis=1) for k in ("a", "e", "i")}


def first_time_over(t, x, level):
    """First t where x >= level (NaN if never)."""
    hit = np.flatnonzero(x >= level)
    return t[hit[0]] if hit.size else np.nan
