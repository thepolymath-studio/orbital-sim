"""Vis-viva: the energy form of the two-body law (eq. 01).

    v^2 = mu (2/r - 1/a)

v is the speed relative to the central body, r the distance from
its centre, a the semi-major axis. Ellipse: a > 0. Parabola:
a -> infinity (v = escape speed). Hyperbola: a < 0, and then

    v^2 = v_inf^2 + 2 mu / r,   v_inf^2 = -mu / a.

It follows from the specific orbital energy, which is constant on
every two-body orbit:  v^2/2 - mu/r = -mu/(2a).
"""
from __future__ import annotations

import numpy as np


def vis_viva(r, a, mu):
    """Speed at distance r on an orbit with axis a.

    v^2 = mu (2/r - 1/a). Pass a = np.inf for the
    parabola; a < 0 is a hyperbola.
    """
    return np.sqrt(mu * (2.0 / r - 1.0 / a))


def semi_major_axis(r, v, mu):
    """Invert vis-viva: a from one (r, v) pair."""
    return 1.0 / (2.0 / r - v ** 2 / mu)


def energy(r, v, mu):
    """Specific orbital energy v^2/2 - mu/r (J/kg)."""
    return 0.5 * v ** 2 - mu / r


def escape_speed(r, mu):
    """Vis-viva with a -> infinity: sqrt(2 mu / r)."""
    return np.sqrt(2.0 * mu / r)


def v_inf_squared(r, v, mu):
    """Hyperbolic excess, v_inf^2 = v^2 - 2 mu / r.

    Positive: the craft escapes (a hyperbola).
    """
    return v ** 2 - 2.0 * mu / r


def hyperbola_speed(r, v_inf, mu):
    """Vis-viva on a hyperbola, a = -mu / v_inf^2."""
    return np.sqrt(v_inf ** 2 + 2.0 * mu / r)
