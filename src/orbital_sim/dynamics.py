"""Planar heliocentric two-body motion with an optional thrust term.

Equation 06 of the post:  r'' = -mu r / r^3 + a_T

State vector y = [x, y, vx, vy, dv] in SI units (m, m/s). The fifth
component accumulates the thrust delta-v, dv = integral of |a_T| dt,
which is what equation 07 (dv = a_T t) describes.

A thrust law is any callable ``thrust(t, r, v) -> a_T`` (m/s^2, 2-vector).
"""
from __future__ import annotations

import numpy as np

from .constants import MU_SUN


def gravity(r, mu=MU_SUN):
    """Two-body acceleration -mu r / r^3 (eq. 01)."""
    return -mu * r / np.linalg.norm(r) ** 3


def no_thrust(t, r, v):
    """Thrust law for a coast: a_T = 0."""
    return np.zeros(2)


def prograde(a_T):
    """Thrust law: constant a_T along the velocity."""
    def thrust(t, r, v):
        return a_T * v / np.linalg.norm(v)
    return thrust


def rhs(t, y, mu, thrust):
    """Equation 06: r'' = -mu r / r^3 + a_T."""
    r, v = y[0:2], y[2:4]
    a_T = thrust(t, r, v)
    a = gravity(r, mu) + a_T
    dv_rate = np.linalg.norm(a_T)
    return np.concatenate([v, a, [dv_rate]])


def circular_state(radius, mu=MU_SUN):
    """State on a circular orbit, on +x, moving +y."""
    v_c = np.sqrt(mu / radius)
    return np.array([radius, 0.0, 0.0, v_c, 0.0])


def specific_energy(y, mu=MU_SUN):
    """Orbital energy per unit mass, v^2/2 - mu/r (J/kg).

    A diagnostic for the tests; its time derivative is v . a_T.
    """
    r = np.hypot(y[0], y[1])
    v2 = y[2] ** 2 + y[3] ** 2
    return 0.5 * v2 - mu / r
