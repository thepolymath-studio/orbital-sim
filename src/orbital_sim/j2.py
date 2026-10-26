"""Earth's gravity with its equatorial bulge (the J2 term), in 3D.

The perturbed two-body equation of the 2026-10 wk4 posts:

    r'' = -mu r / r^3 + a_J2

    a_J2 = -(3/2) J2 mu R^2 / r^5 * [(1 - 5 z^2/r^2) x,
                                     (1 - 5 z^2/r^2) y,
                                     (3 - 5 z^2/r^2) z]

Frame: Earth-centred inertial (ECI), z along Earth's spin axis.
Units: SI (m, s). Positions may be one vector (3,) or a stack
(..., 3); every function works on the last axis.

a_J2 is minus the gradient of the J2 part of the potential,

    Phi = -mu/r + (mu J2 R^2 / 2 r^3) (3 z^2/r^2 - 1),

so E = v^2/2 + Phi is conserved, and so is the z-component of the
angular momentum (the bulge is symmetric about the spin axis).
The tests check both, and a_J2 = -grad Phi by finite differences.

Orbit-averaged, J2 turns the orbital plane about the spin axis
(nodal precession), the closed form in ``node_rate``.
"""
from __future__ import annotations

import numpy as np

from .constants import (DAY, J2_EARTH, MU_EARTH, R_EARTH_EQ,
                        YEAR_TROPICAL)

MU, R_E, J2 = MU_EARTH, R_EARTH_EQ, J2_EARTH


def j2_accel(r, mu=MU, R=R_E, J2=J2):
    """Extra pull of Earth's bulge, a_J2 (m/s^2).

    r: position(s) from Earth's centre, z along the
    spin axis (m), shape (..., 3).
    """
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    k = -1.5 * J2 * mu * R**2 / r2**2.5
    s = 5.0 * z * z / r2
    a = (x * (1 - s), y * (1 - s), z * (3 - s))
    return k[..., None] * np.stack(a, axis=-1)


def kepler_accel(r, mu=MU):
    """Point-mass gravity -mu r / r^3 (m/s^2)."""
    r3 = np.linalg.norm(r, axis=-1, keepdims=True) ** 3
    return -mu * r / r3


def earth_accel(r, j2=True, mu=MU):
    """Kepler plus (optionally) the J2 term."""
    a = kepler_accel(r, mu)
    return a + j2_accel(r, mu) if j2 else a


def potential(r, j2=True, mu=MU, R=R_E, J2=J2):
    """Phi (J/kg) with a = -grad Phi."""
    rn = np.linalg.norm(r, axis=-1)
    phi = -mu / rn
    if j2:
        s = r[..., 2] ** 2 / rn**2
        phi = phi + mu * J2 * R**2 * (3 * s - 1) / (2 * rn**3)
    return phi


def energy(r, v, j2=True, mu=MU):
    """Specific energy v^2/2 + Phi (J/kg), conserved."""
    return 0.5 * np.sum(v * v, axis=-1) + potential(r, j2, mu)


def node_rate(a, e, i, mu=MU, R=R_E, J2=J2):
    """Closed-form nodal precession dOmega/dt (rad/s).

    dOmega/dt = -(3/2) n J2 (R/p)^2 cos i, with the mean
    motion n = sqrt(mu/a^3) and p = a (1 - e^2).
    a in m, i in rad. Negative = the plane turns west.
    """
    n = np.sqrt(mu / a**3)
    p = a * (1 - e**2)
    return -1.5 * n * J2 * (R / p) ** 2 * np.cos(i)


def sun_sync_inclination(a, e=0.0):
    """Inclination (rad) whose plane turns 360 deg a year."""
    want = 2 * np.pi / (YEAR_TROPICAL * DAY)
    cos_i = want / node_rate(a, e, 0.0)
    return np.arccos(cos_i)
