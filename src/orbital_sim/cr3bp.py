"""Planar circular restricted three-body problem (CR3BP).

Two heavy bodies circle their barycentre; a third body is too light
to pull them back. In the frame that turns with the pair, both stay
put (research S3, Murray & Dermott ch. 3, Szebehely ch. 1):

    x'' - 2 y' = dOmega/dx,   y'' + 2 x' = dOmega/dy
    Omega = (x^2 + y^2)/2 + (1 - mu)/r1 + mu/r2,  U_eff = -Omega

Normalised units: separation 1, total mass 1, angular rate 1.
mu = m2 / (m1 + m2). The big body sits at (-mu, 0), the small one
at (1 - mu, 0). State s = [x, y, vx, vy] (rotating frame).
The -2y', +2x' terms are the Coriolis effect; x^2 + y^2 in Omega is
the centrifugal term. One quantity is conserved: the Jacobi
constant C_J = 2 Omega - v^2.

``System`` carries the physical scales so outputs can be labelled
in km, days and J/kg.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import (A_JUPITER, AU, D_MOON, MU_EARTH,
                        MU_JUPITER_SYS, MU_MOON, MU_SUN)

RTOL = 1e-12
ATOL = 1e-13


def distances(x, y, mu):
    """Distances r1, r2 to the big and the small body."""
    r1 = np.hypot(x + mu, y)
    r2 = np.hypot(x - 1 + mu, y)
    return r1, r2


def omega(x, y, mu):
    """Omega: centrifugal term plus both gravity wells."""
    r1, r2 = distances(x, y, mu)
    spin = 0.5 * (x**2 + y**2)
    return spin + (1 - mu) / r1 + mu / r2


def u_eff(x, y, mu):
    """The landscape U_eff = -Omega (energy per unit mass)."""
    return -omega(x, y, mu)


def grad_omega(x, y, mu):
    """(dOmega/dx, dOmega/dy): the push on a craft at rest."""
    r1, r2 = distances(x, y, mu)
    k1 = (1 - mu) / r1**3
    k2 = mu / r2**3
    gx = x - k1 * (x + mu) - k2 * (x - 1 + mu)
    gy = y - k1 * y - k2 * y
    return gx, gy


def rhs(t, s, mu):
    """Equations of motion in the rotating frame."""
    x, y, vx, vy = s
    gx, gy = grad_omega(x, y, mu)
    ax = gx + 2 * vy        # +2 vy: Coriolis
    ay = gy - 2 * vx        # -2 vx: Coriolis
    return np.array([vx, vy, ax, ay])


def jacobi(s, mu):
    """Jacobi constant C_J = 2 Omega - v^2."""
    x, y, vx, vy = s
    return 2 * omega(x, y, mu) - (vx**2 + vy**2)


def hessian_omega(x, y, mu):
    """Second derivatives (Oxx, Oxy, Oyy) of Omega."""
    r1, r2 = distances(x, y, mu)
    a1, a2 = (1 - mu) / r1**3, mu / r2**3
    b1, b2 = 3 * a1 / r1**2, 3 * a2 / r2**2
    dx1, dx2 = x + mu, x - 1 + mu
    oxx = 1 - a1 - a2 + b1 * dx1**2 + b2 * dx2**2
    oyy = 1 - a1 - a2 + (b1 + b2) * y**2
    oxy = (b1 * dx1 + b2 * dx2) * y
    return oxx, oxy, oyy


def hits_body(mu, r1_min, r2_min):
    """Terminal event: closer than either body's radius."""
    def event(t, s, mu_):
        r1, r2 = distances(s[0], s[1], mu)
        return min(r1 - r1_min, r2 - r2_min)
    event.terminal = True
    return event


def fly(s0, t_end, mu, t_eval=None, events=None):
    """Integrate the CR3BP (DOP853, rtol 1e-12)."""
    sol = solve_ivp(
        rhs, (0.0, t_end), np.asarray(s0, float),
        method="DOP853", rtol=RTOL, atol=ATOL,
        args=(mu,), t_eval=t_eval, events=events,
    )
    if sol.status == -1:
        raise RuntimeError(sol.message)
    return sol


def to_inertial(t, s, mu):
    """Rotating-frame states -> inertial (barycentric) frame."""
    x, y, vx, vy = s
    c, sn = np.cos(t), np.sin(t)
    X, Y = c * x - sn * y, sn * x + c * y
    ux, uy = vx - y, vy + x      # add omega x r
    VX, VY = c * ux - sn * uy, sn * ux + c * uy
    return np.array([X, Y, VX, VY])


@dataclass(frozen=True)
class System:
    """A primary pair with its physical scales (SI)."""
    name: str
    gm1: float          # m^3/s^2, big body
    gm2: float          # m^3/s^2, small body
    sep: float          # m, separation (circular)

    @property
    def mu(self):
        """Mass ratio mu = m2 / (m1 + m2)."""
        return self.gm2 / (self.gm1 + self.gm2)

    @property
    def big_to_small(self):
        """Mass ratio m1 / m2 (the Routh test's number)."""
        return self.gm1 / self.gm2

    @property
    def n(self):
        """Angular rate of the pair (rad/s), Kepler's law."""
        gm = self.gm1 + self.gm2
        return np.sqrt(gm / self.sep**3)

    @property
    def time_unit(self):
        """Seconds per normalised time unit, 1/n."""
        return 1.0 / self.n

    @property
    def speed_unit(self):
        """m/s per normalised speed unit, n * sep."""
        return self.n * self.sep

    @property
    def energy_unit(self):
        """J/kg per normalised U_eff unit, (n sep)^2."""
        return self.speed_unit**2


def earth_moon():
    """Earth-Moon: DE430 GMs, NSSDC mean distance."""
    return System("Earth-Moon", MU_EARTH, MU_MOON,
                  D_MOON)


def sun_earth():
    """Sun and Earth+Moon at 1 AU (the JWST pair)."""
    return System("Sun-Earth", MU_SUN,
                  MU_EARTH + MU_MOON, AU)


def sun_jupiter():
    """Sun-Jupiter: DE440 Jupiter-system GM, NSSDC a."""
    return System("Sun-Jupiter", MU_SUN, MU_JUPITER_SYS,
                  A_JUPITER)


__all__ = ["System", "distances", "earth_moon", "fly",
           "grad_omega", "hessian_omega", "hits_body",
           "jacobi", "omega", "rhs", "sun_earth",
           "sun_jupiter",
           "to_inertial", "u_eff"]
