"""Gargantua and Miller's planet as a restricted three-body pair.

Interstellar, 2026-10 wk3. Two views of the same pair:

* the stand-in: the pair "treated as if it were Earth and Moon".
  Earth-Moon mass ratio and body sizes, quoted in the pair's own
  units: distance = the planet's distance from Gargantua, time =
  orbits of the planet. Absolute km and days are not quoted,
  because they would be Earth-Moon numbers;
* the real numbers: Thorne's 1e8 suns, 130 % of Earth's gravity,
  and the planet at Gargantua's radius GM/c^2 (Thorne: the orbit
  hugs the horizon). Newton's model then puts L1 and L2 inside
  the planet, at about half its radius.

Newton is only a stand-in at that distance: there his orbital
speed equals the speed of light. Relativity is needed for the orbit
itself (``kerr_period``). It is not why the planet survives: the
relativistic tide there is stronger, not weaker.
"""
from __future__ import annotations

import numpy as np

from .constants import (C_LIGHT, D_MOON, G, G_MILLER,
                        MU_GARGANTUA, R_EARTH, R_MOON,
                        RHO_EARTH)
from .cr3bp import System, earth_moon
from .lagrange import lagrange_points

ORBIT = 2 * np.pi        # normalised time units per orbit


def stand_in_mu():
    """The stand-in's mass ratio: Earth-Moon's."""
    return earth_moon().mu


def stand_in_radii():
    """(Gargantua, planet) drawn sizes in the stand-in,
    as shares of their distance: Earth-Moon's."""
    return (R_EARTH / D_MOON, R_MOON / D_MOON)


def orbit_radius(gm=MU_GARGANTUA):
    """Thorne's radius of Gargantua, GM/c^2 (m). The
    real planet orbits about this far from Gargantua."""
    return gm / C_LIGHT**2


def planet_radius(g, rho):
    """Radius (m) of a uniform ball of density rho
    with surface gravity g: g = (4/3) pi G rho R."""
    return 3 * g / (4 * np.pi * G * rho)


def planet_gm(g, rho):
    """GM of that ball (m^3/s^2): g R^2."""
    return g * planet_radius(g, rho) ** 2


def miller_real(rho=RHO_EARTH):
    """The pair with real numbers: Gargantua, and a
    planet of density rho at 130 % of Earth's g."""
    return System("Gargantua-Miller (real)",
                  MU_GARGANTUA, planet_gm(G_MILLER, rho),
                  orbit_radius())


def gaps_in_radii(rho):
    """L1's and L2's distance from the planet's centre,
    in planet radii. Below 1 means inside the planet."""
    S = miller_real(rho)
    x1, x2 = lagrange_points(S.mu)[:2, 0]
    small = 1 - S.mu
    R = planet_radius(G_MILLER, rho) / S.sep
    return (small - x1) / R, (x2 - small) / R


def clearing_density(gm=MU_GARGANTUA, a=None):
    """Density (kg/m^3) at which the Hill radius
    a (m / 3M)^(1/3) equals the planet's radius, so
    L1/L2 reach the surface: 9 GM / (4 pi G a^3)."""
    a = orbit_radius(gm) if a is None else a
    return 9 * gm / (4 * np.pi * G * a**3)


def newton_period(gm, r):
    """Kepler's period (s) of a circular orbit."""
    return 2 * np.pi * np.sqrt(r**3 / gm)


def kerr_period(gm, r, spin):
    """Period (s), seen from afar, of a prograde
    circular orbit in a spinning black hole's equator
    (Bardeen, Press & Teukolsky 1972):
    Omega = (c^3/GM) / (x^1.5 + spin), x = r c^2/GM.
    spin = 0 gives Kepler's period exactly."""
    t_g = gm / C_LIGHT**3
    x = r / (gm / C_LIGHT**2)
    return 2 * np.pi * t_g * (x**1.5 + spin)


__all__ = ["ORBIT", "clearing_density", "gaps_in_radii",
           "kerr_period", "miller_real", "newton_period",
           "orbit_radius", "planet_gm", "planet_radius",
           "stand_in_mu", "stand_in_radii"]
