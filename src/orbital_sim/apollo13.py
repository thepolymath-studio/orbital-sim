"""Apollo 13 around the Moon and home: vis-viva on real telemetry.

Patched conics, one body at a time:

  * Moon leg: Moon-centred two-body hyperbola, from the Flight
    Journal's Moon-relative (altitude, speed) calls between
    75:06 GET and the PC+2 burn.
  * Earth leg: Earth-centred two-body conic, from the calls after
    the last course correction, down to the entry interface.

One number fixes each conic: its energy (equivalently a, or v_inf
for the hyperbola). The fit finds the single energy that best
matches every call; vis-viva then predicts the speed anywhere.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, least_squares

from .constants import FT, NMI
from .dynamics import no_thrust
from .propagate import fly
from .visviva import energy, hyperbola_speed, vis_viva


def get_seconds(get):
    """'hhh:mm[:ss.s]' ground elapsed time -> seconds."""
    parts = [float(p) for p in get.split(":")]
    parts += [0.0] * (3 - len(parts))
    h, m, s = parts
    return 3600.0 * h + 60.0 * m + s


def to_radius_speed(alt_nmi, speed_fps, body_radius):
    """A call's (altitude nmi, ft/s) -> (r, v) in SI.

    r is measured from the body's centre: altitude
    plus the body's radius.
    """
    alt = np.asarray(alt_nmi, float) * NMI
    v = np.asarray(speed_fps, float) * FT
    return alt + body_radius, v


def fit_energy(r, v, mu):
    """One conic through every (r, v): best energy.

    Least squares on the relative speed error of
    v = sqrt(2 (eps + mu/r)), i.e. vis-viva.
    """
    def miss(eps):
        return np.sqrt(2 * (eps[0] + mu / r)) / v - 1

    eps0 = np.median(energy(r, v, mu))
    return least_squares(miss, [eps0]).x[0]


def speed_misfit(r, v, eps, mu):
    """Relative speed error of each call vs the conic."""
    return np.sqrt(2 * (eps + mu / r)) / v - 1


def pericynthion_state(v_inf, r_p, mu):
    """Start state at closest approach.

    Perifocal frame: Moon at the origin, +x through
    pericynthion, motion counter-clockwise.
    """
    v_p = hyperbola_speed(r_p, v_inf, mu)
    return np.array([r_p, 0.0, 0.0, v_p, 0.0])


def fly_pass(v_inf, r_p, t_before, t_after, mu):
    """Integrate the hyperbola both ways from closest
    approach (t = 0): back to -t_before, on to +t_after.

    Returns the two dense solutions (inbound, outbound).
    """
    y0 = pericynthion_state(v_inf, r_p, mu)
    inbound = fly(y0, (0.0, -t_before), no_thrust, mu)
    outbound = fly(y0, (0.0, t_after), no_thrust, mu)
    return inbound, outbound


def sample_pass(inbound, outbound, dt):
    """Uniform time grid across both halves of the pass.

    Returns t (s from pericynthion) and states (5, n).
    """
    t0, t1 = inbound.t[-1], outbound.t[-1]
    t = np.arange(np.ceil(t0 / dt), np.floor(t1 / dt) + 1)
    t = t * dt
    Y = np.where(t < 0, inbound.sol(np.minimum(t, 0)),
                 outbound.sol(np.maximum(t, 0)))
    return t, Y


def time_at_radius(sol, r):
    """Time when one half of the pass is at distance r."""
    def gap(t):
        y = sol.sol(t)
        return np.hypot(y[0], y[1]) - r

    lo, hi = sorted((0.0, sol.t[-1]))
    return brentq(gap, lo, hi, xtol=1e-6)


def entry_speed(eps, r_entry, mu):
    """Vis-viva at the entry interface on the return conic.

    The fitted energy fixes a = -mu / (2 eps); then
    v^2 = mu (2/r - 1/a).
    """
    a = -mu / (2.0 * eps)
    return vis_viva(r_entry, a, mu)


def earth_energy_change(v_moon, v_inf, e, x):
    """Patched conic: Earth-frame energy the pass adds.

    The Moon turns v_inf; the change has size
    2 v_inf / e and points away from pericynthion.
    Seen from Earth the energy changes by V . dv_inf.
    x: pericynthion's angle from the anti-Earth point
    toward the Moon's trailing side (rad). x > 0:
    behind the Moon, gain; x < 0: ahead of it, loss.
    """
    dv_inf = 2.0 * v_inf / e
    return v_moon * dv_inf * np.sin(x)


def trailing_offset(peri_lon, sub_earth_lon):
    """x for earth_energy_change, from selenographic
    east longitudes (deg). The trailing apex is 90 E,
    so moving from the anti-Earth point toward smaller
    east longitude is moving behind the Moon."""
    anti = sub_earth_lon + 180.0
    x = (anti - peri_lon + 180.0) % 360.0 - 180.0
    return np.radians(x)


def burn_angle(v_before, dv, v_after):
    """Angle between a burn and the velocity it adds to,
    from the three speeds (law of cosines)."""
    c = (v_after ** 2 - v_before ** 2 - dv ** 2)
    return np.arccos(c / (2.0 * v_before * dv))
