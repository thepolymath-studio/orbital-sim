"""Apollo 13, alternate histories (2026-10-wk2 Sim post).

Same patched-conic model as apollo13.py: the Moon alone pulls on
the craft near the Moon, the Earth alone on the way home. Three
what-ifs, each decided by vis-viva or equation 09 and flown with
the integrator (propagate.fly, DOP853):

  1. a different closest approach, same v_inf: a different turn;
  2. no PC+2 burn: the Moon-relative v_inf the burn added, and the
     band of entry speeds any fall home from the Moon can have;
  3. no free-return burn: only a toy (a straight-line drift).

What this module does NOT do: say where a changed path would meet
the Earth, or when. That needs the Earth and the Moon pulling
together, and belongs to week 3.
"""
from __future__ import annotations

import numpy as np

from .dynamics import no_thrust
from .flyby import hyperbola_state, turn_angle
from .propagate import fly, reach_radius
from .visviva import hyperbola_speed, vis_viva


def aim_offset(v_inf, r_p, mu):
    """Impact parameter b: how far from the Moon's centre the
    incoming straight line points.

    Angular momentum is conserved: b v_inf = r_p v_p.
    """
    v_p = hyperbola_speed(r_p, v_inf, mu)
    return r_p * v_p / v_inf


def what_if_pass(v_inf, alt, mu, radius):
    """Same v_inf, a different closest approach (altitude).

    Returns e and the turn delta (eq. 09), the speed at
    closest approach (vis-viva) and the aim offset b.
    """
    r_p = radius + alt
    e = 1.0 + r_p * v_inf ** 2 / mu
    return {
        "r_p": r_p, "e": e,
        "delta": turn_angle(v_inf, r_p, mu),
        "v_p": hyperbola_speed(r_p, v_inf, mu),
        "b": aim_offset(v_inf, r_p, mu),
    }


def fly_aimed_pass(v_inf, r_p, mu, r_out):
    """Fly one pass that arrives moving along +x at v_inf.

    Every pass shares the arrival direction, so they overlay:
    only the aim (b) and the exit direction (delta) differ.
    Flown from closest approach, both ways, out to r_out.
    Returns the dense solutions (inbound, outbound).
    """
    turn = turn_angle(v_inf, r_p, mu)
    v_in = np.array([v_inf, 0.0])
    pos, vel = hyperbola_state(v_in, turn, r_p, mu, 0.0)
    y0 = np.concatenate([pos, vel, [0.0]])
    t_max = 50.0 * r_out / v_inf
    legs = [fly(y0, (0.0, s * t_max), no_thrust, mu,
                [reach_radius(r_out)]) for s in (-1, 1)]
    return legs[0], legs[1]


def flight_path_angle(r, v, r_p, v_p):
    """Angle of the velocity above the local horizontal, on
    the way out (rad).

    Angular momentum: r v cos(gamma) = r_p v_p.
    """
    c = np.clip(r_p * v_p / (r * v), -1.0, 1.0)
    return np.arccos(c)


def along_velocity(gamma, dv_lvlh):
    """Part of an LVLH burn that points along the velocity.

    LVLH: x forward along the local horizontal, y across the
    orbit plane, z toward the Moon's centre (down).
    """
    x, _, z = dv_lvlh
    return x * np.cos(gamma) - z * np.sin(gamma)


def burn_v_inf(r, v, gamma, dv_lvlh, mu):
    """v_inf after an impulse dv_lvlh at distance r, speed v.

    LVLH as in along_velocity. Only the part along the
    velocity adds speed: v'^2 = v^2 + 2 v dv_along + |dv|^2.
    Then vis-viva gives the speed far away:
    v_inf^2 = v'^2 - 2 mu / r.
    """
    x, y, z = dv_lvlh
    along = x * np.cos(gamma) - z * np.sin(gamma)
    v2 = v ** 2 + 2 * v * along + x * x + y * y + z * z
    return np.sqrt(v2 - 2.0 * mu / r)


def apply_burn(y, dv_lvlh):
    """Planar state y = [x, y, vx, vy, dv] after an LVLH
    impulse. The small out-of-plane part (y) is dropped:
    the model is planar.
    """
    r, v = y[0:2], y[2:4]
    up = r / np.linalg.norm(r)
    spin = np.sign(r[0] * v[1] - r[1] * v[0])
    fwd = spin * np.array([-up[1], up[0]])
    x, _, z = dv_lvlh
    out = np.array(y, dtype=float)
    out[2:4] = v + x * fwd - z * up
    return out


def fly_with_and_without(y_burn, dv_lvlh, t_span, mu):
    """From one state, fly on with and without the burn.

    Returns the dense solutions (with_burn, without_burn).
    """
    burnt = apply_burn(y_burn, dv_lvlh)
    with_burn = fly(burnt, (0.0, t_span), no_thrust, mu)
    without = fly(y_burn, (0.0, t_span), no_thrust, mu)
    return with_burn, without


def entry_speed_band(r_far, r_entry, mu):
    """Entry speed of ANY coast home that stays bound to the
    Earth and whose far point is r_far or beyond.

    Vis-viva at r_entry: the slowest has a = r_far / 2
    (dropped from rest at r_far), the fastest a -> infinity
    (escape speed). The path in between does not matter.
    """
    slow = vis_viva(r_entry, r_far / 2.0, mu)
    fast = vis_viva(r_entry, np.inf, mu)
    return slow, fast


def drift_without_gravity(dv, t):
    """TOY, not a prediction: how far a velocity
    error dv carries the craft in time t, if
    nothing pulled on it. The real answer needs
    the Earth and the Moon pulling together.
    """
    return dv * t
