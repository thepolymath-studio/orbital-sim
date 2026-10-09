"""Patched-conic planetary flyby (equations 08 and 09 of the post).

In the planet's frame the flyby is a hyperbola: the speed at infinity
is the same in and out (eq. 08), only its direction turns, by

    delta = 2 arcsin(1/e),   e = 1 + r_p v_inf^2 / mu_p      (eq. 09)

The flyby is instantaneous on the heliocentric time scale: the ship's
heliocentric velocity jumps from v_p + v_inf_in to v_p + v_inf_out.
"""
from __future__ import annotations

import numpy as np


def turn_angle(v_inf, r_p, mu):
    """Equation 09: delta = 2 arcsin(1/e), in rad."""
    e = 1.0 + r_p * v_inf ** 2 / mu
    return 2.0 * np.arcsin(1.0 / e)


def periapsis_for_turn(v_inf, delta, mu):
    """Inverse of eq. 09: r_p (m) that gives turn delta."""
    e = 1.0 / np.sin(abs(delta) / 2.0)
    return (e - 1.0) * mu / v_inf ** 2


def rotate(vec, angle):
    """Rotate a 2-vector counter-clockwise by angle (rad)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vec[0] - s * vec[1],
                     s * vec[0] + c * vec[1]])


def flyby(v_ship, v_planet, psi):
    """Heliocentric velocity after a flyby that turns v_inf by psi.

    Equation 08: |v_inf| is unchanged; only its direction turns.
    psi > 0 turns counter-clockwise (passing on one side of the
    planet), psi < 0 clockwise (the other side).
    """
    v_inf_in = v_ship - v_planet
    v_inf_out = rotate(v_inf_in, psi)
    return v_planet + v_inf_out


def hyperbola_state(v_inf_in, psi, r_p, mu, f):
    """Planet-frame (r, v) at true anomaly f on the flyby
    hyperbola that turns v_inf_in by psi with periapsis r_p.

    Periapsis lies opposite the velocity change (gravity pulls
    toward the planet); psi > 0 is counter-clockwise motion.
    """
    e = 1.0 + r_p * np.dot(v_inf_in, v_inf_in) / mu
    p = r_p * (1.0 + e)
    dv = rotate(v_inf_in, psi) - v_inf_in
    e_p = -dv / np.linalg.norm(dv)
    e_q = rotate(e_p, np.sign(psi) * np.pi / 2)
    r = p / (1.0 + e * np.cos(f))
    pos = r * (np.cos(f) * e_p + np.sin(f) * e_q)
    vel = np.sqrt(mu / p) * (-np.sin(f) * e_p
                             + (e + np.cos(f)) * e_q)
    return pos, vel


def true_anomaly_at(r, e, r_p):
    """Positive true anomaly where the hyperbola reaches r."""
    p = r_p * (1.0 + e)
    return np.arccos((p / r - 1.0) / e)
