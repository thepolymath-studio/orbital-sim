"""2026-10-wk1 Sim post: Hermes races back to Mars, three ways.

Shared set-up: one start epoch, t = 0, at the Earth encounter
--------------------------------------------------------------
Sun-centred and planar. Earth (1 AU) and Mars (1.5237 AU) move on
circles. Earth is on +x at t = 0, and Mars leads Earth by
`mars_lead`. Nominal phasing: the planets' mean longitudes on the
novel's Earth-flyby date, 6 Jul 2036 (Burke, Table 1; research F21).

Hermes reaches Earth at t = 0 coasting in from Mars's orbit: its
ellipse has aphelion at Mars's radius, and it meets Earth with
v_inf = 6 km/s (NASA Glenn's upper limit for the Earth encounter,
research S15). It is inbound (sunward) and faster than Earth: the
film's "long deceleration to intercept" (F17).

01 Hohmann : impulsive stop (dv = v_inf) onto Earth's orbit, wait
             for the phase angle alpha (eq. 04), then the two
             Hohmann burns (eq. 02, 03), coasting in between.
02 Thrust  : the ion drive only, a_th = 2 mm/s^2 (F5), always on,
             steered along the primer vector (minimum time).
             Hermes passes Earth far away: no gravity assist.
03 Purnell : patched-conic Earth flyby at t = 0 that turns v_inf
             by psi, |psi| <= delta at 300 km periapsis altitude
             (eq. 08, 09), then the same ion drive. psi = 0 (a
             distant pass) is 02, so 03 can never be slower.

Every arrival is a rendezvous: Mars's position and velocity.
Planetary gravity is ignored except in the flyby itself, and the
flyby is instantaneous on the heliocentric time scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .constants import (A_HERMES, AU, MU_EARTH, MU_SUN, R_EARTH,
                        R_EARTH_ORBIT, R_MARS_ORBIT)
from .dynamics import no_thrust
from .ephemeris import mars_lead, mean_motion, planet_state
from .flyby import flyby, periapsis_for_turn, turn_angle
from .hohmann import hohmann
from .primer import (ATOL8, R_MIN, T_UNIT, primer_rhs,
                     primer_start)
from .propagate import RTOL, fly

NOVEL_EARTH_FLYBY = date(2036, 7, 6)   # Burke, Table 1 (F21)
V_INF = 6.0e3                          # m/s, Burke's limit (S15)
FLYBY_ALT_MIN = 300e3                  # m, lowest periapsis (S14)
# Rendezvous tolerance (stated): the ship counts as arrived when
# it is within 10 km of Mars's centre and 1 cm/s of its velocity.
TOL_R = 10e3                           # m
TOL_V = 0.01                           # m/s


@dataclass(frozen=True)
class Scenario:
    """One race: Mars's lead over Earth at t = 0, and Hermes."""
    mars_lead: float                   # rad
    v_inf: float = V_INF               # m/s
    alt_min: float = FLYBY_ALT_MIN     # m
    a_th: float = A_HERMES             # m/s^2
    r_min: float = R_MIN               # m, perihelion floor


def nominal():
    """Mars phasing on the novel's Earth-flyby date."""
    return Scenario(mars_lead=float(mars_lead(NOVEL_EARTH_FLYBY)))


def earth(t):
    """Earth's (r, v) at t (s): on +x at t = 0."""
    return planet_state(t, R_EARTH_ORBIT, 0.0)


def mars(lead):
    """Mars's (r, v) as a function of t, for a given lead."""
    def target(t):
        return planet_state(t, R_MARS_ORBIT, lead)
    return target


def coasting_in(r_p, r=R_EARTH_ORBIT, r_a=R_MARS_ORBIT,
                mu=MU_SUN):
    """Inbound state at radius r on the ellipse r_p .. r_a."""
    a = 0.5 * (r_a + r_p)
    h = np.sqrt(mu * 2 * r_a * r_p / (r_a + r_p))
    v_t = h / r
    v_r = -np.sqrt(mu * (2 / r - 1 / a) - v_t ** 2)
    return np.array([r, 0.0, v_r, v_t])


def inbound_state(v_inf=V_INF):
    """Hermes at Earth, t = 0: in from Mars's orbit, v_inf."""
    _, v_e = earth(0.0)

    def gap(r_p):
        v = coasting_in(r_p)[2:]
        return np.linalg.norm(v - v_e) - v_inf
    r_p = brentq(gap, 0.3 * AU, R_EARTH_ORBIT * (1 - 1e-12),
                 xtol=1e-3)
    return coasting_in(r_p)


def turn_limit(sc):
    """Largest flyby turn: eq. 09 at the lowest periapsis."""
    return turn_angle(sc.v_inf, R_EARTH + sc.alt_min, MU_EARTH)


def after_flyby(sc, psi):
    """Hermes's state just after a flyby turning v_inf by psi."""
    ship = inbound_state(sc.v_inf)
    _, v_e = earth(0.0)
    v = flyby(ship[2:], v_e, psi)
    return np.concatenate([ship[:2], v])


def problem(sc, lead, psi):
    """(start state, target) for the low-thrust solver."""
    return after_flyby(sc, psi), mars(lead)


def hohmann_wait(lead):
    """Eq. 04: wait (s) until Mars leads Earth by alpha."""
    r1, r2 = R_EARTH_ORBIT, R_MARS_ORBIT
    _, _, t_H = hohmann(r1, r2)
    n1, n2 = mean_motion(r1), mean_motion(r2)
    alpha = np.pi - n2 * t_H
    return ((lead - alpha) % (2 * np.pi)) / (n1 - n2)


@dataclass
class Leg:
    """An integrated arc: phase name, times (s), dense solution."""
    phase: str
    t0: float
    t1: float
    sol: object
    thrust: object = None              # thrust law, if on


@dataclass
class Result:
    """One strategy's run, measured against Mars at arrival."""
    name: str
    t_arrive: float                    # s
    dv: float                          # m/s, everything used
    legs: list
    impulses: list                     # (t s, dv m/s)
    miss_r: float = np.nan             # m
    miss_v: float = np.nan             # m/s
    info: dict = field(default_factory=dict)


def _measure(res, y_final, lead):
    """Fill the rendezvous residual against Mars."""
    r_m, v_m = mars(lead)(res.t_arrive)
    res.miss_r = float(np.linalg.norm(y_final[:2] - r_m))
    res.miss_v = float(np.linalg.norm(y_final[2:4] - v_m))
    return res


def prograde_kick(y, dv):
    """Impulsive burn dv along the velocity (copy of y)."""
    y = np.array(y, float)
    y[2:4] *= 1.0 + dv / np.linalg.norm(y[2:4])
    return y


def race_hohmann(sc):
    """01: stop at Earth, wait for the window, two burns.

    Impulses: the stop (dv = v_inf, onto Earth's orbit), burn 1
    (eq. 02) and burn 2 (eq. 03), as (t, dv) in res.impulses.
    """
    dv1, dv2, t_H = hohmann(R_EARTH_ORBIT, R_MARS_ORBIT)
    t_w = hohmann_wait(sc.mars_lead)
    t_a = t_w + t_H
    y = np.concatenate([*earth(0.0), [0.0]])  # stopped
    legs = []
    if t_w > 0:                               # wait
        wait = fly(y, (0.0, t_w), no_thrust)
        legs.append(Leg("wait", 0.0, t_w, wait.sol))
        y = wait.y[:, -1]
    y = prograde_kick(y, dv1)                 # burn 1
    xfer = fly(y, (t_w, t_a), no_thrust)
    legs.append(Leg("transfer", t_w, t_a, xfer.sol))
    y = prograde_kick(xfer.y[:, -1], dv2)     # burn 2
    kicks = [(0.0, sc.v_inf), (t_w, dv1), (t_a, dv2)]
    res = Result("01 Hohmann", t_a, 0.0, legs, kicks)
    res.dv = sum(k[1] for k in kicks)
    res.info = {"t_wait": t_w, "t_H": t_H}
    return _measure(res, y, sc.mars_lead)


def primer_law(ship, z, T, a_th=A_HERMES, mu=MU_SUN):
    """Thrust law a_th p/|p|, p from the primer equation.

    The primer is integrated once (DOP853, rtol 1e-12, dense) and
    then read off at any t, so propagate.fly can fly eq. 06 with
    the same rhs as every other strategy.
    """
    y0 = primer_start(ship, z)
    sol = solve_ivp(primer_rhs, (0.0, T), y0, method="DOP853",
                    rtol=RTOL, atol=ATOL8, dense_output=True,
                    args=(mu, a_th))

    def thrust(t, r, v):
        p = sol.sol(t)[4:6]
        return a_th * p / np.linalg.norm(p)
    return thrust


def race_lowthrust(sc, X, psi, name):
    """02 / 03: fly the primer solution X = (z, T/T_UNIT)."""
    ship = after_flyby(sc, psi)
    T = X[3] * T_UNIT
    law = primer_law(ship, X[:3], T, sc.a_th)
    seg = fly(np.append(ship, 0.0), (0.0, T), law)
    y = seg.y[:, -1]
    r = np.hypot(seg.y[0], seg.y[1])
    info = {"psi": float(psi), "z": [float(v) for v in X[:3]],
            "r_min": float(r.min()), "r_max": float(r.max()),
            "floor_binds": bool(r.min() < 1.01 * sc.r_min),
            "transversality": transversality(sc, X, psi)}
    info.update(flyby_info(sc, psi))
    leg = Leg("thrust", 0.0, T, seg.sol, law)
    res = Result(name, T, float(y[4]), [leg], [], info=info)
    return _measure(res, y, sc.mars_lead)


def transversality(sc, X, psi):
    """sin of the angle between p(0) and v_inf_out.

    Pontryagin: at a best flyby turn strictly inside the limits,
    dT/dpsi = 0 forces p(0) parallel to v_inf_out, so this is 0.
    """
    _, v_e = earth(0.0)
    w = after_flyby(sc, psi)[2:] - v_e
    p = np.array([np.cos(X[0]), np.sin(X[0])])
    return float((p[0] * w[1] - p[1] * w[0]) / np.linalg.norm(w))


def flyby_info(sc, psi):
    """Periapsis, eccentricity and turn of the flyby (if any)."""
    if psi == 0.0:
        return {"flyby": False}
    r_p = periapsis_for_turn(sc.v_inf, psi, MU_EARTH)
    e = 1.0 + r_p * sc.v_inf ** 2 / MU_EARTH
    return {"flyby": True, "r_p": float(r_p),
            "alt_p": float(r_p - R_EARTH), "e": float(e),
            "delta": float(turn_angle(sc.v_inf, r_p, MU_EARTH)),
            "delta_max": float(turn_limit(sc)),
            "dv_flyby": float(2 * sc.v_inf / e)}
