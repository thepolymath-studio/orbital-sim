"""Numerical propagation of equation 06 with scipy's DOP853.

Thrust schedules with on/off switches are integrated arc by arc, so
the integrator never steps across a discontinuity in a_T.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import MU_SUN
from .dynamics import no_thrust, prograde, rhs

RTOL = 1e-12
# absolute floors: 1 m for position, 1e-9 m/s for velocity and dv
ATOL = np.array([1.0, 1.0, 1e-9, 1e-9, 1e-9])


def reach_radius(r_stop):
    """Terminal event: r crosses r_stop outward."""
    def event(t, y, *args):
        return np.hypot(y[0], y[1]) - r_stop
    event.terminal = True
    event.direction = 1
    return event


def fly(y0, t_span, thrust, mu=MU_SUN, events=None):
    """Integrate equation 06 over t_span (s)."""
    sol = solve_ivp(
        rhs, t_span, y0, method="DOP853",
        rtol=RTOL, atol=ATOL, args=(mu, thrust),
        events=events, dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


@dataclass(frozen=True)
class Arc:
    """One arc of a schedule: engine on or off.

    duration : seconds, or None = run until the stop event.
    """
    thrust_on: bool
    duration: float | None


@dataclass
class Segment:
    """An integrated arc: its dense solution and flag."""
    t0: float
    t1: float
    thrust_on: bool
    sol: object


def fly_schedule(y0, arcs, a_T, r_stop, mu=MU_SUN,
                 t_max=5 * 365.25 * 86400.0):
    """Fly burn/coast arcs until r reaches r_stop."""
    segs, y, t = [], np.asarray(y0, float), 0.0
    event = reach_radius(r_stop)
    for arc in arcs:
        on = arc.thrust_on
        law = prograde(a_T) if on else no_thrust
        t_end = t + (arc.duration or t_max)
        sol = fly(y, (t, t_end), law, mu, [event])
        segs.append(Segment(t, sol.t[-1], on, sol))
        t, y = sol.t[-1], sol.y[:, -1]
        if sol.status == 1:      # stop radius reached
            return segs
    raise RuntimeError("schedule ended before r_stop")


def sample(segs, dt):
    """Resample segments on a uniform time grid.

    Returns (t, Y, on): times (s), states (5, n), thrust flags.
    """
    t_end = segs[-1].t1
    t = np.append(np.arange(0.0, t_end, dt), t_end)
    Y = np.empty((5, t.size))
    on = np.zeros(t.size, dtype=bool)
    for s in segs:
        end = s is segs[-1]
        inside = (t < s.t1) | (end & (t <= s.t1))
        m = (t >= s.t0) & inside
        Y[:, m] = s.sol.sol(t[m])
        on[m] = s.thrust_on
    return t, Y, on
