"""Minimum-time low-thrust rendezvous (race strategies 02 and 03).

Steering law
------------
The engine is always on at a_th (a minimum-time problem never
coasts). The thrust points at an angle u(t) from the local horizontal:
u = 0 is along the local prograde horizontal, +90 deg radially out,
180 deg retrograde. u(t) is piecewise linear in t over N equal
segments, so the law has N + 1 node angles.

Optimisation
------------
SLSQP minimises the flight time T over (T, extra start parameters,
node angles) subject to the rendezvous constraints

    r(T) = r_target(T),   v(T) = v_target(T).

Gradients are forward finite differences, integrated as one batch:
all perturbed copies share one DOP853 step sequence, so the
differences are smooth. The batch runs at rtol 1e-10; the final
trajectory is re-flown with propagate.fly at rtol 1e-12, and the
rendezvous residual is measured on that.

This is the simple, direct counterpart of the primer-vector solution
(primer.py). It cannot beat a true optimum, so the gap between the two
says how close a few-segment law gets: see tests/test_lowthrust.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .constants import A_HERMES, DAY, MU_SUN
from .primer import ATOL8, primer_rhs, primer_start

T_UNIT = 100 * DAY        # optimiser's scale for T
R_UNIT = 1.495978707e11   # m, residual scale (1 AU)
V_UNIT = 29.78e3          # m/s, residual scale (Earth's speed)
BATCH_RTOL = 1e-10
FD_STEP = 1e-7
# absolute floors: 1 mm for x, y; 1e-9 m/s for vx, vy
ATOL4 = np.array([1e-3, 1e-3, 1e-9, 1e-9])


def thrust_direction(r, u):
    """Unit vector at angle u from the local horizontal."""
    e_r = r / np.linalg.norm(r)
    e_t = np.array([-e_r[1], e_r[0]])
    return np.cos(u) * e_t + np.sin(u) * e_r


def steering(nodes, T, a_th=A_HERMES):
    """Thrust law: a_th along u(t), linear between nodes."""
    s = np.linspace(0.0, T, len(nodes))

    def thrust(t, r, v):
        u = np.interp(t, s, nodes)
        return a_th * thrust_direction(r, u)
    return thrust


def _rhs_batch(tau, y, T, U, a_th, mu):
    """Eq. 06 for K ships at once, in scaled time tau = t / T."""
    n = U.shape[1] - 1
    x, yy, vx, vy = y.reshape(4, -1)
    r = np.hypot(x, yy)
    j = min(int(tau * n), n - 1)
    w = tau * n - j
    u = (1.0 - w) * U[:, j] + w * U[:, j + 1]
    ex, ey = x / r, yy / r
    c, s = np.cos(u), np.sin(u)
    g = -mu / r ** 2
    ax = g * ex + a_th * (s * ex - c * ey)
    ay = g * ey + a_th * (s * ey + c * ex)
    return np.concatenate([vx, vy, ax, ay]) * np.tile(T, 4)


def _shoot(T, U, Y0, a_th, mu):
    """Final states (K, 4) of K ships flown for T_k each."""
    K = len(T)
    y0 = np.asarray(Y0, float).T.reshape(-1)
    sol = solve_ivp(_rhs_batch, (0.0, 1.0), y0, method="DOP853",
                    rtol=BATCH_RTOL, atol=np.repeat(ATOL4, K),
                    args=(np.asarray(T), U, a_th, mu))
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1].reshape(4, K).T


@dataclass
class Problem:
    """A rendezvous problem for the optimiser.

    start(p) -> state [x, y, vx, vy] at t = 0, for extra params p.
    target(t) -> (r, v) of the target at time t.
    """
    start: object
    target: object
    n_extra: int = 0
    a_th: float = A_HERMES
    mu: float = MU_SUN


def _unpack(x, prob):
    T = x[0] * T_UNIT
    p = x[1:1 + prob.n_extra]
    nodes = x[1 + prob.n_extra:]
    return T, p, nodes


def residual(state, t, prob):
    """Scaled rendezvous miss [dr / AU, dv / V_E] (4,)."""
    r_t, v_t = prob.target(t)
    dr = (state[:2] - r_t) / R_UNIT
    dv = (state[2:] - v_t) / V_UNIT
    return np.concatenate([dr, dv])


class _Batch:
    """Residual and Jacobian from one batched integration."""

    def __init__(self, prob):
        self.prob, self.key, self.c, self.J = prob, None, None, None

    def eval(self, x):
        if self.key == x.tobytes():
            return
        X = np.vstack([x, x + FD_STEP * np.eye(len(x))])
        T, Y0, U = [], [], []
        for xi in X:
            Ti, p, nodes = _unpack(xi, self.prob)
            T.append(Ti)
            Y0.append(self.prob.start(p))
            U.append(nodes)
        Yf = _shoot(T, np.array(U), Y0, self.prob.a_th,
                    self.prob.mu)
        C = np.array([residual(Yf[k], T[k], self.prob)
                      for k in range(len(X))])
        self.key = x.tobytes()
        self.c, self.J = C[0], (C[1:] - C[0]).T / FD_STEP

    def cons(self, x):
        self.eval(x)
        return self.c

    def jac(self, x):
        self.eval(x)
        return self.J


@dataclass
class Solution:
    """Optimised transfer: flight time, extra params, nodes.

    feasible : the rendezvous miss is below 1e-8 (scaled), i.e.
               about 1.5 km and 0.3 mm/s.
    """
    T: float
    extra: np.ndarray
    nodes: np.ndarray
    feasible: bool
    miss: float
    message: str


def optimise(prob, x0, extra_bounds=(), maxiter=200):
    """SLSQP: minimise T subject to the rendezvous constraints."""
    batch = _Batch(prob)
    n_nodes = len(x0) - 1 - prob.n_extra
    bounds = ([(0.05, 20.0)] + list(extra_bounds)
              + [(None, None)] * n_nodes)
    res = minimize(
        lambda x: x[0], np.asarray(x0, float),
        jac=lambda x: np.eye(len(x))[0], method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": batch.cons,
                      "jac": batch.jac}],
        options={"maxiter": maxiter, "ftol": 1e-12})
    miss = float(np.max(np.abs(batch.cons(res.x))))
    T, p, nodes = _unpack(res.x, prob)
    return Solution(T, np.array(p), np.array(nodes),
                    miss < 1e-8, miss, str(res.message))


def primer_angles(ship, z, T, n_seg, a_th=A_HERMES, mu=MU_SUN):
    """The primer steering u(t) sampled at n_seg + 1 nodes.

    Used to warm-start the direct law from the optimal one.
    """
    t = np.linspace(0.0, T, n_seg + 1)
    sol = solve_ivp(primer_rhs, (0.0, T), primer_start(ship, z),
                    method="DOP853", rtol=1e-12, atol=ATOL8,
                    t_eval=t, args=(mu, a_th))
    r, p = sol.y[0:2], sol.y[4:6]
    e_r = r / np.hypot(*r)
    e_t = np.array([-e_r[1], e_r[0]])
    u = np.arctan2((p * e_r).sum(0), (p * e_t).sum(0))
    return np.unwrap(u)


def direct_time(ship, target, z, T, n_seg, a_th=A_HERMES):
    """Earliest rendezvous with the n_seg-segment law, started
    from the primer solution (z, T). Returns a Solution."""
    prob = Problem(start=lambda p: ship, target=target,
                   a_th=a_th)
    u = primer_angles(ship, z, T, n_seg, a_th)
    return optimise(prob, np.concatenate([[T / T_UNIT], u]))
