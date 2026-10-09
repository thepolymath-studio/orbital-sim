"""Minimum-time low-thrust rendezvous by Lawden's primer vector.

Pontryagin's principle for a minimum-time transfer at constant thrust
acceleration a_th says the engine is always on, pointing along the
primer vector p, which obeys (Lawden, 1963)

    p'' = G(r) p,     G = mu / r^3 (3 r_hat r_hat^T - I).

The whole steering history follows from p(0) and p'(0). Only p's
direction matters, so |p(0)| = 1 and the unknowns are

    z = (alpha, q_x, q_y):  p(0) = (cos alpha, sin alpha),
                            p'(0) = n_E (q_x, q_y),

plus the flight time T: four unknowns for the four rendezvous
conditions r(T) = r_M(T), v(T) = v_M(T).

Search: a batched scan over z (every trajectory in one DOP853 call)
records each candidate's closest approach to the target in position
and velocity. The best candidates seed a vectorised damped Newton
iteration. The earliest converged arrival wins. Every converged
candidate is a Pontryagin extremal; taking the earliest of many is
the best extremal found, not a proof of global optimality. The tests
cross-check it against the direct piecewise-linear optimiser.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .constants import AU, DAY, MU_SUN
from .dynamics import gravity
from .ephemeris import mean_motion

N_E = float(mean_motion(AU))   # 1/s, scale for p'(0)
T_UNIT = 100 * DAY             # s, scale for T in the solver
V_UNIT = 29.78e3               # m/s, velocity miss scale
# Perihelion floor (stated assumption): Mercury's mean distance,
# 0.387 AU (NASA NSSDC). The Purnell route dips "well inside Venus's
# orbit" (research F12), so a dive is allowed, a sun-grazer is not.
# Candidates that cross the floor are frozen during the search and
# cannot converge, so where it binds the times are upper bounds.
R_MIN = 0.387 * AU
SCAN_RTOL = 1e-8
NEWTON_RTOL = 1e-11
FD_STEP = 1e-7
MAX_STEP = 0.3                 # Newton step cap (rad, T units)
TOL = 1e-10                    # converged: scaled miss below this
# |p| this small (|p(0)| = 1): the thrust would flip almost instantly
# and the shared step size would collapse for the whole batch. Such
# candidates are frozen during the search; the final trajectories are
# flown without this guard and checked against Mars.
P_MIN = 1e-4
# absolute floors: 1 mm (r), 1e-9 m/s (v), 1e-12 (p), 1e-20 1/s (p')
ATOL8 = np.array([1e-3, 1e-3, 1e-9, 1e-9,
                  1e-12, 1e-12, 1e-20, 1e-20])


def primer_rhs(t, y, mu, a_th):
    """Eq. 06, thrust along the primer p; p'' = G(r) p."""
    r, v, p, q = y[0:2], y[2:4], y[4:6], y[6:8]
    rn = np.linalg.norm(r)
    e = r / rn
    G = mu / rn ** 3 * (3 * np.outer(e, e) - np.eye(2))
    a = gravity(r, mu) + a_th * p / np.linalg.norm(p)
    return np.concatenate([v, a, q, G @ p])


def primer_start(ship, z):
    """[r, v, p, p'] at t = 0 for ship [r, v] and z."""
    alpha, qx, qy = z
    p0 = [np.cos(alpha), np.sin(alpha)]
    q0 = [N_E * qx, N_E * qy]
    return np.concatenate([ship[:4], p0, q0])


def _rhs_batch(tau, y, T, a_th, mu, r_min):
    """primer_rhs for K ships at once, scaled time tau = t / T."""
    x, yy, vx, vy, px, py, qx, qy = y.reshape(8, -1)
    r2 = x * x + yy * yy
    r = np.sqrt(r2)
    k = mu / (r2 * r)
    pn = np.hypot(px, py)
    rp = (x * px + yy * py) / r2
    out = np.concatenate([
        vx, vy,
        -k * x + a_th * px / pn, -k * yy + a_th * py / pn,
        qx, qy,
        k * (3 * rp * x - px), k * (3 * rp * yy - py)])
    out *= np.tile(T, 8)
    dead = (r < r_min) | (pn < P_MIN)  # below floor, or p ~ 0
    out[np.tile(dead, 8)] = 0.0        # frozen: cannot converge
    return out


def fly_batch(ships, Z, T, a_th, mu=MU_SUN, taus=None,
              rtol=NEWTON_RTOL, r_min=0.0):
    """Fly K primer trajectories; states (8, K, len(taus))."""
    ships, Z = np.atleast_2d(ships), np.atleast_2d(Z)
    K = len(Z)
    if K == 0:
        return np.empty((8, 0, 1 if taus is None else len(taus)))
    y0 = np.vstack([ships[:, :4].T, np.cos(Z[:, 0]),
                    np.sin(Z[:, 0]), N_E * Z[:, 1],
                    N_E * Z[:, 2]]).reshape(-1)
    sol = solve_ivp(_rhs_batch, (0.0, 1.0), y0, method="DOP853",
                    rtol=rtol, atol=np.repeat(ATOL8, K),
                    t_eval=taus, args=(np.asarray(T, float), a_th,
                                       mu, r_min))
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.reshape(8, K, -1)


def scaled_miss(r, v, t, target):
    """[dr / AU, dv / V_UNIT] against target(t); r, v (2, ...)."""
    r_t, v_t = target(t)
    return np.concatenate([(r - r_t) / AU, (v - v_t) / V_UNIT])


def scan_grid(n_alpha=16, n_beta=8, kappas=(0.5, 1.0, 2.0)):
    """Grid of z = (alpha, q_x, q_y) for the global scan."""
    al = np.arange(n_alpha) * 2 * np.pi / n_alpha
    be = np.arange(n_beta) * 2 * np.pi / n_beta
    q = [(0.0, 0.0)] + [(k * np.cos(b), k * np.sin(b))
                        for k in kappas for b in be]
    return np.array([(a, qx, qy) for a in al for qx, qy in q])


def scan(ships, target, T_max, a_th, mu=MU_SUN, n_bins=12,
         per_bin=4, r_min=R_MIN):
    """Seeds for the solver, spread over arrival time.

    Flies every ship (rows of `ships`) with every grid z to T_max
    and records the scaled miss against the target along the way.
    In each of n_bins time bins, the per_bin closest passes become
    seeds X = (z, T / T_UNIT). Returns (ship_index, X).
    """
    ships = np.atleast_2d(ships)
    Z = scan_grid()
    idx = np.repeat(np.arange(len(ships)), len(Z))
    ZZ = np.tile(Z, (len(ships), 1))
    taus = np.linspace(0.0, 1.0, 30 * n_bins + 1)[1:]
    Y = fly_batch(ships[idx], ZZ, np.full(len(ZZ), T_max), a_th,
                  mu, taus, SCAN_RTOL, r_min)
    t = taus * T_max
    c = scaled_miss(Y[0:2], Y[2:4], t[None, :], target)
    d = np.sqrt((c ** 2).sum(axis=0)).reshape(len(ZZ), n_bins, 30)
    out = []
    for b in range(n_bins):
        jb = d[:, b].argmin(axis=1)
        best = np.argsort(d[np.arange(len(ZZ)), b, jb])[:per_bin]
        for k in best:
            T = t[30 * b + jb[k]] / T_UNIT
            out.append((idx[k], *ZZ[k], T))
    out = np.array(out)
    return out[:, 0].astype(int), out[:, 1:]


def miss_and_jac(ships, X, target, a_th, mu=MU_SUN,
                 rtol=NEWTON_RTOL, r_min=R_MIN):
    """Scaled miss (K, 4) and its Jacobian (K, 4, 4) wrt X,
    by forward differences in one batched integration."""
    K = len(X)
    P = np.concatenate([X[:, None, :],
                        X[:, None, :] + FD_STEP * np.eye(4)], 1)
    P = P.reshape(-1, 4)
    S = np.repeat(ships, 5, axis=0)
    T = P[:, 3] * T_UNIT
    Y = fly_batch(S, P[:, :3], T, a_th, mu, rtol=rtol,
                  r_min=r_min)[:, :, -1]
    c = scaled_miss(Y[0:2], Y[2:4], T, target).T
    c = c.reshape(K, 5, 4)
    J = (c[:, 1:, :] - c[:, :1, :]) / FD_STEP   # (K, i, k)
    return c[:, 0, :], np.transpose(J, (0, 2, 1))


def solve(ships, X, target, a_th, mu=MU_SUN, iters=60,
          rtol=NEWTON_RTOL, tol=TOL, T_max=7.3, r_min=R_MIN):
    """Vectorised Levenberg-Marquardt on the scaled miss.

    Each row is one candidate (its ship and X = (z, T/T_UNIT));
    a step is accepted only if it reduces that row's miss.
    Returns (X, miss) with miss = max |scaled miss| per row.
    """
    X = np.array(X, float)
    S = np.atleast_2d(ships)
    c, J = miss_and_jac(S, X, target, a_th, mu, rtol, r_min)
    lam = np.full(len(X), 1e-3)
    for _ in range(iters):
        a = np.flatnonzero(np.abs(c).max(axis=1) > tol)
        if a.size == 0:
            break
        Xt = X[a] + _lm_step(J[a], c[a], lam[a])
        Xt[:, 3] = np.clip(Xt[:, 3], 0.05, T_max)
        ct, Jt = miss_and_jac(S[a], Xt, target, a_th, mu, rtol,
                              r_min)
        ok = (ct ** 2).sum(1) < (c[a] ** 2).sum(1)
        X[a[ok]], c[a[ok]], J[a[ok]] = Xt[ok], ct[ok], Jt[ok]
        lam[a] = np.where(ok, lam[a] / 3, lam[a] * 4)
    return X, np.abs(c).max(axis=1)


def _lm_step(J, c, lam):
    """Levenberg-Marquardt steps for a stack of 4x4 systems."""
    JT = np.transpose(J, (0, 2, 1))
    A = JT @ J + lam[:, None, None] * np.eye(4)
    step = -np.linalg.solve(A, (JT @ c[..., None]))[..., 0]
    big = np.maximum(np.abs(step).max(axis=1), MAX_STEP)
    return step * (MAX_STEP / big)[:, None]
