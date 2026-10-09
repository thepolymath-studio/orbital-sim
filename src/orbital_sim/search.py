"""The race's search for the fastest low-thrust rendezvous.

For a start phase (Mars's lead) and a flyby turn psi, the earliest
rendezvous is the earliest primer-vector extremal we can find
(primer.py). Shooting problems like this have many extremals and
narrow basins, so the search uses three sources of seeds and keeps
the earliest converged arrival at every (lead, psi):

1. a scan of the primer's start values (primer.scan);
2. continuation in psi within each lead (0 -> +-delta and back);
3. continuation in lead around the whole circle, both directions,
   for every psi: a solution at one phase seeds its neighbour, and
   the step is halved when a warm start fails.

Then psi is refined continuously at every lead (bounded Brent).
All tasks are deterministic and results are merged in a fixed order,
so the outcome does not depend on how `pmap` schedules them.

The result is the best extremal found, not a global-optimality
proof. Extremals that dip below the perihelion floor (Scenario.r_min,
default primer.R_MIN = 0.387 AU) are discarded, so where that floor
binds the times are upper bounds.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .primer import T_UNIT, TOL, scan, solve
from .race import problem, turn_limit

T_MAX = 7.3            # T_UNIT (730 d): longest arrival considered
JUMP = 0.5             # T_UNIT: a bigger change is a branch jump
DEPTH = 3              # continuation: at most 2^3 sub-steps
WARM_ITERS = 12        # LM iterations for a warm start
PSI_XATOL = 1e-3       # rad: flyby-turn refinement tolerance


def extremals(ships, X, target, sc, iters=40):
    """Converged extremals from seeds X (coarse, then fine)."""
    ships = np.atleast_2d(ships)
    if len(X) == 0:
        return np.empty((0, 4))
    kw = {"T_max": T_MAX, "r_min": sc.r_min}
    Xc, m = solve(ships, X, target, sc.a_th, rtol=1e-9, tol=1e-6,
                  iters=iters, **kw)
    k = m < 1e-6
    Xf, m = solve(ships[k], Xc[k], target, sc.a_th, iters=10, **kw)
    return Xf[m < TOL]


def earliest(X):
    """The row with the smallest T, or None."""
    return X[np.argmin(X[:, 3])] if len(X) else None


def better(a, b):
    """The earlier of two solutions (either may be None)."""
    if a is None or (b is not None and b[3] < a[3]):
        return b
    return a


def seeded(task):
    """Scan-seeded earliest solution at task = (sc, lead, psi)."""
    sc, lead, psi = task
    ship, target = problem(sc, lead, psi)
    _, X = scan(ship, target, T_MAX * T_UNIT, sc.a_th, per_bin=2,
                r_min=sc.r_min)
    ships = np.repeat(ship[None, :], len(X), axis=0)
    return earliest(extremals(ships, X, target, sc))


def track(sc, X, a, b, depth=0):
    """Carry solution X from (lead, psi) = a to b."""
    ship, target = problem(sc, *b)
    sol = earliest(extremals(ship, X[None, :], target, sc,
                             iters=WARM_ITERS))
    near = sol is not None and abs(sol[3] - X[3]) < JUMP
    if near or depth >= DEPTH:
        return sol
    mid = 0.5 * (np.asarray(a) + np.asarray(b))
    m = track(sc, X, a, mid, depth + 1)
    return None if m is None else track(sc, m, mid, b, depth + 1)


def chain(task):
    """Continuation along a path of keys; returns {key: X}."""
    sc, path, start = task
    out, cur = {}, start
    for a, b in zip(path[:-1], path[1:]):
        if cur is None:
            break
        cur = track(sc, cur, a, b)
        if cur is not None:
            out[_key(b)] = cur
    return out


def _key(p):
    """Grid key: lead wrapped to [0, 2 pi), both rounded."""
    return (round(float(p[0]) % (2 * np.pi), 9),
            round(float(p[1]), 9))


def _paths(leads, psis):
    """Continuation paths: in lead (both ways), and in psi."""
    i0 = list(psis).index(0.0)
    ring = np.unwrap(np.append(leads, leads[0]))
    out = []
    for p in psis if len(leads) > 1 else []:
        fwd = [(l, p) for l in ring]
        out += [fwd, fwd[::-1]]
    for l in leads:
        up = [(l, p) for p in psis[i0:]]
        down = [(l, p) for p in psis[:i0 + 1]][::-1]
        out += [up, up[::-1], down, down[::-1]]
    return out


def sweep(sc, leads, psi_fracs, pmap=map, seed_fracs=None,
          rounds=1):
    """Earliest solution at every (lead, psi = f * delta_max).

    Scan seeds at the seed_fracs turns (default: all), then
    continuation fills and improves the whole grid.
    Returns ({key: X or None}, psis); key = _key((lead, psi)).
    """
    psis = [float(f * turn_limit(sc)) for f in psi_fracs]
    fr = psi_fracs if seed_fracs is None else seed_fracs
    seeds = [float(f * turn_limit(sc)) for f in fr]
    grid = [(l, p) for l in leads for p in seeds]
    found = pmap(seeded, [(sc, l, p) for l, p in grid])
    best = {_key((l, p)): None for l in leads for p in psis}
    best.update(zip(map(_key, grid), found))
    for _ in range(rounds):
        tasks = [(sc, path, best[_key(path[0])])
                 for path in _paths(leads, psis)]
        for found in pmap(chain, tasks):
            for k, X in found.items():
                best[k] = better(best.get(k), X)
    return best, psis


def refine_psi(task):
    """Best psi in [-delta, delta] at one lead (bounded Brent).

    task = (sc, lead, psis, Xs): the grid psis and their solutions.
    Each trial psi is warm-started from the nearest psi solved so
    far. Returns (psi, X) for the earliest arrival found.
    """
    sc, lead, psis, Xs = task
    T = [x[3] if x is not None else np.inf for x in Xs]
    i = int(np.argmin(T))
    if not np.isfinite(T[i]):
        return None, None
    lo = psis[max(i - 1, 0)]
    hi = psis[min(i + 1, len(psis) - 1)]
    done = {psis[i]: Xs[i]}

    def t_of(psi):
        p0 = min(done, key=lambda p: abs(p - psi))
        X = track(sc, done[p0], (lead, p0), (lead, psi))
        if X is not None:
            done[psi] = X
        return X[3] if X is not None else np.inf
    minimize_scalar(t_of, bounds=(lo, hi), method="bounded",
                    options={"xatol": PSI_XATOL})
    return min(done.items(), key=lambda c: c[1][3])


PSI_FRACS = (-1.0, -0.5, 0.0, 0.5, 1.0)   # psi grid / delta_max
SEED_FRACS = (-1.0, 0.0, 1.0)             # scanned; rest continued


def fastest_at(sc, pmap=map):
    """02 and 03 at the scenario's own phase (no phase sweep).

    Returns (X02, (psi03, X03)).
    """
    lead = sc.mars_lead
    best, psis = sweep(sc, [lead], PSI_FRACS, pmap, SEED_FRACS)
    Xs = [best[_key((lead, p))] for p in psis]
    psi, X = refine_psi((sc, lead, psis, Xs))
    return Xs[psis.index(0.0)], (psi, X)
