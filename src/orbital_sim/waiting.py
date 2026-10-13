"""Waiting at a balance point: ships parked at L1-L5.

Interstellar: while the Ranger visits Miller's planet, Endurance
waits in orbit around Gargantua. Where could a ship wait near the
planet without burning fuel? This module sets up the experiments on
top of the planar CR3BP (orbital_sim.cr3bp, .lagrange, .balance):

* the waiting test: at each of L1-L5, one ship exactly on the
  point, one slightly off it and one slowly drifting. L4/L5 hold
  them (the Coriolis effect turns them back), and the saddles
  L1-L3 let them slide off;
* the chaos test: a cloud of ships, all a hair apart, at rest on
  a saddle point. The saddle splits them, and the cloud's width
  grows by a factor e every e-folding time.

Everything is in normalised CR3BP units: distance = the pair's
separation, speed = n x separation, time = 1/n (one orbit of the
pair = 2 pi). +x points from the big body to the small one, and +y
along the small body's motion.
"""
from __future__ import annotations

import numpy as np

from .cr3bp import distances
from .lagrange import (NAMES, growth_rate, lagrange_points,
                       linearise)

GOLDEN = np.pi * (3 - np.sqrt(5))     # golden angle, rad


def misplace(point, dr=0.0, dv=0.0):
    """A ship at `point`, dr further along +x (the
    big-to-small direction), moving at dv along +y
    (the small body's direction of motion)."""
    x, y = point
    return np.array([x + dr, y, 0.0, dv])


def waiting_starts(mu, dr, dv):
    """Three ships per point, L1..L5: exactly there,
    dr off, dv off. Shape (15, 4), point-major."""
    tries = [(0.0, 0.0), (dr, 0.0), (0.0, dv)]
    return np.array([misplace(p, a, b)
                     for p in lagrange_points(mu)
                     for a, b in tries])


def drift(track, point):
    """Distance from `point` along a (4, n) track."""
    return np.hypot(track[0] - point[0],
                    track[1] - point[1])


def first_time_over(t, d, limit):
    """First t at which d exceeds limit (NaN if
    never)."""
    over = np.flatnonzero(d > limit)
    return t[over[0]] if over.size else np.nan


def efold_orbits(mu, name):
    """Drift e-folding time at a point, in orbits of
    the pair: the gap grows x e (2.718) in this time.
    Returns inf at the stable L4/L5."""
    x, y = lagrange_points(mu)[NAMES.index(name)]
    lam = growth_rate(x, y, mu)
    if lam < 1e-9:
        return np.inf
    return 1.0 / lam / (2 * np.pi)


def sunflower(centre, diameter, n):
    """n points spread evenly over a disc (a sunflower
    pattern: deterministic, no random numbers)."""
    k = np.arange(n) + 0.5
    r = 0.5 * diameter * np.sqrt(k / n)
    a = k * GOLDEN
    return centre + np.c_[r * np.cos(a), r * np.sin(a)]


def point_cloud(mu, name, n, diameter):
    """n ships at rest, all within `diameter` of each
    other, centred on the named point. Shape (n, 4)."""
    p = lagrange_points(mu)[NAMES.index(name)]
    xy = sunflower(p, diameter, n)
    return np.c_[xy, np.zeros((n, 2))]


def saddle_side(mu, states, name):
    """+1 for ships on the small body's side of the
    saddle's dividing line, -1 on the other side.
    Linear theory: the sign of each start's part along
    the unstable direction."""
    p = lagrange_points(mu)[NAMES.index(name)]
    A = linearise(*p, mu)
    lam, V = np.linalg.eig(A)
    lt, W = np.linalg.eig(A.T)
    v = V[:, np.argmax(lam.real)].real
    w = W[:, np.argmax(lt.real)].real
    toward = np.sign(1 - mu - p[0])   # small body
    v = v * toward * np.sign(v[0])
    part = (states - np.r_[p, 0, 0]) @ w / (w @ v)
    return np.sign(part)


def departure(track, point, limit):
    """+1 if the ship first passes `limit` from point
    on its +x side, -1 on its -x side, 0 if never."""
    d = drift(track, point)
    over = np.flatnonzero(d > limit)
    if not over.size:
        return 0
    return int(np.sign(track[0, over[0]] - point[0]))


def spread(tracks):
    """Widest gap between any two ships at each time.
    tracks: (n, 4, t) states. Returns shape (t,)."""
    xy = tracks[:, :2, :]
    widest = np.zeros(xy.shape[-1])
    for ship in xy:
        gap = np.hypot(*(xy - ship).transpose(1, 0, 2))
        widest = np.fmax(widest, np.nanmax(gap, axis=0))
    return widest


def hill_radius(mu):
    """The small body's sphere of control,
    (mu/3)^(1/3) of the separation."""
    return (mu / 3) ** (1 / 3)


def closest_to_small(track, mu):
    """Smallest distance to the small body."""
    _, r2 = distances(track[0], track[1], mu)
    return np.nanmin(r2)


def where_now(state, mu):
    """Label of a state: near the small body (inside
    its Hill radius), inside or outside its orbit."""
    r1, r2 = distances(state[0], state[1], mu)
    rh = hill_radius(mu)
    if r2 < rh:
        return "near the small body"
    if r1 < 1 - mu - rh:
        return "inside the orbit"
    if r1 > 1 - mu + rh:
        return "outside the orbit"
    return "along the orbit"
