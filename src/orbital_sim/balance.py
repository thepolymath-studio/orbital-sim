"""2026-10-wk3 scenes: tracer dots near L1 and L4, and the landscape.

* L1 cloud: dots at rest in the rotating frame, a hair apart on the
  Earth-Moon line around L1. The saddle sends each one downhill to
  the Moon's side or back to Earth's, and neighbours 1 km apart end
  up on unrelated paths (sensitive dependence).
* L4 ring: dots on L4's slow libration mode (the linear eigenvector
  with the long period), spread in phase. They circle the hilltop
  and stay: the Coriolis effect turns them back.
* U_eff on a grid, for the contour landscape.

All in normalised CR3BP units; ``System`` converts to km and days.
"""
from __future__ import annotations

import numpy as np

from .cr3bp import fly, hits_body, u_eff
from .lagrange import lagrange_points, linearise


def l1_cloud(mu, n, gap):
    """n states at rest, spaced `gap` along x, about L1."""
    x_l1 = lagrange_points(mu)[0, 0]
    dx = (np.arange(n) - (n - 1) / 2) * gap
    s = np.zeros((n, 4))
    s[:, 0] = x_l1 + dx
    return s


def slow_mode(mu):
    """L4's long-period eigenvector, x amplitude 1."""
    lam, vec = np.linalg.eig(linearise(
        *lagrange_points(mu)[3], mu))
    up = np.where(lam.imag > 0, lam.imag, np.inf)
    k = np.argmin(up)
    return vec[:, k] / vec[0, k], lam[k].imag


def l4_ring(mu, n, amp):
    """n states on L4's slow mode, phases 360/n apart."""
    mode, _ = slow_mode(mu)
    phase = np.exp(2j * np.pi * np.arange(n) / n)
    l4 = np.r_[lagrange_points(mu)[3], 0.0, 0.0]
    return l4 + amp * (np.outer(phase, mode)).real


def fly_cloud(states, t, mu, radii):
    """Fly each state on times t; NaN after an impact.

    radii : (r_big, r_small) body radii, normalised.
    Returns (n, 4, t.size) states and impact times
    (NaN when no impact).
    """
    hit = hits_body(mu, *radii)
    out = np.full((len(states), 4, t.size), np.nan)
    t_hit = np.full(len(states), np.nan)
    for i, s0 in enumerate(states):
        sol = fly(s0, t[-1], mu, t_eval=t, events=[hit])
        out[i, :, :sol.t.size] = sol.y
        if sol.status == 1:
            t_hit[i] = sol.t_events[0][0]
    return out, t_hit


def ueff_grid(mu, half, n):
    """U_eff on an n x n grid over [-half, half]^2."""
    x = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x, x)   # U[i, j] at x[j], y[i]
    return x, x, u_eff(X, Y, mu)
