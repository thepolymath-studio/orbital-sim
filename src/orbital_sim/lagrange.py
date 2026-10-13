"""Lagrange points of the CR3BP and their linear stability.

The five balance points are where the landscape is flat,
grad U_eff = 0 (equation 11 of the post). L1-L3 lie on the x axis
(roots of dOmega/dx there, found by bracketing); L4 and L5 close
equilateral triangles with the two bodies.

Stability: linearise the equations of motion about a point. A
saddle of U_eff with a real eigenvalue pair is unstable; L4/L5 are
hilltops of U_eff, yet linearly stable when Coriolis wins, i.e.
when m1/m2 > (25 + 3 sqrt 69)/2 = 24.96 (Routh 1875).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .cr3bp import grad_omega, hessian_omega

NAMES = ("L1", "L2", "L3", "L4", "L5")
EPS = 1e-12


def axis_push(x, mu):
    """dOmega/dx along the line through both bodies."""
    return grad_omega(x, 0.0, mu)[0]


def collinear_points(mu):
    """x of L1 (between), L2 (beyond small), L3 (far side)."""
    big, small = -mu, 1 - mu
    brackets = [(big + EPS, small - EPS),
                (small + EPS, 2.0),
                (-2.0, big - EPS)]
    return [brentq(axis_push, a, b, args=(mu,),
                   xtol=1e-15, rtol=1e-15)
            for a, b in brackets]


def lagrange_points(mu):
    """All five points, shape (5, 2), rotating frame."""
    x1, x2, x3 = collinear_points(mu)
    h = np.sqrt(3) / 2
    return np.array([[x1, 0.0], [x2, 0.0], [x3, 0.0],
                     [0.5 - mu, h], [0.5 - mu, -h]])


def linearise(x, y, mu):
    """4x4 matrix of the motion near (x, y)."""
    oxx, oxy, oyy = hessian_omega(x, y, mu)
    return np.array([[0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [oxx, oxy, 0, 2],
                     [oxy, oyy, -2, 0]])


def landscape_shape(x, y, mu):
    """'saddle', 'hilltop' or 'well' of U_eff at (x, y)."""
    oxx, oxy, oyy = hessian_omega(x, y, mu)
    curv = np.linalg.eigvalsh([[oxx, oxy], [oxy, oyy]])
    if curv.min() > 0:       # Omega min = U_eff max
        return "hilltop"
    return "saddle" if curv.max() > 0 else "well"


def growth_rate(x, y, mu):
    """Largest real part of the eigenvalues (1/time unit)."""
    lam = np.linalg.eigvals(linearise(x, y, mu))
    return lam.real.max()


def routh_mu():
    """Critical mu: L4/L5 stable only for mu below it."""
    return 0.5 * (1 - np.sqrt(23 / 27))


def routh_ratio():
    """Critical big-to-small mass ratio, about 24.96."""
    mu = routh_mu()
    return (1 - mu) / mu
