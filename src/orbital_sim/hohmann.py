"""Hohmann transfer between circular coplanar orbits (S3 in research).

Used as the baseline the continuous-thrust runs are compared with,
and as an analytic check on the integrator.
"""
from __future__ import annotations

import numpy as np

from .constants import MU_SUN


def hohmann(r1, r2, mu=MU_SUN):
    """Return (dv1, dv2, t_H) in m/s, m/s, s."""
    a_t = 0.5 * (r1 + r2)
    v_c1, v_c2 = np.sqrt(mu / r1), np.sqrt(mu / r2)
    v_t1 = v_c1 * np.sqrt(2 * r2 / (r1 + r2))
    v_t2 = v_c2 * np.sqrt(2 * r1 / (r1 + r2))
    t_H = np.pi * np.sqrt(a_t ** 3 / mu)
    return v_t1 - v_c1, v_c2 - v_t2, t_H
