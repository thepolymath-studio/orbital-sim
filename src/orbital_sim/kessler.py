"""Two plain scalings behind Kessler's argument (no simulation).

1. Collisions grow with the square of the number of objects.
   Kessler & Cour-Palais (1978), J. Geophys. Res. 83(A6), 2637:
   the collision frequency between catalogued objects varies as
   N^2 (research S1, S2). The counting reason: every collision
   needs a pair, and N objects make N (N - 1) / 2 pairs, so
   doubling N roughly quadruples the pairs. ``pair_count`` is that
   count. It is a scaling statement, not a forecast.

2. How fast two objects on circular orbits at the same height meet:
   both move at v, so the closing speed is 2 v sin(theta/2) for a
   crossing angle theta (vector difference of two equal speeds).
   Head-on (180 deg) gives 2 v; right angles give sqrt(2) v.
"""
from __future__ import annotations

import numpy as np

MPH = 0.44704   # m/s per mile per hour (exact)


def pair_count(n):
    """Number of different pairs among n objects."""
    return n * (n - 1) / 2


def pair_growth(n, factor=2):
    """How many times more pairs with factor x n objects."""
    return pair_count(factor * n) / pair_count(n)


def closing_speed(v, angle):
    """Speed of approach of two speeds v at angle (rad)."""
    return 2 * v * np.sin(angle / 2)


def crossing_angle(v_rel, v):
    """Angle (rad) at which two speeds v close at v_rel."""
    return 2 * np.arcsin(v_rel / (2 * v))
