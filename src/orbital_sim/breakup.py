"""A satellite breakup: the NASA Standard Breakup Model (collision).

Source: Johnson, Krisko, Liou, Anz-Meador (2001), "NASA's new
breakup model of EVOLVE 4.0", Adv. Space Res. 28(9), 1377-1384,
eqs. 4, 6 and 12 and appendix B; the same parameters are coded in
ESA's open implementation (github.com/esa/NASA-breakup-model-cpp,
UtilityAreaMassRatio.h, Collision.cpp) and restated in
arXiv:2111.12196, sec. 2.3.1.

Three draws per piece, all from published fits:

1. size Lc (m): cumulative count N(>Lc) = 0.1 M^0.75 Lc^-1.71
   (catastrophic collision, M = mass that breaks up, kg);
2. area-to-mass ratio chi = log10(A/M) [A/M in m^2/kg]: for
   spacecraft pieces above 11 cm a mix of two normal
   distributions whose weight, means and widths depend on
   lambda = log10(Lc);
3. kick speed dv (m/s): log10(dv) is normal with mean
   0.9 chi + 2.9 and width 0.4, in a random direction.

We keep the large-fragment branch (Lc from 11 cm to 1 m): that is
the trackable debris (about fist-sized and up), so the 8-11 cm
bridge and the small-fragment branch are not needed. Pieces whose
new orbit dips below MIN_PERIGEE_ALT are counted as falling back
(drag pulls them down; the model has no air) and are not followed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import R_EARTH_EQ
from .elements import elements

L_MIN, L_MAX = 0.11, 1.0        # m, large-fragment branch
SLOPE = 1.71                    # N(>Lc) ~ Lc^-1.71
MIN_PERIGEE_ALT = 200e3         # m, below this: falls back


def fragment_count(mass, l_min=L_MIN):
    """Pieces larger than l_min (m) from mass (kg)."""
    return 0.1 * mass**0.75 * l_min**-SLOPE


def mass_for_count(n, l_min=L_MIN):
    """Mass (kg) whose breakup gives n pieces > l_min."""
    return (n / (0.1 * l_min**-SLOPE)) ** (1 / 0.75)


def sizes(rng, n, lo=L_MIN, hi=L_MAX):
    """Lc (m) from the -1.71 power law, inverse CDF."""
    a, b = lo**-SLOPE, hi**-SLOPE
    u = rng.random(n)
    return (a - u * (a - b)) ** (-1 / SLOPE)


def am_params(lam):
    """Weight and the two normals (spacecraft, > 11 cm).

    Each parameter is constant, then a straight ramp in
    lambda = log10(Lc), then constant again (appendix B).
    """
    ramp = np.interp
    alpha = ramp(lam, [-1.95, 0.55], [0.0, 1.0])
    mu1 = ramp(lam, [-1.1, 0.0], [-0.6, -0.95])
    sd1 = ramp(lam, [-1.3, -0.3], [0.1, 0.3])
    mu2 = ramp(lam, [-0.7, -0.1], [-1.2, -2.0])
    sd2 = ramp(lam, [-0.5, -0.3], [0.5, 0.3])
    return alpha, mu1, sd1, mu2, sd2


def area_to_mass(rng, lc):
    """chi = log10(A/M), from the two-normal mix."""
    alpha, mu1, sd1, mu2, sd2 = am_params(np.log10(lc))
    first = rng.random(lc.size) < alpha
    return np.where(first, rng.normal(mu1, sd1),
                    rng.normal(mu2, sd2))


def kick(rng, chi):
    """Velocity kick (m/s) of each piece: NASA's fit.

    Lighter, flatter pieces (bigger chi) fly off faster.
    """
    log_dv = rng.normal(0.9 * chi + 2.9, 0.4)
    speed = 10.0 ** log_dv                     # m/s
    d = rng.normal(size=(chi.size, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return speed[:, None] * d


@dataclass
class Cloud:
    """Pieces that stay in orbit, plus what fell back."""
    r: np.ndarray        # (n, 3) m, all at the breakup point
    v: np.ndarray        # (n, 3) m/s
    lc: np.ndarray       # (n,) m
    chi: np.ndarray      # (n,) log10(m^2/kg)
    dv: np.ndarray       # (n, 3) m/s, the kick
    n_drawn: int         # pieces generated in total
    n_fell: int          # perigee below MIN_PERIGEE_ALT
    n_escaped: int       # unbound (none expected)


def break_up(rng, r0, v0, n_keep, min_alt=MIN_PERIGEE_ALT):
    """Break one satellite at (r0, v0) into pieces.

    Draws pieces in batches until n_keep stay in orbit
    (bound, perigee above min_alt). Deterministic for a
    seeded rng.
    """
    keep = {k: [] for k in ("lc", "chi", "dv")}
    drawn = fell = esc = got = 0
    while got < n_keep:
        lc = sizes(rng, n_keep)
        chi = area_to_mass(rng, lc)
        dv = kick(rng, chi)
        r = np.broadcast_to(r0, dv.shape)
        with np.errstate(invalid="ignore"):   # unbound: M = NaN
            el = elements(r, v0 + dv)
        bound = el["a"] > 0
        rp = el["a"] * (1 - el["e"]) - R_EARTH_EQ
        ok = bound & (rp > min_alt)
        for i in range(lc.size):
            if got == n_keep:
                break
            drawn += 1
            fell += bool(bound[i] and not ok[i])
            esc += bool(not bound[i])
            if ok[i]:
                got += 1
                for k, x in zip(keep, (lc, chi, dv)):
                    keep[k].append(x[i])
    lc, chi, dv = (np.array(keep[k]) for k in keep)
    r = np.tile(r0, (n_keep, 1))
    return Cloud(r, v0 + dv, lc, chi, dv, drawn, fell, esc)
