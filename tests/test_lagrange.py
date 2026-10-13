"""Lagrange points and stability: closed forms, an independent
quintic solve, sourced distances, the Routh threshold."""
import numpy as np
import pytest

from orbital_sim.cr3bp import (earth_moon, grad_omega, jacobi,
                               sun_earth, sun_jupiter)
from orbital_sim.lagrange import (NAMES, growth_rate,
                                  lagrange_points, landscape_shape,
                                  linearise, routh_mu, routh_ratio)

EM, SE, SJ = earth_moon(), sun_earth(), sun_jupiter()
MUS = [EM.mu, SE.mu, SJ.mu, 0.01, 0.2, 0.5]


@pytest.mark.parametrize("mu", MUS)
def test_gradient_vanishes_at_all_five(mu):
    """grad U_eff = 0 (equation 11) at each point."""
    for x, y in lagrange_points(mu):
        assert np.hypot(*grad_omega(x, y, mu)) < 1e-13


def _quintic_x(mu, point):
    """Independent route: the textbook quintics in gamma."""
    m = mu
    polys = {   # Szebehely ch. 4; gamma = distance to a body
        "L1": [1, -(3 - m), 3 - 2 * m, -m, 2 * m, -m],
        "L2": [1, 3 - m, 3 - 2 * m, -m, -2 * m, -m],
        "L3": [1, 2 + m, 1 + 2 * m, -(1 - m), -2 * (1 - m),
               -(1 - m)],
    }
    roots = np.roots(polys[point])
    g = roots[(abs(roots.imag) < 1e-9) & (roots.real > 0)].real
    g = g.min()
    return {"L1": 1 - m - g, "L2": 1 - m + g,
            "L3": -m - g}[point]


@pytest.mark.parametrize("mu", MUS[:-1])
def test_collinear_points_match_quintic_roots(mu):
    pts = lagrange_points(mu)
    for i, name in enumerate(("L1", "L2", "L3")):
        assert pts[i, 0] == pytest.approx(_quintic_x(mu, name),
                                          abs=1e-11)


@pytest.mark.parametrize("mu", MUS)
def test_triangle_points_closed_form(mu):
    """L4/L5 at (1/2 - mu, +-sqrt(3)/2): equilateral."""
    l4, l5 = lagrange_points(mu)[3:]
    assert l4 == pytest.approx([0.5 - mu, np.sqrt(3) / 2],
                               abs=1e-15)
    assert l5 == pytest.approx([0.5 - mu, -np.sqrt(3) / 2],
                               abs=1e-15)
    for p in (l4, l5):
        assert np.hypot(p[0] + mu, p[1]) == pytest.approx(1)
        assert np.hypot(p[0] - 1 + mu, p[1]) == pytest.approx(1)
    # C_J at L4 = 3 - mu + mu^2
    assert jacobi(np.r_[l4, 0, 0], mu) == pytest.approx(
        3 - mu + mu ** 2, abs=1e-14)


def test_hill_limit_small_mu():
    """L1, L2 at distance (mu/3)^(1/3) from the small body."""
    mu = 1e-9
    x1, x2 = lagrange_points(mu)[:2, 0]
    h = (mu / 3) ** (1 / 3)
    assert 1 - mu - x1 == pytest.approx(h, rel=1e-3)
    assert x2 - 1 + mu == pytest.approx(h, rel=1e-3)


def test_earth_moon_matches_sourced_distances():
    """Research S7 (Wikipedia, 100 km rounding): from Earth's
    centre L1 326,400, L2 448,900, L3 381,700 km."""
    mu, km = EM.mu, EM.sep / 1e3
    pts = lagrange_points(mu)
    from_earth = np.hypot(pts[:, 0] + mu, pts[:, 1]) * km
    assert from_earth[:3] == pytest.approx(
        [326_400, 448_900, 381_700], abs=50)
    # L3 measured from the barycentre would NOT match
    assert abs(abs(pts[2, 0]) * km - 381_700) > 4_000


def test_earth_moon_jacobi_published():
    """C_J at L1..L4, mu = 0.01215 (Koon, Lo, Marsden & Ross)."""
    pts = lagrange_points(0.01215)
    c = [jacobi(np.r_[p, 0, 0], 0.01215) for p in pts[:4]]
    assert c == pytest.approx([3.1883, 3.1722, 3.0121, 2.9880],
                              abs=6e-5)


def test_sun_earth_l2_and_drift_time():
    """S7: L2 about 1.5 million km beyond Earth; S6: unstable
    with a time constant of about 23 days at L1/L2."""
    mu, km, day = SE.mu, SE.sep / 1e3, SE.time_unit / 86400
    pts = lagrange_points(mu)
    assert (pts[1, 0] - 1 + mu) * km == pytest.approx(1.5e6,
                                                     rel=0.01)
    for x, y in pts[:2]:
        assert day / growth_rate(x, y, mu) == pytest.approx(
            23, abs=0.5)


def test_sun_jupiter_triangle_at_60_degrees():
    mu = SJ.mu
    for p, sign in zip(lagrange_points(mu)[3:], (1, -1)):
        ang = np.degrees(np.arctan2(p[1], p[0] + mu))
        assert ang == pytest.approx(sign * 60, abs=1e-12)
    assert SJ.big_to_small == pytest.approx(1047.35, abs=0.01)


@pytest.mark.parametrize("mu", [EM.mu, SJ.mu, 0.2])
def test_collinear_points_are_unstable_saddles(mu):
    for x, y in lagrange_points(mu)[:3]:
        assert landscape_shape(x, y, mu) == "saddle"
        assert growth_rate(x, y, mu) > 1e-3
        lam = np.linalg.eigvals(linearise(x, y, mu))
        real = lam[abs(lam.imag) < 1e-12].real
        assert sorted(real) == pytest.approx(
            [-real.max(), real.max()])     # one +-lambda pair


@pytest.mark.parametrize("mu", MUS)
def test_triangle_points_are_hilltops(mu):
    """L4/L5 are maxima of U_eff for every mu: not bowls."""
    for x, y in lagrange_points(mu)[3:]:
        assert landscape_shape(x, y, mu) == "hilltop"


def test_l4_characteristic_polynomial():
    """lam^4 + lam^2 + (27/4) mu (1 - mu) = 0 at L4."""
    for mu in (EM.mu, SJ.mu, 0.03):
        lam = np.linalg.eigvals(linearise(*lagrange_points(mu)[3],
                                          mu))
        k = 27 / 4 * mu * (1 - mu)
        assert np.abs(lam ** 4 + lam ** 2 + k).max() < 1e-12


def test_routh_threshold():
    """Stable iff m1/m2 > (25 + 3 sqrt 69)/2 = 24.96."""
    assert routh_ratio() == pytest.approx((25 + 3 * 69 ** .5) / 2,
                                          rel=1e-14)
    assert routh_ratio() == pytest.approx(24.96, abs=5e-3)
    mu_c = routh_mu()

    def unstable(mu):
        return growth_rate(*lagrange_points(mu)[3], mu) > 1e-9
    lo, hi = 0.01, 0.1          # bisect the stability switch
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if unstable(mid) else (mid, hi)
    assert lo == pytest.approx(mu_c, abs=1e-8)
    assert not unstable(mu_c * 0.999) and unstable(mu_c * 1.001)


def test_real_pairs_on_the_right_side_of_routh():
    """Earth/Moon 81.3 and Sun/Jupiter 1047 are both > 24.96."""
    for S in (EM, SJ):
        assert S.big_to_small > routh_ratio()
        for x, y in lagrange_points(S.mu)[3:]:
            assert growth_rate(x, y, S.mu) < 1e-9
    assert EM.big_to_small == pytest.approx(81.30, abs=0.005)


def test_names():
    assert NAMES == ("L1", "L2", "L3", "L4", "L5")
