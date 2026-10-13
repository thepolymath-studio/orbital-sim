"""Gargantua + Miller's planet: the stand-in's proportions and the
real numbers (L1/L2 inside the planet, Thorne's 1/M^2 scaling, the
Kerr period's Newtonian limit)."""
import numpy as np
import pytest

from orbital_sim.constants import (AU, C_LIGHT, D_MOON, G,
                                   G_MILLER, MU_EARTH,
                                   MU_GARGANTUA, MU_MOON, MU_SUN,
                                   R_EARTH, R_MOON, RHO_EARTH)
from orbital_sim.cr3bp import earth_moon
from orbital_sim.lagrange import lagrange_points, routh_mu
from orbital_sim.miller import (clearing_density, gaps_in_radii,
                                kerr_period, miller_real,
                                newton_period, orbit_radius,
                                planet_gm, planet_radius,
                                stand_in_mu, stand_in_radii)
from orbital_sim.waiting import hill_radius

RHOS = (1_000.0, RHO_EARTH, 2 * RHO_EARTH, 3 * RHO_EARTH)


# ---- the stand-in -------------------------------------------------

def test_stand_in_is_earth_moon():
    mu = stand_in_mu()
    assert mu == pytest.approx(MU_MOON / (MU_EARTH + MU_MOON))
    assert (1 - mu) / mu == pytest.approx(81.30, abs=0.005)
    assert mu < routh_mu()            # L4/L5 stable in the stand-in
    rb, rs = stand_in_radii()
    assert rb == pytest.approx(R_EARTH / D_MOON)
    assert rs == pytest.approx(0.00452, abs=5e-6)


# ---- the real numbers ---------------------------------------------

def test_orbit_radius_is_earths_orbit_size():
    """Thorne: Gargantua's radius ~ Earth's orbit (150e6 km)."""
    a = orbit_radius()
    assert a == pytest.approx(1e8 * MU_SUN / C_LIGHT**2)
    assert a / AU == pytest.approx(1.0, abs=0.02)


def test_planet_ball_reproduces_earth():
    """Earth's mean g and density give Earth's radius."""
    g = MU_EARTH / R_EARTH**2
    assert planet_radius(g, RHO_EARTH) == pytest.approx(
        R_EARTH, rel=2e-3)
    assert planet_gm(g, RHO_EARTH) == pytest.approx(
        MU_EARTH, rel=5e-3)


@pytest.mark.parametrize("rho", RHOS)
def test_surface_gravity_is_130_percent(rho):
    R = planet_radius(G_MILLER, rho)
    assert planet_gm(G_MILLER, rho) / R**2 == pytest.approx(
        1.30 * 9.80665)


@pytest.mark.parametrize("rho", RHOS)
def test_gaps_match_hill_estimate(rho):
    """L1/L2 gap / R = (rho / rho_clear)^(1/3) for tiny mu,
    and both gaps are equal to 1e-4 (Hill's symmetry)."""
    g1, g2 = gaps_in_radii(rho)
    want = (rho / clearing_density()) ** (1 / 3)
    assert g1 == pytest.approx(want, rel=1e-4)
    assert g2 == pytest.approx(want, rel=1e-4)


def test_l1_l2_inside_the_planet_at_any_plausible_density():
    """Earth-like: half the radius; even 3x Earth: inside."""
    assert gaps_in_radii(RHO_EARTH)[0] == pytest.approx(
        0.4995, abs=5e-4)
    for rho in RHOS:
        assert max(gaps_in_radii(rho)) < 1
    assert gaps_in_radii(RHOS[0])[0] < gaps_in_radii(RHOS[-1])[0]


def test_l1_reaches_surface_at_clearing_density():
    rho = clearing_density()
    assert gaps_in_radii(rho)[0] == pytest.approx(1.0, rel=1e-4)
    assert 1.0 < gaps_in_radii(1.1 * rho)[0]
    assert rho == pytest.approx(44_231, abs=1)
    assert rho / RHO_EARTH == pytest.approx(8.0, abs=0.05)


def test_clearing_density_is_three_mean_densities():
    """rho_clear = 3 x Gargantua's mass spread over a ball of
    radius a: the Hill-sphere condition."""
    a = orbit_radius()
    mean = MU_GARGANTUA / G / (4 / 3 * np.pi * a**3)
    assert clearing_density() == pytest.approx(3 * mean)


def test_tide_weakens_as_one_over_mass_squared():
    """Thorne's scaling: at a = GM/c^2 the tide ~ GM/a^3
    ~ 1/M^2, so the density needed drops 4x for 2x mass."""
    r1 = clearing_density(MU_GARGANTUA)
    r2 = clearing_density(2 * MU_GARGANTUA)
    assert r1 / r2 == pytest.approx(4.0)


def test_real_mass_ratio_and_routh():
    S = miller_real(RHO_EARTH)
    assert S.mu == pytest.approx(6.57e-14, rel=1e-3)
    assert S.big_to_small == pytest.approx(1.522e13, rel=1e-3)
    assert S.mu / routh_mu() < 1e-11


def test_all_points_sit_at_the_planets_distance():
    """Every balance point is within one Hill radius of the
    planet's own distance from Gargantua."""
    S = miller_real(RHO_EARTH)
    p = lagrange_points(S.mu)
    r = np.hypot(p[:, 0] + S.mu, p[:, 1])
    rh = hill_radius(S.mu)
    assert np.abs(r - 1).max() == pytest.approx(rh, rel=1e-3)


def test_newton_speed_is_light_speed_there():
    a = orbit_radius()
    assert np.sqrt(MU_GARGANTUA / a) == pytest.approx(C_LIGHT)


@pytest.mark.parametrize("x", [2.0, 6.0, 50.0])
def test_kerr_period_newtonian_limit(x):
    """No spin: the coordinate period is Kepler's exactly."""
    r = x * orbit_radius()
    assert kerr_period(MU_GARGANTUA, r, 0.0) == pytest.approx(
        newton_period(MU_GARGANTUA, r))


def test_kerr_period_at_the_horizon_is_twice_newton():
    """Maximal spin, r = GM/c^2: P = 4 pi GM/c^3 = 1.72 h,
    twice Newton's 0.86 h at the same distance."""
    a = orbit_radius()
    pk = kerr_period(MU_GARGANTUA, a, 1.0)
    assert pk == pytest.approx(4 * np.pi * MU_GARGANTUA
                               / C_LIGHT**3)
    assert pk / newton_period(MU_GARGANTUA, a) == pytest.approx(2)
    assert pk / 3600 == pytest.approx(1.72, abs=0.005)


def test_earth_moon_unchanged_by_stand_in():
    """The stand-in borrows ratios only; Earth-Moon itself is
    still the sourced system (used for validation)."""
    assert earth_moon().mu == stand_in_mu()


def test_real_growth_rates_hit_small_mu_limits():
    """With the real mu (~7e-14): L1/L2 grow at Hill's
    sqrt(1 + 2 sqrt 7) = 2.508 per time unit (e-fold
    0.0635 orbits), L3 at sqrt(21 mu / 8), and L4's slow
    swing has frequency sqrt(27 mu / 4)."""
    from orbital_sim.lagrange import growth_rate, linearise
    mu = miller_real(RHO_EARTH).mu
    p = lagrange_points(mu)
    hill = np.sqrt(1 + 2 * np.sqrt(7))
    for q in p[:2]:
        # corrections are O(r_H) ~ 3e-5
        assert growth_rate(*q, mu) == pytest.approx(hill,
                                                    rel=1e-4)
    assert 1 / hill / (2 * np.pi) == pytest.approx(0.0635,
                                                   abs=5e-5)
    assert growth_rate(*p[2], mu) == pytest.approx(
        np.sqrt(21 * mu / 8), rel=1e-2)
    w = np.sort(np.abs(np.linalg.eigvals(
        linearise(*p[3], mu)).imag))
    assert w[0] == pytest.approx(np.sqrt(27 * mu / 4), rel=1e-2)
