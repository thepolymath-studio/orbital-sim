"""The NASA Standard Breakup Model draws (Johnson et al. 2001).

The parameters are checked against an independent, literal
transcription of the piecewise formulas in ESA's open
implementation (UtilityAreaMassRatio.h), and the sampled
distributions against their defining moments.
"""
import numpy as np
import pytest

from orbital_sim.breakup import (L_MAX, L_MIN, MIN_PERIGEE_ALT,
                                 am_params, area_to_mass, break_up,
                                 fragment_count, kick,
                                 mass_for_count, sizes)
from orbital_sim.constants import R_EARTH_EQ
from orbital_sim.elements import circular_state, elements


def esa_params(lam):
    """Literal piecewise formulas (spacecraft, Lc > 11 cm)."""
    def pw(x, lo, hi, vlo, vhi, f):
        return vlo if x <= lo else vhi if x >= hi else f(x)
    return (
        pw(lam, -1.95, 0.55, 0.0, 1.0, lambda x: 0.3 + 0.4 * (x + 1.2)),
        pw(lam, -1.1, 0.0, -0.6, -0.95,
           lambda x: -0.6 - 0.318 * (x + 1.1)),
        pw(lam, -1.3, -0.3, 0.1, 0.3, lambda x: 0.1 + 0.2 * (x + 1.3)),
        pw(lam, -0.7, -0.1, -1.2, -2.0,
           lambda x: -1.2 - 1.333 * (x + 0.7)),
        pw(lam, -0.5, -0.3, 0.5, 0.3, lambda x: 0.5 - (x + 0.5)))


@pytest.mark.parametrize("lam", np.linspace(-1.2, 0.2, 29))
def test_area_to_mass_parameters_match_published_formulas(lam):
    ours = am_params(np.array([lam]))
    for a, b in zip(ours, esa_params(lam)):
        # published slopes are rounded (0.318, 1.333)
        assert a[0] == pytest.approx(b, abs=1e-3)


def test_fragment_count_formula():
    # N(>Lc) = 0.1 M^0.75 Lc^-1.71: 1000 kg, 10 cm -> 912
    assert fragment_count(1000, 0.1) == pytest.approx(
        0.1 * 1000**0.75 * 0.1**-1.71)
    assert fragment_count(1000, 0.1) == pytest.approx(912.0, abs=1)
    assert mass_for_count(fragment_count(650.0)) == pytest.approx(650)


def test_sizes_follow_the_power_law():
    rng = np.random.default_rng(1)
    lc = sizes(rng, 200_000)
    assert lc.min() >= L_MIN and lc.max() <= L_MAX
    for x in (0.15, 0.2, 0.5):
        want = (x**-1.71 - L_MAX**-1.71) / (L_MIN**-1.71
                                            - L_MAX**-1.71)
        assert np.mean(lc > x) == pytest.approx(want, abs=0.005)


def test_area_to_mass_mixture_moments():
    rng = np.random.default_rng(2)
    lc = np.full(400_000, 0.3)
    chi = area_to_mass(rng, lc)
    al, m1, s1, m2, s2 = (p for p in esa_params(np.log10(0.3)))
    mean = al * m1 + (1 - al) * m2
    var = (al * (s1**2 + m1**2) + (1 - al) * (s2**2 + m2**2)
           - mean**2)
    assert chi.mean() == pytest.approx(mean, abs=0.005)
    assert chi.var() == pytest.approx(var, rel=0.02)


def test_kick_speed_law_and_isotropy():
    rng = np.random.default_rng(3)
    chi = np.full(200_000, -1.0)
    dv = kick(rng, chi)
    log_dv = np.log10(np.linalg.norm(dv, axis=1))
    # log10(dv) ~ N(0.9 chi + 2.9, 0.4): median 100 m/s here
    assert log_dv.mean() == pytest.approx(2.0, abs=0.005)
    assert log_dv.std() == pytest.approx(0.4, abs=0.005)
    d = dv / np.linalg.norm(dv, axis=1, keepdims=True)
    assert np.all(np.abs(d.mean(axis=0)) < 0.01)
    assert np.allclose((d**2).mean(axis=0), 1 / 3, atol=0.005)
    # lighter pieces (bigger chi) fly faster
    fast = kick(rng, np.full(50_000, -0.5))
    assert (np.median(np.linalg.norm(fast, axis=1))
            > np.median(np.linalg.norm(dv, axis=1)))


def test_break_up_keeps_only_pieces_that_stay_up():
    r0, v0 = circular_state(600e3, np.radians(51.6))
    c = break_up(np.random.default_rng(5), r0, v0, 300)
    assert len(c.lc) == 300 and c.r.shape == (300, 3)
    assert np.all(c.r == r0)
    assert np.allclose(c.v - v0, c.dv)
    el = elements(c.r, c.v)
    assert np.all(el["a"] > 0)
    rp = el["a"] * (1 - el["e"]) - R_EARTH_EQ
    assert np.all(rp > MIN_PERIGEE_ALT)
    assert c.n_drawn == 300 + c.n_fell + c.n_escaped
    assert 0 < c.n_fell < c.n_drawn


def test_break_up_is_deterministic():
    r0, v0 = circular_state(600e3, np.radians(51.6))
    a = break_up(np.random.default_rng(9), r0, v0, 50)
    b = break_up(np.random.default_rng(9), r0, v0, 50)
    assert np.array_equal(a.v, b.v) and a.n_drawn == b.n_drawn
