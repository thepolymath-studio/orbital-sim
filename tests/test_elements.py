"""State <-> orbital elements, and the orbit paths for drawing."""
import numpy as np
import pytest

from orbital_sim.constants import MU_EARTH, R_EARTH_EQ
from orbital_sim.elements import (circular_state, elements,
                                  kepler_period, orbit_paths,
                                  perigee_radius,
                                  state_from_elements)

RNG = np.random.default_rng(3)


def random_elements(n=50):
    return np.column_stack([
        RNG.uniform(6.8e6, 3e7, n),        # a
        RNG.uniform(0.001, 0.6, n),        # e
        RNG.uniform(0.05, np.pi - 0.05, n),  # i
        RNG.uniform(0, 2 * np.pi, n),      # raan
        RNG.uniform(0, 2 * np.pi, n),      # argp
        RNG.uniform(0, 2 * np.pi, n)])     # nu


def ang_diff(x, y):
    return np.abs((x - y + np.pi) % (2 * np.pi) - np.pi)


def test_round_trip():
    E = random_elements()
    rv = [state_from_elements(*row) for row in E]
    r = np.array([p for p, _ in rv])
    v = np.array([q for _, q in rv])
    el = elements(r, v)
    assert np.allclose(el["a"], E[:, 0], rtol=1e-10)
    assert np.allclose(el["e"], E[:, 1], rtol=1e-9)
    for k, col in (("i", 2), ("raan", 3), ("argp", 4), ("nu", 5)):
        assert np.all(ang_diff(el[k], E[:, col]) < 1e-8), k
    u = (E[:, 4] + E[:, 5])
    assert np.all(ang_diff(el["u"], u) < 1e-8)


def test_mean_anomaly_by_kepler_equation():
    # M = E - e sin E: solving Kepler's equation for M must
    # give back the true anomaly we started from
    a, e = 8.0e6, 0.3
    nus = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    M = []
    for nu in nus:
        r, v = state_from_elements(a, e, 0.4, 0.1, 0.2, nu)
        M.append(elements(r, v)["M"])
    M = np.array(M)
    # inverse check: solve Kepler for E from M, get nu back
    for m, nu in zip(M, nus):
        Ecc = m
        for _ in range(50):
            Ecc = m + e * np.sin(Ecc)
        nu2 = 2 * np.arctan(np.sqrt((1 + e) / (1 - e))
                            * np.tan(Ecc / 2))
        assert ang_diff(nu2, nu) < 1e-10


def test_circular_start():
    r, v = circular_state(600e3, np.radians(51.6))
    el = elements(r, v)
    assert el["a"] == pytest.approx(R_EARTH_EQ + 600e3, rel=1e-12)
    assert el["e"] < 1e-12
    assert np.degrees(el["i"]) == pytest.approx(51.6, abs=1e-10)
    assert ang_diff(el["raan"], 0.0) < 1e-12
    assert ang_diff(el["u"], 0.0) < 1e-12
    assert perigee_radius(r, v) == pytest.approx(el["a"])


def test_kepler_period_600km():
    # 2 pi sqrt(a^3/mu), the post's ~96.7 min at 600 km
    T = kepler_period(R_EARTH_EQ + 600e3) / 60
    assert T == pytest.approx(96.687, abs=0.001)


def test_orbit_paths_lie_on_the_ellipse():
    E = random_elements(10)
    rv = [state_from_elements(*row) for row in E]
    r = np.array([p for p, _ in rv])
    v = np.array([q for _, q in rv])
    P = orbit_paths(r, v, n=720)
    h = np.cross(r, v)
    for k in range(len(E)):
        a, e = E[k, 0], E[k, 1]
        d = np.linalg.norm(P[k], axis=1)
        assert d.min() == pytest.approx(a * (1 - e), rel=1e-4)
        assert d.max() == pytest.approx(a * (1 + e), rel=1e-4)
        # in the orbit plane, and through the current state
        assert np.max(np.abs(P[k] @ h[k])) < 1e-6 * np.linalg.norm(
            h[k]) * a
        step = 2 * np.pi * a * (1 + e) / 720
        gap = np.linalg.norm(P[k] - r[k], axis=1).min()
        assert gap < step
