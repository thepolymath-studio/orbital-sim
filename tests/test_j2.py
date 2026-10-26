"""The J2 acceleration and the closed-form nodal precession.

Catches: a wrong sign or factor in a_J2 (gradient check, signs over
the equator and the pole), a wrong precession formula (published
worked example, sun-synchronous tilt, polar/retrograde limits), and
an integrator that does not obey them (energy, polar angular
momentum, simulated vs closed-form turning rate).
"""
import numpy as np
import pytest

from orbital_sim.cloud import (fitted_rate, mean_elements,
                               propagate_cloud, read_elements,
                               unwrap)
from orbital_sim.constants import (DAY, J2_EARTH, MU_EARTH,
                                   R_EARTH_EQ)
from orbital_sim.elements import circular_state, kepler_period
from orbital_sim.j2 import (earth_accel, energy, j2_accel,
                            kepler_accel, node_rate, potential,
                            sun_sync_inclination)

DEG_DAY = np.degrees(1.0) * DAY
RNG = np.random.default_rng(7)


def random_points(n=20):
    d = RNG.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * RNG.uniform(6.6e6, 4.2e7, size=(n, 1))


def test_acceleration_is_minus_gradient_of_potential():
    h = 10.0   # m: truncation ~ (h/r)^2, roundoff ~ 1e-16 r/h
    for r in random_points():
        grad = np.zeros(3)
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            grad[k] = (potential(r + e) - potential(r - e)) / (2 * h)
        a = earth_accel(r)
        assert np.linalg.norm(a + grad) < 1e-7 * np.linalg.norm(a)
        # the J2 part alone, too (Phi_J2 = Phi - Phi_kepler)
        gj = np.zeros(3)
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            f = (potential(r + e) - potential(r + e, j2=False)
                 - potential(r - e) + potential(r - e, j2=False))
            gj[k] = f / (2 * h)
        aj = j2_accel(r)
        assert np.linalg.norm(aj + gj) < 1e-4 * np.linalg.norm(aj)


def test_bulge_pulls_harder_over_equator_less_over_pole():
    r = R_EARTH_EQ + 600e3
    eq = j2_accel(np.array([r, 0.0, 0.0]))
    pole = j2_accel(np.array([0.0, 0.0, r]))
    share = 1.5 * J2_EARTH * (R_EARTH_EQ / r) ** 2
    g = MU_EARTH / r**2
    # over the equator: inward (adds to gravity), 3/2 J2 (R/r)^2
    assert eq[0] == pytest.approx(-share * g, rel=1e-12)
    assert eq[1] == eq[2] == 0.0
    # over the pole: outward, twice as large
    assert pole[2] == pytest.approx(+2 * share * g, rel=1e-12)
    # about a thousandth of gravity at 600 km
    assert 1e-3 < share < 2e-3


def test_j2_off_is_kepler_and_stacks_work():
    r = random_points(5)
    assert np.allclose(earth_accel(r, j2=False), kepler_accel(r))
    one = np.array([j2_accel(p) for p in r])
    assert np.allclose(j2_accel(r), one, rtol=1e-14)
    assert np.all(j2_accel(r, J2=0.0) == 0)


def test_node_rate_published_example():
    # Wikipedia "Nodal precession", worked example (same R_E
    # and J2 as ours): circular, 800 km, 56 deg -> -3.683 deg/day
    a = R_EARTH_EQ + 800e3
    rate = node_rate(a, 0.0, np.radians(56)) * DEG_DAY
    assert rate == pytest.approx(-3.683, abs=0.005)


def test_sun_synchronous_tilt():
    # Wikipedia "Sun-synchronous orbit": a = 7200 km (about
    # 800 km up) gives a sun-synchronous tilt of 98.7 deg
    i = np.degrees(sun_sync_inclination(7200e3))
    assert i == pytest.approx(98.7, abs=0.05)
    # by construction it turns 360 deg per tropical year
    rate = node_rate(7200e3, 0.0, np.radians(i))
    assert rate * DEG_DAY == pytest.approx(0.98565, rel=1e-5)


def test_node_rate_limits():
    a = R_EARTH_EQ + 600e3
    assert node_rate(a, 0, np.radians(90)) == pytest.approx(0, abs=1e-20)
    assert node_rate(a, 0, np.radians(51.6)) < 0      # west
    assert node_rate(a, 0, np.radians(128.4)) > 0     # east
    # farther out it slows as a^-3.5
    r2 = node_rate(2 * a, 0, 0.5) / node_rate(a, 0, 0.5)
    assert r2 == pytest.approx(2 ** -3.5, rel=1e-12)


def fly_one(alt, inc, days, j2=True, e_kick=0.0):
    r, v = circular_state(alt, np.radians(inc))
    v = v * (1 + e_kick)
    t = np.linspace(0, days * DAY, int(days * 24 * 4) + 1)
    s = propagate_cloud(r[None], v[None], t, j2)
    return t, s


def test_energy_and_polar_spin_conserved_with_j2():
    # an eccentric orbit (e ~ 0.1) for 5 days
    t, s = fly_one(600e3, 51.6, 5, e_kick=0.05)
    r, v = s[0, :, :3], s[0, :, 3:]
    E = energy(r, v)
    hz = np.cross(r, v)[:, 2]
    # DOP853 at rtol 1e-11 over ~5,000 steps: drift ~2e-10
    assert np.max(np.abs(E / E[0] - 1)) < 1e-9
    assert np.max(np.abs(hz / hz[0] - 1)) < 1e-9
    # ... while the plane itself turns (hx, hy change)
    h = np.cross(r, v)
    turn = np.arctan2(h[-1, 0], -h[-1, 1]) - np.arctan2(h[0, 0],
                                                        -h[0, 1])
    assert abs(turn) > np.radians(10)


def test_no_j2_no_turning_and_kepler_period():
    r0, v0 = circular_state(600e3, np.radians(51.6))
    T = kepler_period(np.linalg.norm(r0))
    s = propagate_cloud(r0[None], v0[None], np.array([0, T]),
                        j2=False)
    assert np.linalg.norm(s[0, -1, :3] - r0) < 0.01   # m
    t, s = fly_one(600e3, 51.6, 3, j2=False)
    el = read_elements(s)
    assert np.ptp(unwrap(el["raan"])) < 1e-9


# The closed form is first order in J2, so differences of order J2
# (~1e-3) remain. The largest piece, the gap between averaged and
# "mean" semi-major axis, scales with (1 - 3/2 sin^2 i): largest at
# low tilt, near zero at 51.6 deg (the film cloud's tilt).
@pytest.mark.parametrize("alt, inc, tol", [
    (600e3, 51.6, 1e-3), (410e3, 51.64, 1e-3),
    (800e3, 98.6, 3e-3), (600e3, 28.5, 4e-3)])
def test_simulated_precession_matches_closed_form(alt, inc, tol):
    t, s = fly_one(alt, inc, 20)
    el = read_elements(s)
    sim = fitted_rate(t, unwrap(el["raan"]))[0]
    m = mean_elements(el)
    cf = node_rate(m["a"], m["e"], m["i"])[0]
    assert sim == pytest.approx(cf, rel=tol)
