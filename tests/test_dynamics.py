"""Equation 06 and its integrator: conservation and analytic limits."""
import numpy as np
import pytest
from scipy.integrate import quad

from orbital_sim import (A_HERMES, AU, DAY, MU_SUN, R_EARTH_ORBIT,
                         circular_state, fly, no_thrust, prograde,
                         specific_energy)


def kepler_period(a, mu=MU_SUN):
    return 2 * np.pi * np.sqrt(a ** 3 / mu)


def ang_mom(y):
    return y[0] * y[3] - y[1] * y[2]


def eccentric_state(r_peri, e, mu=MU_SUN):
    """Perihelion state of an ellipse with eccentricity e."""
    v_p = np.sqrt(mu / r_peri * (1 + e))
    return np.array([r_peri, 0.0, 0.0, v_p, 0.0])


def test_energy_and_momentum_conserved_thrust_off():
    # e = 0.5 ellipse, 10 orbits: stresses perihelion passages.
    r_p, e = R_EARTH_ORBIT, 0.5
    a = r_p / (1 - e)
    y0 = eccentric_state(r_p, e)
    sol = fly(y0, (0, 10 * kepler_period(a)), no_thrust)
    E = specific_energy(sol.y)
    h = ang_mom(sol.y)
    # DOP853 at rtol 1e-12: drift should sit near 1e-11 over 10 orbits
    assert np.max(np.abs(E / E[0] - 1)) < 1e-10
    assert np.max(np.abs(h / h[0] - 1)) < 1e-10
    # and the energy is the analytic -mu/(2a)
    assert E[0] == pytest.approx(-MU_SUN / (2 * a), rel=1e-12)


def test_circular_orbit_stays_circular_and_obeys_kepler():
    r0 = R_EARTH_ORBIT
    T = kepler_period(r0)
    y0 = circular_state(r0)
    sol = fly(y0, (0, 5 * T), no_thrust)
    t = np.linspace(0, 5 * T, 2001)
    Y = sol.sol(t)
    r = np.hypot(Y[0], Y[1])
    v = np.hypot(Y[2], Y[3])
    assert np.max(np.abs(r / r0 - 1)) < 1e-9
    assert np.max(np.abs(v / y0[3] - 1)) < 1e-9
    # Kepler's third law: back at the start after each period
    for k in (1, 5):
        yk = sol.sol(k * T)
        assert np.hypot(*(yk[:2] - y0[:2])) / r0 < 1e-8
    # one year at 1 AU (sidereal year 365.256 d, to 0.01 %)
    assert T / DAY == pytest.approx(365.256, rel=1e-4)


def test_thrust_power_equals_energy_rate():
    """dE/dt = v . a_T, checked by finite differences."""
    y0 = circular_state(R_EARTH_ORBIT)
    law = prograde(A_HERMES)
    sol = fly(y0, (0, 60 * DAY), law)
    h = 600.0
    for t in np.linspace(1 * DAY, 59 * DAY, 12):
        dE = (specific_energy(sol.sol(t + h))
              - specific_energy(sol.sol(t - h))) / (2 * h)
        y = sol.sol(t)
        power = np.dot(y[2:4], law(t, y[:2], y[2:4]))
        assert dE == pytest.approx(power, rel=1e-6)


def test_work_energy_theorem_over_a_burn():
    """E(t) - E(0) equals the integral of v . a_T."""
    y0 = circular_state(R_EARTH_ORBIT)
    law = prograde(A_HERMES)
    t_end = 60 * DAY
    sol = fly(y0, (0, t_end), law)

    def power(t):
        y = sol.sol(t)
        return np.dot(y[2:4], law(t, y[:2], y[2:4]))

    work, _ = quad(power, 0, t_end, limit=200, epsabs=0,
                   epsrel=1e-11)
    gain = specific_energy(sol.y[:, -1]) - specific_energy(y0)
    assert gain == pytest.approx(work, rel=1e-8)
    assert gain > 0          # prograde thrust raises the orbit


def test_dv_equals_aT_t_without_gravity():
    """Equation 07 in its pure form: mu = 0, thrust along v."""
    y0 = circular_state(AU)
    t_end = 30 * DAY
    sol = fly(y0, (0, t_end), prograde(A_HERMES), mu=0.0)
    v0 = np.hypot(*y0[2:4])
    v1 = np.hypot(*sol.y[2:4, -1])
    assert v1 - v0 == pytest.approx(A_HERMES * t_end, rel=1e-10)
    assert sol.y[4, -1] == pytest.approx(A_HERMES * t_end,
                                         rel=1e-10)


def test_dv_state_accumulates_aT_t_with_gravity():
    """With gravity the speed change differs, but the thrust dv
    (the budget the engine pays) is still a_T t."""
    y0 = circular_state(R_EARTH_ORBIT)
    t_end = 45 * DAY
    sol = fly(y0, (0, t_end), prograde(A_HERMES))
    t = np.linspace(0, t_end, 50)
    np.testing.assert_allclose(sol.sol(t)[4], A_HERMES * t,
                               rtol=1e-10, atol=1e-6)


def test_low_thrust_limit_matches_circular_speed_difference():
    """Slow tangential spiral: dv -> v_c(r1) - v_c(r2).

    Classic low-thrust result (Edelbaum, coplanar case), stated for
    the osculating semi-major axis a = -mu / (2E): r itself carries
    an epicyclic wobble of ~2 (a_T/g) r. A wrong thrust direction
    or magnitude would miss by far more than 0.1 %.
    """
    a_T = 1e-5                       # m/s^2, a_T/g_sun ~ 0.002
    r1 = R_EARTH_ORBIT
    y0 = circular_state(r1)
    sol = fly(y0, (0, 4 * 365.25 * DAY), prograde(a_T))
    for y in sol.y[:, ::max(1, sol.y.shape[1] // 10)].T:
        a = -MU_SUN / (2 * specific_energy(y))
        expected = np.sqrt(MU_SUN / r1) - np.sqrt(MU_SUN / a)
        # residual is a bounded wobble (< 0.4 m/s), not a drift
        assert y[4] == pytest.approx(expected, abs=1.0)
    a_end = -MU_SUN / (2 * specific_energy(sol.y[:, -1]))
    assert a_end > 1.08 * AU         # the spiral really went out
