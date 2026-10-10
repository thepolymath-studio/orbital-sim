"""Vis-viva: special cases, inverses, and agreement with the integrator."""
import numpy as np
import pytest

from orbital_sim import fly, no_thrust
from orbital_sim.constants import MU_EARTH, MU_MOON, R_MOON
from orbital_sim.dynamics import specific_energy
from orbital_sim.visviva import (energy, escape_speed,
                                 hyperbola_speed, semi_major_axis,
                                 v_inf_squared, vis_viva)

R0 = 7.0e6


def test_circular_limit():
    """a = r: v = sqrt(mu / r), the circular speed."""
    for r in (R0, 4.2e7, 3.844e8):
        assert vis_viva(r, r, MU_EARTH) == pytest.approx(
            np.sqrt(MU_EARTH / r), rel=1e-15)


def test_parabola_is_escape_speed():
    assert vis_viva(R0, np.inf, MU_EARTH) == pytest.approx(
        escape_speed(R0, MU_EARTH), rel=1e-15)
    assert v_inf_squared(R0, escape_speed(R0, MU_EARTH),
                         MU_EARTH) == pytest.approx(0, abs=1e-6)


def test_hyperbola_negative_a():
    """a = -mu / v_inf^2 turns vis-viva into v^2 = v_inf^2 + 2mu/r."""
    for v_inf in (300.0, 1046.0, 3000.0):
        a = -MU_MOON / v_inf ** 2
        r = np.linspace(R_MOON, 50 * R_MOON, 7)
        assert np.allclose(vis_viva(r, a, MU_MOON),
                           hyperbola_speed(r, v_inf, MU_MOON),
                           rtol=1e-14)
        assert np.allclose(v_inf_squared(
            r, hyperbola_speed(r, v_inf, MU_MOON), MU_MOON),
            v_inf ** 2, rtol=1e-9)


@pytest.mark.parametrize("a", [1.2e7, 4.0e8, -2.0e7])
def test_semi_major_axis_inverts_vis_viva(a):
    r = 9.0e6
    v = vis_viva(r, a, MU_EARTH)
    assert semi_major_axis(r, v, MU_EARTH) == pytest.approx(
        a, rel=1e-12)
    assert energy(r, v, MU_EARTH) == pytest.approx(
        -MU_EARTH / (2 * a), rel=1e-12)


def _ellipse(r_p, a, mu, n_orbits):
    """Start at periapsis with the vis-viva speed; integrate."""
    v_p = vis_viva(r_p, a, mu)
    y0 = np.array([r_p, 0.0, 0.0, v_p, 0.0])
    period = 2 * np.pi * np.sqrt(a ** 3 / mu)
    sol = fly(y0, (0.0, n_orbits * period), no_thrust, mu)
    return sol, period


def test_integrator_obeys_vis_viva_on_an_ellipse():
    """e = 0.6 ellipse round Earth, 3 orbits.

    At every one of 4001 samples the integrated speed equals
    vis-viva at the integrated r (rel 1e-10), and energy is
    conserved (rel 1e-10). Half an orbit after each periapsis
    the integrator sits at the apoapsis 2a - r_p implied by
    vis-viva's a (to 1 m), moving purely sideways.
    """
    r_p, a, n = 8.0e6, 2.0e7, 3
    sol, period = _ellipse(r_p, a, MU_EARTH, n)
    Y = sol.sol(np.linspace(0, n * period, 4001))
    r = np.hypot(Y[0], Y[1])
    v = np.hypot(Y[2], Y[3])
    assert np.allclose(v, vis_viva(r, a, MU_EARTH), rtol=1e-10)
    E = specific_energy(Y, MU_EARTH)
    assert np.ptp(E) / abs(E[0]) < 1e-10
    for k in range(n):
        y = sol.sol((k + 0.5) * period)
        assert np.hypot(y[0], y[1]) == pytest.approx(
            2 * a - r_p, abs=1.0)
        v_radial = (y[0] * y[2] + y[1] * y[3]) / np.hypot(y[0], y[1])
        assert abs(v_radial) < 1e-6


def test_integrator_obeys_vis_viva_on_a_hyperbola():
    """The same check on a Moon hyperbola (v_inf 1 km/s)."""
    v_inf, r_p = 1.0e3, 2.0e6
    a = -MU_MOON / v_inf ** 2
    y0 = np.array([r_p, 0, 0, vis_viva(r_p, a, MU_MOON), 0])
    for span in (-6 * 3600.0, 6 * 3600.0):
        sol = fly(y0, (0.0, span), no_thrust, MU_MOON)
        r = np.hypot(sol.y[0], sol.y[1])
        v = np.hypot(sol.y[2], sol.y[3])
        assert np.allclose(v, vis_viva(r, a, MU_MOON),
                           rtol=1e-10)
        assert r.min() == pytest.approx(r_p, rel=1e-12)
