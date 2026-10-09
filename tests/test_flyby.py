"""Patched-conic flyby: equations 08 and 09, closed form and integrated."""
import numpy as np
import pytest

from orbital_sim import fly, no_thrust
from orbital_sim.constants import MU_EARTH, R_EARTH
from orbital_sim.flyby import (flyby, hyperbola_state,
                               periapsis_for_turn, rotate,
                               true_anomaly_at, turn_angle)
from orbital_sim.propagate import reach_radius

R_P = R_EARTH + 300e3          # research S14: 300 km altitude


@pytest.mark.parametrize("v_inf, e, delta_deg, dv_kms", [
    (6e3, 1.60, 77, 7.5),      # research S14, first case
    (3e3, 1.15, 121, 5.2),     # research S14, second case
])
def test_turn_angle_matches_research_s14(v_inf, e, delta_deg,
                                         dv_kms):
    delta = turn_angle(v_inf, R_P, MU_EARTH)
    e_got = 1 + R_P * v_inf ** 2 / MU_EARTH
    assert e_got == pytest.approx(e, abs=0.005)
    assert np.degrees(delta) == pytest.approx(delta_deg, abs=0.5)
    # |dv_helio| = 2 v_inf sin(delta/2) = 2 v_inf / e  (S13)
    dv = 2 * v_inf * np.sin(delta / 2)
    assert dv == pytest.approx(2 * v_inf / e_got, rel=1e-12)
    assert dv / 1e3 == pytest.approx(dv_kms, abs=0.05)


def test_periapsis_for_turn_inverts_eq09():
    for v_inf in (2e3, 6e3, 9e3):
        for r_p in (R_P, 2 * R_P, 30 * R_P):
            d = turn_angle(v_inf, r_p, MU_EARTH)
            back = periapsis_for_turn(v_inf, d, MU_EARTH)
            assert back == pytest.approx(r_p, rel=1e-10)
            assert periapsis_for_turn(v_inf, -d, MU_EARTH) == \
                pytest.approx(r_p, rel=1e-10)


def test_flyby_keeps_v_inf_and_turns_by_psi():
    """Eq. 08 in the Sun's frame: only v_inf's direction turns."""
    v_planet = np.array([0.0, 29.78e3])
    v_ship = np.array([-5.6e3, 31.9e3])
    for psi in (-1.2, -0.3, 0.0, 0.7, 1.3):
        v_out = flyby(v_ship, v_planet, psi)
        a, b = v_ship - v_planet, v_out - v_planet
        assert np.linalg.norm(b) == pytest.approx(
            np.linalg.norm(a), rel=1e-14)
        ang = np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)
        assert ang == pytest.approx(psi, abs=1e-13)


@pytest.mark.parametrize("psi_sign", [1, -1])
def test_integrated_hyperbola_obeys_eq08_and_eq09(psi_sign):
    """Fly the hyperbola with the integrator in Earth's frame.

    From r = 1e9 m (beyond Earth's sphere of influence, ~9.2e8 m)
    in, round periapsis, and out to the same radius:
      - the speed out equals the speed in (energy: 1e-10);
      - the closest approach is r_p (1e-8);
      - the state out matches the analytic conic (1e-8);
      - the velocity turns by delta up to the finite-distance
        offset, which shrinks like mu / (r v_inf^2) (~7e-3 here).
    """
    v_inf = 6e3
    v_in = rotate(np.array([v_inf, 0.0]), 0.4)
    delta = turn_angle(v_inf, R_P, MU_EARTH)
    psi = psi_sign * delta
    e = 1 + R_P * v_inf ** 2 / MU_EARTH
    f1 = true_anomaly_at(1e9, e, R_P)
    r0, v0 = hyperbola_state(v_in, psi, R_P, MU_EARTH, -f1)
    y0 = np.concatenate([r0, v0, [0.0]])
    sol = fly(y0, (0, 30 * 86400.0), no_thrust, MU_EARTH,
              [reach_radius(1e9)])
    y1 = sol.y[:, -1]
    r1, v1 = hyperbola_state(v_in, psi, R_P, MU_EARTH, f1)
    s0, s1 = np.linalg.norm(v0), np.linalg.norm(y1[2:4])
    assert s1 == pytest.approx(s0, rel=1e-10)
    r_min = np.min(np.hypot(*sol.sol(np.linspace(
        0, sol.t[-1], 20001))[:2]))
    assert r_min == pytest.approx(R_P, rel=1e-6)
    assert np.allclose(y1[:2], r1, rtol=0, atol=1e-8 * 1e9)
    assert np.allclose(y1[2:4], v1, rtol=0, atol=1e-8 * s0)
    turn = np.arctan2(v0[0] * y1[3] - v0[1] * y1[2],
                      v0 @ y1[2:4])
    assert np.sign(turn) == psi_sign
    off = MU_EARTH / (1e9 * v_inf ** 2)
    assert abs(abs(turn) - delta) < 2 * off


def test_asymptotes_are_v_inf_in_and_out():
    """Far out on the conic, velocity -> v_inf_in / v_inf_out."""
    v_inf = 4e3
    v_in = rotate(np.array([v_inf, 0.0]), -1.0)
    psi = 0.9
    r_p = periapsis_for_turn(v_inf, psi, MU_EARTH)
    e = 1 + r_p * v_inf ** 2 / MU_EARTH
    f = true_anomaly_at(1e15, e, r_p)
    _, va = hyperbola_state(v_in, psi, r_p, MU_EARTH, -f)
    _, vb = hyperbola_state(v_in, psi, r_p, MU_EARTH, f)
    assert np.allclose(va, v_in, atol=1e-3)
    assert np.allclose(vb, rotate(v_in, psi), atol=1e-3)
