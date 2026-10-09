"""Hohmann baseline: closed form vs research S4, and vs the integrator."""
import numpy as np
import pytest

from orbital_sim import (MU_SUN, R_EARTH_ORBIT, R_MARS_ORBIT,
                         circular_state, fly, hohmann, no_thrust)


def test_earth_mars_values_match_research_s4():
    dv1, dv2, _ = hohmann(R_EARTH_ORBIT, R_MARS_ORBIT)
    assert dv1 / 1e3 == pytest.approx(2.94, abs=0.005)
    assert dv2 / 1e3 == pytest.approx(2.65, abs=0.005)
    assert dv1 > dv2          # "burn 01 large, burn 02 small"


def test_integrated_transfer_arrives_at_aphelion():
    r1, r2 = R_EARTH_ORBIT, R_MARS_ORBIT
    dv1, dv2, t_H = hohmann(r1, r2)
    y0 = circular_state(r1)
    y0[3] += dv1              # tangential impulse at perihelion
    y = fly(y0, (0, t_H), no_thrust).y[:, -1]
    r = np.hypot(y[0], y[1])
    v_r = (y[0] * y[2] + y[1] * y[3]) / r
    assert r == pytest.approx(r2, rel=1e-9)
    assert abs(v_r) < 1e-4    # m/s: turning point
    assert np.degrees(np.arctan2(y[1], y[0])) == pytest.approx(
        180.0, abs=1e-6)
    v = np.hypot(y[2], y[3])
    assert v + dv2 == pytest.approx(np.sqrt(MU_SUN / r2), rel=1e-9)
