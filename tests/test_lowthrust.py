"""Direct piecewise-linear steering law: the optimality cross-check."""
import numpy as np
import pytest

from orbital_sim import A_HERMES, AU, DAY, MU_SUN, circular_state, fly
from orbital_sim.lowthrust import _shoot, steering, thrust_direction


def test_u_zero_is_prograde_on_a_circular_orbit():
    y = circular_state(AU)
    u = thrust_direction(y[:2], 0.0)
    v = y[2:4] / np.linalg.norm(y[2:4])
    assert np.allclose(u, v, atol=1e-15)
    assert np.allclose(thrust_direction(y[:2], np.pi / 2),
                       y[:2] / AU, atol=1e-15)


def test_batched_shoot_matches_fly_for_different_T():
    """Each batch member keeps its own T (regression: the time
    scaling must follow the [x.., y.., vx.., vy..] layout)."""
    y0 = circular_state(AU)
    U = np.array([np.radians([80, 120, 170]),
                  np.radians([10, 40, 200]),
                  np.radians([0, 0, 0])])
    T = np.array([150, 230, 60]) * DAY
    Y = _shoot(T, U, [y0[:4]] * 3, A_HERMES, MU_SUN)
    for k in range(3):
        ref = fly(y0, (0, T[k]), steering(U[k], T[k])).y[:4, -1]
        assert np.allclose(Y[k, :2], ref[:2], rtol=0, atol=2e3)
        assert np.allclose(Y[k, 2:], ref[2:], rtol=0, atol=1e-3)
