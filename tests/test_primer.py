"""Primer-vector steering: the equations behind strategies 02 and 03."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from orbital_sim import A_HERMES, AU, DAY, MU_SUN, circular_state
from orbital_sim.dynamics import gravity
from orbital_sim.primer import (ATOL8, N_E, T_UNIT, fly_batch,
                                primer_rhs, primer_start)

SHIP = circular_state(AU)[:4]
Z = np.array([0.7, -0.4, 0.9])


def hamiltonian(y, a_th=A_HERMES, mu=MU_SUN):
    """H = lambda_r . v + lambda_v . (g + a_th u), with
    lambda_v = -p and lambda_r = p' (so u = p / |p| minimises H)."""
    r, v, p, q = y[0:2], y[2:4], y[4:6], y[6:8]
    return q @ v - p @ gravity(r, mu) - a_th * np.linalg.norm(p)


def test_thrust_has_magnitude_a_th_along_p():
    y = primer_start(SHIP, Z)
    a = primer_rhs(0.0, y, MU_SUN, A_HERMES)[2:4]
    a_T = a - gravity(y[:2])
    assert np.linalg.norm(a_T) == pytest.approx(A_HERMES, rel=1e-14)
    p = y[4:6]
    assert a_T @ p / np.linalg.norm(p) == pytest.approx(A_HERMES,
                                                       rel=1e-14)


def test_hamiltonian_is_constant_along_an_extremal():
    """Autonomous dynamics: H is a first integral. A sign error
    in G (p'' = G p) or in the steering would break this."""
    y0 = primer_start(SHIP, Z)
    sol = solve_ivp(primer_rhs, (0, 300 * DAY), y0, "DOP853",
                    rtol=1e-12, atol=ATOL8, args=(MU_SUN, A_HERMES),
                    t_eval=np.linspace(0, 300 * DAY, 61))
    H = np.array([hamiltonian(y) for y in sol.y.T])
    scale = np.abs(N_E * np.linalg.norm(SHIP[2:]))
    assert np.max(np.abs(H - H[0])) / scale < 1e-9


def test_batched_rhs_matches_single_rhs():
    """fly_batch (scaled time, K ships at once) = primer_rhs."""
    Zs = np.array([Z, [2.0, 0.3, -0.2], [4.0, 0.0, 0.0]])
    T = np.array([150, 220, 90]) * DAY
    Y = fly_batch(np.repeat(SHIP[None], 3, 0), Zs, T, A_HERMES,
                  rtol=1e-12)[:, :, -1]
    for k in range(3):
        y0 = primer_start(SHIP, Zs[k])
        ref = solve_ivp(primer_rhs, (0, T[k]), y0, "DOP853",
                        rtol=1e-12, atol=ATOL8,
                        args=(MU_SUN, A_HERMES)).y[:, -1]
        assert np.allclose(Y[:2, k], ref[:2], rtol=0, atol=1e3)
        assert np.allclose(Y[2:4, k], ref[2:4], rtol=0, atol=1e-4)


def test_primer_start_layout():
    y = primer_start(SHIP, Z)
    assert np.allclose(y[:4], SHIP)
    assert np.allclose(y[4:6], [np.cos(Z[0]), np.sin(Z[0])])
    assert np.allclose(y[6:8], N_E * Z[1:])
    assert T_UNIT == 100 * DAY
