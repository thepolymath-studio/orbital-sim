"""CR3BP equations of motion: Jacobi constant, an independent
inertial-frame integration, the two-body limit and derivatives."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from orbital_sim.cr3bp import (earth_moon, fly, grad_omega,
                               hessian_omega, jacobi, omega,
                               sun_jupiter, to_inertial, u_eff)
from orbital_sim.lagrange import lagrange_points

MU = earth_moon().mu
# Launched from L1 at about 21 m/s: the neck is open and the
# path crosses between the Earth and Moon sides four times.
X_L1 = lagrange_points(MU)[0, 0]
S_CHAOS = np.array([X_L1, 0.0, -0.02, 0.005])


def test_jacobi_conserved_on_chaotic_path():
    """C_J is the one conserved quantity. 140 units = 608 d;
    rtol 1e-12 over ~4,000 steps allows a few 1e-11."""
    t = np.linspace(0, 140.0, 4001)
    sol = fly(S_CHAOS, t[-1], MU, t_eval=t)
    assert np.ptp(jacobi(sol.y, MU)) < 2e-10
    side = np.sign(sol.y[0] - X_L1)
    assert np.count_nonzero(np.diff(side)) >= 4


def _inertial_rhs(t, s, mu):
    """Third body in the inertial frame; primaries on circles."""
    p1 = -mu * np.array([np.cos(t), np.sin(t)])
    p2 = (1 - mu) * np.array([np.cos(t), np.sin(t)])
    r = s[:2]
    a = -(1 - mu) * (r - p1) / np.linalg.norm(r - p1) ** 3
    a -= mu * (r - p2) / np.linalg.norm(r - p2) ** 3
    return np.r_[s[2:], a]


@pytest.mark.parametrize("mu", [MU, sun_jupiter().mu, 0.3])
def test_rotating_frame_matches_inertial_integration(mu):
    """Coriolis + centrifugal terms reproduce plain Newton."""
    s0 = np.array([0.5, 0.4, 0.1, -0.2])
    t = np.linspace(0, 8.0, 201)
    rot = fly(s0, t[-1], mu, t_eval=t)
    s0_in = to_inertial(0.0, s0, mu)
    ine = solve_ivp(_inertial_rhs, (0, t[-1]), s0_in, args=(mu,),
                    t_eval=t, method="DOP853", rtol=1e-12,
                    atol=1e-13)
    assert np.allclose(to_inertial(t, rot.y, mu), ine.y,
                       atol=1e-8)


def test_inertial_check_catches_flipped_coriolis():
    """Coriolis does no work, so C_J cannot see its sign; the
    inertial comparison can."""
    def bad(t, s):
        gx, gy = grad_omega(s[0], s[1], MU)
        return [s[2], s[3], gx - 2 * s[3], gy + 2 * s[2]]
    s0 = np.array([0.5, 0.4, 0.1, -0.2])
    t = np.linspace(0, 8.0, 201)
    rot = solve_ivp(bad, (0, 8.0), s0, t_eval=t, rtol=1e-12,
                    atol=1e-13, method="DOP853")
    ine = solve_ivp(_inertial_rhs, (0, 8.0),
                    to_inertial(0.0, s0, MU), args=(MU,),
                    t_eval=t, rtol=1e-12, atol=1e-13)
    assert np.abs(to_inertial(t, rot.y, MU) - ine.y).max() > 0.1


def test_two_body_limit_circular_orbit():
    """mu = 0: a circle of radius r turns at r^-1.5 - 1."""
    r = 0.6
    w = r ** -1.5
    s0 = [r, 0.0, 0.0, r * (w - 1)]
    t = 2 * np.pi / abs(w - 1)     # one synodic period
    sol = fly(s0, t, 0.0)
    assert np.allclose(sol.y[:, -1], s0, atol=1e-9)
    assert np.allclose(np.hypot(sol.y[0], sol.y[1]), r,
                       atol=1e-10)


def test_gradient_and_hessian_match_finite_differences():
    h = 1e-6
    for x, y in [(0.3, 0.2), (-0.7, -0.4), (1.2, 0.1)]:
        gx, gy = grad_omega(x, y, MU)
        fx = (omega(x + h, y, MU) - omega(x - h, y, MU)) / (2 * h)
        fy = (omega(x, y + h, MU) - omega(x, y - h, MU)) / (2 * h)
        assert (gx, gy) == pytest.approx((fx, fy), abs=1e-8)
        oxx, oxy, oyy = hessian_omega(x, y, MU)
        g = lambda a, b: np.array(grad_omega(a, b, MU))
        dgx = (g(x + h, y) - g(x - h, y)) / (2 * h)
        dgy = (g(x, y + h) - g(x, y - h)) / (2 * h)
        assert (oxx, oxy) == pytest.approx(dgx, abs=1e-6)
        assert (oxy, oyy) == pytest.approx(dgy, abs=1e-6)


def test_u_eff_shape():
    """Wells at both bodies, ground falling away far out."""
    assert u_eff(-MU + 1e-3, 0, MU) < -900
    assert u_eff(1 - MU + 1e-3, 0, MU) < -10
    far = [u_eff(r, 0.0, MU) for r in (2, 4, 8)]
    assert far[0] > far[1] > far[2]


def test_physical_scales_earth_moon():
    """Kepler period from n: 27.28 d. The real sidereal month,
    27.32 d, is slightly longer (the Sun's pull, a = 384,748 km);
    the circular model with D = 384,400 km is within 0.2%."""
    em = earth_moon()
    month = 2 * np.pi * em.time_unit / 86400
    assert month == pytest.approx(27.32, rel=2e-3)
    assert em.big_to_small == pytest.approx(81.30, abs=0.01)
    assert em.speed_unit == pytest.approx(1024.5, abs=0.5)
