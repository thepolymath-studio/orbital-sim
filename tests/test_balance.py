"""Hook scenes: the L1 cloud splits and spreads, the L4 ring
librates and stays, Jacobi holds, runs are deterministic."""
import numpy as np
import pytest

from orbital_sim.balance import (fly_cloud, l1_cloud, l4_ring,
                                 slow_mode, ueff_grid)
from orbital_sim.constants import R_EARTH, R_MOON
from orbital_sim.cr3bp import earth_moon, jacobi, u_eff
from orbital_sim.lagrange import lagrange_points

EM = earth_moon()
MU, KM, TU = EM.mu, EM.sep / 1e3, EM.time_unit / 86400
RADII = (R_EARTH / EM.sep, R_MOON / EM.sep)
X_L1 = lagrange_points(MU)[0, 0]
L4 = lagrange_points(MU)[3]


@pytest.fixture(scope="module")
def l1_run():
    t = np.linspace(0, 90 / TU, 361)
    return t, *fly_cloud(l1_cloud(MU, 10, 1 / KM), t, MU, RADII)


def test_l1_cloud_starts_a_hair_apart():
    s = l1_cloud(MU, 10, 1 / KM)
    assert np.diff(s[:, 0]) * KM == pytest.approx(np.ones(9))
    assert np.abs(s[:, 1:]).max() == 0          # at rest


def test_l1_cloud_splits_both_ways_and_stays(l1_run):
    """Downhill both ways off the saddle; C_J > C_L1, so the
    neck is closed and no dot crosses back."""
    t, y, hit = l1_run
    side = np.sign(y[:, 0, 1:] - X_L1)
    assert (side == side[:, :1]).all()          # no crossing
    assert (side[:, 0] < 0).sum() == 5 and (side[:, 0] > 0).sum() == 5
    assert np.isnan(hit).all()
    c_l1 = jacobi(np.r_[X_L1, 0, 0, 0], MU)
    assert (jacobi(y[:, :, 0].T, MU) > c_l1).all()


def test_l1_neighbours_diverge(l1_run):
    """Dots 1 km apart end tens of thousands of km apart."""
    t, y, _ = l1_run
    gap = np.hypot(*(y[1:, :2, -1] - y[:-1, :2, -1]).T) * KM
    assert np.median(gap) > 1e4


def test_l1_early_growth_matches_eigenvalue(l1_run):
    """Separation grows like exp(lambda t), lambda = 2.93/unit."""
    t, y, _ = l1_run
    d = np.abs(y[5, 0] - X_L1) * KM       # starts 0.5 km
    i = np.searchsorted(t * TU, [2.0, 6.0])
    rate = np.log(d[i[1]] / d[i[0]]) / (t[i[1]] - t[i[0]])
    assert rate == pytest.approx(2.932, rel=0.02)


def test_l4_ring_on_slow_mode_and_stays():
    mode, w = slow_mode(MU)
    assert 2 * np.pi / w * TU == pytest.approx(91.5, abs=0.05)
    s = l4_ring(MU, 6, 2e4 / KM)
    assert np.abs(s[:, 0] - L4[0]).max() * KM == pytest.approx(
        2e4, rel=1e-12)
    t = np.linspace(0, 730 / TU, 1461)
    y, hit = fly_cloud(s, t, MU, RADII)
    d = np.hypot(y[:, 0] - L4[0], y[:, 1] - L4[1]) * KM
    assert np.isnan(hit).all() and d.max() < 40_000
    c = jacobi(y.transpose(1, 0, 2), MU)
    assert np.ptp(c, axis=1).max() < 1e-10


def test_l4_off_mode_rest_start_wanders():
    """Contrast (why the slow mode): 10,000 km off L4 at rest
    toward Earth's side leaves the L4 region."""
    s = np.r_[L4[0] + 1e4 / KM, L4[1], 0, 0][None]
    t = np.linspace(0, 730 / TU, 731)
    y, _ = fly_cloud(s, t, MU, RADII)
    d = np.hypot(y[0, 0] - L4[0], y[0, 1] - L4[1]) * KM
    assert np.nanmax(d) > 3e5


def test_deterministic():
    s = l1_cloud(MU, 3, 1 / KM)
    t = np.linspace(0, 5.0, 11)
    a, _ = fly_cloud(s, t, MU, RADII)
    b, _ = fly_cloud(s, t, MU, RADII)
    assert np.array_equal(a, b)


def test_grid_layout():
    x, y, U = ueff_grid(MU, 1.5, 31)
    assert U.shape == (31, 31)
    assert U[3, 7] == u_eff(x[7], y[3], MU)
