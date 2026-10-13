"""Waiting near Miller's planet (stand-in, Earth-Moon proportions):
the balance points hold or drop a ship as linear theory says, and
the L1 cloud splits on the saddle's unstable direction and widens at
the e-folding rate."""
import numpy as np
import pytest

from orbital_sim import cr3bp
from orbital_sim.balance import fly_cloud
from orbital_sim.constants import DAY
from orbital_sim.cr3bp import earth_moon, jacobi
from orbital_sim.lagrange import growth_rate, lagrange_points
from orbital_sim.miller import ORBIT, stand_in_mu, stand_in_radii
from orbital_sim.waiting import (closest_to_small, departure,
                                 drift, efold_orbits,
                                 first_time_over, hill_radius,
                                 misplace, point_cloud, saddle_side,
                                 spread, sunflower, waiting_starts,
                                 where_now)

MU = stand_in_mu()
RADII = stand_in_radii()
PTS = lagrange_points(MU)
D = 1e-6                                  # the post's small error


def orbits(o0, o1, n):
    return np.linspace(o0, o1, n) * ORBIT


# ---- points and e-folding ----------------------------------------

def test_efold_orbits_matches_growth_rate():
    lam = growth_rate(*PTS[0], MU)
    assert efold_orbits(MU, "L1") == pytest.approx(
        1 / lam / ORBIT)
    assert np.isinf(efold_orbits(MU, "L4"))
    assert np.isinf(efold_orbits(MU, "L5"))


def test_efold_in_days_matches_earth_moon():
    """Same numbers as the concept post's Earth-Moon check:
    1.48 d at L1, 2.01 d at L2 (one model month 27.28 d)."""
    em = earth_moon()
    month = ORBIT * em.time_unit / DAY
    assert efold_orbits(MU, "L1") * month == pytest.approx(
        1.48, abs=0.005)
    assert efold_orbits(MU, "L2") * month == pytest.approx(
        2.01, abs=0.005)


# ---- the waiting test --------------------------------------------

def test_misplace_offsets():
    s = misplace(PTS[0], 2e-6, 3e-6)
    assert s[0] - PTS[0, 0] == pytest.approx(2e-6)
    assert s[3] == 3e-6 and s[2] == 0.0 and s[1] == PTS[0, 1]
    w = waiting_starts(MU, D, D)
    assert w.shape == (15, 4)
    assert np.array_equal(w[::3, :2], PTS)
    assert np.array_equal(w[::3, 2:], np.zeros((5, 2)))


@pytest.fixture(scope="module")
def wait10():
    t = orbits(0, 10, 1001)
    tracks, hit = fly_cloud(waiting_starts(MU, D, D), t, MU,
                            RADII)
    return t, tracks, hit


def test_exact_ships_stay_put(wait10):
    """L3-L5 exactly placed: within 1e-9 a for 10 orbits;
    L1/L2 within 1e-8 a for one orbit (then the computer's
    rounding, ~1e-16, has grown: saddles)."""
    t, tracks, hit = wait10
    for k in (6, 9, 12):
        assert drift(tracks[k], PTS[k // 3]).max() < 1e-9
    i1 = np.searchsorted(t, ORBIT)
    for k in (0, 3):
        assert drift(tracks[k], PTS[k // 3])[:i1].max() < 1e-8
    assert np.isnan(hit).all()


def test_saddle_gap_grows_at_eigenvalue_rate(wait10):
    """Linear phase at L1/L2: gap ~ exp(lambda t), fitted
    0.2 -> 0.5 orbits (the decaying mode has died out and
    the gap is still under 1 % of a)."""
    t, tracks, _ = wait10
    i, j = np.searchsorted(t, [0.2 * ORBIT, 0.5 * ORBIT])
    for k, p in ((1, 0), (4, 1)):
        d = drift(tracks[k], PTS[p])
        assert d[j] < 0.01
        rate = np.log(d[j] / d[i]) / (t[j] - t[i])
        assert rate == pytest.approx(growth_rate(*PTS[p], MU),
                                     rel=0.01)


def test_misplaced_ships_leave_l1_l2_l3_only(wait10):
    t, tracks, _ = wait10
    for k in range(15):
        if k % 3 == 0:
            continue
        gone = first_time_over(t, drift(tracks[k], PTS[k // 3]),
                               0.01)
        assert np.isnan(gone) == (k >= 9)
        if k < 6:
            assert gone < ORBIT          # L1, L2: within an orbit


def test_l4_l5_hold_misplaced_ships_for_300_orbits():
    """Bounded swing: under 30x the start error, and the
    second half never goes farther than the first."""
    t = orbits(0, 300, 6001)
    starts = waiting_starts(MU, D, D)[[10, 11, 13, 14]]
    tracks, hit = fly_cloud(starts, t, MU, RADII)
    assert np.isnan(hit).all()
    for tr, p in zip(tracks, (3, 3, 4, 4)):
        d = drift(tr, PTS[p])
        h = d.size // 2
        assert d.max() < 30 * D
        assert d[h:].max() < 1.01 * d[:h].max()


def test_l1_planet_side_ship_trapped_by_jacobi():
    """C_J above L1's: a ship placed planet-side of L1 at
    rest can never leave the planet's region."""
    s0 = misplace(PTS[0], D, 0.0)
    c_l1 = jacobi(np.r_[PTS[0], 0, 0], MU)
    assert jacobi(s0, MU) > c_l1
    t = orbits(0, 10, 1001)
    tr, _ = fly_cloud(s0[None], t, MU, RADII)
    _, r2 = cr3bp.distances(tr[0, 0], tr[0, 1], MU)
    assert np.nanmax(r2) < 1.2 * hill_radius(MU)
    assert where_now(tr[0, :, -1], MU) == "near the small body"


# ---- the L1 cloud ------------------------------------------------

def test_sunflower_disc_fits_inside_diameter():
    xy = sunflower(np.zeros(2), 1.0, 100)
    gaps = np.hypot(*(xy[:, None] - xy[None]).T)
    assert gaps.max() < 1.0 and np.hypot(*xy.T).max() < 0.5
    assert np.array_equal(xy, sunflower(np.zeros(2), 1.0, 100))


def test_point_cloud_centred_and_at_rest():
    c = point_cloud(MU, "L1", 50, D)
    assert np.allclose(c[:, :2].mean(axis=0), PTS[0], atol=D / 5)
    assert spread(c[:, :, None])[0] < D
    assert not c[:, 2:].any()


@pytest.fixture(scope="module")
def cloud():
    starts = point_cloud(MU, "L1", 24, D)
    t = orbits(0, 2, 201)
    tracks, hit = fly_cloud(starts, t, MU, RADII)
    return starts, t, tracks, hit


def test_cloud_split_decided_by_saddle_direction(cloud):
    """Which way a ship leaves L1 (planet side = +x) is the
    sign of its start along the unstable direction."""
    starts, t, tracks, hit = cloud
    side = saddle_side(MU, starts, "L1")
    went = np.array([departure(tr, PTS[0], 0.01)
                     for tr in tracks])
    assert np.array_equal(went, side)
    assert 0 < (side > 0).sum() < len(side)
    assert np.isnan(hit).all()


def test_saddle_side_orientation_at_l2():
    """At L2 the planet lies on the -x side: a ship nudged
    planet-ward along the unstable direction scores +1."""
    s = np.array([misplace(PTS[1], -D), misplace(PTS[1], D)])
    assert list(saddle_side(MU, s, "L2")) == [1.0, -1.0]


def test_cloud_widens_at_l1_efold_rate(cloud):
    starts, t, tracks, _ = cloud
    w = spread(tracks)
    assert w[0] < D
    i, j = np.searchsorted(t, [0.1 * ORBIT, 0.4 * ORBIT])
    rate = np.log(w[j] / w[i]) / (t[j] - t[i])
    assert rate == pytest.approx(growth_rate(*PTS[0], MU),
                                 rel=0.03)
    assert w[-1] > 0.5                    # half the distance


def test_cloud_conserves_jacobi(cloud):
    _, _, tracks, _ = cloud
    c = jacobi(tracks.transpose(1, 0, 2), MU)
    assert np.ptp(c, axis=1).max() < 1e-10


def test_cloud_fates_converged_in_tolerance(cloud, monkeypatch):
    """A 100x looser tolerance gives the same sides and the
    same 1 %-of-a time (to one sample)."""
    starts, t, tracks, _ = cloud
    monkeypatch.setattr(cr3bp, "RTOL", 1e-10)
    loose, _ = fly_cloud(starts, t, MU, RADII)
    for a, b in zip(tracks, loose):
        assert departure(a, PTS[0], 0.01) == departure(
            b, PTS[0], 0.01)
    ta = first_time_over(t, spread(tracks), 0.01)
    tb = first_time_over(t, spread(loose), 0.01)
    assert abs(ta - tb) <= t[1] - t[0]


def test_departure_zero_if_never_far():
    tr = np.tile(np.r_[PTS[3], 0, 0][:, None], (1, 5))
    assert departure(tr, PTS[3], 0.01) == 0


def test_deterministic():
    s = point_cloud(MU, "L1", 4, D)
    t = orbits(0, 1, 51)
    a, _ = fly_cloud(s, t, MU, RADII)
    b, _ = fly_cloud(s, t, MU, RADII)
    assert np.array_equal(a, b)


def test_where_now_labels_and_closest():
    rh = hill_radius(MU)
    small = 1 - MU
    assert where_now([small + rh / 2, 0], MU) == \
        "near the small body"
    assert where_now([small - 3 * rh, 0], MU) == \
        "inside the orbit"
    assert where_now([small + 3 * rh, 0], MU) == \
        "outside the orbit"
    assert where_now(PTS[3], MU) == "along the orbit"
    tr = np.array([[small + 0.1, small + 0.2], [0, 0],
                   [0, 0], [0, 0]])
    assert closest_to_small(tr, MU) == pytest.approx(0.1)
