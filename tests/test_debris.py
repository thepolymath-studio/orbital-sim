"""The film's cloud: batch integration, returns, ring and shell.

Catches: a batch integration that is less accurate than one piece
alone, a cloud that does not conserve energy and polar angular
momentum, a pass/return measurement that is off (checked against
pure Kepler, where every piece must come back to the exact breakup
point after its own period), a ring that secretly depends on J2,
and a turning rate that disagrees with the closed form.
"""
import numpy as np
import pytest

from orbital_sim import debris as D
from orbital_sim.cloud import (first_time_over, fitted_rate,
                               middle_range, pass_times,
                               propagate_cloud, read_elements,
                               unwrap)
from orbital_sim.constants import DAY, R_EARTH_EQ
from orbital_sim.elements import elements, kepler_period
from orbital_sim.j2 import energy, node_rate
from orbital_sim.propagate import RTOL as TIGHT


@pytest.fixture(scope="module")
def cloud():
    return D.film_breakup()


def test_film_breakup_is_deterministic(cloud):
    again = D.film_breakup()
    assert np.array_equal(cloud.v, again.v)
    assert len(cloud.lc) == D.N_PIECES == 500


def test_orbit_numbers_600km():
    o = D.orbit_numbers()
    assert o["period_min"] == pytest.approx(96.687, abs=0.001)
    assert o["speed_kms"] == pytest.approx(7.5579, abs=1e-4)
    # closed-form turning of the 51.6 deg plane (research S11)
    assert o["node_rate_deg_day"] == pytest.approx(-4.52, abs=0.005)


def test_batch_equals_single_piece_runs(cloud):
    # include the most eccentric pieces: they set the step size
    el = elements(cloud.r, cloud.v)
    pick = list(np.argsort(el["e"])[-3:]) + [0, 1, 2]
    t = np.linspace(0, 3 * DAY, 73)
    batch = propagate_cloud(cloud.r, cloud.v, t)[pick]
    for k, p in enumerate(pick):
        one = propagate_cloud(cloud.r[p:p + 1], cloud.v[p:p + 1],
                              t, rtol=TIGHT)[0]
        # 3 days ~ 45 orbits, ~ 1e6 km flown: within 50 m
        assert np.max(np.linalg.norm(batch[k, :, :3] - one[:, :3],
                                     axis=1)) < 50.0


def test_cloud_conserves_energy_and_polar_spin(cloud):
    t = np.linspace(0, DAY, 97)
    s = propagate_cloud(cloud.r, cloud.v, t)
    r, v = s[..., :3], s[..., 3:]
    E = energy(r, v)
    hz = np.cross(r, v)[..., 2]
    assert np.max(np.abs(E / E[:, :1] - 1)) < 1e-9
    assert np.max(np.abs(hz / hz[:, :1] - 1)) < 1e-9


def test_pass_times_on_a_known_angle():
    t = np.linspace(0, 1000, 1001)
    u = np.array([2 * np.pi * t / 100, 2 * np.pi * t / 250])
    assert pass_times(t, u, 1) == pytest.approx([100, 250])
    assert pass_times(t, u, 3) == pytest.approx([300, 750])
    assert np.isnan(pass_times(t, u, 5)[1])


def test_helpers():
    x = np.linspace(0, 100, 100_001)
    assert middle_range(x, 0.9) == pytest.approx(90, abs=1e-6)
    assert middle_range(x, 0.5) == pytest.approx(50, abs=1e-6)
    t = np.linspace(0, 10, 11)
    ang = np.array([2 * t + 1, -0.5 * t])
    assert fitted_rate(t, ang) == pytest.approx([2, -0.5])
    assert first_time_over(t, t**2, 30) == 6
    assert np.isnan(first_time_over(t, t, 99))


def kepler_day(cloud, j2):
    t = np.arange(0, 0.6 * DAY, 60.0)
    s, sp = D.fly(cloud, t, j2)
    return t, s, read_elements(s), read_elements(sp)


def test_without_j2_every_piece_returns_to_the_breakup_point(cloud):
    t, s, el, elp = kepler_day(cloud, j2=False)
    pn = D.pass_numbers(t, el, elp, s, ks=(1, 2))
    # the intact orbit is back after exactly Kepler's period
    T = kepler_period(R_EARTH_EQ + D.ALT) / 60
    assert pn[1]["parent_return_min"] == pytest.approx(T, abs=1e-3)
    # each piece after its own period, at the breakup point
    a = elements(cloud.r, cloud.v)["a"]
    t1 = pass_times(t, unwrap(el["u"]), 1)
    ok = np.isfinite(t1)
    assert ok.mean() > 0.95
    assert np.allclose(t1[ok], kepler_period(a[ok]), atol=0.5)
    assert pn[1]["miss_median_km"] < 0.5
    assert pn[2]["miss_median_km"] < 0.5


def test_with_j2_the_return_spot_drifts_with_the_turning_plane(
        cloud):
    t, s, el, elp = kepler_day(cloud, j2=True)
    pn = D.pass_numbers(t, el, elp, s, ks=(1, 2, 4))
    r0 = R_EARTH_EQ + D.ALT
    rate = abs(node_rate(r0, 0.0, D.INC))
    for k in (1, 2, 4):
        want = r0 * rate * pn[k]["parent_return_min"] * 60 / 1e3
        assert pn[k]["miss_median_km"] == pytest.approx(want, rel=0.05)
    # the bunch loosens pass by pass (Kepler: in proportion)
    w = [pn[k]["window90_min"] for k in (1, 2, 4)]
    assert w[1] == pytest.approx(2 * w[0], rel=0.02)
    assert w[2] == pytest.approx(4 * w[0], rel=0.02)


def test_ring_comes_from_period_differences_not_j2(cloud):
    out = {}
    for j2 in (True, False):
        t, s, el, elp = kepler_day(cloud, j2)
        out[j2] = D.ring_numbers(t, el, elp)["ring_h"]
    est = D.kepler_ring_h(cloud)
    assert out[True] == pytest.approx(out[False], rel=0.01)
    assert out[False] == pytest.approx(est, rel=0.01)


def test_cloud_turning_matches_closed_form(cloud):
    # 4 samples a day, so the time averages of a, e, i see many
    # orbital phases (the closed form wants mean elements)
    t = np.arange(0, 30.01, 0.25) * DAY
    s, _ = D.fly(cloud, t)
    el = read_elements(s)
    sim, cf = D.precession_check(t, el)
    assert np.max(np.abs(sim / cf - 1)) < 3e-3
    assert middle_range(sim) == pytest.approx(middle_range(cf),
                                              rel=0.01)
    # the spread grows in proportion to time (secular, not noise)
    sh = D.shell_numbers(t, el)
    w = sh["spread90_rad"]
    assert w[120] == pytest.approx(2 * w[60], rel=0.03)


def test_shell_metric_on_known_rates():
    rates = np.linspace(-1.0, 0.0, 1001)          # deg/day
    t = np.arange(0, 800) * DAY
    raan = np.radians(rates[:, None] * t[None] / DAY) % (2 * np.pi)
    sh = D.shell_numbers(t, {"raan": raan})
    # middle 90 % of rates spans 0.9 deg/day -> 400 days
    assert sh["shell_days"] == pytest.approx(400, abs=1)
    assert sh["half_shell_days"] == pytest.approx(200, abs=1)


def test_tilt_matters_as_cos_i():
    out = D.tilt_sensitivity((28.5, 51.6, 82.5))
    days = [out[k]["shell_days"] for k in ("28.5", "51.6", "82.5")]
    assert days[0] < days[1] < days[2]
