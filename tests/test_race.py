"""2026-10-wk1 Sim post: the race back to Mars (strategies 01-03)."""
from datetime import date

import numpy as np
import pytest

from orbital_sim import (A_HERMES, AU, DAY, MU_SUN, R_EARTH_ORBIT,
                         R_MARS_ORBIT, hohmann)
from orbital_sim import race, search
from orbital_sim.constants import MU_EARTH, R_EARTH
from orbital_sim.ephemeris import mars_lead, mean_longitude, mean_motion


# ------------------------------------------------------ set-up checks
def test_mean_longitudes_reproduce_j2000_and_synodic_period():
    """Standish elements at J2000; eq. 05 synodic period 779.9 d."""
    d0 = date(2000, 1, 1)          # 00:00, half a day before J2000
    assert mean_longitude("earth", d0) == pytest.approx(
        100.46457166 - 35999.37244981 * 0.5 / 36525, abs=1e-9)
    d1 = date(2036, 7, 6)
    lead_rate = mars_lead(d1) - mars_lead(date(2036, 7, 5))
    S = 2 * np.pi / ((-lead_rate) % (2 * np.pi)) * DAY
    assert S / DAY == pytest.approx(779.9, abs=0.2)
    n_e, n_m = mean_motion(R_EARTH_ORBIT), mean_motion(R_MARS_ORBIT)
    assert 2 * np.pi / (n_e - n_m) / DAY == pytest.approx(779.9,
                                                          abs=1.5)


def test_nominal_phase_is_the_novels_flyby_date():
    sc = race.nominal()
    assert np.degrees(sc.mars_lead) == pytest.approx(219.63, abs=0.01)


def test_inbound_state_is_coming_from_mars_at_6_km_s():
    y = race.inbound_state()
    r, v = y[:2], y[2:]
    _, v_e = race.earth(0.0)
    assert np.linalg.norm(r) == pytest.approx(R_EARTH_ORBIT, rel=1e-15)
    assert np.linalg.norm(v - v_e) == pytest.approx(6e3, rel=1e-9)
    assert v[0] < 0                  # inbound: moving sunward
    assert v[1] > v_e[1]             # faster than Earth along track
    # aphelion of the coasting ellipse is Mars's orbit
    eps = 0.5 * v @ v - MU_SUN / np.linalg.norm(r)
    a = -MU_SUN / (2 * eps)
    h = r[0] * v[1] - r[1] * v[0]
    e = np.sqrt(1 - h ** 2 / (MU_SUN * a))
    assert a * (1 + e) == pytest.approx(R_MARS_ORBIT, rel=1e-9)


def test_flyby_turn_limit_is_eq09_at_300_km():
    sc = race.nominal()
    e = 1 + (R_EARTH + 300e3) * 6e3 ** 2 / MU_EARTH
    assert race.turn_limit(sc) == pytest.approx(2 * np.arcsin(1 / e),
                                                rel=1e-14)
    assert np.degrees(race.turn_limit(sc)) == pytest.approx(77.2,
                                                            abs=0.05)


def test_after_flyby_keeps_v_inf():
    sc = race.nominal()
    _, v_e = race.earth(0.0)
    for psi in (-1.0, 0.0, 0.4, race.turn_limit(sc)):
        y = race.after_flyby(sc, psi)
        assert np.linalg.norm(y[2:] - v_e) == pytest.approx(6e3,
                                                            rel=1e-12)


# --------------------------------------------------------- 01 Hohmann
@pytest.fixture(scope="module")
def r01():
    return race.race_hohmann(race.nominal())


def test_hohmann_window_puts_mars_alpha_ahead(r01):
    """Eq. 04: at departure Mars leads Earth by alpha = pi - n2 t_H."""
    t_w = r01.info["t_wait"]
    _, _, t_H = hohmann(R_EARTH_ORBIT, R_MARS_ORBIT)
    alpha = np.pi - mean_motion(R_MARS_ORBIT) * t_H
    assert np.degrees(alpha) == pytest.approx(44.35, abs=0.01)
    r_e, _ = race.earth(t_w)
    r_m, _ = race.mars(race.nominal().mars_lead)(t_w)
    lead = np.arctan2(r_e[0] * r_m[1] - r_e[1] * r_m[0], r_e @ r_m)
    assert lead == pytest.approx(alpha, abs=1e-12)
    assert 0 <= t_w < 2 * np.pi / (mean_motion(R_EARTH_ORBIT)
                                   - mean_motion(R_MARS_ORBIT))


def test_hohmann_leg_matches_closed_form(r01):
    dv1, dv2, t_H = hohmann(R_EARTH_ORBIT, R_MARS_ORBIT)
    assert r01.info["t_H"] == t_H
    assert r01.t_arrive == pytest.approx(r01.info["t_wait"] + t_H,
                                         rel=1e-15)
    assert [k[1] for k in r01.impulses] == [6e3, dv1, dv2]
    assert r01.dv == pytest.approx(6e3 + 2944.8 + 2649.0, abs=1.0)
    # integrated: the ellipse ends on Mars within 100 m, 1e-5 m/s
    assert r01.miss_r < 100.0
    assert r01.miss_v < 1e-5


# ----------------------------------------------- 02 thrust, 03 Purnell
@pytest.fixture(scope="module")
def lowthrust():
    """02 and 03 at the nominal phase (scan + psi continuation)."""
    sc = race.nominal()
    X2, (psi3, X3) = search.fastest_at(sc)
    return {
        "sc": sc,
        "02": race.race_lowthrust(sc, X2, 0.0, "02"),
        "03": race.race_lowthrust(sc, X3, psi3, "03"),
    }


@pytest.mark.parametrize("name", ["01", "02", "03"])
def test_each_strategy_reaches_mars_within_tolerance(name, r01,
                                                     lowthrust):
    res = r01 if name == "01" else lowthrust[name]
    assert res.miss_r < race.TOL_R          # 10 km
    assert res.miss_v < race.TOL_V          # 1 cm/s


@pytest.mark.parametrize("name", ["02", "03"])
def test_ion_drive_dv_equals_a_th_t(name, lowthrust):
    """Eq. 07: always on, so dv = a_th T exactly."""
    res = lowthrust[name]
    assert res.dv == pytest.approx(A_HERMES * res.t_arrive, rel=1e-10)


def test_purnell_never_slower_than_thrust_only(lowthrust):
    assert lowthrust["03"].t_arrive <= lowthrust["02"].t_arrive


def test_ranking_at_nominal_phase(r01, lowthrust):
    T = [r01.t_arrive, lowthrust["02"].t_arrive,
         lowthrust["03"].t_arrive]
    assert T[2] <= T[1] < T[0]
    assert T[0] / T[2] == pytest.approx(2.72, abs=0.02)


def test_flyby_is_physical(lowthrust):
    info = lowthrust["03"].info
    assert info["flyby"]
    assert info["alt_p"] >= 300e3 - 1.0
    assert abs(info["psi"]) <= race.turn_limit(lowthrust["sc"])


@pytest.mark.parametrize("name", ["02", "03"])
def test_route_stays_outside_the_perihelion_floor(name, lowthrust):
    info = lowthrust[name].info
    assert info["r_min"] >= 0.999 * lowthrust["sc"].r_min
    assert info["r_max"] <= 1.0001 * R_MARS_ORBIT


def test_interior_flyby_optimum_satisfies_transversality():
    """Away from the floor, the best turn psi is strictly inside
    the limits and Pontryagin's condition p(0) || v_inf_out holds.
    At Mars lead 84.6 deg the 02 route stays beyond 0.8 AU."""
    sc = race.Scenario(mars_lead=np.radians(84.6))
    _, (psi, X) = search.fastest_at(sc)
    res = race.race_lowthrust(sc, X, psi, "03")
    assert abs(psi) < 0.9 * race.turn_limit(sc)
    assert not res.info["floor_binds"]
    assert abs(res.info["transversality"]) < 1e-4


# ------------------------------------------------------- determinism
def test_search_is_deterministic():
    sc = race.nominal()
    a = search.seeded((sc, sc.mars_lead, 0.0))
    b = search.seeded((sc, sc.mars_lead, 0.0))
    assert a is not None
    assert np.array_equal(a, b)
    ra = race.race_lowthrust(sc, a, 0.0, "02")
    rb = race.race_lowthrust(sc, b, 0.0, "02")
    assert ra.t_arrive == rb.t_arrive and ra.miss_r == rb.miss_r
