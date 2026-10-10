"""Apollo 13 (2026-10-wk2): vis-viva against the real tracking calls.

Data: NASA Apollo Flight Journal air-to-ground calls and the Apollo
13 Mission Report (MSC-02680), see orbital_sim/apollo13_data.py.
Each test would fail for a wrong model: centre distance instead of
altitude, a wrong mu, a speed-up "slingshot", a broken integrator.
"""
import numpy as np
import pytest

from orbital_sim import apollo13_data as D
from orbital_sim.apollo13 import (entry_speed, fit_energy, fly_pass,
                                  get_seconds, sample_pass,
                                  speed_misfit, time_at_radius,
                                  to_radius_speed)
from orbital_sim.constants import (FT, H_ENTRY, MU_EARTH, MU_MOON,
                                   NMI, R_EARTH_EQ, R_MOON)
from orbital_sim.dynamics import specific_energy
from orbital_sim.flyby import hyperbola_state, turn_angle
from orbital_sim.visviva import escape_speed, hyperbola_speed

R_P = R_MOON + D.PERI_ALT_NMI * NMI          # table 4-III


def _calls(table, radius):
    alt = np.array([row[1] for row in table], float)
    fps = np.array([row[2] for row in table], float)
    return to_radius_speed(alt, fps, radius)


@pytest.fixture(scope="module")
def moon():
    r, v = _calls(D.MOON_PASS, R_MOON)
    eps = fit_energy(r, v, MU_MOON)
    return r, v, eps, np.sqrt(2 * eps)


@pytest.fixture(scope="module")
def moon_pass(moon):
    _, _, _, v_inf = moon
    return fly_pass(v_inf, R_P, 3 * 3600.0, 3 * 3600.0, MU_MOON)


def test_units_exact():
    assert FT == 0.3048 and NMI == 1852.0
    assert H_ENTRY == pytest.approx(121_920.0, abs=1e-9)
    assert get_seconds("077:20:57") == 77 * 3600 + 20 * 60 + 57
    assert get_seconds("079:27:38.95") == pytest.approx(
        286058.95, abs=1e-9)


def test_moon_calls_are_altitudes_on_one_hyperbola(moon):
    """One v_inf fits all 20 calls to 0.21 % in speed when the
    distance is an altitude. Read as centre distance, the same
    calls give no single conic (misfit > 10 %) and five of
    them would even be bound to the Moon (v_inf^2 < 0)."""
    r, v, eps, v_inf = moon
    assert len(r) == 20
    assert np.max(np.abs(speed_misfit(r, v, eps, MU_MOON))) < 0.0025
    assert v_inf == pytest.approx(1046.0, abs=0.5)
    r_c, _ = _calls(D.MOON_PASS, 0.0)
    eps_c = fit_energy(r_c, v, MU_MOON)
    miss_c = speed_misfit(r_c, v, eps_c, MU_MOON)
    assert np.max(np.abs(miss_c)) > 0.10
    assert np.sum(v ** 2 - 2 * MU_MOON / r_c < 0) == 5


def test_moon_fit_inbound_equals_outbound(moon):
    """Equation 08: v_inf in = v_inf out. Fit each leg alone."""
    r, v, _, _ = moon
    out = np.array([row[3] == "out" for row in D.MOON_PASS])
    v_in = np.sqrt(2 * fit_energy(r[~out], v[~out], MU_MOON))
    v_out = np.sqrt(2 * fit_energy(r[out], v[out], MU_MOON))
    assert v_out == pytest.approx(v_in, rel=0.005)


def test_pericynthion_speed_matches_mission_report(moon):
    """Vis-viva at the report's 136.2 nmi vs its 8,053.4 ft/s."""
    _, _, _, v_inf = moon
    v_p = hyperbola_speed(R_P, v_inf, MU_MOON)
    assert v_p / FT == pytest.approx(D.PERI_SPEED_FPS, rel=5e-4)
    assert v_p > escape_speed(R_P, MU_MOON)     # a hyperbola
    v_afj = hyperbola_speed(R_MOON + D.PERI_ALT_AFJ_KM * 1e3,
                            v_inf, MU_MOON)
    assert v_afj / FT == pytest.approx(D.PERI_SPEED_FPS, rel=1e-3)


def test_independent_pc2_state_from_mission_report(moon):
    """Table 4-II's tracking state at PC+2 ignition (not used in
    the fit) lies on the fitted hyperbola to 0.05 %."""
    _, _, _, v_inf = moon
    r = R_MOON + D.PC2_ALT_NMI * NMI
    v = hyperbola_speed(r, v_inf, MU_MOON)
    assert v / FT == pytest.approx(D.PC2_SPEED_FPS, rel=5e-4)


def test_turn_angle_eq09_and_integrated(moon):
    """delta = 2 arcsin(1/e) ~ 87.7 deg, and the integrator's
    velocity, flown far out (2e9 m) both ways, turns by it."""
    _, _, _, v_inf = moon
    e = 1 + R_P * v_inf ** 2 / MU_MOON
    delta = turn_angle(v_inf, R_P, MU_MOON)
    assert e == pytest.approx(1.444, abs=5e-4)
    assert np.degrees(delta) == pytest.approx(87.66, abs=0.01)
    span = 30 * 86400.0
    inb, outb = fly_pass(v_inf, R_P, span, span, MU_MOON)
    a, b = inb.y[2:4, -1], outb.y[2:4, -1]
    turn = np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)
    assert np.degrees(turn) == pytest.approx(
        np.degrees(delta), abs=0.1)
    assert np.hypot(*b) == pytest.approx(np.hypot(*a), rel=1e-10)


def test_integrated_pass_conserves_energy_and_hits_r_p(moon,
                                                       moon_pass):
    _, _, eps, v_inf = moon
    t, Y = sample_pass(*moon_pass, 20.0)
    E = specific_energy(Y, MU_MOON)
    assert np.ptp(E) / abs(E.mean()) < 1e-10
    assert E.mean() == pytest.approx(eps, rel=1e-9)
    r = np.hypot(Y[0], Y[1])
    assert r.min() == pytest.approx(R_P, rel=1e-12)
    assert t[np.argmin(r)] == 0.0
    assert np.all(np.diff(t) == 20.0)


def test_integrator_matches_closed_form_conic(moon, moon_pass):
    """Integrated positions = week 1's closed-form hyperbola
    (flyby.hyperbola_state) at the same true anomaly, to 1 m."""
    _, _, _, v_inf = moon
    e = 1 + R_P * v_inf ** 2 / MU_MOON
    v_in = np.array([0.0, v_inf])
    psi = turn_angle(v_inf, R_P, MU_MOON)
    t, Y = sample_pass(*moon_pass, 600.0)
    for k in range(0, t.size, 3):
        f = np.arctan2(Y[1, k], Y[0, k])
        pos, vel = hyperbola_state(v_in, psi, R_P, MU_MOON, f)
        # rotate the closed form so its periapsis lies on +x
        ang = np.arctan2(*hyperbola_state(
            v_in, psi, R_P, MU_MOON, 0.0)[0][::-1])
        c, s = np.cos(-ang), np.sin(-ang)
        p = np.array([c * pos[0] - s * pos[1],
                      s * pos[0] + c * pos[1]])
        assert np.allclose(p, Y[0:2, k], atol=1.0)
        assert np.hypot(*vel) == pytest.approx(
            np.hypot(Y[2, k], Y[3, k]), rel=1e-10)
    assert e > 1


def test_call_times_fit_the_hyperbola(moon, moon_pass):
    """Clock check, independent of the speed fit: the calls'
    GETs (spoken to the minute) agree with the integrated time
    of flight to < 40 s, and the fitted pericynthion GET is
    within 10 s of the report's 77:20:57."""
    r, _, _, _ = moon
    inb, outb = moon_pass
    t = np.array([time_at_radius(inb if row[3] == "in" else outb,
                                 ri)
                  for row, ri in zip(D.MOON_PASS, r)])
    get = np.array([get_seconds(row[0]) for row in D.MOON_PASS])
    t_peri = np.mean(get - t)
    assert np.max(np.abs(get - t - t_peri)) < 40.0
    assert t_peri == pytest.approx(get_seconds(D.PERI_GET), abs=10)


def test_entry_speed_matches_mission_report():
    """Return conic from 18 Earth calls -> entry interface speed
    at 400,000 ft within 0.1 % of 36,210.9 ft/s, just below
    escape speed there."""
    r, v = _calls(D.EARTH_RETURN, R_EARTH_EQ)
    eps = fit_energy(r, v, MU_EARTH)
    assert np.max(np.abs(speed_misfit(r, v, eps, MU_EARTH))) < 0.003
    assert eps < 0                                 # bound to Earth
    r_ei = R_EARTH_EQ + H_ENTRY
    v_ei = entry_speed(eps, r_ei, MU_EARTH)
    assert v_ei / FT == pytest.approx(D.ENTRY_SPEED_FPS, rel=1e-3)
    v_esc = escape_speed(r_ei, MU_EARTH)
    assert 0.995 < v_ei / v_esc < 1.0
    assert 0.995 < D.ENTRY_SPEED_FPS * FT / v_esc < 1.0
    r_c, _ = _calls(D.EARTH_RETURN, 0.0)
    eps_c = fit_energy(r_c, v, MU_EARTH)
    assert np.max(np.abs(speed_misfit(r_c, v, eps_c,
                                      MU_EARTH))) > 0.10


def test_fits_are_deterministic(moon):
    r, v, eps, _ = moon
    assert fit_energy(r, v, MU_MOON) == eps


# ---- Earth frame: does the Moon pass itself add speed? ----

from orbital_sim.apollo13 import (burn_angle, earth_energy_change,
                                  trailing_offset)
from orbital_sim.flyby import rotate


@pytest.mark.parametrize("V_ang, vin_ang, sign", [
    (90, 200, 1), (90, 200, -1), (-30, 100, 1), (10, 250, -1)])
def test_earth_energy_change_equals_vector_sum(V_ang, vin_ang,
                                               sign):
    """Closed form V . dv_inf == (1/2)|V+v_out|^2 - (1/2)|V+v_in|^2
    for any geometry, with x read off week 1's hyperbola."""
    V = 967.0 * np.array([np.cos(np.radians(V_ang)),
                          np.sin(np.radians(V_ang))])
    v_inf = 1046.0
    v_in = v_inf * np.array([np.cos(np.radians(vin_ang)),
                             np.sin(np.radians(vin_ang))])
    e = 1 + R_P * v_inf ** 2 / MU_MOON
    psi = sign * turn_angle(v_inf, R_P, MU_MOON)
    v_out = rotate(v_in, psi)
    want = 0.5 * (V + v_out) @ (V + v_out) - 0.5 * (V + v_in) @ (V + v_in)
    p_hat = hyperbola_state(v_in, psi, R_P, MU_MOON, 0.0)[0] / R_P
    x = np.arcsin(-(V / np.hypot(*V)) @ p_hat)
    got = earth_energy_change(np.hypot(*V), v_inf, e, x)
    assert got == pytest.approx(want, rel=1e-9, abs=1e-6)


def test_trailing_offset_conventions():
    """Trailing apex 90 E: a pericynthion there is fully behind
    the Moon (+90 deg), at 90 W fully ahead (-90 deg)."""
    assert np.degrees(trailing_offset(90.0, 0.0)) == \
        pytest.approx(90.0)
    assert np.degrees(trailing_offset(-90.0, 0.0)) == \
        pytest.approx(-90.0)
    assert np.degrees(trailing_offset(180.0, 0.0)) == \
        pytest.approx(0.0, abs=1e-12)
    x = trailing_offset(D.PERI_LON_EAST_DEG,
                        D.SUB_EARTH_LON_EAST_DEG)
    assert np.degrees(x) == pytest.approx(0.23, abs=1e-9)


def test_burn_angle_law_of_cosines():
    v, dv = np.array([1385.0, 0.0]), 262.0
    for a in (0.0, 0.4, 1.02, 2.5):
        after = v + dv * np.array([np.cos(a), np.sin(a)])
        assert burn_angle(1385.0, dv, np.hypot(*after)) == \
            pytest.approx(a, abs=1e-6)


@pytest.fixture(scope="module")
def earth_frame():
    import importlib.util
    from pathlib import Path
    f = (Path(__file__).resolve().parents[1] / "examples"
         / "2026-10-wk2_free_return.py")
    spec = importlib.util.spec_from_file_location("w2", f)
    w2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w2)
    moon, state = w2.moon_leg()
    earth, _ = w2.earth_leg()
    return w2.earth_frame(state, earth["eps_J_kg"])


def test_moon_pass_adds_almost_no_earth_frame_energy(earth_frame):
    """Pericynthion lay within 0.5 deg of the anti-Earth point,
    so the pass changed the Earth-frame energy by <= 0.03 km2/s2
    even with 1 deg of frame error: < 15 % of the 0.217 the
    calls show between 59:51 and entry, out of a possible 1.40."""
    f = earth_frame
    assert abs(f["d_eps_pass_km2s2"]) + \
        f["d_eps_pass_per_deg_km2s2"] < 0.035
    assert f["d_eps_calls_km2s2"] > 0.2
    assert f["d_eps_max_km2s2"] > 1.3


def test_patched_geometry_matches_earth_calls(earth_frame):
    """Inbound: patched Earth-frame energy at the Moon vs the
    farthest-out pre-pass call (0.02). Outbound: pass + PC+2
    (direction from the post-burn Moon calls) vs the entry
    fit (0.01). So the PC+2 burn carries the gain."""
    f = earth_frame
    assert f["eps_pre_patched_km2s2"] == pytest.approx(
        f["eps_call_59_51_km2s2"], abs=0.02)
    assert f["eps_post_patched_km2s2"] == pytest.approx(
        f["eps_entry_fit_km2s2"], abs=0.01)
    assert f["d_eps_burn_km2s2"] == pytest.approx(
        f["d_eps_calls_km2s2"], abs=0.01)
