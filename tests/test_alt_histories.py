"""Apollo 13 alternate histories (2026-10-wk2 Sim post).

Each test would fail for a wrong model: a turn that grows with
altitude, a burn counted as if it all went along the velocity, a
flipped LVLH axis, an entry speed that depends on the path, or an
integrator that drifts. Data: orbital_sim/apollo13_data.py.
"""
import numpy as np
import pytest

from orbital_sim import apollo13_data as D
from orbital_sim.apollo13 import (fit_energy, get_seconds,
                                  sample_pass, speed_misfit,
                                  time_at_radius, to_radius_speed)
from orbital_sim.constants import (D_MOON, FT, H_ENTRY, MU_EARTH,
                                   MU_MOON, NMI, R_EARTH_EQ, R_MOON)
from orbital_sim.dynamics import no_thrust, specific_energy
from orbital_sim.flyby import hyperbola_state, turn_angle
from orbital_sim.propagate import fly
from orbital_sim.visviva import escape_speed, hyperbola_speed
from orbital_sim.whatif import (aim_offset, along_velocity,
                                apply_burn, burn_v_inf,
                                drift_without_gravity,
                                entry_speed_band, flight_path_angle,
                                fly_aimed_pass, fly_with_and_without,
                                what_if_pass)

R_P = R_MOON + D.PERI_ALT_NMI * NMI          # table 4-III
R_EI = R_EARTH_EQ + H_ENTRY
ALTS = [0.0, 50e3, 150e3, D.PERI_ALT_NMI * NMI, 500e3, 1000e3,
        3000e3]


def _calls(table):
    alt = np.array([row[1] for row in table], float)
    fps = np.array([row[2] for row in table], float)
    return to_radius_speed(alt, fps, R_MOON)


@pytest.fixture(scope="module")
def v_inf():
    r, v = _calls(D.MOON_PASS)
    return np.sqrt(2 * fit_energy(r, v, MU_MOON))


def _heading(vel):
    return np.degrees(np.arctan2(vel[1], vel[0]))


# ---- 1. a different Moon pass ---------------------------------

def test_aim_offset_is_angular_momentum(v_inf):
    """b v_inf = r_p v_p; b^2 = r_p^2 + 2 mu r_p / v_inf^2;
    gravity focuses (b > r_p), and with no gravity b = r_p."""
    for r_p in (R_MOON, R_P, 5 * R_MOON):
        b = aim_offset(v_inf, r_p, MU_MOON)
        v_p = hyperbola_speed(r_p, v_inf, MU_MOON)
        assert b * v_inf == pytest.approx(r_p * v_p, rel=1e-14)
        assert b ** 2 == pytest.approx(
            r_p ** 2 + 2 * MU_MOON * r_p / v_inf ** 2, rel=1e-12)
        assert b > r_p
        assert aim_offset(v_inf, r_p, 0.0) == r_p


def test_turn_shrinks_with_altitude(v_inf):
    """Closer pass, bigger turn (eq. 09). The real pass is the
    concept post's 87.66 deg; grazing gives the most, 92.21."""
    p = [what_if_pass(v_inf, h, MU_MOON, R_MOON) for h in ALTS]
    d = np.degrees([q["delta"] for q in p])
    v = np.array([q["v_p"] for q in p])
    assert np.all(np.diff(d) < 0) and np.all(np.diff(v) < 0)
    assert d[3] == pytest.approx(87.66, abs=0.005)
    assert d[0] == pytest.approx(92.21, abs=0.005)
    assert d[-1] == pytest.approx(58.17, abs=0.005)
    assert p[0]["r_p"] == R_MOON


def test_turn_sensitivity_matches_derivative(v_inf):
    """d delta / d r_p = -2 (v_inf^2/mu) / (e sqrt(e^2-1)):
    about -1.70 deg per 100 km at the real pass."""
    e = 1 + R_P * v_inf ** 2 / MU_MOON
    exact = -2 * v_inf ** 2 / MU_MOON / (e * np.sqrt(e * e - 1))
    h = 10.0
    num = (turn_angle(v_inf, R_P + h, MU_MOON)
           - turn_angle(v_inf, R_P - h, MU_MOON)) / (2 * h)
    assert num == pytest.approx(exact, rel=1e-6)
    assert np.degrees(exact) * 1e5 == pytest.approx(-1.70,
                                                    abs=0.005)


@pytest.mark.parametrize("alt", ALTS)
def test_integrated_pass_is_the_conic(v_inf, alt):
    """Each pass, flown with DOP853: closest approach = r_p to
    1 m, energy kept to 1e-10, positions on the closed-form
    hyperbola to 1 m, and it arrives along +x."""
    r_p = R_MOON + alt
    inb, outb = fly_aimed_pass(v_inf, r_p, MU_MOON, 15_000e3)
    t, Y = sample_pass(inb, outb, 300.0)
    r = np.hypot(Y[0], Y[1])
    assert r.min() == pytest.approx(r_p, abs=1.0)
    E = specific_energy(Y, MU_MOON)
    assert np.ptp(E) / abs(E.mean()) < 1e-10
    v_in = np.array([v_inf, 0.0])
    psi = turn_angle(v_inf, r_p, MU_MOON)
    peri = hyperbola_state(v_in, psi, r_p, MU_MOON, 0.0)[0]
    for k in range(0, t.size, 4):
        p = Y[0:2, k]
        f = np.arctan2(peri[0] * p[1] - peri[1] * p[0], peri @ p)
        pos, _ = hyperbola_state(v_in, psi, r_p, MU_MOON, f)
        assert np.allclose(pos, p, atol=1.0)
    h = Y[0] * Y[3] - Y[1] * Y[2]
    assert np.all(h > 0)                  # counter-clockwise


def test_far_out_headings_are_0_and_delta(v_inf):
    """Flown far out (1e11 m), every pass arrives along +x and
    leaves at heading delta: the overlay frame is honest."""
    for alt in (0.0, D.PERI_ALT_NMI * NMI, 3000e3):
        r_p = R_MOON + alt
        inb, outb = fly_aimed_pass(v_inf, r_p, MU_MOON, 1e11)
        d = np.degrees(turn_angle(v_inf, r_p, MU_MOON))
        assert _heading(inb.y[2:4, -1]) == pytest.approx(0, abs=0.01)
        assert _heading(outb.y[2:4, -1]) == pytest.approx(d, abs=0.01)


def _periapsis():
    def ev(t, y, *a):
        return y[0] * y[2] + y[1] * y[3]      # r . v
    ev.terminal, ev.direction = True, 1
    return ev


@pytest.mark.parametrize("scale, hits", [(0.99, True),
                                          (1.01, False)])
def test_aim_inside_grazing_offset_hits_moon(v_inf, scale, hits):
    """Independent of eq. 09: start 2e8 m out on a straight
    aim b = scale x b_graze; the integrator decides."""
    b = scale * aim_offset(v_inf, R_MOON, MU_MOON)
    x0 = 2e8
    r0 = np.hypot(x0, b)
    v0 = hyperbola_speed(r0, v_inf, MU_MOON)
    # set the angular momentum to b v_inf exactly
    y_off = b * v_inf / v0
    y0 = np.array([-x0, -y_off, v0, 0.0, 0.0])
    sol = fly(y0, (0, 1e7), no_thrust, MU_MOON, [_periapsis()])
    r_min = np.hypot(*sol.y[0:2, -1])
    assert (r_min < R_MOON) == hits


# ---- 2. no PC+2 burn ------------------------------------------

def test_burn_identities():
    """No burn: same v_inf. All along v: v' = v + dv. Square
    to v: speeds add in quadrature. Then vis-viva."""
    r, v, mu = 1.2e7, 1400.0, MU_MOON
    v_inf0 = np.sqrt(v * v - 2 * mu / r)
    g = np.radians(70.0)
    assert burn_v_inf(r, v, g, np.zeros(3), mu) == \
        pytest.approx(v_inf0, rel=1e-14)
    dv = 260.0
    along = np.array([dv * np.cos(g), 0, -dv * np.sin(g)])
    assert along_velocity(g, along) == pytest.approx(dv)
    assert burn_v_inf(r, v, g, along, mu) == pytest.approx(
        np.sqrt((v + dv) ** 2 - 2 * mu / r), rel=1e-12)
    side = np.array([0.0, dv, 0.0])
    assert burn_v_inf(r, v, g, side, mu) == pytest.approx(
        np.sqrt(v * v + dv * dv - 2 * mu / r), rel=1e-12)


def test_apply_burn_matches_burn_v_inf():
    """The planar impulse on a state gives the energy that
    burn_v_inf predicts (in-plane burn, outbound state)."""
    r_p, v_inf = R_P, 1046.0
    inb, outb = fly_aimed_pass(v_inf, r_p, MU_MOON, 15_000e3)
    y = outb.sol(7000.0)
    r, v = np.hypot(*y[0:2]), np.hypot(*y[2:4])
    v_p = hyperbola_speed(r_p, v_inf, MU_MOON)
    g = flight_path_angle(r, v, r_p, v_p)
    assert np.sin(g) == pytest.approx(
        (y[0:2] @ y[2:4]) / (r * v), rel=1e-9)
    dv = np.array([250.0, 0.0, -65.0])
    y2 = apply_burn(y, dv)
    E = specific_energy(y2, MU_MOON)
    assert np.sqrt(2 * E) == pytest.approx(
        burn_v_inf(r, v, g, dv, MU_MOON), rel=1e-12)


@pytest.fixture(scope="module")
def pc2(v_inf):
    """The real pass to the PC+2 midpoint (impulse there)."""
    r, _ = _calls(D.MOON_PASS)
    inb, outb = fly_aimed_pass(v_inf, R_P, MU_MOON, 15_000e3)
    legs = [row[3] for row in D.MOON_PASS]
    t = np.array([time_at_radius(inb if g == "in" else outb, ri)
                  for g, ri in zip(legs, r)])
    get = np.array([get_seconds(row[0]) for row in D.MOON_PASS])
    t_peri = np.mean(get - t)
    t_mid = get_seconds(D.PC2_GET) + D.PC2_BURN_S / 2
    y = outb.sol(t_mid - t_peri)
    rb, vb = np.hypot(*y[0:2]), np.hypot(*y[2:4])
    g = flight_path_angle(rb, vb, R_P, np.hypot(*outb.y[2:4, 0]))
    pad = np.array(D.PC2_PAD_LVLH_FPS)
    dv = pad * FT * D.PC2_DV_FPS / np.linalg.norm(pad)
    return {"y": y, "r": rb, "v": vb, "g": g, "dv": dv,
            "t_mid": t_mid}


def test_post_pc2_calls_fit_one_faster_hyperbola(v_inf):
    """The six calls after the burn sit on ONE new hyperbola
    (0.03 %), with v_inf 1.242 km/s: 19 % above 1.046."""
    r, v = _calls(D.MOON_AFTER_PC2)
    eps = fit_energy(r, v, MU_MOON)
    assert np.max(np.abs(speed_misfit(r, v, eps, MU_MOON))) < 5e-4
    v_after = np.sqrt(2 * eps)
    assert v_after == pytest.approx(1241.8, abs=0.5)
    assert v_after / v_inf - 1 == pytest.approx(0.187, abs=0.001)
    # the old hyperbola misses them by > 4 %
    old = speed_misfit(r, v, v_inf ** 2 / 2, MU_MOON)
    assert np.min(np.abs(old)) > 0.04


def test_pad_burn_direction_predicts_new_v_inf(pc2, v_inf):
    """The PAD's LVLH burn on the fitted hyperbola predicts the
    calls' v_inf to 0.1 %. The burn is ~59 deg off the velocity.
    Wrong models fail: z flipped (up), or all 860.5 ft/s along
    the velocity."""
    r, v = _calls(D.MOON_AFTER_PC2)
    v_calls = np.sqrt(2 * fit_energy(r, v, MU_MOON))
    p = pc2
    pred = burn_v_inf(p["r"], p["v"], p["g"], p["dv"], MU_MOON)
    assert pred == pytest.approx(v_calls, rel=1e-3)
    ang = np.arccos(along_velocity(p["g"], p["dv"])
                    / np.linalg.norm(p["dv"]))
    assert np.degrees(ang) == pytest.approx(58.6, abs=0.05)
    flip = p["dv"] * np.array([1, 1, -1])
    bad = burn_v_inf(p["r"], p["v"], p["g"], flip, MU_MOON)
    assert abs(bad / v_calls - 1) > 0.05
    allv = burn_v_inf(p["r"], p["v"], 0.0,
                      np.array([np.linalg.norm(p["dv"]), 0, 0]),
                      MU_MOON)
    assert allv / v_calls - 1 > 0.05


def test_post_pc2_clock(pc2):
    """Independent of the speed fit: flown on from the burn,
    the path reaches each call's distance within 45 s of its
    GET (calls are spoken to the minute)."""
    w, wo = fly_with_and_without(pc2["y"], pc2["dv"], 3 * 3600.0,
                                 MU_MOON)
    r, _ = _calls(D.MOON_AFTER_PC2)
    t = np.array([time_at_radius(w, ri) for ri in r])
    get = np.array([get_seconds(row[0])
                    for row in D.MOON_AFTER_PC2])
    assert np.max(np.abs(get - pc2["t_mid"] - t)) < 45.0
    # without the burn the energy (v_inf) is unchanged
    E = specific_energy(wo.y, MU_MOON)
    E0 = specific_energy(pc2["y"], MU_MOON)
    assert np.ptp(E) / abs(E0) < 1e-10
    assert E.mean() == pytest.approx(E0, rel=1e-10)


def _inward(r_stop):
    def ev(t, y, *a):
        return np.hypot(y[0], y[1]) - r_stop
    ev.terminal, ev.direction = True, -1
    return ev


def test_entry_speed_depends_on_energy_not_path():
    """Falls from the Moon's distance at the same speed but in
    different directions reach 400,000 ft at the SAME speed
    (integrated), equal to vis-viva. Dropped from rest: the
    slow edge of the band."""
    speeds = []
    for v0, ang in [(0.0, 0.0), (200.0, 0.0), (200.0, 20.0),
                    (200.0, 40.0)]:
        a = np.radians(180.0 - ang)          # inward, tilted
        y0 = np.array([D_MOON, 0.0, v0 * np.cos(a),
                       v0 * np.sin(a), 0.0])
        sol = fly(y0, (0, 30 * 86400.0), no_thrust, MU_EARTH,
                  [_inward(R_EI)])
        assert sol.status == 1
        speeds.append(np.hypot(*sol.y[2:4, -1]))
    slow, fast = entry_speed_band(D_MOON, R_EI, MU_EARTH)
    assert speeds[0] == pytest.approx(slow, rel=1e-9)
    want = np.sqrt(200.0 ** 2 + slow ** 2)   # same energy
    assert speeds[1:] == pytest.approx([want] * 3, rel=1e-9)
    assert fast == escape_speed(R_EI, MU_EARTH)


def test_entry_band_contains_nasa():
    """Any bound fall from 384,400 km: 10.98-11.075 km/s, a
    0.85 % band. NASA's 36,210.9 ft/s sits inside."""
    slow, fast = entry_speed_band(D_MOON, R_EI, MU_EARTH)
    assert slow / 1e3 == pytest.approx(10.980, abs=5e-4)
    assert 1 - slow / fast == pytest.approx(0.0085, abs=5e-5)
    assert slow < D.ENTRY_SPEED_FPS * FT < fast
    # farther far point -> narrower band
    assert entry_speed_band(2 * D_MOON, R_EI, MU_EARTH)[0] > slow


def test_sourced_hours_are_arithmetic():
    """The ~9 h is the Mission Report's (152 h vs 142:53:00),
    not a model output; the arithmetic matches its text."""
    saved = D.NO_PC2_LANDING_H - get_seconds(
        D.PC2_PRED_LANDING_GET) / 3600
    assert saved == pytest.approx(D.MR_HOURS_SAVED, abs=0.25)


# ---- 3. no free-return burn (toy only) ------------------------

def test_toy_drift_sizes():
    """37.8 ft/s = 11.52 m/s is 1.2 % of the 940 m/s Earth-
    relative speed; a straight-line drift (TOY) over the 15.85 h
    to the Moon is 658 km, over the 81.2 h to entry 3,367 km."""
    dv = D.FR_BURN_DV_FPS * FT
    assert dv == pytest.approx(11.521, abs=5e-4)
    v = D.EARTH_AROUND_FR_BURN[1][2] * FT
    assert dv / v == pytest.approx(0.01226, abs=5e-5)
    t0 = get_seconds(D.FR_BURN_GET)
    t_p = get_seconds(D.PERI_GET) - t0
    t_e = get_seconds(D.ENTRY_GET) - t0
    assert drift_without_gravity(dv, t_p) / 1e3 == \
        pytest.approx(657.6, abs=0.1)
    assert drift_without_gravity(dv, t_e) / 1e3 == \
        pytest.approx(3367.3, abs=0.1)
    assert drift_without_gravity(dv, 2 * t_p) == \
        2 * drift_without_gravity(dv, t_p)


def test_runs_are_deterministic(v_inf):
    a = fly_aimed_pass(v_inf, R_P, MU_MOON, 15_000e3)[1]
    b = fly_aimed_pass(v_inf, R_P, MU_MOON, 15_000e3)[1]
    assert np.array_equal(a.y, b.y)
