"""Week 2026-10-wk1 scenarios: Hermes-like thrust from Earth to Mars."""
import numpy as np
import pytest

from orbital_sim import (A_HERMES, DAY, HOUR, MU_SUN, R_MARS_ORBIT,
                         sample, specific_energy)
from orbital_sim.dynamics import no_thrust, prograde
from orbital_sim.scenarios import BURN_COAST, THRUST_TO_MARS, to_mars


@pytest.fixture(scope="module")
def cont():
    return to_mars(THRUST_TO_MARS)


@pytest.fixture(scope="module")
def bc():
    return to_mars(BURN_COAST)


def burn_time(segs):
    return sum(s.t1 - s.t0 for s in segs if s.thrust_on)


def end_state(segs):
    return segs[-1].sol.y[:, -1]


def test_hermes_rate_is_7_2_m_s_per_hour():
    assert A_HERMES * HOUR == pytest.approx(7.2, rel=1e-12)


@pytest.mark.parametrize("name", ["cont", "bc"])
def test_run_stops_exactly_at_mars_radius(name, request):
    y = end_state(request.getfixturevalue(name))
    assert np.hypot(y[0], y[1]) == pytest.approx(R_MARS_ORBIT,
                                                 rel=1e-12)


@pytest.mark.parametrize("name", ["cont", "bc"])
def test_dv_equals_aT_times_burn_time(name, request):
    """Equation 07, summed over the burns (coasts add nothing)."""
    segs = request.getfixturevalue(name)
    dv = end_state(segs)[4]
    assert dv == pytest.approx(A_HERMES * burn_time(segs), rel=1e-10)


def test_burn_coast_schedule_is_respected(bc):
    assert [s.thrust_on for s in bc] == [True, False, True]
    assert bc[0].t1 == pytest.approx(40 * DAY)
    assert bc[1].t1 - bc[1].t0 == pytest.approx(40 * DAY)
    # dv does not grow during the coast
    y_a, y_b = bc[1].sol.y[4, 0], bc[1].sol.y[4, -1]
    assert y_b == pytest.approx(y_a, abs=1e-9)


def test_coast_arc_conserves_energy(bc):
    s = bc[1]
    E = specific_energy(s.sol.y)
    assert np.max(np.abs(E / E[0] - 1)) < 1e-11


def test_energy_gain_equals_thrust_work(cont):
    """Whole run: E_end - E_0 = integral of a_T |v| dt."""
    from scipy.integrate import quad
    s = cont[0]
    law = prograde(A_HERMES)

    def power(t):
        y = s.sol.sol(t)
        return np.dot(y[2:4], law(t, y[:2], y[2:4]))

    work, _ = quad(power, s.t0, s.t1, limit=400, epsabs=0,
                   epsrel=1e-11)
    gain = specific_energy(s.sol.y[:, -1]) - specific_energy(
        s.sol.y[:, 0])
    assert gain == pytest.approx(work, rel=1e-8)


def test_thrust_beats_hohmann_budget_but_costs_more_dv(cont):
    """Sanity scale (S10): the prograde run spends more dv than the
    Hohmann pair (5.59 km/s) - it arrives fast, not matched."""
    from orbital_sim import hohmann, R_EARTH_ORBIT
    dv1, dv2, _ = hohmann(R_EARTH_ORBIT, R_MARS_ORBIT)
    assert end_state(cont)[4] > dv1 + dv2


def test_sampling_is_uniform_and_flags_match(bc):
    dt = 0.25 * DAY
    t, Y, on = sample(bc, dt)
    assert np.allclose(np.diff(t)[:-1], dt)
    assert t[-1] == bc[-1].t1
    assert np.all(on[t < bc[0].t1])
    assert not np.any(on[(t >= bc[1].t0) & (t < bc[1].t1)])
    assert np.all(on[t >= bc[2].t0])
    r = np.hypot(Y[0], Y[1])
    assert r[-1] == pytest.approx(R_MARS_ORBIT, rel=1e-12)


def test_deterministic(cont):
    again = to_mars(THRUST_TO_MARS)
    assert np.array_equal(end_state(again), end_state(cont))
    assert again[-1].t1 == cont[-1].t1
