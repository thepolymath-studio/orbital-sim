"""Kessler's N^2 counting and the closing speed of crossings."""
import numpy as np
import pytest

from orbital_sim.kessler import (MPH, closing_speed,
                                 crossing_angle, pair_count,
                                 pair_growth)


def test_pairs_brute_force():
    for n in (2, 3, 7, 40):
        brute = sum(1 for i in range(n) for j in range(i + 1, n))
        assert pair_count(n) == brute


def test_doubling_quadruples_pairs():
    # exactly 2 (2N - 1) / (N - 1), which tends to 4
    for n in (10, 1000, 10**6):
        assert pair_growth(n) == pytest.approx(
            2 * (2 * n - 1) / (n - 1))
    assert pair_growth(10**6) == pytest.approx(4, rel=1e-5)
    assert pair_growth(1000, factor=3) == pytest.approx(9, rel=3e-3)


def test_closing_speed_limits():
    v = 7.558
    assert closing_speed(v, 0.0) == 0.0
    assert closing_speed(v, np.pi) == pytest.approx(2 * v)
    assert closing_speed(v, np.pi / 2) == pytest.approx(np.sqrt(2) * v)
    # equal to the vector difference for any angle
    for th in np.linspace(0, np.pi, 7):
        a = v * np.array([1.0, 0.0])
        b = v * np.array([np.cos(th), np.sin(th)])
        assert closing_speed(v, th) == pytest.approx(
            np.linalg.norm(a - b), abs=1e-12)
        assert crossing_angle(closing_speed(v, th), v) == \
            pytest.approx(th, abs=1e-7)


def test_mph():
    assert 20_000 * MPH == pytest.approx(8940.8)
