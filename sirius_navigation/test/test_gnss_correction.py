import math

import numpy as np
import pytest

from sirius_navigation.gnss_correction import Datum, OffsetFilter, OffsetNoise

ERC = Datum(50.06623240, 19.91320025, heading=math.radians(-19.198))


def test_datum_origin_and_axes():
    assert np.allclose(ERC.to_map(ERC.latitude, ERC.longitude), 0.0, atol=1e-6)
    # 1e-4 deg north: ~11.13 m north, the map x axis points 19.2 deg south of east
    north = Datum(ERC.latitude, ERC.longitude).to_map(ERC.latitude + 1e-4, ERC.longitude)
    assert north[0] == pytest.approx(0.0, abs=1e-3) and north[1] == pytest.approx(11.13, abs=0.02)
    p = ERC.to_map(ERC.latitude + 1e-4, ERC.longitude)
    assert np.linalg.norm(p) == pytest.approx(11.13, abs=0.02)
    assert math.degrees(math.atan2(p[1], p[0])) == pytest.approx(90 + 19.198, abs=0.01)


def test_filter_averages_noise_when_still():
    rng = np.random.default_rng(0)
    f = OffsetFilter()
    truth = np.array([0.7, -0.4])
    for i in range(400):                                  # 20 s at 20 Hz
        f.predict(i * 0.05, 0.0, 0.0)
        f.update(i * 0.05, truth + rng.normal(0, 0.85, 2), 0.85 ** 2)
    offset, var = f.state()
    assert np.linalg.norm(offset - truth) < 0.25
    assert var < 0.05


def test_filter_follows_slip_while_turning():
    rng = np.random.default_rng(1)
    f = OffsetFilter(OffsetNoise())
    t = 0.0
    for _ in range(400):                                  # converged while still
        t += 0.05
        f.predict(t, 0.0, 0.0)
        f.update(t, rng.normal(0, 0.85, 2), 0.85 ** 2)
    for i in range(240):                                  # turning, slides 0.8 m in 12 s
        t += 0.05
        f.predict(t, 0.0, 0.5)
        f.update(t, np.array([0.8 * i / 240, 0.0]) + rng.normal(0, 0.85, 2), 0.85 ** 2)
    for _ in range(100):                                  # 5 s still after the turn
        t += 0.05
        f.predict(t, 0.0, 0.0)
        f.update(t, np.array([0.8, 0.0]) + rng.normal(0, 0.85, 2), 0.85 ** 2)
    assert np.linalg.norm(f.state()[0] - [0.8, 0.0]) < 0.3


def test_outlier_rejected_then_sudden_slip_accepted():
    f = OffsetFilter(max_rejections=3)
    for i in range(100):
        f.predict(i * 0.05, 0.0, 0.0)
        f.update(i * 0.05, np.zeros(2), 0.85 ** 2)
    assert not f.update(5.0, np.array([20.0, 0.0]), 0.85 ** 2)     # one wild fix
    assert np.allclose(f.state()[0], 0.0, atol=1e-6)
    for i in range(4):                                              # but it persists: a slip
        accepted = f.update(5.0 + i * 0.05, np.array([5.0, 0.0]), 0.85 ** 2)
    assert accepted and f.state()[0][0] > 2.0
