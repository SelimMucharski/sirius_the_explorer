import math
import random

import pytest

from sirius_the_explorer.probe_supervisor.probe_supervisor.domain.orientation import Orientation, sample_orientation


def _quat_from_rpy(r, p, y):
    cr, sr, cp, sp = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def test_unit_quaternion():
    rng = random.Random(0)
    for _ in range(100):
        q = sample_orientation(rng)
        assert math.isclose(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2, 1.0, abs_tol=1e-9)


def test_axis_tilt_is_uniform():
    # For a uniform rotation the rotated z axis' z-component is uniform on [-1, 1]
    rng = random.Random(2)
    zs = []
    for _ in range(5000):
        q = sample_orientation(rng)
        zs.append(1 - 2 * (q.x ** 2 + q.y ** 2))
    assert abs(sum(zs) / len(zs)) < 0.05
    assert sum(z < 0 for z in zs) / len(zs) == pytest.approx(0.5, abs=0.05)


def test_rpy_round_trip():
    rng = random.Random(3)
    for _ in range(200):
        q = sample_orientation(rng)
        q2 = _quat_from_rpy(*q.to_rpy())
        dot = abs(q.x * q2[0] + q.y * q2[1] + q.z * q2[2] + q.w * q2[3])
        assert dot == pytest.approx(1.0, abs=1e-9)  # same rotation (q and -q are equal)


def test_identity_rpy():
    assert Orientation(0, 0, 0, 1).to_rpy() == (0.0, 0.0, 0.0)
