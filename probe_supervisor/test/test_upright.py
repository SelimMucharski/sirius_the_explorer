import math
import random

import numpy as np

from probe_supervisor.domain.upright import (
    Footprints, HeightGrid, PROBE_LENGTH, sample_upright, UprightParams)


def ground(z=0.3):
    return HeightGrid(np.full((50, 50), z), 0.0, 0.0, 0.1)


def test_no_upright_draws_nothing_from_rng():
    rng = random.Random(3)
    assert sample_upright(rng, 2.0, 2.0, ground(), UprightParams(0.0)) is None
    assert rng.random() == random.Random(3).random()


def test_upright_buried_tip_down():
    p = UprightParams(1.0, 0.3, 0.3, 0.0)
    z, roll, pitch, _ = sample_upright(random.Random(1), 2.0, 2.0, ground(0.3), p)
    assert roll == 0.0 and pitch == 0.0
    tip = z - PROBE_LENGTH / 2
    assert math.isclose(tip, 0.3 - 0.3 * PROBE_LENGTH)


def test_upright_tilt_bounded_and_footprints_respected():
    p = UprightParams(1.0, 0.25, 0.45, 12.0)
    rng = random.Random(5)
    for _ in range(50):
        _, roll, pitch, _ = sample_upright(rng, 2.0, 2.0, ground(), p)
        assert math.hypot(roll, pitch) <= math.radians(12.0) + 1e-9
    foot = Footprints([(2.0, 2.0, 0.2)])
    assert sample_upright(random.Random(1), 2.1, 2.0, ground(), p, foot) is None
    assert sample_upright(random.Random(1), 3.0, 3.0, ground(), p, foot) is not None
    assert sample_upright(random.Random(1), 30.0, 3.0, ground(), p) is None   # outside the map
