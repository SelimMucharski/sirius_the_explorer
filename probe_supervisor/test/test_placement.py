import math
import random

import pytest

from probe_supervisor.domain.placement import (
    KeepOutCircle, PlacementConstraints, sample_positions)
from probe_supervisor.domain.spawn_area import RectangleArea


AREA = RectangleArea(-5.0, 5.0, -5.0, 5.0)


def _min_pairwise(points):
    return min(math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1:])


def test_points_inside_area_and_separated():
    pts = sample_positions(AREA, 20, [], PlacementConstraints(min_separation=0.5),
                           random.Random(0))
    assert len(pts) == 20
    assert all(AREA.contains(x, y) for x, y in pts)
    assert _min_pairwise(pts) >= 0.5


def test_keep_out_circle_respected():
    keep_out = KeepOutCircle(0.0, 0.0, 2.0)
    pts = sample_positions(AREA, 30, [], PlacementConstraints(keep_out=(keep_out,)),
                           random.Random(1))
    assert len(pts) == 30
    assert not any(keep_out.contains(x, y) for x, y in pts)


def test_separation_from_existing_probes():
    existing = [(0.0, 0.0)]
    pts = sample_positions(RectangleArea(-1, 1, -1, 1), 5, existing,
                           PlacementConstraints(min_separation=0.8), random.Random(2))
    assert all(math.dist(p, existing[0]) >= 0.8 for p in pts)


def test_returns_fewer_when_area_saturated():
    tiny = RectangleArea(0.0, 0.5, 0.0, 0.5)
    pts = sample_positions(tiny, 10, [], PlacementConstraints(min_separation=1.0),
                           random.Random(3))
    assert len(pts) == 1


def test_seed_is_reproducible():
    c = PlacementConstraints()
    assert (sample_positions(AREA, 5, [], c, random.Random(42))
            == sample_positions(AREA, 5, [], c, random.Random(42)))


def test_empty_rectangle_rejected():
    with pytest.raises(ValueError):
        RectangleArea(1.0, 0.0, 0.0, 1.0)
