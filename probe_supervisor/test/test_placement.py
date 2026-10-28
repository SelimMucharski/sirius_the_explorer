import math
import random

import pytest

from probe_supervisor.domain.placement import (
    KeepOutCircle, PlacementConstraints, sample_positions)
from probe_supervisor.domain.spawn_area import MaskedArea, RectangleArea, TraversabilityMask


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


def _mask(tmp_path, free_rows):
    """Traversability map YAML + PNG from rows of '#' (blocked) / '.' (free), row 0 = top."""
    from PIL import Image
    h, w = len(free_rows), len(free_rows[0])
    img = Image.new('L', (w, h))
    img.putdata([255 if ch == '.' else 0 for row in free_rows for ch in row])
    img.save(tmp_path / 'm.png')
    (tmp_path / 'm.yaml').write_text(
        'image: m.png\nresolution: 1.0\norigin: [0.0, 0.0, 0.0]\nnegate: 0\n'
        'occupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n')
    return TraversabilityMask.load(str(tmp_path / 'm.yaml'))


def test_mask_orientation_and_bounds(tmp_path):
    mask = _mask(tmp_path, ['..#',
                            '###'])
    assert mask.is_free(0.5, 1.5) and mask.is_free(1.5, 1.9)   # top row = larger y
    assert not mask.is_free(2.5, 1.5)
    assert not mask.is_free(0.5, 0.5)
    assert not mask.is_free(-0.5, 1.5) and not mask.is_free(0.5, 2.5)


def test_masked_area_keeps_clearance(tmp_path):
    rows = ['#' * 10] + ['#' + '.' * 8 + '#'] * 8 + ['#' * 10]
    mask = _mask(tmp_path, rows)
    area = MaskedArea(RectangleArea(0.0, 10.0, 0.0, 10.0), mask, clearance=1.5)
    pts = sample_positions(area, 20, [], PlacementConstraints(min_separation=0.0),
                           random.Random(3))
    assert len(pts) == 20
    assert all(2.5 <= x <= 7.5 and 2.5 <= y <= 7.5 for x, y in pts)


def test_spawn_mask_checked_at_point(tmp_path):
    rows = ['.' * 10] * 10
    mask = _mask(tmp_path, rows)
    sub = tmp_path / 'spawn'
    sub.mkdir()
    spawn = _mask(sub, ['#' * 10] * 5 + ['.' * 10] * 5)        # only y < 5 visible
    area = MaskedArea(RectangleArea(0.0, 10.0, 0.0, 10.0), mask, clearance=0.0,
                      spawn_mask=spawn)
    pts = sample_positions(area, 20, [], PlacementConstraints(min_separation=0.0),
                           random.Random(4))
    assert len(pts) == 20 and all(y < 5.0 for _, y in pts)
