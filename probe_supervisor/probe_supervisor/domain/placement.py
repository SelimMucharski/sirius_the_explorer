"""Random probe placement with spacing constraints. Pure Python, no ROS dependencies."""

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from sirius_the_explorer.probe_supervisor.probe_supervisor.domain.spawn_area import SpawnArea

Point = Tuple[float, float]


@dataclass(frozen=True)
class KeepOutCircle:
    """Region where probes must not appear (e.g. around the robot's spawn point)."""

    x: float
    y: float
    radius: float

    def contains(self, x: float, y: float) -> bool:
        return math.hypot(x - self.x, y - self.y) < self.radius


@dataclass(frozen=True)
class PlacementConstraints:
    min_separation: float = 0.5
    keep_out: Sequence[KeepOutCircle] = field(default_factory=tuple)
    max_attempts_per_probe: int = 200


def sample_positions(
    area: SpawnArea,
    count: int,
    existing: Sequence[Point],
    constraints: PlacementConstraints,
    rng: random.Random,
) -> List[Point]:
    """Rejection-sample up to `count` points inside `area`.

    Each point is at least `min_separation` from every `existing` point and
    every other new point, and outside all keep-out circles. Returns fewer
    than `count` points if the area is too crowded to fit them.
    """
    placed: List[Point] = []
    occupied: List[Point] = list(existing)

    for _ in range(count):
        for _ in range(constraints.max_attempts_per_probe):
            x, y = area.sample(rng)
            if not area.contains(x, y):
                continue
            if any(k.contains(x, y) for k in constraints.keep_out):
                continue
            if any(math.hypot(x - ox, y - oy) < constraints.min_separation
                   for ox, oy in occupied):
                continue
            placed.append((x, y))
            occupied.append((x, y))
            break
        else:
            break  # area saturated; further probes would fail as well

    return placed
