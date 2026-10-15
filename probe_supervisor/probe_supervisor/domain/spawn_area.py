"""Where probes are allowed to spawn. Pure Python, no ROS dependencies."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


class SpawnArea(ABC):
    """Port: a 2D region (world frame) that probes may be placed in.

    Implement a new subclass to restrict spawning differently, e.g. an
    OccupancyGridArea built from a traversability map (free cells only).
    """

    @abstractmethod
    def contains(self, x: float, y: float) -> bool:
        """True if a probe may be placed at (x, y)."""

    @abstractmethod
    def sample(self, rng: random.Random) -> Tuple[float, float]:
        """Draw a candidate (x, y); it is checked with contains() afterwards."""


@dataclass(frozen=True)
class RectangleArea(SpawnArea):
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self):
        if self.min_x > self.max_x or self.min_y > self.max_y:
            raise ValueError(f'Empty rectangle: {self}')

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def sample(self, rng: random.Random) -> Tuple[float, float]:
        return rng.uniform(self.min_x, self.max_x), rng.uniform(self.min_y, self.max_y)
