"""Where probes are allowed to spawn. Pure Python, no ROS dependencies."""

import math
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import yaml
from PIL import Image


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


class TraversabilityMask:
    """
    Binary traversability map in the nav2_map_server format (as kalman_nav2/maps).

    Image (white = traversable with negate 0) + YAML {image, resolution, origin, negate,
    free_thresh}. Image row 0 is the top edge (largest y). Outside the image: not traversable.
    """

    def __init__(self, free, resolution: float, origin_x: float, origin_y: float):
        self.free = free                        # free[row][col], row 0 = top
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.height = len(free)
        self.width = len(free[0]) if free else 0

    @staticmethod
    def load(yaml_path: str) -> 'TraversabilityMask':
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        image = meta['image']
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(yaml_path), image)
        img = Image.open(image).convert('L')
        w, h = img.size
        px = list(img.tobytes())
        negate = bool(meta.get('negate', 0))
        free_thresh = float(meta.get('free_thresh', 0.25))

        def is_free(v):
            occ = v / 255.0 if negate else 1.0 - v / 255.0
            return occ < free_thresh
        free = [[is_free(px[r * w + c]) for c in range(w)] for r in range(h)]
        return TraversabilityMask(free, float(meta['resolution']),
                                  float(meta['origin'][0]), float(meta['origin'][1]))

    def is_free(self, x: float, y: float) -> bool:
        col = math.floor((x - self.origin_x) / self.resolution)
        row = self.height - 1 - math.floor((y - self.origin_y) / self.resolution)
        return 0 <= row < self.height and 0 <= col < self.width and self.free[row][col]

    def is_free_around(self, x: float, y: float, clearance: float, n: int = 16) -> bool:
        """(x, y) and points on circles of radius clearance and clearance/2 are traversable."""
        if not self.is_free(x, y):
            return False
        for r in (clearance, clearance / 2):
            if r <= 0:
                continue
            for k in range(n):
                a = 2 * math.pi * k / n
                if not self.is_free(x + r * math.cos(a), y + r * math.sin(a)):
                    return False
        return True


@dataclass(frozen=True)
class MaskedArea(SpawnArea):
    """
    A rectangle restricted to a traversability map, at least `clearance` from its edge.

    Probes must lie where the rover can drive up to them (ERC Probing: samples are placed on
    accessible terrain), so the search prior and the spawn distribution match.
    """

    rectangle: RectangleArea
    mask: TraversabilityMask
    clearance: float = 0.5
    # optional map of spots where a probe stays visible (no clearance: checked at the point)
    spawn_mask: Optional[TraversabilityMask] = None

    def contains(self, x: float, y: float) -> bool:
        return (self.rectangle.contains(x, y)
                and self.mask.is_free_around(x, y, self.clearance)
                and (self.spawn_mask is None or self.spawn_mask.is_free(x, y)))

    def sample(self, rng: random.Random) -> Tuple[float, float]:
        return self.rectangle.sample(rng)
