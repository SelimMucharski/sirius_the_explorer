"""Probes stuck upright into the ground (as at ERC). Pure Python + numpy, no ROS."""

import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import yaml

PROBE_LENGTH = 0.2          # [m] probe.sdf: z from the tip (-0.1) to the head face (+0.1)


@dataclass
class HeightGrid:
    """Visual-terrain height map (ig_search format: {image: .npy, origin: [x, y], resolution})."""

    h: np.ndarray               # [row = y, col = x]
    origin_x: float
    origin_y: float
    res: float

    @staticmethod
    def load(yaml_path: str) -> 'HeightGrid':
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        npy = meta['image']
        if not os.path.isabs(npy):
            npy = os.path.join(os.path.dirname(yaml_path), npy)
        return HeightGrid(np.load(npy), float(meta['origin'][0]), float(meta['origin'][1]),
                          float(meta['resolution']))

    def at(self, x: float, y: float) -> float:
        """Bilinear height; nan outside the map or next to missing data."""
        fx = (x - self.origin_x) / self.res - 0.5
        fy = (y - self.origin_y) / self.res - 0.5
        j0, i0 = int(math.floor(fx)), int(math.floor(fy))
        if not (0 <= i0 < self.h.shape[0] - 1 and 0 <= j0 < self.h.shape[1] - 1):
            return float('nan')
        tx, ty = fx - j0, fy - i0
        q = self.h[i0:i0 + 2, j0:j0 + 2]
        return float((q[0, 0] * (1 - tx) + q[0, 1] * tx) * (1 - ty)
                     + (q[1, 0] * (1 - tx) + q[1, 1] * tx) * ty)


@dataclass
class Footprints:
    """Discs (x, y, r) of stones and slabs: an upright probe is not stuck into them."""

    discs: List[Tuple[float, float, float]] = field(default_factory=list)

    @staticmethod
    def load(yaml_path: str) -> 'Footprints':
        with open(yaml_path) as f:
            meta = yaml.safe_load(f) or {}
        return Footprints([tuple(map(float, d)) for d in meta.get('discs', [])])

    def blocked(self, x: float, y: float, margin: float = 0.03) -> bool:
        return any(math.hypot(x - a, y - b) < r + margin for a, b, r in self.discs)


@dataclass(frozen=True)
class UprightParams:
    fraction: float = 0.0                   # share of probes stuck upright
    bury_min: float = 0.25                  # buried part of the length (tip down)
    bury_max: float = 0.45
    max_tilt_deg: float = 12.0              # from vertical, random direction


def sample_upright(rng: random.Random, x: float, y: float, ground: HeightGrid,
                   params: UprightParams,
                   footprints: Optional[Footprints] = None) -> Optional[Tuple[float, ...]]:
    """
    Upright pose (z, roll, pitch, yaw) of the probe model at (x, y), or None (lie it down).

    The model z axis goes tip -> head, so tip down = no rotation; the model origin is the
    middle of the probe. Draws from `rng` only when the probe is chosen to stand, so
    fraction = 0 keeps the random sequence (and the lying orientations) unchanged.
    """
    if params.fraction <= 0.0 or rng.random() >= params.fraction:
        return None
    zg = ground.at(x, y)
    if not math.isfinite(zg) or (footprints is not None and footprints.blocked(x, y)):
        return None
    bury = rng.uniform(params.bury_min, params.bury_max) * PROBE_LENGTH
    tilt = math.radians(params.max_tilt_deg) * math.sqrt(rng.random())
    az = rng.uniform(-math.pi, math.pi)
    roll, pitch = -tilt * math.sin(az), tilt * math.cos(az)
    z = zg - bury + 0.5 * PROBE_LENGTH * math.cos(tilt)
    return z, roll, pitch, rng.uniform(-math.pi, math.pi)
