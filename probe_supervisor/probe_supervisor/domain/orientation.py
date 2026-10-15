"""Random probe orientations. Pure Python, no ROS dependencies."""

import math
import random
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Orientation:
    """Unit quaternion (x, y, z, w)."""

    x: float
    y: float
    z: float
    w: float

    def to_rpy(self) -> Tuple[float, float, float]:
        """Roll, pitch, yaw (rad), fixed-axis XYZ as used by SDF / ros_gz_sim create."""
        x, y, z, w = self.x, self.y, self.z, self.w
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw


def sample_orientation(rng: random.Random) -> Orientation:
    """Uniformly random 3D rotation (Shoemake's method); the probe falls and settles on its own."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    a, b = math.sqrt(1 - u1), math.sqrt(u1)
    return Orientation(a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2),
                       b * math.sin(2 * math.pi * u3), b * math.cos(2 * math.pi * u3))
