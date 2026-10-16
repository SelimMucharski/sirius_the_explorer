"""GNSS correction of the odometry drift: map -> odom as a 2D translation. No ROS imports.

The odometry (wheel velocities + gyro) keeps a good heading - the gyro integrates the yaw
accurately - but the position drifts: a skid-steered rover slides when it turns, most on
slopes, and no odometry sees that. GNSS gives the antenna position in the map frame with
~1 m noise and no drift. The map -> odom transform is therefore estimated as a translation
only, d = p_map(antenna, GNSS) - p_odom(antenna, odometry), by a Kalman filter on d with a
random-walk model; its process noise grows while the rover turns or drives (when the slip
happens), so the correction follows the slip quickly and averages the GNSS noise when the
rover stands. The heading stays the odometry's (fusing it with GNSS in a full map EKF pulled
the yaw by up to ~10 deg, as the filter explained lateral GNSS noise by rotation).
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return np.array([(n + alt) * math.cos(lat) * math.cos(lon),
                     (n + alt) * math.cos(lat) * math.sin(lon),
                     (n * (1.0 - WGS84_E2) + alt) * math.sin(lat)])


@dataclass(frozen=True)
class Datum:
    """Map frame origin.

    Latitude, longitude [deg], altitude [m] and the ENU heading of the map x axis [rad]
    (Gazebo: the world's spherical_coordinates, heading_deg in radians).
    """

    latitude: float
    longitude: float
    altitude: float = 0.0
    heading: float = 0.0

    def to_map(self, lat_deg: float, lon_deg: float, alt: float = 0.0) -> np.ndarray:
        """Geodetic position -> (x, y) in the map frame (local tangent plane at the datum)."""
        lat0, lon0 = math.radians(self.latitude), math.radians(self.longitude)
        d = geodetic_to_ecef(lat_deg, lon_deg, alt) - geodetic_to_ecef(
            self.latitude, self.longitude, self.altitude)
        east = -math.sin(lon0) * d[0] + math.cos(lon0) * d[1]
        north = (-math.sin(lat0) * math.cos(lon0) * d[0]
                 - math.sin(lat0) * math.sin(lon0) * d[1] + math.cos(lat0) * d[2])
        c, s = math.cos(self.heading), math.sin(self.heading)
        return np.array([c * east + s * north, -s * east + c * north])


@dataclass(frozen=True)
class OffsetNoise:
    """Random-walk process noise of the map -> odom offset [m^2/s].

    A base rate + a rate per rad/s of turn rate + a rate per m/s of speed.
    """

    still: float = 1e-4
    per_turn_rate: float = 0.02
    per_speed: float = 2e-3

    def rate(self, speed: float, turn_rate: float) -> float:
        return self.still + self.per_turn_rate * abs(turn_rate) + self.per_speed * abs(speed)


class OffsetFilter:
    """Kalman filter of the 2D offset d = p_map - p_odom (independent x and y)."""

    def __init__(self, noise: OffsetNoise = OffsetNoise(), gate_sigma: float = 4.0,
                 max_rejections: int = 20):
        self._noise = noise
        self._gate = gate_sigma
        self._max_rejections = max_rejections
        self._rejections = 0
        self.offset: Optional[np.ndarray] = None
        self.variance = 0.0
        self._last_t: Optional[float] = None

    def predict(self, t: float, speed: float, turn_rate: float) -> None:
        if self.offset is not None and self._last_t is not None and t > self._last_t:
            self.variance += self._noise.rate(speed, turn_rate) * (t - self._last_t)
        self._last_t = t if self._last_t is None else max(t, self._last_t)

    def update(self, t: float, measured: np.ndarray, variance: float) -> bool:
        """
        Fuse one measured offset (GNSS antenna in map - antenna in odom) with its variance.

        Returns False when the measurement is rejected as an outlier. After max_rejections
        rejections in a row the filter takes the innovation as a sudden slip: its variance
        grows by the squared innovation and the measurement is fused.
        """
        measured = np.asarray(measured, float)
        if self.offset is None:
            self.offset, self.variance, self._last_t = measured.copy(), variance, t
            return True
        innovation = float(np.linalg.norm(measured - self.offset))
        s = self.variance + variance
        if innovation > self._gate * math.sqrt(s):
            self._rejections += 1
            if self._rejections <= self._max_rejections:
                return False
            self.variance += innovation ** 2
            s = self.variance + variance
        self._rejections = 0
        k = self.variance / s
        self.offset = self.offset + k * (measured - self.offset)
        self.variance = (1.0 - k) * self.variance
        return True

    def state(self) -> Tuple[np.ndarray, float]:
        return (np.zeros(2) if self.offset is None else self.offset), self.variance
