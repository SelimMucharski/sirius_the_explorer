"""
Simulation-only ground truth: model poses straight from Gazebo (gz-transport).

Reads /world/<world>/pose/info, which carries every model's full 3D world pose with its name -
unlike /ground_truth (OdometryPublisher in 2D mode: z = 0, no roll/pitch) and the per-probe pose
bridges of probe_supervisor (separate processes that can miss updates). The Gazebo world frame
is 'map' in the simulation.
"""

from bisect import bisect_left
from collections import deque
import threading

import numpy as np


def _pose_T(p):
    x, y, z, w = p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


class GzWorldPoses:
    """Latest poses of models named probe_prefix* and a time-indexed history of the robot."""

    def __init__(self, world='erc2025', robot='sirius', probe_prefix='probe_object_',
                 history=4000, rate=None):
        """rate - max. messages/s handled (gz-transport throttling before parsing); None - all."""
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node, SubscribeOptions
        self._robot, self._prefix = robot, probe_prefix
        self._lock = threading.Lock()
        self._probes = {}
        self._robot_hist = deque(maxlen=history)
        self._node = Node()
        options = SubscribeOptions()
        if rate:
            # /world/<w>/pose/info comes every physics step; parsing it in Python cost
            # ~25% of a core per subscriber in the simulation
            options.msgs_per_sec = int(rate)
        if not self._node.subscribe(Pose_V, f'/world/{world}/pose/info', self._on_poses,
                                    options):
            raise RuntimeError(f'cannot subscribe to /world/{world}/pose/info')

    def _on_poses(self, msg):
        t = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec
        probes, robot = {}, None
        for p in msg.pose:
            if p.name.startswith(self._prefix):
                probes[p.name] = _pose_T(p)
            elif p.name == self._robot:
                robot = _pose_T(p)
        with self._lock:
            self._probes = probes
            if robot is not None:
                self._robot_hist.append((t, robot))

    def probes(self):
        """Return the world poses of the probes, sorted by name."""
        with self._lock:
            return [self._probes[k] for k in sorted(self._probes)]

    def robot_at(self, t_ns, tolerance_ns=50_000_000):
        """Robot (model frame = base_link) world pose closest to sim time t_ns, or None."""
        with self._lock:
            hist = list(self._robot_hist)
        if not hist:
            return None
        ts = [h[0] for h in hist]
        i = min(bisect_left(ts, t_ns), len(ts) - 1)
        k = min((i, max(i - 1, 0)), key=lambda j: abs(ts[j] - t_ns))
        return hist[k][1] if abs(ts[k] - t_ns) <= tolerance_ns else None

    def robot_latest(self):
        with self._lock:
            return self._robot_hist[-1][1] if self._robot_hist else None
