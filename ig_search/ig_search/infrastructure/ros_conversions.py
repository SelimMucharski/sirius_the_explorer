"""Konwersje między wiadomościami ROS a typami domeny (macierze, CameraPose, SamplePose)."""
import math

from ig_search.application.ports import SamplePose
from ig_search.domain.visibility import CameraPose
import numpy as np

# Ramka optyczna (z - oś optyczna, x - w prawo, y - w dół) -> ramka kamery domeny
# (x - oś optyczna, y - w lewo, z - w górę): kolumny = osie domeny w ramce optycznej.
OPTICAL_TO_BODY = np.array([[0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0]])


def quat_to_R(x, y, z, w) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_to_quat(yaw: float):
    """(x, y, z, w) obrotu wokół osi z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def transform_to_T(tf) -> np.ndarray:
    """geometry_msgs/TransformStamped -> macierz 4x4."""
    r, t = tf.transform.rotation, tf.transform.translation
    T = np.eye(4)
    T[:3, :3] = quat_to_R(r.x, r.y, r.z, r.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def T_to_q(T: np.ndarray) -> tuple:
    """Poza planarna (x, y, psi) z macierzy 4x4 base_link w ramce map."""
    return float(T[0, 3]), float(T[1, 3]), float(math.atan2(T[1, 0], T[0, 0]))


def optical_camera_pose(T_map_optical: np.ndarray) -> CameraPose:
    """Poza ramki optycznej kamery -> CameraPose domeny (x w przód, z w górę)."""
    return CameraPose(T_map_optical[:3, 3].copy(), T_map_optical[:3, :3] @ OPTICAL_TO_BODY)


def detection_to_sample(pose) -> SamplePose:
    """geometry_msgs/Pose z probe_detector (oś z orientacji = oś próbki) -> SamplePose."""
    p, o = pose.position, pose.orientation
    R = quat_to_R(o.x, o.y, o.z, o.w)
    return SamplePose(np.array([p.x, p.y, p.z]), R[:, 2].copy())
