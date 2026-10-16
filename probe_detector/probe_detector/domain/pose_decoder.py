"""
Probe pose from pose3d keypoints (pixel position + predicted depth).

Pure numpy.

Every keypoint is back-projected with its predicted depth and the axis P_i = F + z_i * u is fitted
by weighted least squares with the known positions z_i of the points along the axis (F = face
centre). Same decoder as oak_probe_vision (scripts/gazebo/probe_pose_sim.py), where it was
evaluated against the Gazebo ground truth.
"""

import numpy as np

from probe_detector.domain.probe import (
    CameraIntrinsics, KEYPOINT_AXIS_Z, PROBE_LENGTH, ProbeKeypoints, ProbePose)


def keypoint_weights(kps: np.ndarray, cam: CameraIntrinsics, w_min: float = 0.05) -> np.ndarray:
    """
    Weight the keypoints by the network confidence.

    Points predicted outside the image (they have no 2D labels in training) get w_min.
    """
    w = np.clip(kps[:, 2], w_min, 1.0)
    inside = ((kps[:, 0] > 2) & (kps[:, 0] < cam.width - 2)
              & (kps[:, 1] > 2) & (kps[:, 1] < cam.height - 2))
    return np.where(inside, w, w_min)


def fit_axis(points: np.ndarray, weights: np.ndarray):
    """
    Weighted least squares points_i = c + (z_i - zm) * v (z_i = KEYPOINT_AXIS_Z).

    Returns the weighted centroid c, its axis coordinate zm and v (|v| = 1 for a correct
    depth scale).
    """
    z, w = KEYPOINT_AXIS_Z, weights
    zm = (w * z).sum() / w.sum()
    pm = (w[:, None] * points).sum(0) / w.sum()
    v = (w[:, None] * (z - zm)[:, None] * (points - pm)).sum(0) / (w * (z - zm) ** 2).sum()
    return pm, zm, v


def decode_probe_pose(det: ProbeKeypoints, cam: CameraIntrinsics) -> ProbePose:
    kps = np.asarray(det.keypoints, float)
    points = cam.backproject(kps[:, :2], np.exp(kps[:, 3]))
    pm, zm, v = fit_axis(points, keypoint_weights(kps, cam))
    u = v / max(np.linalg.norm(v), 1e-9)
    face = pm - zm * u
    spread = float(np.linalg.norm(points - (face + np.outer(KEYPOINT_AXIS_Z, u)), axis=1).mean())
    return ProbePose(centre=face - PROBE_LENGTH / 2 * u, face=face, axis=u, spread=spread,
                     score=float(det.score))
