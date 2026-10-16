"""Use case: probe poses in the camera frame from one image."""

from dataclasses import dataclass
from typing import List

import numpy as np

from probe_detector.domain.ports import ProbeKeypointDetector
from probe_detector.domain.pose_decoder import decode_probe_pose
from probe_detector.domain.probe import CameraIntrinsics, ProbePose


@dataclass(frozen=True)
class PoseFilter:
    min_score: float = 0.3
    max_spread: float = 0.05         # [m] keypoints inconsistent with a straight probe -> drop
    min_depth: float = 0.03          # [m]
    # two detections whose centres are closer are one probe (e.g. a probe split in two by a
    # gripper jaw gives two boxes); the higher score is kept
    duplicate_distance: float = 0.06  # [m]


class EstimateProbePoses:

    def __init__(self, detector: ProbeKeypointDetector, pose_filter: PoseFilter = PoseFilter()):
        self._detector = detector
        self._filter = pose_filter

    def execute(self, image_bgr: np.ndarray, camera: CameraIntrinsics) -> List[ProbePose]:
        out = []
        for det in self._detector.detect(image_bgr):
            if det.score < self._filter.min_score:
                continue
            pose = decode_probe_pose(det, camera)
            if pose.spread > self._filter.max_spread or pose.centre[2] < self._filter.min_depth:
                continue
            out.append(pose)
        kept = []
        for pose in sorted(out, key=lambda p: -p.score):
            dist = self._filter.duplicate_distance
            if all(np.linalg.norm(pose.centre - k.centre) >= dist for k in kept):
                kept.append(pose)
        return kept
