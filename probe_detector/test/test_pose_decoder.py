import numpy as np
from probe_detector.application.estimate_probe_poses import EstimateProbePoses, PoseFilter
from probe_detector.domain.evaluation import axis_angle_deg, match
from probe_detector.domain.ports import ProbeKeypointDetector
from probe_detector.domain.pose_decoder import decode_probe_pose
from probe_detector.domain.probe import (
    CameraIntrinsics, KEYPOINT_AXIS_Z, ProbeKeypoints, ProbePose)
import pytest

CAM = CameraIntrinsics(fx=904.9, fy=904.9, cx=640.0, cy=400.0, width=1280, height=800)


def detection(face, axis, score):
    return ProbeKeypoints(keypoints_of(np.array(face), np.array(axis)), (0, 0, 1, 1), score)


def keypoints_of(face, axis, conf=0.9):
    """Exact model output for a probe: projected keypoints with their true log depth."""
    axis = axis / np.linalg.norm(axis)
    pts = face + np.outer(KEYPOINT_AXIS_Z, axis)
    uv = CAM.project(pts)
    return np.c_[uv, np.full(4, conf), np.log(pts[:, 2])]


@pytest.mark.parametrize('face,axis', [
    (np.array([0.05, -0.02, 0.40]), np.array([1.0, 0.2, 0.3])),
    (np.array([-0.10, 0.08, 0.25]), np.array([0.0, -1.0, 0.1])),
    (np.array([0.0, 0.0, 0.60]), np.array([0.1, 0.0, -1.0])),       # end-on, face towards camera
])
def test_decoder_recovers_exact_pose(face, axis):
    pose = decode_probe_pose(ProbeKeypoints(keypoints_of(face, axis), (0, 0, 1, 1), 0.9), CAM)
    u = axis / np.linalg.norm(axis)
    assert np.allclose(pose.face, face, atol=1e-6)
    assert np.allclose(pose.axis, u, atol=1e-6)
    assert np.allclose(pose.centre, face - 0.1 * u, atol=1e-6)
    assert pose.spread < 1e-6


def test_quaternion_maps_z_to_axis():
    for u in (np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), np.array([0.3, -0.5, 0.8])):
        u = u / np.linalg.norm(u)
        x, y, z, w = ProbePose(np.zeros(3), np.zeros(3), u, 0.0, 1.0).quaternion()
        zc = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
        assert np.allclose(zc, u, atol=1e-6)


class FakeDetector(ProbeKeypointDetector):
    def __init__(self, dets):
        self.dets = dets

    def detect(self, image_bgr):
        return self.dets


def test_use_case_filters_score_and_inconsistent_keypoints():
    good = detection([0, 0, 0.5], [1.0, 0, 0], 0.8)
    weak = ProbeKeypoints(good.keypoints, (0, 0, 1, 1), 0.1)
    bent = good.keypoints.copy()
    bent[0, 3] += 0.5                                       # tip depth far off the axis
    bad = ProbeKeypoints(bent, (0, 0, 1, 1), 0.9)
    use_case = EstimateProbePoses(FakeDetector([weak, good, bad]), PoseFilter(0.3, 0.02))
    poses = use_case.execute(None, CAM)
    assert len(poses) == 1 and poses[0].score == pytest.approx(0.8)


def test_match_and_axis_angle():
    gt = np.array([[0, 0, 1.0], [1, 0, 3.0]])
    det = np.array([[1.1, 0, 3.2], [0.05, 0, 1.02], [5, 5, 5]])
    assert sorted(match(gt, det)) == [(0, 1), (1, 0)]
    assert axis_angle_deg(np.array([0, 0, 1.0]), np.array([0, 0, -1.0])) == pytest.approx(180)


def test_use_case_merges_duplicates():
    a = detection([0, 0, 0.5], [1.0, 0, 0], 0.7)
    b = detection([0.02, 0, 0.5], [1.0, 0, 0], 0.9)
    c = detection([0.3, 0, 0.5], [1.0, 0, 0], 0.5)
    poses = EstimateProbePoses(FakeDetector([a, b, c])).execute(None, CAM)
    assert [p.score for p in poses] == pytest.approx([0.9, 0.5])
