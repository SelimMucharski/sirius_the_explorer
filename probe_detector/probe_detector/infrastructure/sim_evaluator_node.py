"""
Simulation-only: compares the probe detectors with the Gazebo ground truth.

Ground truth: probe and robot model poses straight from Gazebo (gz_world_poses.GzWorldPoses,
full 3D) and base_link -> camera from TF (robot_state_publisher, includes the arm). So the
result does not depend on the EKF drift.

Per camera and detection message: every probe whose tip, centre or face projects into the
image at Z < max_range counts as "in view" (occlusion by the terrain or the gripper is not
known, so the recall is a lower bound). Detections are matched to all probes in the world
(domain.evaluation.match), so a detection of a probe outside the counted view is not a false
positive. A still camera is sampled at most every still_period. Results are split by the
scenario phase (/probe_sim_scenario/phase). Writes <output_dir>/<camera>_matches.csv,
<camera>_false_positives.csv and summary.txt every report_period seconds and on shutdown.
"""

import csv
import os

import numpy as np
from probe_detector.domain.evaluation import (
    axis_angle_deg, in_view, match, MatchGate, summarize)
from probe_detector.infrastructure.gz_world_poses import GzWorldPoses
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

BINS = {
    'front': [0, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0],
    'wrist': [0, 0.2, 0.3, 0.45, 0.6, 1.0, 1.5, 3.0],
}
MATCH_KEYS = ['t', 'phase', 'probe', 'moving', 'Z', 'matched', 'score', 'err_centre',
              'err_depth', 'err_lateral', 'err_axis', 'gt', 'det']


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_to_T(p):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


def transform_to_T(tf):
    r, t = tf.transform.rotation, tf.transform.translation
    T = np.eye(4)
    T[:3, :3] = quat_to_R(r.x, r.y, r.z, r.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _errors(rows, key):
    return summarize([r[key] for r in rows])


class SimEvaluatorNode(Node):

    def __init__(self):
        super().__init__('probe_sim_evaluator')
        self.declare_parameter('cameras', ['front', 'wrist'])
        self.declare_parameter('detections_topic', '/{camera}/probe_detector/detections')
        self.declare_parameter('camera_info_topics',
                               ['/front/color/camera_info', '/wrist/left/camera_info'])
        self.declare_parameter('world', 'erc2025')
        self.declare_parameter('robot_model', 'sirius')
        self.declare_parameter('probe_name_prefix', 'probe_object_')
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('output_dir', os.path.expanduser('~/probe_sim_eval'))
        self.declare_parameter('report_period', 15.0)
        # [s sim] save the debug image with the ground truth drawn in, 0 = off
        self.declare_parameter('snapshot_period', 0.0)
        # a still camera sees the same scene every frame: sample it at most every still_period
        self.declare_parameter('still_period', 1.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cams = list(p('cameras'))
        self.out = p('output_dir')
        os.makedirs(self.out, exist_ok=True)
        self.tf = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)
        self.gt = GzWorldPoses(p('world'), p('robot_model'), p('probe_name_prefix'))
        self.info = {}
        self.rows = {c: [] for c in self.cams}
        self.fps = {c: [] for c in self.cams}
        self.frames = {c: 0 for c in self.cams}
        self.phase_frames = {}
        self.debug = {c: {} for c in self.cams}
        self.pending = {c: {} for c in self.cams}
        self.last_snap = {c: -1e9 for c in self.cams}
        self.last_still = {c: -1e9 for c in self.cams}
        topic = p('detections_topic')
        if p('snapshot_period') > 0:
            os.makedirs(os.path.join(self.out, 'snapshots'), exist_ok=True)
            for cam in self.cams:
                debug_topic = topic.format(camera=cam).rsplit('/', 1)[0] + '/debug_image'
                self.create_subscription(Image, debug_topic, lambda m, c=cam: self._on_debug(c, m),
                                         qos_profile_sensor_data)
        for cam, info in zip(self.cams, p('camera_info_topics')):
            self.create_subscription(CameraInfo, info,
                                     lambda m, c=cam: self.info.__setitem__(c, m), 10)
            self.create_subscription(Detection3DArray, topic.format(camera=cam),
                                     lambda m, c=cam: self._on_dets(c, m), 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.phase = ''
        self.create_subscription(String, '/probe_sim_scenario/phase',
                                 lambda m: setattr(self, 'phase', m.data), latched)
        self.create_timer(p('report_period'), self.write)
        self.get_logger().info(f'evaluating {self.cams} -> {self.out}')

    # ------------------------------------------------------------ snapshots
    def _on_debug(self, cam, msg):
        t = Time.from_msg(msg.header.stamp).nanoseconds
        if t in self.pending[cam]:                      # detections arrived first
            self._draw(cam, msg, *self.pending[cam].pop(t))
            return
        buf = self.debug[cam]
        buf[t] = msg
        for k in sorted(buf)[:-10]:
            del buf[k]

    def _snapshot(self, cam, stamp, *args):
        ts = args[-1]
        if ts - self.last_snap[cam] < self.get_parameter('snapshot_period').value:
            return
        self.last_snap[cam] = ts
        t = stamp.nanoseconds
        msg = self.debug[cam].pop(t, None)
        if msg is None:                                 # debug image not here yet
            self.pending[cam][t] = args
            for k in sorted(self.pending[cam])[:-10]:
                del self.pending[cam][k]
            return
        self._draw(cam, msg, *args)

    def _draw(self, cam, msg, info, all_c, all_u, view, pairs, ts):
        import cv2
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
        k = info.k
        for g, (c, u) in enumerate(zip(all_c, all_u)):
            pts = np.stack([c - 0.1 * u, c + 0.1 * u])
            if np.any(pts[:, 2] < 0.05):
                continue
            uv = np.stack([k[0] * pts[:, 0] / pts[:, 2] + k[2],
                           k[4] * pts[:, 1] / pts[:, 2] + k[5]], 1).astype(int)
            col = (255, 255, 255) if view[g] else (160, 160, 160)
            cv2.line(img, tuple(uv[0]), tuple(uv[1]), col, 1, cv2.LINE_AA)
            cv2.circle(img, tuple(uv[1]), 4, col, 1, cv2.LINE_AA)
            label = f'gt{g} {"det" if g in pairs else "miss"} Z={c[2]:.2f}'
            cv2.putText(img, label, tuple(uv[1] + 5), 0, 0.4, col, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.out, 'snapshots', f'{cam}_{ts:08.2f}.jpg'), img)

    # ------------------------------------------------------------ evaluation
    def _camera_moving(self, frame, stamp, T_map_cam, dt=0.1, max_lin=0.02, max_ang=0.05):
        """
        Tell whether the camera moves faster than max_lin [m/s] or max_ang [rad/s].

        The ground truth is less exact while the arm / rover moves: TF and the poses are
        sampled, not rendered.
        """
        prev = stamp - Duration(seconds=dt)
        T_map_base = self.gt.robot_at(prev.nanoseconds)
        try:
            T0 = transform_to_T(self.tf.lookup_transform('base_link', frame, prev))
        except TransformException:
            return True
        if T_map_base is None:
            return True
        T0 = T_map_base @ T0
        lin = np.linalg.norm(T_map_cam[:3, 3] - T0[:3, 3]) / dt
        c = (np.trace(T0[:3, :3].T @ T_map_cam[:3, :3]) - 1) / 2
        ang = np.arccos(np.clip(c, -1, 1)) / dt
        return bool(lin > max_lin or ang > max_ang)

    def _on_dets(self, cam, msg):
        info = self.info.get(cam)
        stamp = Time.from_msg(msg.header.stamp)
        T_map_base = self.gt.robot_at(stamp.nanoseconds)
        probes = self.gt.probes()
        if info is None or T_map_base is None or not probes:
            return
        try:
            tf = self.tf.lookup_transform('base_link', msg.header.frame_id, stamp,
                                          Duration(seconds=0.2))
        except TransformException as e:
            self.get_logger().warn(f'{cam}: {e}', throttle_duration_sec=5.0)
            return
        T_map_cam = T_map_base @ transform_to_T(tf)
        T_cam_map = np.linalg.inv(T_map_cam)
        moving = self._camera_moving(msg.header.frame_id, stamp, T_map_cam)
        k, max_range = info.k, self.get_parameter('max_range').value
        all_c, all_u, view = [], [], []
        for Tp in probes:
            T = T_cam_map @ Tp
            c, u = T[:3, 3], T[:3, 2]
            all_c.append(c)
            all_u.append(u)
            # in view: tip, centre or face inside the image (partly visible probes count)
            view.append(c[2] < max_range and any(
                in_view(c + s * u, k[0], k[4], k[2], k[5], info.width, info.height)
                for s in (-0.1, 0.0, 0.1)))
        all_c, all_u = np.array(all_c).reshape(-1, 3), np.array(all_u).reshape(-1, 3)
        det_c, det_u, score = [], [], []
        for d in msg.detections:
            T = pose_to_T(d.results[0].pose.pose)
            det_c.append(T[:3, 3])
            det_u.append(T[:3, 2])
            score.append(d.results[0].hypothesis.score)
        det_c, det_u = np.array(det_c).reshape(-1, 3), np.array(det_u).reshape(-1, 3)
        pairs = dict(match(all_c, det_c, MatchGate()))
        ts = stamp.nanoseconds * 1e-9
        if not moving:
            if ts - self.last_still[cam] < self.get_parameter('still_period').value:
                return
            self.last_still[cam] = ts
        if self.get_parameter('snapshot_period').value > 0:
            self._snapshot(cam, stamp, info, all_c, all_u, view, pairs, ts)
        self.frames[cam] += 1
        key = (cam, self.phase)
        self.phase_frames[key] = self.phase_frames.get(key, 0) + 1
        for g in (g for g in range(len(all_c)) if view[g]):
            row = {'t': ts, 'phase': self.phase, 'probe': g, 'moving': moving, 'Z': all_c[g, 2],
                   'matched': g in pairs, 'gt': all_c[g].round(3).tolist()}
            if g in pairs:
                d = pairs[g]
                e = det_c[d] - all_c[g]
                row.update(score=score[d], err_centre=np.linalg.norm(e) * 1000,
                           err_depth=abs(e[2]) * 1000, err_lateral=np.hypot(e[0], e[1]) * 1000,
                           err_axis=axis_angle_deg(det_u[d], all_u[g]),
                           det=det_c[d].round(3).tolist())
            self.rows[cam].append(row)
        for d in set(range(len(det_c))) - set(pairs.values()):
            self.fps[cam].append({'t': ts, 'phase': self.phase, 'score': score[d],
                                  'det': det_c[d].round(3).tolist()})

    # ------------------------------------------------------------ report
    def _summary(self, cam):
        rows, fps, frames = self.rows[cam], self.fps[cam], self.frames[cam]
        m = [r for r in rows if r['matched']]
        lines = [
            f'== {cam}: {frames} frames, {len(rows)} probe-in-view instances, {len(m)} detected '
            f'(recall in view {len(m) / max(len(rows), 1):.3f}), {len(fps)} false positives '
            f'({len(fps) / max(frames, 1):.3f}/frame)',
            f'   all: centre [mm] {_errors(m, "err_centre")} | depth {_errors(m, "err_depth")} | '
            f'lateral {_errors(m, "err_lateral")} | axis [deg] {_errors(m, "err_axis")} | '
            f'axis flipped {sum(r["err_axis"] > 90 for r in m)}']
        for ph in sorted({r['phase'] for r in rows} | {f['phase'] for f in fps}):
            g = [r for r in rows if r['phase'] == ph]
            gm = [r for r in g if r['matched']]
            nf = self.phase_frames.get((cam, ph), 0)
            n_fp = sum(f['phase'] == ph for f in fps)
            lines.append(
                f'   phase {ph or "-"}: {nf} frames, recall {len(gm) / max(len(g), 1):.3f} '
                f'(n={len(g)}), FP/frame {n_fp / max(nf, 1):.3f} | centre '
                f'{_errors(gm, "err_centre")} | axis {_errors(gm, "err_axis")}')
        st = [r for r in m if not r['moving']]
        lines.append(f'   camera still ({len(st)} of {len(m)}): centre [mm] '
                     f'{_errors(st, "err_centre")} | depth {_errors(st, "err_depth")} | '
                     f'axis [deg] {_errors(st, "err_axis")}')
        b = BINS.get(cam, BINS['front'])
        for lo, hi in zip(b[:-1], b[1:]):
            g = [r for r in rows if lo <= r['Z'] < hi]
            if g:
                gm = [r for r in g if r['matched']]
                lines.append(f'   Z {lo}-{hi} m: n={len(g)} recall {len(gm) / len(g):.3f} | '
                             f'centre {_errors(gm, "err_centre")} | '
                             f'axis {_errors(gm, "err_axis")}')
        return lines

    def write(self):
        lines = []
        for cam in self.cams:
            lines += self._summary(cam)
            with open(os.path.join(self.out, f'{cam}_matches.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, MATCH_KEYS)
                w.writeheader()
                w.writerows(self.rows[cam])
            with open(os.path.join(self.out, f'{cam}_false_positives.csv'), 'w',
                      newline='') as f:
                w = csv.DictWriter(f, ['t', 'phase', 'score', 'det'])
                w.writeheader()
                w.writerows(self.fps[cam])
        txt = '\n'.join(lines)
        with open(os.path.join(self.out, 'summary.txt'), 'w') as f:
            f.write(txt + '\n')
        if rclpy.ok():
            self.get_logger().info('\n' + txt)


def main(args=None):
    rclpy.init(args=args)
    node = SimEvaluatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.write()
        node.destroy_node()
        rclpy.try_shutdown()
