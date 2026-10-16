"""
Lifecycle node: probe poses from one camera (pose3d model).

Subscribes   image (sensor_msgs/Image), camera_info (sensor_msgs/CameraInfo) - remap per camera.
Publishes    ~/poses       geometry_msgs/PoseArray       probe centres; orientation z axis = probe
                                                         axis tip -> face
             ~/detections  vision_msgs/Detection3DArray  same + score, box 0.03 x 0.03 x 0.2 m
             ~/markers     visualization_msgs/MarkerArray
             ~/debug_image sensor_msgs/Image             only with publish_debug_image
Frame: the camera optical frame of the image, or target_frame (TF at the image stamp).

configure: loads the model on the GPU (a few seconds); activate: subscribes.
"""

import os

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray
import numpy as np
from probe_detector.application.estimate_probe_poses import EstimateProbePoses, PoseFilter
from probe_detector.domain.probe import CameraIntrinsics, PROBE_DIAMETER, PROBE_LENGTH
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray


def image_to_bgr(msg: Image) -> np.ndarray:
    """
    Convert a sensor_msgs/Image to a BGR uint8 array.

    Not cv_bridge: the ROS build of cv_bridge is compiled against NumPy 1.x and crashes next to
    the NumPy 2 that Ultralytics needs.
    """
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(msg.encoding)
    if ch is None:
        raise ValueError(f'unsupported image encoding {msg.encoding}')
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    a = a.reshape(msg.height, msg.width, ch)
    if msg.encoding == 'mono8':
        return np.repeat(a, 3, axis=2)
    if msg.encoding.startswith('rgb'):
        return np.ascontiguousarray(a[..., 2::-1])
    return a[..., :3].copy()                    # writable (the message buffer is read-only)


def bgr_to_image(img: np.ndarray, header) -> Image:
    out = Image(header=header, height=img.shape[0], width=img.shape[1], encoding='bgr8',
                step=img.shape[1] * 3)
    out.data = img.tobytes()
    return out


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def _rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class ProbeDetectorNode(LifecycleNode):

    def __init__(self, **kwargs):
        super().__init__('probe_detector', **kwargs)
        # model file (required); a relative path is relative to this package's share directory
        self.declare_parameter('weights', '')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', '')
        self.declare_parameter('detector_conf', 0.25)
        self.declare_parameter('min_score', 0.3)
        self.declare_parameter('max_spread', 0.05)
        self.declare_parameter('duplicate_distance', 0.06)
        self.declare_parameter('target_frame', '')
        self.declare_parameter('tf_timeout', 0.1)
        self.declare_parameter('publish_debug_image', False)
        self._use_case = None
        self._camera = None
        self._subs = []
        self._pubs = {}
        self._tf_buffer = None
        self._tf_listener = None

    # ------------------------------------------------------------ lifecycle
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        from probe_detector.infrastructure.yolo_detector import YoloPose3dDetector
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        weights = p('weights')
        if not weights:
            self.get_logger().error('parameter "weights" is empty')
            return TransitionCallbackReturn.FAILURE
        if not os.path.isabs(weights):
            weights = os.path.join(get_package_share_directory('probe_detector'), weights)
        try:
            detector = YoloPose3dDetector(weights, p('imgsz'), p('detector_conf'), p('device'))
            detector.warmup(p('imgsz'), p('imgsz'))
        except Exception as e:  # noqa: BLE001 - any loading error fails the transition
            self.get_logger().error(f'cannot load {weights}: {e}')
            return TransitionCallbackReturn.FAILURE
        pose_filter = PoseFilter(p('min_score'), p('max_spread'),
                                 duplicate_distance=p('duplicate_distance'))
        self._use_case = EstimateProbePoses(detector, pose_filter)
        if p('target_frame'):
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)
        self._pubs = {
            'poses': self.create_lifecycle_publisher(PoseArray, '~/poses', 10),
            'detections': self.create_lifecycle_publisher(Detection3DArray, '~/detections', 10),
            'markers': self.create_lifecycle_publisher(MarkerArray, '~/markers', 10)}
        if p('publish_debug_image'):
            self._pubs['debug'] = self.create_lifecycle_publisher(Image, '~/debug_image',
                                                                  qos_profile_sensor_data)
        self.get_logger().info(f'loaded {weights} (imgsz {p("imgsz")})')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._subs = [
            self.create_subscription(CameraInfo, 'camera_info', self._on_info, 10),
            self.create_subscription(Image, 'image', self._on_image, qos_profile_sensor_data),
        ]
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        for s in self._subs:
            self.destroy_subscription(s)
        self._subs = []
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        for pub in self._pubs.values():
            self.destroy_lifecycle_publisher(pub)
        self._pubs, self._use_case, self._camera = {}, None, None
        self._tf_buffer = self._tf_listener = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------ callbacks
    def _on_info(self, msg: CameraInfo) -> None:
        self._camera = CameraIntrinsics.from_k(msg.k, msg.width, msg.height)

    def _on_image(self, msg: Image) -> None:
        if self._camera is None:
            self.get_logger().warn('no camera_info yet', throttle_duration_sec=5.0)
            return
        try:
            img = image_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        poses = self._use_case.execute(img, self._camera)
        frame, q_tf, t_tf = msg.header.frame_id, None, None
        target = self.get_parameter('target_frame').value
        if target and target != frame:
            try:
                tf = self._tf_buffer.lookup_transform(
                    target, frame, Time.from_msg(msg.header.stamp),
                    Duration(seconds=self.get_parameter('tf_timeout').value))
            except TransformException as e:
                self.get_logger().warn(f'TF {frame} -> {target}: {e}', throttle_duration_sec=5.0)
                return
            r, t = tf.transform.rotation, tf.transform.translation
            q_tf, t_tf, frame = (r.x, r.y, r.z, r.w), np.array([t.x, t.y, t.z]), target
        self._publish(msg, frame, poses, q_tf, t_tf)
        if 'debug' in self._pubs:
            self._publish_debug(msg, img, poses)

    # ------------------------------------------------------------ output
    def _publish(self, msg, frame, poses, q_tf, t_tf):
        header = Header(stamp=msg.header.stamp, frame_id=frame)
        arr, dets = PoseArray(header=header), Detection3DArray(header=header)
        markers = MarkerArray()
        markers.markers.append(Marker(header=header, ns='probes', action=Marker.DELETEALL))
        for i, p in enumerate(poses):
            q, c = p.quaternion(), p.centre
            if q_tf is not None:
                q, c = _quat_mul(q_tf, q), _rotate(q_tf, c) + t_tf
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in c)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
            arr.poses.append(pose)
            d = Detection3D(header=header)
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = 'probe', p.score
            h.pose.pose = pose
            d.results.append(h)
            d.bbox.center = pose
            d.bbox.size.x = d.bbox.size.y = PROBE_DIAMETER
            d.bbox.size.z = PROBE_LENGTH
            d.id = str(i)
            dets.detections.append(d)
            m = Marker(header=header, ns='probes', id=i, type=Marker.CYLINDER,
                       action=Marker.ADD, pose=pose)
            m.scale.x = m.scale.y = PROBE_DIAMETER
            m.scale.z = PROBE_LENGTH
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.85, 0.0, 0.9
            markers.markers.append(m)
        self._pubs['poses'].publish(arr)
        self._pubs['detections'].publish(dets)
        self._pubs['markers'].publish(markers)

    def _publish_debug(self, msg, img, poses):
        import cv2
        cam = self._camera
        for p in poses:
            pts = np.stack([p.face + z * p.axis for z in (-PROBE_LENGTH, 0.0)])
            if np.all(pts[:, 2] > 0.01):
                uv = cam.project(pts).astype(int)
                cv2.line(img, tuple(uv[0]), tuple(uv[1]), (0, 215, 255), 2, cv2.LINE_AA)
                c = p.centre
                label = f'{p.score:.2f} ({c[0]:+.2f} {c[1]:+.2f} {c[2]:.2f})'
                cv2.putText(img, label, tuple(uv[1] + 6), 0, 0.45, (0, 215, 255), 1,
                            cv2.LINE_AA)
        self._pubs['debug'].publish(bgr_to_image(img, msg.header))


def main(args=None):
    rclpy.init(args=args)
    node = ProbeDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
