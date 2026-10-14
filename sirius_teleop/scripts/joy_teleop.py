#!/usr/bin/env python3
"""ROS 2 adapter: /joy -> /cmd_vel (base), MoveIt Servo (arm), gripper trajectory.

Hold LB and use the left stick to drive. Hold RB to jog the arm with MoveIt Servo
(left stick: x/y, right stick: z/yaw, d-pad: pitch/roll). A opens, B closes the gripper.
"""

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import TwistStamped
from moveit_msgs.srv import ServoCommandType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from sirius_teleop.mapping import (
    GRIPPER_OPEN,
    DriveLimits,
    GamepadLayout,
    GamepadTeleop,
)


class JoyTeleopNode(Node):

    def __init__(self):
        super().__init__('joy_teleop')
        defaults = GamepadLayout()
        layout = GamepadLayout(**{
            name: self.declare_parameter(f'layout.{name}', value).value
            for name, value in vars(defaults).items()})
        limits = DriveLimits(
            max_linear=self.declare_parameter('max_linear', 0.5).value,
            max_angular=self.declare_parameter('max_angular', 1.0).value,
            deadzone=self.declare_parameter('deadzone', 0.05).value)
        self._base_frame = self.declare_parameter('base_frame', 'base_link').value
        self._arm_frame = self.declare_parameter('arm_frame', 'root_link').value
        self._gripper_open = self.declare_parameter('gripper_open_position', 0.0326).value
        self._gripper_closed = self.declare_parameter('gripper_closed_position', 0.0).value
        self._gripper_joints = self.declare_parameter(
            'gripper_joints', ['joint_gripper_left', 'joint_gripper_right']).value
        self._gripper_time = self.declare_parameter('gripper_motion_time', 0.5).value

        self._teleop = GamepadTeleop(layout, limits)
        self._cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self._servo_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self._gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)
        self._servo_mode = self.create_client(
            ServoCommandType, '/servo_node/switch_command_type')
        self._servo_ready = False
        self.create_subscription(Joy, '/joy', self._on_joy, 10)

    def _ensure_twist_mode(self) -> None:
        # Servo starts in joint-jog mode; switch it to twist once it is available.
        if self._servo_ready or not self._servo_mode.service_is_ready():
            return
        self._servo_ready = True
        self._servo_mode.call_async(ServoCommandType.Request(command_type=ServoCommandType.Request.TWIST))

    def _on_joy(self, msg: Joy) -> None:
        command = self._teleop.update(msg.axes, msg.buttons)
        now = self.get_clock().now().to_msg()

        if command.drive is not None:
            out = TwistStamped()
            out.header.stamp = now
            out.header.frame_id = self._base_frame
            out.twist.linear.x, out.twist.angular.z = command.drive
            self._cmd_vel_pub.publish(out)

        if command.arm is not None:
            self._ensure_twist_mode()
            out = TwistStamped()
            out.header.stamp = now
            out.header.frame_id = self._arm_frame
            (out.twist.linear.x, out.twist.linear.y, out.twist.linear.z,
             out.twist.angular.x, out.twist.angular.y, out.twist.angular.z) = command.arm
            self._servo_pub.publish(out)

        if command.gripper is not None:
            self._send_gripper(command.gripper == GRIPPER_OPEN)

    def _send_gripper(self, open_gripper: bool) -> None:
        position = self._gripper_open if open_gripper else self._gripper_closed
        seconds = int(self._gripper_time)
        point = JointTrajectoryPoint(
            positions=[position] * len(self._gripper_joints),
            time_from_start=Duration(sec=seconds,
                                     nanosec=int((self._gripper_time - seconds) * 1e9)))
        self._gripper_pub.publish(JointTrajectory(joint_names=self._gripper_joints, points=[point]))


def main(args=None):
    rclpy.init(args=args)
    node = JoyTeleopNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise  # a real error; otherwise spin was interrupted by shutdown (SIGINT race)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
