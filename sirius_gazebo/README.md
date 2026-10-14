# sirius_gazebo package
Gazebo Harmonic (gz-sim 8) simulation of the Sirius rover: worlds, ros_gz bridge,
ros2_control (gz_ros2_control), plus a full bringup with Nav2 and MoveIt.

Requires `ros-jazzy-gz-ros2-control`.

## Launching

### Simulation + ros2_control only
```bash
ros2 launch sirius_gazebo sirius_gazebo.launch.py                    # base only
ros2 launch sirius_gazebo sirius_gazebo.launch.py manipulator:=6DOF  # with the arm
```
Arguments: `manipulator` (`none`/`6DOF`), `world`, `headless`, `x`, `y`, `z`, `yaw`.

### Full stack: simulation + localization + Nav2 + MoveIt
```bash
ros2 launch sirius_gazebo sirius_sim.launch.py manipulator:=6DOF
```
Arguments: as above plus `nav2`, `moveit` (ignored for `manipulator:=none`), `rviz`,
`teleop` (gamepad, see [sirius_teleop](../sirius_teleop/README.md)).

## Data flow
| Source | Topic / frame |
|--------|---------------|
| Nav2 / teleop | `/cmd_vel` (`TwistStamped`) → `sirius_base_controller` |
| `sirius_base_controller` | `/odometry/wheels` |
| `ekf_odom` (wheels + IMU yaw rate) | `/odometry/filtered`, TF `odom → base_link` |
| static | TF `map → odom` |
| front OAK-D-W | `/front/depth/image_raw`, `/front/depth/camera_info`, `/front/color/image_raw`, `/front/points` |
| `obstacle_cloud` | `/front/obstacles` → Nav2 costmaps |
| IMU / GNSS | `/imu/data`, `/gps/fix` |
| Gazebo | `/clock`, `/ground_truth` |
| MoveIt | `/manip_controller/follow_joint_trajectory` |
| MoveIt / user | `/gripper_controller/follow_joint_trajectory` or `/gripper_controller/joint_trajectory` (both jaws; opening per jaw: 0 = closed, 0.0326 m = open) |

Driving by hand:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
```

Gripper by hand (or use the `open` / `closed` states of the `gripper` group in MoveIt):
```bash
ros2 topic pub --once /gripper_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [joint_gripper_left, joint_gripper_right], points: [{positions: [0.0326, 0.0326], time_from_start: {sec: 1}}]}"
```

## File structure
- [worlds](worlds) - SDF worlds (`empty.sdf`).
- [config](config) - gz_ros2_control controllers, bridge topics, EKF.
- [launch](launch) - `sirius_gazebo.launch.py` (sim + ros2_control),
  `sirius_sim.launch.py` (full stack).
