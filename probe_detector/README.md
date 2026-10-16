# probe_detector

Probe pose estimation from the rover cameras with the pose3d models trained in
`oak_probe_vision` (Gazebo renders of the erc2025 world, `scripts/gazebo/`):

| node | camera | input | model |
|------|--------|-------|-------|
| `/front/probe_detector` | OAK-D W, front | `/front/color/image_raw`, `/front/color/camera_info` (640x360) | `models/probe_front_oak_d_w.pt` (imgsz 640) |
| `/wrist/probe_detector` | OAK-D SR, wrist | `/wrist/left/image_raw`, `/wrist/left/camera_info` (1280x800) | `models/probe_wrist_oak_d_sr.pt` (imgsz 960) |

Each node publishes in the camera optical frame (or `target_frame`, via TF):

* `~/poses` (`geometry_msgs/PoseArray`): probe centre (middle of the 200 mm probe); the
  orientation's z axis is the probe axis, tip -> face. The probe is rotationally symmetric, so
  this is its full pose.
* `~/detections` (`vision_msgs/Detection3DArray`): the same with the score and a 0.03 x 0.03 x 0.2 m box.
* `~/markers` (RViz) and, with `publish_debug_image`, `~/debug_image`.

The model outputs 4 keypoints along the axis with a predicted depth each; the domain decoder
back-projects them with the camera K and fits the axis (`domain/pose_decoder.py`). Detections
with inconsistent keypoints (`max_spread`) and duplicates closer than `duplicate_distance`
are dropped. Parameters: `config/probe_detectors.yaml`.

```bash
ros2 launch probe_detector probe_detectors.launch.py use_sim_time:=true [cameras:=wrist] [debug_image:=true]
```

The nodes are lifecycle nodes (configure loads the model on the GPU). Requires `ultralytics`
and `torch` (pip). `cv_bridge` is not used: its ROS build is compiled against NumPy 1.x and
crashes next to the NumPy 2 that Ultralytics needs.

## Layout

```
probe_detector/
  domain/          probe value objects, pose decoder, detector port, evaluation helpers (no ROS)
  application/     EstimateProbePoses use case (filters, duplicate suppression)
  infrastructure/  YOLO adapter (+ pose3d_patch.py, a copy of oak_probe_vision's pose3d.py),
                   lifecycle node, simulation-only ground truth + evaluator
  presentation/    simulation test scenario
```

## End-to-end simulation test

```bash
ros2 launch probe_detector sim_probe_test.launch.py output_dir:=/tmp/probe_eval \
    [debug_image:=true snapshot_period:=4.0] [headless:=false]
```

Starts the simulation (erc2025, 6DOF arm, no Nav2/MoveIt), the probe spawner, both detectors,
the evaluator and the scenario: probes are spawned ahead of the rover and within the arm's
reach; the arm points the wrist camera at the near probes from 0.25 / 0.4 / 0.6 m with the
gripper open, then closed; the rover turns twice in place and drives 1.6 m forward and back.
The evaluator compares every detection with the ground truth read straight from Gazebo
(`/world/erc2025/pose/info` over gz-transport: full 3D model poses; `/ground_truth` is 2D and
the spawner's per-probe pose bridges can miss updates) and writes `summary.txt`, per-detection
CSVs and, with snapshots, debug images with the ground truth drawn in. A probe counts as in
view when its tip, centre or face projects into the image; occlusion is not checked, so the
recall is a lower bound.
