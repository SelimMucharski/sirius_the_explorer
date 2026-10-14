# sirius_description package
This package provides the physical description (URDF/Xacro) of the Sirius 2 rover. Migrated from [ROS1 Noetic package](https://github.com/SKA-Robotics/sirius_common/tree/noetic/sirius_description)

## TO DO
- Add IP cameras frames to TF

## Launching

### Mobile platform (Base only)
To start publishing the robot description state, use the following command:
```bash
ros2 launch sirius_description sirius_description.launch.py
```

### Mobile platform + Manipulator
To publish the robot description together with a manipulator, use the `manipulator` argument.

**Available options:** `none`, `6DOF` (default).

Example (launching with 6DOF arm):
```bash
ros2 launch sirius_description sirius_description.launch.py manipulator:=6DOF
```

## File structure
The package is organized into the following directories:
- [config](config) - Configuration files (mechanical dimensions, inertia, OAK-D-W camera/IMU calibration).
- [meshes](meshes) - Visual and collision files (.stl).
- [robots](robots) - Main top-level .xacro file for the Sirius 2 rover.
- [urdf](urdf) - Macro definitions and component descriptions (submodules).
- [launch](launch) - Launch files.