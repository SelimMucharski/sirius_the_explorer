# sirius_teleop package
Gamepad teleoperation of the Sirius rover in simulation.

```bash
ros2 launch sirius_teleop teleop.launch.py   # or sirius_sim.launch.py teleop:=true
```

| Input | Action |
|-------|--------|
| hold **LB** + left stick | drive (`/cmd_vel`, TwistStamped) |
| hold **RB** + left stick | arm: x / y (MoveIt Servo, `root_link` frame) |
| hold **RB** + right stick | arm: z / yaw |
| hold **RB** + d-pad | arm: pitch / roll |
| **A** / **B** | open / close gripper |

Indices follow the `joy` package (SDL mapping); adjust `layout.*` in
[config/joy_teleop.yaml](config/joy_teleop.yaml) if your pad differs
(`ros2 topic echo /joy`).

Driving publishes only while LB is held, so it does not fight Nav2 for `/cmd_vel`.
