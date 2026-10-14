"""Gamepad -> teleoperation commands (pure logic, no ROS imports).

One gamepad drives the base and jogs the arm, each behind its own dead-man
button, and opens/closes the gripper with two buttons.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Twist6 = Tuple[float, float, float, float, float, float]  # vx, vy, vz, wx, wy, wz

GRIPPER_OPEN = 'open'
GRIPPER_CLOSE = 'close'


@dataclass(frozen=True)
class GamepadLayout:
    """Axis / button indices of sensor_msgs/Joy (defaults: joy package, SDL mapping)."""

    axis_left_x: int = 0       # +1 = left
    axis_left_y: int = 1       # +1 = up
    axis_right_x: int = 2
    axis_right_y: int = 3
    button_gripper_open: int = 0   # A
    button_gripper_close: int = 1  # B
    button_drive: int = 9          # LB, dead-man for driving
    button_arm: int = 10           # RB, dead-man for the arm
    button_up: int = 11            # d-pad
    button_down: int = 12
    button_left: int = 13
    button_right: int = 14


@dataclass(frozen=True)
class DriveLimits:
    max_linear: float = 0.5   # [m/s]
    max_angular: float = 1.0  # [rad/s]
    deadzone: float = 0.05

    def __post_init__(self):
        if self.max_linear <= 0.0 or self.max_angular <= 0.0:
            raise ValueError('drive limits must be positive')
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError('deadzone must be in [0, 1)')


@dataclass(frozen=True)
class TeleopCommand:
    """What to send this cycle; None means "send nothing" for that output."""

    drive: Optional[Tuple[float, float]] = None   # (linear x [m/s], angular z [rad/s])
    arm: Optional[Twist6] = None                  # unitless [-1, 1] for MoveIt Servo
    gripper: Optional[str] = None                 # GRIPPER_OPEN / GRIPPER_CLOSE


def _deadzone(value: float, deadzone: float) -> float:
    return 0.0 if abs(value) < deadzone else value


def _get(values: Sequence[float], index: int) -> float:
    return float(values[index]) if 0 <= index < len(values) else 0.0


class GamepadTeleop:
    """Stateful mapper: dead-man handling and button edge detection."""

    def __init__(self, layout: GamepadLayout = GamepadLayout(),
                 limits: DriveLimits = DriveLimits()):
        self._layout = layout
        self._limits = limits
        self._driving = False
        self._jogging = False
        self._previous_buttons: Sequence[float] = ()

    def update(self, axes: Sequence[float], buttons: Sequence[float]) -> TeleopCommand:
        lay, lim = self._layout, self._limits

        def axis(index):
            return _deadzone(_get(axes, index), lim.deadzone)

        def pressed(index):
            return _get(buttons, index) > 0.5

        def rising(index):
            return pressed(index) and not _get(self._previous_buttons, index) > 0.5

        drive = None
        if pressed(lay.button_drive):
            drive = (axis(lay.axis_left_y) * lim.max_linear,
                     axis(lay.axis_left_x) * lim.max_angular)
        elif self._driving:
            drive = (0.0, 0.0)  # stop once when the dead-man is released
        self._driving = pressed(lay.button_drive)

        arm = None
        if pressed(lay.button_arm):
            pitch = float(pressed(lay.button_up)) - float(pressed(lay.button_down))
            roll = float(pressed(lay.button_right)) - float(pressed(lay.button_left))
            arm = (axis(lay.axis_left_y), axis(lay.axis_left_x), axis(lay.axis_right_y),
                   roll, pitch, axis(lay.axis_right_x))
        elif self._jogging:
            arm = (0.0,) * 6
        self._jogging = pressed(lay.button_arm)

        gripper = None
        if rising(lay.button_gripper_open):
            gripper = GRIPPER_OPEN
        elif rising(lay.button_gripper_close):
            gripper = GRIPPER_CLOSE

        self._previous_buttons = tuple(buttons)
        return TeleopCommand(drive=drive, arm=arm, gripper=gripper)
