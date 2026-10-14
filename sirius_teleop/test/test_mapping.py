import pytest

from sirius_teleop.mapping import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    DriveLimits,
    GamepadLayout,
    GamepadTeleop,
)

LAYOUT = GamepadLayout()
N_AXES, N_BUTTONS = 6, 15


def joy(axes=None, pressed=()):
    a = [0.0] * N_AXES
    for index, value in (axes or {}).items():
        a[index] = value
    b = [0] * N_BUTTONS
    for index in pressed:
        b[index] = 1
    return a, b


def test_invalid_limits():
    with pytest.raises(ValueError):
        DriveLimits(max_linear=0.0)
    with pytest.raises(ValueError):
        DriveLimits(deadzone=1.0)


def test_no_deadman_no_commands():
    teleop = GamepadTeleop()
    cmd = teleop.update(*joy({LAYOUT.axis_left_y: 1.0}))
    assert cmd.drive is None and cmd.arm is None and cmd.gripper is None


def test_drive_scaled_and_stops_once_on_release():
    teleop = GamepadTeleop(limits=DriveLimits(max_linear=0.5, max_angular=1.0))
    cmd = teleop.update(*joy({LAYOUT.axis_left_y: 1.0, LAYOUT.axis_left_x: -0.5},
                             pressed=[LAYOUT.button_drive]))
    assert cmd.drive == pytest.approx((0.5, -0.5))
    assert teleop.update(*joy()).drive == (0.0, 0.0)
    assert teleop.update(*joy()).drive is None


def test_deadzone():
    teleop = GamepadTeleop(limits=DriveLimits(deadzone=0.1))
    cmd = teleop.update(*joy({LAYOUT.axis_left_y: 0.05}, pressed=[LAYOUT.button_drive]))
    assert cmd.drive == (0.0, 0.0)


def test_arm_twist_and_stop():
    teleop = GamepadTeleop()
    cmd = teleop.update(*joy({LAYOUT.axis_left_y: 1.0, LAYOUT.axis_right_y: -1.0},
                             pressed=[LAYOUT.button_arm, LAYOUT.button_up, LAYOUT.button_left]))
    assert cmd.arm == (1.0, 0.0, -1.0, -1.0, 1.0, 0.0)
    assert cmd.drive is None
    assert teleop.update(*joy()).arm == (0.0,) * 6


def test_gripper_on_rising_edge_only():
    teleop = GamepadTeleop()
    assert teleop.update(*joy(pressed=[LAYOUT.button_gripper_open])).gripper == GRIPPER_OPEN
    assert teleop.update(*joy(pressed=[LAYOUT.button_gripper_open])).gripper is None
    assert teleop.update(*joy(pressed=[LAYOUT.button_gripper_close])).gripper == GRIPPER_CLOSE


def test_short_joy_message_is_tolerated():
    teleop = GamepadTeleop()
    cmd = teleop.update([0.0], [0])
    assert cmd.drive is None and cmd.arm is None
