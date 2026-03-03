import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rigbo import RigboConfig

logger = logging.getLogger(__name__)


class Rigbo(Robot):
    """Rigbo: single arm + omni base (no head)."""

    config_class = RigboConfig
    name = "rigbo"

    def __init__(self, config: RigboConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys

        self.speed_levels = [
            {"xy": 0.1, "theta": 30},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.3, "theta": 90},
        ]
        self.speed_index = 0

        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # --- Single Arm ---
        self.bus_arm = FeetechMotorsBus(
            port=config.port_arm,
            motors={
                "arm_shoulder_pan": Motor(1, "sts3215", norm_mode),
                "arm_shoulder_lift": Motor(2, "sts3215", norm_mode),
                "arm_elbow_flex": Motor(3, "sts3215", norm_mode),
                "arm_wrist_flex": Motor(4, "sts3215", norm_mode),
                "arm_wrist_roll": Motor(5, "sts3215", norm_mode),
                "arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # --- Base ---
        self.bus_base = FeetechMotorsBus(
            port=config.port_base,
            motors={
                "base_left_wheel": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )

        self.arm_motors = list(self.bus_arm.motors.keys())
        self.base_motors = list(self.bus_base.motors.keys())

        self.cameras = make_cameras_from_configs(config.cameras)

    # ---------------- Features ----------------

    @cached_property
    def observation_features(self):
        return dict.fromkeys(
            (
                "arm_shoulder_pan.pos",
                "arm_shoulder_lift.pos",
                "arm_elbow_flex.pos",
                "arm_wrist_flex.pos",
                "arm_wrist_roll.pos",
                "arm_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    action_features = observation_features

    @property
    def is_connected(self):
        return self.bus_arm.is_connected and self.bus_base.is_connected

    # ---------------- Lifecycle ----------------

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Rigbo already connected")

        self.bus_arm.connect()
        self.bus_base.connect()

        self.bus_arm.enable_torque()
        self.bus_base.enable_torque()

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("Rigbo not connected")

        self.bus_arm.disconnect(self.config.disable_torque_on_disconnect)
        self.bus_base.disconnect(self.config.disable_torque_on_disconnect)

    # ---------------- Base Control ----------------

    def _from_keyboard_to_base_action(self, pressed_keys):
        speed = self.speed_levels[self.speed_index]

        x = y = theta = 0.0
        if self.teleop_keys["forward"] in pressed_keys:
            x += speed["xy"]
        if self.teleop_keys["backward"] in pressed_keys:
            x -= speed["xy"]
        if self.teleop_keys["left"] in pressed_keys:
            y += speed["xy"]
        if self.teleop_keys["right"] in pressed_keys:
            y -= speed["xy"]
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta += speed["theta"]
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta -= speed["theta"]

        return {"x.vel": x, "y.vel": y, "theta.vel": theta}

    # ---------------- I/O ----------------

    def get_observation(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("Rigbo not connected")

        arm = self.bus_arm.sync_read("Present_Position", self.arm_motors)
        base = self.bus_base.sync_read("Present_Velocity", self.base_motors)

        return {
            **{f"{k}.pos": v for k, v in arm.items()},
            "x.vel": base["base_left_wheel"],  # simplified
            "y.vel": base["base_back_wheel"],
            "theta.vel": base["base_right_wheel"],
        }

    def send_action(self, action: dict[str, Any]):
        arm = {k.replace(".pos", ""): v for k, v in action.items() if k.endswith(".pos")}
        base = {k: v for k, v in action.items() if k.endswith(".vel")}

        if arm:
            self.bus_arm.sync_write("Goal_Position", arm)
        if base:
            self.bus_base.sync_write("Goal_Velocity", base)

        return action