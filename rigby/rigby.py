# lerobot/robots/rigby/rigby.py
import logging
import time
from functools import cached_property
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.utils import ensure_safe_goal_position

from .config_rigby import RigbyConfig

logger = logging.getLogger(__name__)


class Rigby(Robot):
    config_class = RigbyConfig
    name = "rigby"

    def __init__(self, config: RigbyConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys

        # Same speed ladder as XLerobot pattern
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.3, "theta": 90},
        ]
        self.speed_index = 0

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # --- bus_left: left arm ---
        calibration_left = (
            {k: self.calibration.get(k) for k in (
                "left_arm_shoulder_pan",
                "left_arm_shoulder_lift",
                "left_arm_elbow_flex",
                "left_arm_wrist_flex",
                "left_arm_wrist_roll",
                "left_arm_gripper",
            )}
            if self.calibration.get("left_arm_shoulder_pan") is not None
            else self.calibration
        )

        self.bus_left = FeetechMotorsBus(
            port=self.config.port_left,
            motors={
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration_left,
        )

        # --- bus_right: right arm ---
        calibration_right = (
            {k: self.calibration.get(k) for k in (
                "right_arm_shoulder_pan",
                "right_arm_shoulder_lift",
                "right_arm_elbow_flex",
                "right_arm_wrist_flex",
                "right_arm_wrist_roll",
                "right_arm_gripper",
            )}
            if self.calibration.get("right_arm_shoulder_pan") is not None
            else self.calibration
        )

        self.bus_right = FeetechMotorsBus(
            port=self.config.port_right,
            motors={
                "right_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "right_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "right_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "right_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "right_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "right_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration_right,
        )

        # --- bus_base: base wheels ---
        calibration_base = (
            {k: self.calibration.get(k) for k in ("base_left_wheel", "base_back_wheel", "base_right_wheel")}
            if self.calibration.get("base_left_wheel") is not None
            else self.calibration
        )

        self.bus_base = FeetechMotorsBus(
            port=self.config.port_base,
            motors={
                "base_left_wheel": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=calibration_base,
        )

        self.left_arm_motors = list(self.bus_left.motors.keys())
        self.right_arm_motors = list(self.bus_right.motors.keys())
        self.base_motors = list(self.bus_base.motors.keys())

        self.cameras = make_cameras_from_configs(config.cameras)

    # -------- features (no head) --------
    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "left_arm_shoulder_pan.pos",
                "left_arm_shoulder_lift.pos",
                "left_arm_elbow_flex.pos",
                "left_arm_wrist_flex.pos",
                "left_arm_wrist_roll.pos",
                "left_arm_gripper.pos",
                "right_arm_shoulder_pan.pos",
                "right_arm_shoulder_lift.pos",
                "right_arm_elbow_flex.pos",
                "right_arm_wrist_flex.pos",
                "right_arm_wrist_roll.pos",
                "right_arm_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus_left.is_connected and self.bus_right.is_connected and self.bus_base.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return self.bus_left.is_calibrated and self.bus_right.is_calibrated and self.bus_base.is_calibrated

    # -------- lifecycle --------
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus_left.connect()
        self.bus_right.connect()
        self.bus_base.connect()

        if calibrate and (not self.calibration_fpath.is_file()):
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")

        # Left arm
        self.bus_left.disable_torque()
        for name in self.left_arm_motors:
            self.bus_left.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move left arm joints to middle of range and press ENTER....")
        homing_left = self.bus_left.set_half_turn_homings(self.left_arm_motors)

        print("Move left arm joints through full ranges. Press ENTER to stop...")
        mins_left, maxs_left = self.bus_left.record_ranges_of_motion(self.left_arm_motors)

        cal_left = {}
        for name, motor in self.bus_left.motors.items():
            cal_left[name] = MotorCalibration(
                id=motor.id, drive_mode=0, homing_offset=homing_left[name], range_min=mins_left[name], range_max=maxs_left[name]
            )
        self.bus_left.write_calibration(cal_left)

        # Right arm
        self.bus_right.disable_torque()
        for name in self.right_arm_motors:
            self.bus_right.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move right arm joints to middle of range and press ENTER....")
        homing_right = self.bus_right.set_half_turn_homings(self.right_arm_motors)

        print("Move right arm joints through full ranges. Press ENTER to stop...")
        mins_right, maxs_right = self.bus_right.record_ranges_of_motion(self.right_arm_motors)

        cal_right = {}
        for name, motor in self.bus_right.motors.items():
            cal_right[name] = MotorCalibration(
                id=motor.id, drive_mode=0, homing_offset=homing_right[name], range_min=mins_right[name], range_max=maxs_right[name]
            )
        self.bus_right.write_calibration(cal_right)

        # Base wheels (fixed full-turn range)
        self.bus_base.disable_torque()
        for name in self.base_motors:
            self.bus_base.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        cal_base = {}
        for name, motor in self.bus_base.motors.items():
            cal_base[name] = MotorCalibration(
                id=motor.id, drive_mode=0, homing_offset=0, range_min=0, range_max=4095
            )
        self.bus_base.write_calibration(cal_base)

        self.calibration = {**cal_left, **cal_right, **cal_base}
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self):
        # Configure and set operating modes
        self.bus_left.disable_torque()
        self.bus_left.configure_motors()
        self.bus_right.disable_torque()
        self.bus_right.configure_motors()
        self.bus_base.disable_torque()
        self.bus_base.configure_motors()

        for name in self.left_arm_motors:
            self.bus_left.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.bus_left.write("P_Coefficient", name, 16)
            self.bus_left.write("I_Coefficient", name, 0)
            self.bus_left.write("D_Coefficient", name, 43)

        for name in self.right_arm_motors:
            self.bus_right.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.bus_right.write("P_Coefficient", name, 16)
            self.bus_right.write("I_Coefficient", name, 0)
            self.bus_right.write("D_Coefficient", name, 43)

        for name in self.base_motors:
            self.bus_base.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        self.bus_left.enable_torque()
        self.bus_right.enable_torque()
        self.bus_base.enable_torque()

    # -------- base kinematics (same as XLerobot pattern) --------
    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_int = int(round(degps * steps_per_deg))
        return max(min(speed_int, 0x7FFF), -0x8000)

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        return raw_speed / steps_per_deg

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        theta_rad = theta * (np.pi / 180.0)
        velocity_vector = np.array([x, y, theta_rad])
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            wheel_degps *= (max_raw / max_raw_computed)

        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]
        return {"base_left_wheel": wheel_raw[0], "base_back_wheel": wheel_raw[1], "base_right_wheel": wheel_raw[2]}

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        wheel_degps = np.array(
            [self._raw_to_degps(left_wheel_speed), self._raw_to_degps(back_wheel_speed), self._raw_to_degps(right_wheel_speed)]
        )
        wheel_radps = wheel_degps * (np.pi / 180.0)
        wheel_linear_speeds = wheel_radps * wheel_radius

        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        velocity_vector = np.linalg.inv(m).dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)
        return {"x.vel": float(x), "y.vel": float(y), "theta.vel": float(theta)}

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)

        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]
        theta_speed = speed_setting["theta"]

        x_cmd = y_cmd = theta_cmd = 0.0
        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed

        return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}

    # -------- I/O --------
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_arm_pos = self.bus_left.sync_read("Present_Position", self.left_arm_motors)
        right_arm_pos = self.bus_right.sync_read("Present_Position", self.right_arm_motors)
        base_wheel_vel = self.bus_base.sync_read("Present_Velocity", self.base_motors)

        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )

        left_arm_state = {f"{k}.pos": v for k, v in left_arm_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_arm_pos.items()}

        camera_obs = self.get_camera_observation()
        return {**left_arm_state, **right_arm_state, **base_vel, **camera_obs}

    def get_camera_observation(self):
        obs_dict = {}
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_arm_pos = {k: v for k, v in action.items() if k.startswith("left_arm_") and k.endswith(".pos")}
        right_arm_pos = {k: v for k, v in action.items() if k.startswith("right_arm_") and k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel.get("x.vel", 0.0),
            base_goal_vel.get("y.vel", 0.0),
            base_goal_vel.get("theta.vel", 0.0),
        )

        if self.config.max_relative_target is not None:
            # present positions for clamp
            present_left = self.bus_left.sync_read("Present_Position", self.left_arm_motors)
            present_right = self.bus_right.sync_read("Present_Position", self.right_arm_motors)
            present = {**present_left, **present_right}

            # Build "goal, present" pairs keyed by raw motor name (strip '.pos')
            goal_present = {
                key.replace(".pos", ""): (g_pos, present[key.replace(".pos", "")])
                for key, g_pos in chain(left_arm_pos.items(), right_arm_pos.items())
            }
            safe = ensure_safe_goal_position(goal_present, self.config.max_relative_target)

            # Put back into .pos dicts
            left_arm_pos = {f"{k}.pos": v for k, v in safe.items() if k.startswith("left_arm_")}
            right_arm_pos = {f"{k}.pos": v for k, v in safe.items() if k.startswith("right_arm_")}

        left_raw = {k.replace(".pos", ""): v for k, v in left_arm_pos.items()}
        right_raw = {k.replace(".pos", ""): v for k, v in right_arm_pos.items()}

        if left_raw:
            self.bus_left.sync_write("Goal_Position", left_raw)
        if right_raw:
            self.bus_right.sync_write("Goal_Position", right_raw)
        if base_wheel_goal_vel:
            self.bus_base.sync_write("Goal_Velocity", base_wheel_goal_vel)

        return {**left_arm_pos, **right_arm_pos, **base_goal_vel}

    def stop_base(self):
        self.bus_base.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()

        self.bus_left.disconnect(self.config.disable_torque_on_disconnect)
        self.bus_right.disconnect(self.config.disable_torque_on_disconnect)
        self.bus_base.disconnect(self.config.disable_torque_on_disconnect)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")