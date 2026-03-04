#!/usr/bin/env python

import logging
import time
from functools import cached_property
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rigbo import RigboConfig

logger = logging.getLogger(__name__)


class Rigbo(Robot):
    """
    One arm + one omniwheel base.

    Goal: be as close to xlerobot.py as possible.
    Changes vs XLerobot:
      - Removed right arm
      - Removed head
      - Kept base + keyboard teleop contract identical (x.vel/y.vel/theta.vel)
      - Kept state/action naming identical for a single arm: left_arm_*.pos
    """

    config_class = RigboConfig
    name = "rigbo"

    def __init__(self, config: RigboConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys

        # Define three speed levels and a current index (same as xlerobot.py)
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0  # Start at slow

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # Match xlerobot.py behavior: if calibration contains a key, build a calibration dict scoped to this bus
        if self.calibration.get("left_arm_shoulder_pan") is not None:
            calibration1 = {
                "left_arm_shoulder_pan": self.calibration.get("left_arm_shoulder_pan"),
                "left_arm_shoulder_lift": self.calibration.get("left_arm_shoulder_lift"),
                "left_arm_elbow_flex": self.calibration.get("left_arm_elbow_flex"),
                "left_arm_wrist_flex": self.calibration.get("left_arm_wrist_flex"),
                "left_arm_wrist_roll": self.calibration.get("left_arm_wrist_roll"),
                "left_arm_gripper": self.calibration.get("left_arm_gripper"),
            }
        else:
            calibration1 = self.calibration

        # BUS 1: left arm
        self.bus1 = FeetechMotorsBus(
            port=self.config.port_arm,
            motors={
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration1,
        )

        if self.calibration.get("base_left_wheel") is not None:
            calibration2 = {
                "base_left_wheel": self.calibration.get("base_left_wheel"),
                "base_back_wheel": self.calibration.get("base_back_wheel"),
                "base_right_wheel": self.calibration.get("base_right_wheel"),
            }
        else:
            calibration2 = self.calibration

        # BUS 2: base
        self.bus2 = FeetechMotorsBus(
            port=self.config.port_base,
            motors={
                "base_left_wheel": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=calibration2,
        )

        self.left_arm_motors = [motor for motor in self.bus1.motors if motor.startswith("left_arm")]
        self.base_motors = [motor for motor in self.bus2.motors if motor.startswith("base")]

        self.cameras = make_cameras_from_configs(config.cameras)

    # ---------- Feature typing (same pattern as xlerobot.py) ----------

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
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus1.is_connected and self.bus2.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    # ---------- Connect / calibration flow (mirrors xlerobot.py) ----------

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus1.connect()
        self.bus2.connect()

        # Check if calibration file exists and ask user if they want to restore it
        if self.calibration_fpath.is_file():
            logger.info(f"Calibration file found at {self.calibration_fpath}")
            user_input = input(
                "Press ENTER to restore calibration from file, or type 'c' and press ENTER to run manual calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Attempting to restore calibration from file...")
                try:
                    # Load calibration data into bus memory
                    self.bus1.calibration = {k: v for k, v in self.calibration.items() if k in self.bus1.motors}
                    self.bus2.calibration = {k: v for k, v in self.calibration.items() if k in self.bus2.motors}
                    logger.info("Calibration data loaded into bus memory successfully!")

                    # Write calibration data to motors
                    self.bus1.write_calibration({k: v for k, v in self.calibration.items() if k in self.bus1.motors})
                    self.bus2.write_calibration({k: v for k, v in self.calibration.items() if k in self.bus2.motors})
                    logger.info("Calibration restored successfully from file!")

                except Exception as e:
                    logger.warning(f"Failed to restore calibration from file: {e}")
                    if calibrate:
                        logger.info("Proceeding with manual calibration...")
                        self.calibrate()
            else:
                logger.info("User chose manual calibration...")
                if calibrate:
                    self.calibrate()
        elif calibrate:
            logger.info("No calibration file found, proceeding with manual calibration...")
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus1.is_calibrated and self.bus2.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")

        # ---- calibrate left arm (bus1) ----
        left_motors = self.left_arm_motors
        self.bus1.disable_torque()
        for name in left_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move left arm motors to the middle of their range of motion and press ENTER....")

        homing_offsets = self.bus1.set_half_turn_homings(left_motors)

        print(
            "Move all left arm joints sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus1.record_ranges_of_motion(left_motors)

        calibration_left: dict[str, MotorCalibration] = {}
        for name, motor in self.bus1.motors.items():
            calibration_left[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.bus1.write_calibration(calibration_left)

        # ---- calibrate base (bus2) ----
        # Wheels are full-turn: assign a known full range and homing offset 0 (same approach used in xlerobot.py for wheels)
        self.bus2.disable_torque()
        for name in self.base_motors:
            # keep base in velocity mode normally; during calibration torque is off anyway
            pass

        calibration_base: dict[str, MotorCalibration] = {}
        for name, motor in self.bus2.motors.items():
            calibration_base[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                range_min=0,
                range_max=4095,
            )

        self.bus2.write_calibration(calibration_base)

        # Save merged calibration like xlerobot.py
        self.calibration = {**calibration_left, **calibration_base}
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self):
        # Match xlerobot.py structure: disable torque, configure_motors, then set modes and coefficients.

        # bus 1 (arm)
        self.bus1.disable_torque()
        self.bus1.configure_motors()

        for name in self.left_arm_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Coefficients consistent with xlerobot.py
            self.bus1.write("P_Coefficient", name, 16)
            self.bus1.write("I_Coefficient", name, 0)
            self.bus1.write("D_Coefficient", name, 43)

        # bus 2 (base)
        self.bus2.disable_torque()
        self.bus2.configure_motors()

        for name in self.base_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        # enable torque after config
        self.bus1.enable_torque()
        self.bus2.enable_torque()

    # ---------- Base kinematics (copied in style from xlerobot.py) ----------

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
        elif speed_int < -0x8000:
            speed_int = -0x8000
        return speed_int

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
    ) -> dict[str, int]:
        theta_rad = theta * (np.pi / 180.0)
        velocity_vector = np.array([x, y, theta_rad])

        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats) if raw_floats else 0.0
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]
        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed: int,
        back_wheel_speed: int,
        right_wheel_speed: int,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, float]:
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )
        wheel_radps = wheel_degps * (np.pi / 180.0)
        wheel_linear_speeds = wheel_radps * wheel_radius

        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        m_inv = np.linalg.inv(m)

        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)

        return {"x.vel": float(x), "y.vel": float(y), "theta.vel": float(theta)}

    # ---------- Keyboard mapping (same pattern as xlerobot.py) ----------

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray) -> dict[str, float]:
        # speed up/down (missing in your current rigbo.py)
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)

        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]
        theta_speed = speed_setting["theta"]

        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0

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

    # ---------- Observation / action IO (same layout as xlerobot.py) ----------

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        left_arm_pos = self.bus1.sync_read("Present_Position", self.left_arm_motors)
        base_wheel_vel = self.bus2.sync_read("Present_Velocity", self.base_motors)

        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )

        left_arm_state = {f"{k}.pos": v for k, v in left_arm_pos.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        camera_obs = self.get_camera_observation()
        obs_dict = {**left_arm_state, **base_vel, **camera_obs}
        return obs_dict

    def get_camera_observation(self):
        obs_dict = {}
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Mirrors xlerobot.py:
        - extracts left_arm_*.pos and base .vel
        - converts body vel -> wheel raw
        - clamps if max_relative_target set
        - writes Goal_Position / Goal_Velocity
        - returns action actually sent
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_arm_pos = {k: v for k, v in action.items() if k.startswith("left_arm_") and k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel.get("x.vel", 0.0),
            base_goal_vel.get("y.vel", 0.0),
            base_goal_vel.get("theta.vel", 0.0),
        )

        if self.config.max_relative_target is not None:
            present_pos_left = self.bus1.sync_read("Present_Position", self.left_arm_motors)
            goal_present_pos = {key: (g_pos, present_pos_left[key.replace(".pos", "")]) for key, g_pos in left_arm_pos.items()}
            safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            left_arm_pos = safe_goal_pos

        left_arm_pos_raw = {k.replace(".pos", ""): v for k, v in left_arm_pos.items()}

        if left_arm_pos_raw:
            self.bus1.sync_write("Goal_Position", left_arm_pos_raw)
        if base_wheel_goal_vel:
            self.bus2.sync_write("Goal_Velocity", base_wheel_goal_vel)

        return {
            **left_arm_pos,
            **base_goal_vel,
        }

    def stop_base(self):
        self.bus2.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.bus1.disconnect(self.config.disable_torque_on_disconnect)
        self.bus2.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")