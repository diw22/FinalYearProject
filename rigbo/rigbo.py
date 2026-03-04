# lerobot/robots/rigbo/rigbo.py

import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rigbo import RigboConfig

logger = logging.getLogger(__name__)


class Rigbo(Robot):
    """
    Rigbo: single arm + omni base (no head).

    Goal: behave as close to the XLerobot keyboard teleop stack as possible.
    - action space: left_arm_*.pos + x.vel/y.vel/theta.vel
    - observation:  left_arm_*.pos + x.vel/y.vel/theta.vel
    """

    config_class = RigboConfig
    name = "rigbo"

    def __init__(self, config: RigboConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys

        # Match the XLerobot speed ladder exactly
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0

        arm_norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # --- Single Arm (use LEFT naming to match teleop script and XLerobot convention) ---
        self.bus_arm = FeetechMotorsBus(
            port=config.port_arm,
            motors={
                "left_arm_shoulder_pan": Motor(1, "sts3215", arm_norm_mode),
                "left_arm_shoulder_lift": Motor(2, "sts3215", arm_norm_mode),
                "left_arm_elbow_flex": Motor(3, "sts3215", arm_norm_mode),
                "left_arm_wrist_flex": Motor(4, "sts3215", arm_norm_mode),
                "left_arm_wrist_roll": Motor(5, "sts3215", arm_norm_mode),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # --- Base (3-wheel omni) ---
        self.bus_base = FeetechMotorsBus(
            port=config.port_base,
            motors={
                "base_left_wheel": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )

        self.left_arm_motors = list(self.bus_arm.motors.keys())
        self.base_motors = list(self.bus_base.motors.keys())

        self.cameras = make_cameras_from_configs(config.cameras)

    # ---------------- Features ----------------

    @cached_property
    def observation_features(self):
        # EXACT contract expected by your teleop script + “XLerobot style”
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

        # Configure modes BEFORE torque (prevents weird behavior)
        self.configure()

        self.bus_arm.enable_torque()
        self.bus_base.enable_torque()

    def configure(self):
        """
        Keep this consistent with the baseline behavior:
        - Arm in POSITION mode
        - Base in VELOCITY mode
        - Basic PID values for position control (safe defaults)
        """
        if not (self.bus_arm.is_connected and self.bus_base.is_connected):
            # allow calling pre-connect if someone wants, but do nothing
            return

        # Arm: position mode + PID
        self.bus_arm.sync_write(
            "Operating_Mode",
            dict.fromkeys(self.left_arm_motors, OperatingMode.POSITION),
            num_retry=5,
        )
        # These match typical baseline tuning patterns; adjust only if your hardware requires it
        self.bus_arm.sync_write("Position_P_Gain", dict.fromkeys(self.left_arm_motors, 16), num_retry=5)
        self.bus_arm.sync_write("Position_I_Gain", dict.fromkeys(self.left_arm_motors, 0), num_retry=5)
        self.bus_arm.sync_write("Position_D_Gain", dict.fromkeys(self.left_arm_motors, 43), num_retry=5)

        # Base: velocity mode
        self.bus_base.sync_write(
            "Operating_Mode",
            dict.fromkeys(self.base_motors, OperatingMode.VELOCITY),
            num_retry=5,
        )

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("Rigbo not connected")

        self.bus_arm.disconnect(self.config.disable_torque_on_disconnect)
        self.bus_base.disconnect(self.config.disable_torque_on_disconnect)

    # ---------------- Base Kinematics (IDENTICAL TO BASELINE STYLE) ----------------

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap to signed 16-bit range
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
        """
        Convert body-frame velocities:
          x (m/s), y (m/s), theta (deg/s)
        into wheel raw commands for:
          base_left_wheel, base_back_wheel, base_right_wheel
        """
        theta_rad = theta * (np.pi / 180.0)
        velocity_vector = np.array([x, y, theta_rad])

        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        wheel_linear_speeds = m.dot(velocity_vector)          # m/s
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius  # rad/s
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)  # deg/s

        # Scale if any wheel exceeds max_raw in steps
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
        """
        Convert wheel raw velocities back into body-frame:
          x.vel (m/s), y.vel (m/s), theta.vel (deg/s)
        """
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )
        wheel_radps = wheel_degps * (np.pi / 180.0)
        wheel_linear_speeds = wheel_radps * wheel_radius  # m/s

        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)

        return {"x.vel": float(x), "y.vel": float(y), "theta.vel": float(theta)}

    # ---------------- Keyboard Base Mapping (IDENTICAL PATTERN) ----------------

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        # Speed control (match baseline behavior)
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

    # ---------------- I/O ----------------

    def get_observation(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("Rigbo not connected")

        arm = self.bus_arm.sync_read("Present_Position", self.left_arm_motors)
        base = self.bus_base.sync_read("Present_Velocity", self.base_motors)

        # Convert wheel raw velocities -> body velocities
        body_vel = self._wheel_raw_to_body(
            base["base_left_wheel"],
            base["base_back_wheel"],
            base["base_right_wheel"],
        )

        return {
            **{f"{k}.pos": float(v) for k, v in arm.items()},
            **body_vel,
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Writes:
          - left_arm_*.pos -> Goal_Position (optionally clamped)
          - x.vel/y.vel/theta.vel -> converted -> wheel Goal_Velocity

        Returns the action actually sent (clamped if max_relative_target is set).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("Rigbo not connected")

        # Separate arm/base portions (keep exact key contract)
        left_arm_pos = {k: v for k, v in action.items() if k.startswith("left_arm_") and k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        # Convert base body velocities to wheel raw
        base_wheel_goal_vel = self._body_to_wheel_raw(
            float(base_goal_vel.get("x.vel", 0.0)),
            float(base_goal_vel.get("y.vel", 0.0)),
            float(base_goal_vel.get("theta.vel", 0.0)),
        )

        # Optional clamp (match baseline semantics)
        if self.config.max_relative_target is not None and left_arm_pos:
            present_pos_left = self.bus_arm.sync_read("Present_Position", self.left_arm_motors)
            # Build (goal, present) mapping for ensure_safe_goal_position
            goal_present_pos = {
                key: (float(g_pos), float(present_pos_left[key.replace(".pos", "")]))
                for key, g_pos in left_arm_pos.items()
            }
            safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            left_arm_pos = {k: v for k, v in safe_goal_pos.items()}

        # Strip ".pos" for bus write
        left_arm_pos_raw = {k.replace(".pos", ""): v for k, v in left_arm_pos.items()}

        if left_arm_pos_raw:
            self.bus_arm.sync_write("Goal_Position", left_arm_pos_raw)
        if base_wheel_goal_vel:
            self.bus_base.sync_write("Goal_Velocity", base_wheel_goal_vel)

        # Return action actually sent (same pattern as baseline)
        return {**left_arm_pos, **base_goal_vel}

    def stop_base(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("Rigbo not connected")
        self.bus_base.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")