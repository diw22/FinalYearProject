from __future__ import annotations

from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

from .config_rigbo_leader_keyboard import RigboLeaderKeyboardTeleopConfig


class RigboLeaderKeyboardTeleop(Teleoperator):
    config_class = RigboLeaderKeyboardTeleopConfig
    name = "rigbo_leader_keyboard"

    def __init__(self, config: RigboLeaderKeyboardTeleopConfig):
        super().__init__(config)
        self.config = config

        leader_cfg = SOLeaderTeleopConfig(
            id=config.leader_id,
            port=config.leader_port,
            use_degrees=config.leader_use_degrees,
        )
        keyboard_cfg = KeyboardTeleopConfig(
            id=config.keyboard_id,
        )

        self.leader = SOLeader(leader_cfg)
        self.keyboard = KeyboardTeleop(keyboard_cfg)

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "left_arm_shoulder_pan.pos": float,
            "left_arm_shoulder_lift.pos": float,
            "left_arm_elbow_flex.pos": float,
            "left_arm_wrist_flex.pos": float,
            "left_arm_wrist_roll.pos": float,
            "left_arm_gripper.pos": float,
            "x.vel": float,
            "y.vel": float,
            "theta.vel": float,
        }
    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.leader.is_connected and self.keyboard.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.leader.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.leader.connect(calibrate=calibrate)
        self.keyboard.connect()

    def calibrate(self) -> None:
        # keyboard does not need calibration
        if not self.leader.is_calibrated:
            self.leader.calibrate()

    def configure(self) -> None:
        # delegate to leader if needed
        self.leader.configure()

    def disconnect(self) -> None:
        try:
            self.keyboard.disconnect()
        finally:
            self.leader.disconnect()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # no-op for now
        return

    def _map_arm_action(self, leader_action: dict[str, Any]) -> dict[str, Any]:
        mapped = {}
        for key, value in leader_action.items():
            out_key = self.config.arm_key_map.get(key)
            if out_key is not None:
                mapped[out_key] = value
        return mapped

    def _keyboard_to_base_action(self, pressed_keys: set[str]) -> dict[str, float]:
        x = 0.0
        y = 0.0
        theta = 0.0

        if self.config.forward_key in pressed_keys:
            x += self.config.xy_speed
        if self.config.backward_key in pressed_keys:
            x -= self.config.xy_speed
        if self.config.left_key in pressed_keys:
            y += self.config.xy_speed
        if self.config.right_key in pressed_keys:
            y -= self.config.xy_speed
        if self.config.rotate_left_key in pressed_keys:
            theta += self.config.theta_speed
        if self.config.rotate_right_key in pressed_keys:
            theta -= self.config.theta_speed

        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }

    def get_action(self) -> dict[str, Any]:
        leader_action = self.leader.get_action()
        arm_action = self._map_arm_action(leader_action)

        keyboard_action = self.keyboard.get_action()
        pressed_keys = set(keyboard_action.keys())
        base_action = self._keyboard_to_base_action(pressed_keys)
        if arm_action:
            print("leader raw:", leader_action)
            print("leader mapped:", arm_action)

        return {**arm_action, **base_action}
