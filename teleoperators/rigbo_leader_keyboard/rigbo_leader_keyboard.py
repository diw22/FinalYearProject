from __future__ import annotations

from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.keyboard.keyboard import KeyboardTeleop
from lerobot.teleoperators.so_leader.so_leader import SOLeader

from .config_rigbo_leader_keyboard import RigboLeaderKeyboardTeleopConfig


class RigboLeaderKeyboardTeleop(Teleoperator):
    config_class = RigboLeaderKeyboardTeleopConfig
    name = "rigbo_leader_keyboard"

    def __init__(self, config: RigboLeaderKeyboardTeleopConfig):
        super().__init__(config)
        self.config = config

        self.leader = SOLeader(config.leader)
        self.keyboard = KeyboardTeleop(config.keyboard)

    @property
    def is_connected(self) -> bool:
        return self.leader.is_connected and self.keyboard.is_connected

    def connect(self) -> None:
        self.leader.connect()
        self.keyboard.connect()

    def disconnect(self) -> None:
        # disconnect both even if one fails
        errors = []
        try:
            self.keyboard.disconnect()
        except Exception as e:
            errors.append(e)

        try:
            self.leader.disconnect()
        except Exception as e:
            errors.append(e)

        if errors:
            raise errors[0]

    def _map_arm_action(self, leader_action: dict[str, Any]) -> dict[str, Any]:
        mapped = {}
        for key, value in leader_action.items():
            out_key = self.config.arm_key_map.get(key)
            if out_key is not None:
                mapped[out_key] = value
        return mapped

    def _keyboard_to_base_action(self, pressed_keys: set[str]) -> dict[str, float]:
        x = y = theta = 0.0

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

        return {**arm_action, **base_action}