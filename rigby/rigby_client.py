# lerobot/robots/rigby/rigby_client.py
import base64
import json
import logging
from functools import cached_property
from typing import Any, Dict, Optional

import cv2
import numpy as np
import zmq

from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rigby import RigbyClientConfig


class RigbyClient(Robot):
    config_class = RigbyClientConfig
    name = "rigby_client"

    def __init__(self, config: RigbyClientConfig):
        super().__init__(config)
        self.config = config

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations

        self.teleop_keys = config.teleop_keys
        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_frames: dict[str, Any] = {}
        self.last_remote_state: dict[str, Any] = {}

        self.speed_levels = [
            {"xy": 0.1, "theta": 30},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.3, "theta": 90},
        ]
        self.speed_index = 0

        self._is_connected = False

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        # no head motors
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

    @cached_property
    def _state_order(self) -> tuple[str, ...]:
        return tuple(self._state_ft.keys())

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError("RigbyClient already connected.")

        self.zmq_context = zmq.Context()

        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_cmd_socket.connect(f"tcp://{self.remote_ip}:{self.port_zmq_cmd}")
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_observation_socket.connect(f"tcp://{self.remote_ip}:{self.port_zmq_observations}")
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)

        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.connect_timeout_s * 1000))
        if self.zmq_observation_socket not in socks or socks[self.zmq_observation_socket] != zmq.POLLIN:
            raise DeviceNotConnectedError("Timeout waiting for Rigby host observations.")

        self._is_connected = True

    def configure(self):
        pass

    def calibrate(self):
        pass

    def _poll_and_get_latest_message(self) -> Optional[str]:
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.polling_timeout_ms))
        if self.zmq_observation_socket not in socks:
            return None

        last_msg = None
        while True:
            try:
                msg = self.zmq_observation_socket.recv_string(zmq.NOBLOCK)
                last_msg = msg
            except zmq.Again:
                break
        return last_msg

    def _parse_observation_json(self, obs_string: str) -> Optional[Dict[str, Any]]:
        try:
            obs = dict(json.loads(obs_string))
        except json.JSONDecodeError as e:
            logging.error(f"Bad observation JSON: {e}")
            return None

        frames = {}
        for cam_name in self.config.cameras.keys():
            b64 = obs.get(cam_name, "")
            if not b64:
                frames[cam_name] = None
                continue
            try:
                raw = base64.b64decode(b64)
                arr = np.frombuffer(raw, dtype=np.uint8)
                frames[cam_name] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception as e:
                logging.error(f"Frame decode failed for {cam_name}: {e}")
                frames[cam_name] = None

        for cam_name in self.config.cameras.keys():
            obs.pop(cam_name, None)

        return {"frames": frames, "state": obs}

    def _get_data(self):
        msg = self._poll_and_get_latest_message()
        if msg is None:
            return self.last_frames, self.last_remote_state

        parsed = self._parse_observation_json(msg)
        if parsed is None:
            return self.last_frames, self.last_remote_state

        self.last_frames = parsed["frames"]
        self.last_remote_state = parsed["state"]
        return self.last_frames, self.last_remote_state

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("RigbyClient not connected.")

        frames, state = self._get_data()
        obs = dict(state)

        for cam_name, frame in frames.items():
            if frame is None:
                frame = np.zeros((640, 480, 3), dtype=np.uint8)
            obs[cam_name] = frame

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("RigbyClient not connected.")

        self.zmq_cmd_socket.send_string(json.dumps(action))

        actions = np.array([action.get(k, 0.0) for k in self._state_order], dtype=np.float32)
        action_sent = {k: float(actions[i]) for i, k in enumerate(self._state_order)}
        action_sent["action"] = actions
        return action_sent

    def disconnect(self):
        if not self._is_connected:
            raise DeviceNotConnectedError("RigbyClient not connected.")

        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False