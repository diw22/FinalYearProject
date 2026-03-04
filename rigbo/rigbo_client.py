# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE:
# This client mirrors XLerobotClient as closely as possible (ZMQ PUSH/PULL, CONFLATE,
# poll+drain to last msg, JSON obs, optional base64 camera frames).
# Adjust `_state_ft` keys if your rigbo_host uses different names.

import base64
import json
import logging
from functools import cached_property
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import zmq

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_rigbo import RigboConfig, RigboClientConfig  # expected to exist like xlerobot


class RigboClient(Robot):
    config_class = RigboClientConfig
    name = "rigbo_client"

    def __init__(self, config: RigboClientConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.robot_type = config.type

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations

        # If your Rigbo client config has teleop_keys like XLerobotClient, keep it.
        # Otherwise, it can be absent; we guard usage in _from_keyboard_to_base_action.
        self.teleop_keys = getattr(config, "teleop_keys", {})

        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_frames: Dict[str, np.ndarray] = {}
        self.last_remote_state: Dict[str, Any] = {}

        # Optional speed ladder (kept identical to XLerobotClient)
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0

        self._is_connected = False
        self._is_calibrated = True  # host generally owns calibration
        self.logs = {}

    # ---- Feature specs (mirror XLerobotClient) ----

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        """
        These keys MUST match what rigbo_host sends in its observation JSON
        and what it expects in its action JSON.

        This list is derived from your teleop script's motor naming (left arm)
        plus base velocity keys. :contentReference[oaicite:2]{index=2}
        """
        return dict.fromkeys(
            (
                "left_arm_shoulder_pan.pos",
                "left_arm_shoulder_lift.pos",
                "left_arm_elbow_flex.pos",
                "left_arm_wrist_flex.pos",
                "left_arm_wrist_roll.pos",
                "left_arm_gripper.pos",
                # Mobile base (if Rigbo has one; your teleop calls _from_keyboard_to_base_action)
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
        # Same convention as XLerobotClient: cameras stored in config.cameras as name -> cfg(height,width)
        cameras = getattr(self.config, "cameras", {}) or {}
        return {name: (cfg.height, cfg.width, 3) for name, cfg in cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    # ---- Robot interface ----

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        # If your framework expects calibration gating, you can toggle this.
        return self._is_calibrated

    def configure(self) -> None:
        # Keep as a no-op to satisfy abstract interface.
        return None

    def calibrate(self) -> None:
        # If host handles calibration, mark calibrated.
        self._is_calibrated = True

    def connect(self) -> None:
        """Establish ZMQ sockets with the remote Rigbo host."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "RigboClient is already connected. Do not run `robot.connect()` twice."
            )

        self.zmq_context = zmq.Context()

        # Commands: PUSH -> host PULL
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        zmq_cmd_locator = f"tcp://{self.remote_ip}:{self.port_zmq_cmd}"
        self.zmq_cmd_socket.connect(zmq_cmd_locator)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        # Observations: PULL <- host PUSH
        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        zmq_obs_locator = f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        self.zmq_observation_socket.connect(zmq_obs_locator)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)

        # Wait for first observation to confirm connectivity
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.connect_timeout_s * 1000))
        if self.zmq_observation_socket not in socks or socks[self.zmq_observation_socket] != zmq.POLLIN:
            raise DeviceNotConnectedError("Timeout waiting for Rigbo Host observation stream.")

        self._is_connected = True

    def disconnect(self) -> None:
        """Cleans ZMQ comms."""
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "RigboClient is not connected. You need to run `robot.connect()` before disconnecting."
            )

        try:
            if self.zmq_observation_socket is not None:
                self.zmq_observation_socket.close()
            if self.zmq_cmd_socket is not None:
                self.zmq_cmd_socket.close()
            if self.zmq_context is not None:
                self.zmq_context.term()
        finally:
            self.zmq_observation_socket = None
            self.zmq_cmd_socket = None
            self.zmq_context = None
            self._is_connected = False

    # ---- Observation handling (identical pattern) ----

    def _poll_and_get_latest_message(self) -> Optional[str]:
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)

        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logging.error(f"ZMQ polling error: {e}")
            return None

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
            return json.loads(obs_string)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON observation: {e}")
            return None

    def _decode_image_from_b64(self, image_b64: str) -> Optional[np.ndarray]:
        if not image_b64:
            return None
        try:
            jpg_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return frame
        except (TypeError, ValueError) as e:
            logging.error(f"Error decoding base64 image data: {e}")
            return None

    def _remote_state_from_obs(
        self, observation: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        flat_state = {key: observation.get(key, 0.0) for key in self._state_order}
        state_vec = np.array([flat_state[key] for key in self._state_order], dtype=np.float32)

        obs_dict: Dict[str, Any] = {**flat_state, "observation.state": state_vec}

        current_frames: Dict[str, np.ndarray] = {}
        for cam_name, image_b64 in observation.items():
            if cam_name not in self._cameras_ft:
                continue
            frame = self._decode_image_from_b64(image_b64)
            if frame is not None:
                current_frames[cam_name] = frame

        return current_frames, obs_dict

    def _get_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        latest_message_str = self._poll_and_get_latest_message()
        if latest_message_str is None:
            return self.last_frames, self.last_remote_state

        observation = self._parse_observation_json(latest_message_str)
        if observation is None:
            return self.last_frames, self.last_remote_state

        try:
            new_frames, new_state = self._remote_state_from_obs(observation)
        except Exception as e:
            logging.error(f"Error processing observation data, serving last observation: {e}")
            return self.last_frames, self.last_remote_state

        self.last_frames = new_frames
        self.last_remote_state = new_state
        return new_frames, new_state

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("RigboClient is not connected. You need to run `robot.connect()`.")

        frames, obs_dict = self._get_data()

        for cam_name, frame in frames.items():
            if frame is None:
                frame = np.zeros((640, 480, 3), dtype=np.uint8)
            obs_dict[cam_name] = frame

        return obs_dict

    # ---- Action handling (identical pattern) ----

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray) -> dict[str, float]:
        """
        Mirrors XLerobotClient base control. If Rigbo doesn't have a base, you can
        return {} or remove x/y/theta from _state_ft.
        """
        if not self.teleop_keys:
            return {}

        # Speed control
        if self.teleop_keys.get("speed_up") in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys.get("speed_down") in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)

        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]
        theta_speed = speed_setting["theta"]

        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0

        if self.teleop_keys.get("forward") in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys.get("backward") in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys.get("left") in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys.get("right") in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys.get("rotate_left") in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys.get("rotate_right") in pressed_keys:
            theta_cmd -= theta_speed

        return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("RigboClient is not connected. You need to run `robot.connect()`.")

        # Send action JSON as-is (motor space)
        self.zmq_cmd_socket.send_string(json.dumps(action))

        # Provide a numpy-packed action vector (same recording trick as XLerobotClient)
        actions = np.array([action.get(k, 0.0) for k in self._state_order], dtype=np.float32)

        action_sent = {key: actions[i] for i, key in enumerate(self._state_order)}
        action_sent["action"] = actions
        return action_sent