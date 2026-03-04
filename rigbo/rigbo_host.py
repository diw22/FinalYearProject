#!/usr/bin/env python

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

import base64
import json
import logging
import time

import cv2
import zmq

from .rigbo import Rigbo
from .config_rigbo import RigboConfig, RigboHostConfig


class RigboHost:
    """
    Near 1:1 structure with XLerobotHost:
    - ZMQ PULL for commands, ZMQ PUSH for observations
    - CONFLATE on both sockets
    - watchdog to stop the base if no commands arrive
    """
    def __init__(self, config: RigboHostConfig):
        self.zmq_context = zmq.Context()

        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def _stop_base_safe(robot: Rigbo):
    """
    XLerobotHost calls robot.stop_base(). Rigbo might or might not implement it.
    If it doesn't, fall back to sending zero velocities using the same keys your client uses.
    """
    if hasattr(robot, "stop_base") and callable(getattr(robot, "stop_base")):
        robot.stop_base()
        return

    # Fallback: set base velocities to zero (only if your Rigbo action space supports these keys)
    try:
        robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    except Exception:
        # Last resort: do nothing (better than crashing host loop)
        pass


def main():
    logging.info("Configuring Rigbo")
    robot_config = RigboConfig(id="rigbo_host")
    robot = Rigbo(robot_config)

    logging.info("Connecting Rigbo")
    robot.connect()

    logging.info("Starting Rigbo HostAgent")
    host_config = RigboHostConfig()
    host = RigboHost(host_config)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")

    try:
        start = time.perf_counter()
        duration = 0.0
        while duration < host.connection_time_s:
            loop_start_time = time.time()

            # 1) Receive latest command (non-blocking)
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                _action_sent = robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            # 2) Watchdog: stop base if commands stall
            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                _stop_base_safe(robot)

            # 3) Fetch observation
            last_observation = robot.get_observation()

            # 4) Encode camera frames to base64 like XLerobotHost
            #    (Your client expects base64-jpg strings for camera keys, and ignores non-camera keys.)
            cameras = getattr(robot, "cameras", {}) or {}
            for cam_key, _cam in cameras.items():
                try:
                    ret, buffer = cv2.imencode(
                        ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    )
                    if ret:
                        last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                    else:
                        last_observation[cam_key] = ""
                except Exception:
                    last_observation[cam_key] = ""

            # 5) Send observation to remote client (non-blocking)
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")

            # 6) Loop timing
            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0.0))
            duration = time.perf_counter() - start

        logging.info("Cycle time reached.")

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Exiting...")
    finally:
        logging.info("Shutting down Rigbo Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished Rigbo Host cleanly")


if __name__ == "__main__":
    main()