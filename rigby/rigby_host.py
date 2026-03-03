# lerobot/robots/rigby/rigby_host.py
#!/usr/bin/env python
import base64
import json
import logging
import time

import cv2
import zmq

from .config_rigby import RigbyConfig, RigbyHostConfig
from .rigby import Rigby


class RigbyHost:
    def __init__(self, config: RigbyHostConfig):
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


def main():
    logging.info("Configuring Rigby")
    robot_config = RigbyConfig(id="rigby_host")
    robot = Rigby(robot_config)

    logging.info("Connecting Rigby")
    robot.connect()

    logging.info("Starting Rigby Host")
    host_config = RigbyHostConfig()
    host = RigbyHost(host_config)

    last_cmd_time = time.time()
    watchdog_active = False

    try:
        start = time.perf_counter()
        duration = 0.0
        while duration < host.connection_time_s:
            loop_start = time.time()

            # Receive latest command (CONFLATE keeps only most recent)
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                pass
            except Exception as e:
                logging.error("Command receive failed: %s", e)

            # Watchdog
            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"No command for > {host.watchdog_timeout_ms}ms. Stopping base."
                )
                watchdog_active = True
                robot.stop_base()

            obs = robot.get_observation()

            # Encode camera frames to base64 jpg strings
            for cam_key in robot.cameras.keys():
                ret, buffer = cv2.imencode(".jpg", obs[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                obs[cam_key] = base64.b64encode(buffer).decode("utf-8") if ret else ""

            try:
                host.zmq_observation_socket.send_string(json.dumps(obs), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            # Rate limit
            elapsed = time.time() - loop_start
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start

    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        host.disconnect()
        logging.info("Rigby host shut down cleanly")


if __name__ == "__main__":
    main()