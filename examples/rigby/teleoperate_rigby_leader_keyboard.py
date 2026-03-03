#!/usr/bin/env python
import time

from lerobot.robots.rigby import RigbyClient, RigbyClientConfig  # ZMQ client -> host Rigby
# Or local Rigby hardware:
# from lerobot.robots.rigby import Rigby, RigbyConfig

from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def _prefix_arm(action: dict, prefix: str) -> dict:
    """
    Leader arm returns keys like 'shoulder_pan.pos', etc (see LeKiwi example),
    so we prefix into Rigby keys: 'left_arm_shoulder_pan.pos', etc. :contentReference[oaicite:2]{index=2}
    """
    return {f"{prefix}_{k}": v for k, v in action.items()}


def main():
    # --- Robot (Rigby follower) ---
    # ZMQ client (recommended if Rigby is on another machine running rigby_host)
    robot_config = RigbyClientConfig(remote_ip="192.168.1.10", id="rigby_client")
    robot = RigbyClient(robot_config)

    # Or local hardware (uncomment):
    # robot_config = RigbyConfig(id="rigby_local")
    # robot = Rigby(robot_config)

    # --- Teleoperators ---
    left_leader_cfg = SO100LeaderConfig(port="/dev/tty.usbmodemLEFT", id="left_leader")
    right_leader_cfg = SO100LeaderConfig(port="/dev/tty.usbmodemRIGHT", id="right_leader")
    keyboard_cfg = KeyboardTeleopConfig(id="keyboard")

    left_leader = SO100Leader(left_leader_cfg)
    right_leader = SO100Leader(right_leader_cfg)
    keyboard = KeyboardTeleop(keyboard_cfg)

    # Connect
    # (Rigby host should already be running if using client)
    robot.connect()
    left_leader.connect()
    right_leader.connect()
    keyboard.connect()

    init_rerun(session_name="rigby_leader_keyboard_teleop")

    if not robot.is_connected or not left_leader.is_connected or not right_leader.is_connected or not keyboard.is_connected:
        raise RuntimeError("Robot or teleop is not connected!")

    print("Starting Rigby teleop loop (leaders -> arms, keyboard -> base)...")
    while True:
        t0 = time.perf_counter()

        # Observe robot
        observation = robot.get_observation()

        # Leaders -> arms
        left_action = _prefix_arm(left_leader.get_action(), "left_arm")
        right_action = _prefix_arm(right_leader.get_action(), "right_arm")

        # Keyboard -> base
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        action = {**left_action, **right_action, **(base_action or {})}

        # Send
        _ = robot.send_action(action)

        # Visualize (same pattern as example teleoperate) :contentReference[oaicite:3]{index=3}
        log_rerun_data(observation=observation, action=action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()