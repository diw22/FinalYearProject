#!/usr/bin/env python
from dataclasses import dataclass

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from lerobot.robots.rigby import RigbyClient, RigbyClientConfig  # ZMQ client -> host Rigby
# Or local:
# from lerobot.robots.rigby import Rigby, RigbyConfig


NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Rigby teleop demo task"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"


def _prefix_arm(action: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in action.items()}


@dataclass
class RigbyLeadersKeyboardTeleop:
    """
    A tiny adapter that looks like a teleoperator (connect/disconnect/is_connected/get_action),
    but internally merges:
      - left leader joints -> left_arm_*
      - right leader joints -> right_arm_*
      - keyboard -> base (x/y/theta vel)
    """
    robot: any
    left_leader: SO100Leader
    right_leader: SO100Leader
    keyboard: KeyboardTeleop

    @property
    def is_connected(self) -> bool:
        return self.left_leader.is_connected and self.right_leader.is_connected and self.keyboard.is_connected

    def connect(self) -> None:
        self.left_leader.connect()
        self.right_leader.connect()
        self.keyboard.connect()

    def disconnect(self) -> None:
        self.left_leader.disconnect()
        self.right_leader.disconnect()
        self.keyboard.disconnect()

    def get_action(self) -> dict:
        left_action = _prefix_arm(self.left_leader.get_action(), "left_arm")
        right_action = _prefix_arm(self.right_leader.get_action(), "right_arm")
        keyboard_keys = self.keyboard.get_action()
        base_action = self.robot._from_keyboard_to_base_action(keyboard_keys) or {}
        return {**left_action, **right_action, **base_action}


def main():
    # --- Rigby follower robot ---
    robot_config = RigbyClientConfig(remote_ip="192.168.1.10", id="rigby_client")
    robot = RigbyClient(robot_config)

    # Or local:
    # robot_config = RigbyConfig(id="rigby_local")
    # robot = Rigby(robot_config)

    # --- Leaders + keyboard ---
    left_leader = SO100Leader(SO100LeaderConfig(port="/dev/tty.usbmodemLEFT", id="left_leader"))
    right_leader = SO100Leader(SO100LeaderConfig(port="/dev/tty.usbmodemRIGHT", id="right_leader"))
    keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="keyboard"))

    teleop = RigbyLeadersKeyboardTeleop(robot=robot, left_leader=left_leader, right_leader=right_leader, keyboard=keyboard)

    # Processors (same pattern as example record.py) :contentReference[oaicite:6]{index=6}
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Dataset feature schema (same pattern) :contentReference[oaicite:7]{index=7}
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Connect
    robot.connect()
    teleop.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="rigby_record")

    if not robot.is_connected or not teleop.is_connected:
        raise RuntimeError("Robot or teleop is not connected!")

    print("Starting record loop...")
    recorded = 0
    while recorded < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {recorded + 1} / {NUM_EPISODES}")

        # Main record episode :contentReference[oaicite:8]{index=8}
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            teleop=teleop,  # single merged teleop
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Optional reset period between episodes (same idea as example) :contentReference[oaicite:9]{index=9}
        if not events["stop_recording"] and ((recorded < NUM_EPISODES - 1) or events["rerecord_episode"]):
            log_say("Resetting (not recorded as an episode)...")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded += 1

    # Cleanup
    log_say("Stop recording")
    robot.disconnect()
    teleop.disconnect()
    listener.stop()

    dataset.finalize()
    dataset.push_to_hub()


if __name__ == "__main__":
    main()