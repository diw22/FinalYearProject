from dataclasses import dataclass, field

from lerobot.teleoperators.teleoperator import TeleoperatorConfig
from lerobot.teleoperators.keyboard.config_keyboard import KeyboardTeleopConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig


@TeleoperatorConfig.register_subclass("rigbo_leader_keyboard")
@dataclass
class RigboLeaderKeyboardTeleopConfig(TeleoperatorConfig):
    leader: SOLeaderTeleopConfig = field(
        default_factory=lambda: SOLeaderTeleopConfig(type="so101_leader")
    )
    keyboard: KeyboardTeleopConfig = field(default_factory=KeyboardTeleopConfig)

    forward_key: str = "w"
    backward_key: str = "s"
    left_key: str = "a"
    right_key: str = "d"
    rotate_left_key: str = "q"
    rotate_right_key: str = "e"

    xy_speed: float = 0.2
    theta_speed: float = 60.0

    arm_key_map: dict[str, str] = field(
        default_factory=lambda: {
            "shoulder_pan.pos": "arm_shoulder_pan.pos",
            "shoulder_lift.pos": "arm_shoulder_lift.pos",
            "elbow_flex.pos": "arm_elbow_flex.pos",
            "wrist_flex.pos": "arm_wrist_flex.pos",
            "wrist_roll.pos": "arm_wrist_roll.pos",
            "gripper.pos": "arm_gripper.pos",
            # include prefixed variants too in case leader already uses them
            "arm_shoulder_pan.pos": "arm_shoulder_pan.pos",
            "arm_shoulder_lift.pos": "arm_shoulder_lift.pos",
            "arm_elbow_flex.pos": "arm_elbow_flex.pos",
            "arm_wrist_flex.pos": "arm_wrist_flex.pos",
            "arm_wrist_roll.pos": "arm_wrist_roll.pos",
            "arm_gripper.pos": "arm_gripper.pos",
        }
    )