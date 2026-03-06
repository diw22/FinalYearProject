from dataclasses import dataclass, field

from lerobot.teleoperators.teleoperator import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rigbo_leader_keyboard")
@dataclass
class RigboLeaderKeyboardTeleopConfig(TeleoperatorConfig):
    # SO101 leader settings
    leader_port: str = ""
    leader_id: str = "rigbo_leader"
    leader_use_degrees: bool = False

    # keyboard settings
    keyboard_id: str = "rigbo_keyboard"

    # base key bindings
    forward_key: str = "w"
    backward_key: str = "s"
    left_key: str = "a"
    right_key: str = "d"
    rotate_left_key: str = "q"
    rotate_right_key: str = "e"

    # base command magnitudes
    xy_speed: float = 0.2
    theta_speed: float = 60.0

    # leader motor names -> Rigbo action names
    arm_key_map: dict[str, str] = field(
        default_factory=lambda: {
	    "shoulder_pan.pos": "left_arm_shoulder_pan.pos",
	    "shoulder_lift.pos": "left_arm_shoulder_lift.pos",
	    "elbow_flex.pos": "left_arm_elbow_flex.pos",
	    "wrist_flex.pos": "left_arm_wrist_flex.pos",
	    "wrist_roll.pos": "left_arm_wrist_roll.pos",
	    "gripper.pos": "left_arm_gripper.pos",

	    "left_arm_shoulder_pan.pos": "left_arm_shoulder_pan.pos",
	    "left_arm_shoulder_lift.pos": "left_arm_shoulder_lift.pos",
	    "left_arm_elbow_flex.pos": "left_arm_elbow_flex.pos",
	    "left_arm_wrist_flex.pos": "left_arm_wrist_flex.pos",
	    "left_arm_wrist_roll.pos": "left_arm_wrist_roll.pos",
	    "left_arm_gripper.pos": "left_arm_gripper.pos",

	    # optional, keep these too if the leader ever emits arm_* names
	    "arm_shoulder_pan.pos": "left_arm_shoulder_pan.pos",
	    "arm_shoulder_lift.pos": "left_arm_shoulder_lift.pos",
	    "arm_elbow_flex.pos": "left_arm_elbow_flex.pos",
	    "arm_wrist_flex.pos": "left_arm_wrist_flex.pos",
	    "arm_wrist_roll.pos": "left_arm_wrist_roll.pos",
    	    "arm_gripper.pos": "left_arm_gripper.pos",
        }
    )
