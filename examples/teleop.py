# To run on the host (Rigby):
"""
PYTHONPATH=src python -m lerobot.robots.rigby.rigby_host --robot.id=rigby_host
"""

# To run teleop:
"""
PYTHONPATH=src python -m examples.rigby.teleoperate_keyboard
"""

import time
import numpy as np
import math

from lerobot.robots.rigby import RigbyConfig, Rigby
# For ZMQ connection (uncomment if you want to run via host/client)
# from lerobot.robots.rigby import RigbyClient, RigbyClientConfig

from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.kinematics.so101_kinematics import SO101Kinematics

from lerobot.utils.rerun import init_rerun, log_rerun_data


# ----------------------------
# Keymaps (NO HEAD)
# ----------------------------
LEFT_KEYMAP = {
    "shoulder_pan+": "q",
    "shoulder_pan-": "e",
    "wrist_roll+": "r",
    "wrist_roll-": "f",
    "gripper+": "t",
    "gripper-": "g",
    "x+": "w",
    "x-": "s",
    "y+": "a",
    "y-": "d",
    "pitch+": "z",
    "pitch-": "x",
    "reset": "c",
    "triangle": "y",  # Rectangle trajectory key
}

RIGHT_KEYMAP = {
    "shoulder_pan+": "7",
    "shoulder_pan-": "9",
    "wrist_roll+": "/",
    "wrist_roll-": "*",
    "gripper+": "+",
    "gripper-": "-",
    "x+": "8",
    "x-": "2",
    "y+": "4",
    "y-": "6",
    "pitch+": "1",
    "pitch-": "3",
    "reset": "0",
    "triangle": "Y",  # Rectangle trajectory key
}

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}

RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}


# ----------------------------
# Helpers
# ----------------------------
def busy_wait(dt: float):
    """Spin-wait for dt seconds. (Optional; can be replaced with time.sleep)"""
    start = time.perf_counter()
    while time.perf_counter() - start < dt:
        pass


class RectangularTrajectory:
    def __init__(self, width=0.10, height=0.10, segment_duration=1.0):
        self.width = width
        self.height = height
        self.segment_duration = segment_duration
        self.total_duration = 4 * segment_duration

    def get_position(self, t: float):
        if t >= self.total_duration:
            t = self.total_duration

        segment = int(t // self.segment_duration)
        segment_t = (t % self.segment_duration) / self.segment_duration

        # smooth interpolation (cosine)
        s = 0.5 * (1 - math.cos(math.pi * segment_t))

        if segment == 0:  # right
            return (s * self.width, 0)
        elif segment == 1:  # up
            return (self.width, s * self.height)
        elif segment == 2:  # left
            return (self.width * (1 - s), self.height)
        else:  # down
            return (0, self.height * (1 - s))


class SimpleTeleopArm:
    def __init__(self, kinematics, joint_map, initial_obs, prefix="left"):
        self.kinematics = kinematics
        self.joint_map = joint_map
        self.prefix = prefix

        # These are the same "task-space" defaults you had
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.xy_step = 0.0081

        self.degree_step = 3
        self.pitch = 0.0

        # Targets
        self.target_positions = {}
        for key, motor_name in self.joint_map.items():
            obs_key = f"{motor_name}.pos"
            self.target_positions[key] = float(initial_obs.get(obs_key, 0.0))

        # Ensure we have wrist_flex target present
        if "wrist_flex" not in self.target_positions:
            self.target_positions["wrist_flex"] = 0.0

        self.rect_traj = RectangularTrajectory(width=0.10, height=0.10, segment_duration=1.0)

    def handle_keys(self, key_state: dict):
        # Shoulder pan
        if key_state.get("shoulder_pan+"):
            self.target_positions["shoulder_pan"] += self.degree_step
        if key_state.get("shoulder_pan-"):
            self.target_positions["shoulder_pan"] -= self.degree_step

        # Wrist roll
        if key_state.get("wrist_roll+"):
            self.target_positions["wrist_roll"] += self.degree_step
        if key_state.get("wrist_roll-"):
            self.target_positions["wrist_roll"] -= self.degree_step

        # Gripper
        if key_state.get("gripper+"):
            self.target_positions["gripper"] += self.degree_step
        if key_state.get("gripper-"):
            self.target_positions["gripper"] -= self.degree_step

        # Pitch (affects wrist_flex coupling)
        if key_state.get("pitch+"):
            self.pitch += self.degree_step
        if key_state.get("pitch-"):
            self.pitch -= self.degree_step

        # Task-space XY
        if key_state.get("x+"):
            self.current_x += self.xy_step
        if key_state.get("x-"):
            self.current_x -= self.xy_step
        if key_state.get("y+"):
            self.current_y += self.xy_step
        if key_state.get("y-"):
            self.current_y -= self.xy_step

        # IK update (planar)
        joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
        self.target_positions["shoulder_lift"] = float(joint2)
        self.target_positions["elbow_flex"] = float(joint3)

        # Wrist flex coupling
        self.target_positions["wrist_flex"] = -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + self.pitch

    def p_control_action(self, robot, kp=0.2):
        obs = robot.get_observation()
        action = {}
        for joint_name, motor_name in self.joint_map.items():
            obs_key = f"{motor_name}.pos"
            cur = float(obs.get(obs_key, 0.0))
            tgt = float(self.target_positions[joint_name])
            cmd = cur + kp * (tgt - cur)
            action[obs_key] = cmd
        return action

    def move_to_zero_position(self, robot, kp=0.2):
        for k in self.target_positions:
            self.target_positions[k] = 0.0
        # reset task-space defaults + pitch
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0

        # force wrist_flex to 0 explicitly
        self.target_positions["wrist_flex"] = 0.0

        action = self.p_control_action(robot, kp=kp)
        robot.send_action(action)
        print(f"[{self.prefix.upper()}] Moved toward zero.")

    def execute_rectangular_trajectory(self, robot, fps=50, kp=0.2):
        start_time = time.time()
        print(f"[{self.prefix.upper()}] Executing rectangular trajectory...")
        while True:
            t = time.time() - start_time
            if t > self.rect_traj.total_duration:
                break

            dx, dy = self.rect_traj.get_position(t)
            target_x = self.current_x + dx
            target_y = self.current_y + dy

            joint2, joint3 = self.kinematics.inverse_kinematics(target_x, target_y)
            self.target_positions["shoulder_lift"] = float(joint2)
            self.target_positions["elbow_flex"] = float(joint3)
            self.target_positions["wrist_flex"] = -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + self.pitch

            action = self.p_control_action(robot, kp=kp)
            robot.send_action(action)

            obs = robot.get_observation()
            log_rerun_data(obs, action)

            # Optional timing control
            # busy_wait(1.0 / fps)

        print(f"[{self.prefix.upper()}] Rectangle trajectory done.")


# ----------------------------
# Main
# ----------------------------
def main():
    FPS = 50

    # Choose one:
    # 1) Local/wired:
    robot_config = RigbyConfig()
    robot = Rigby(robot_config)

    # 2) ZMQ client (uncomment):
    # ip = "192.168.1.123"
    # robot_config = RigbyClientConfig(remote_ip=ip, id="rigby_client")
    # robot = RigbyClient(robot_config)

    try:
        robot.connect()
        print("[MAIN] Connected to Rigby")
    except Exception as e:
        print(f"[MAIN] Failed to connect to Rigby: {e}")
        print(robot_config)
        print(robot)
        return

    init_rerun(session_name="rigby_teleop_keyboard")

    keyboard = KeyboardTeleop(KeyboardTeleopConfig())
    keyboard.connect()

    obs = robot.get_observation()
    left_arm = SimpleTeleopArm(SO101Kinematics(), LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = SimpleTeleopArm(SO101Kinematics(), RIGHT_JOINT_MAP, obs, prefix="right")

    # Move arms toward zero at start
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)

    try:
        while True:
            pressed_keys = set(keyboard.get_action().keys())

            left_key_state = {action: (key in pressed_keys) for action, key in LEFT_KEYMAP.items()}
            right_key_state = {action: (key in pressed_keys) for action, key in RIGHT_KEYMAP.items()}

            # Rectangle trajectory triggers
            if left_key_state.get("triangle"):
                print("[MAIN] Left rectangle triggered")
                left_arm.execute_rectangular_trajectory(robot, fps=FPS)
                continue

            if right_key_state.get("triangle"):
                print("[MAIN] Right rectangle triggered")
                right_arm.execute_rectangular_trajectory(robot, fps=FPS)
                continue

            # Resets
            if left_key_state.get("reset"):
                left_arm.move_to_zero_position(robot)
                continue

            if right_key_state.get("reset"):
                right_arm.move_to_zero_position(robot)
                continue

            # Update targets from held keys
            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)

            left_action = left_arm.p_control_action(robot)
            right_action = right_arm.p_control_action(robot)

            # Base action (same pattern as before)
            keyboard_keys = np.array(list(pressed_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

            action = {**left_action, **right_action, **base_action}
            robot.send_action(action)

            obs = robot.get_observation()
            log_rerun_data(obs, action)

            # Optional fixed-rate loop:
            # busy_wait(1.0 / FPS)

    finally:
        robot.disconnect()
        keyboard.disconnect()
        print("Teleoperation ended.")


if __name__ == "__main__":
    main()