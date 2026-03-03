# To run on the host (Rigbo):
"""
PYTHONPATH=src python -m lerobot.robots.rigbo.rigbo_host --robot.id=rigbo_host
"""

# To run teleop:
"""
PYTHONPATH=src python -m examples.rigbo.teleoperate_keyboard
"""

import time
import numpy as np
import math

from lerobot.robots.rigbo import RigboConfig, Rigbo
# For ZMQ connection (uncomment if you want to run via host/client)
# from lerobot.robots.rigbo import RigboClient, RigboClientConfig

from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.model.SO101Robot import SO101Kinematics

from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# ----------------------------
# Keymaps (SINGLE ARM)
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

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
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

        # Task-space defaults
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.xy_step = 0.0081

        self.degree_step = 3
        self.pitch = 0.0

        # Targets init from observation (prevents startup jump)
        self.target_positions = {}
        for key, motor_name in self.joint_map.items():
            obs_key = f"{motor_name}.pos"
            self.target_positions[key] = float(initial_obs.get(obs_key, 0.0))

        # Ensure wrist_flex target present
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
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            - self.target_positions["elbow_flex"]
            + self.pitch
        )

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
            self.target_positions["wrist_flex"] = (
                -self.target_positions["shoulder_lift"]
                - self.target_positions["elbow_flex"]
                + self.pitch
            )

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
    robot_config = RigboConfig()
    robot = Rigbo(robot_config)

    # 2) ZMQ client (uncomment):
    # ip = "192.168.1.123"
    # robot_config = RigboClientConfig(remote_ip=ip, id="rigbo_client")
    # robot = RigboClient(robot_config)

    try:
        robot.connect()
        print("[MAIN] Connected to Rigbo")
    except Exception as e:
        print(f"[MAIN] Failed to connect to Rigbo: {e}")
        print(robot_config)
        print(robot)
        return

    init_rerun(session_name="rigbo_teleop_keyboard_single_arm")

    keyboard = KeyboardTeleop(KeyboardTeleopConfig())
    keyboard.connect()

    obs = robot.get_observation()
    arm = SimpleTeleopArm(SO101Kinematics(), LEFT_JOINT_MAP, obs, prefix="left")

    # Move arm toward zero at start
    arm.move_to_zero_position(robot)

    try:
        while True:
            pressed_keys = set(keyboard.get_action().keys())

            key_state = {action: (key in pressed_keys) for action, key in LEFT_KEYMAP.items()}

            # Rectangle trajectory trigger
            if key_state.get("triangle"):
                print("[MAIN] Rectangle triggered")
                arm.execute_rectangular_trajectory(robot, fps=FPS)
                continue

            # Reset
            if key_state.get("reset"):
                arm.move_to_zero_position(robot)
                continue

            # Update targets from held keys
            arm.handle_keys(key_state)

            arm_action = arm.p_control_action(robot)

            # Base action (same pattern as before)
            keyboard_keys = np.array(list(pressed_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

            action = {**arm_action, **base_action}
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