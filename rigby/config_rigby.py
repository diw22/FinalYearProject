# lerobot/robots/rigby/config_rigby.py
from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


def rigby_cameras_config() -> dict[str, CameraConfig]:
    return {
        # Add cameras if you want, or leave empty
        # "front": CameraConfig(...),
    }


@RobotConfig.register_subclass("rigby")
@dataclass
class RigbyConfig(RobotConfig):
    # 3 physical serial buses
    port_left: str = "/dev/ttyACM0"
    port_right: str = "/dev/ttyACM1"
    port_base: str = "/dev/ttyACM2"

    disable_torque_on_disconnect: bool = True
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=rigby_cameras_config)

    # match XLerobot’s default behavior
    use_degrees: bool = False

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            "forward": "i",
            "backward": "k",
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            "speed_up": "n",
            "speed_down": "m",
            "quit": "b",
        }
    )


@dataclass
class RigbyHostConfig:
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    connection_time_s: int = 3600
    watchdog_timeout_ms: int = 500
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("rigby_client")
@dataclass
class RigbyClientConfig(RobotConfig):
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            "forward": "i",
            "backward": "k",
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            "speed_up": "n",
            "speed_down": "m",
            "quit": "b",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=rigby_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5