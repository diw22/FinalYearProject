from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


def rigbo_cameras_config() -> dict[str, CameraConfig]:
    return {}


@RobotConfig.register_subclass("rigbo")
@dataclass
class RigboConfig(RobotConfig):
    port_arm: str = "/dev/ttyACM0"
    port_base: str = "/dev/ttyACM1"

    disable_torque_on_disconnect: bool = True
    max_relative_target: int | None = None
    use_degrees: bool = False

    cameras: dict[str, CameraConfig] = field(default_factory=rigbo_cameras_config)

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
class RigboHostConfig:
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    connection_time_s: int = 3600
    watchdog_timeout_ms: int = 500
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("rigbo_client")
@dataclass
class RigboClientConfig(RobotConfig):
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5

    cameras: dict[str, CameraConfig] = field(default_factory=rigbo_cameras_config)