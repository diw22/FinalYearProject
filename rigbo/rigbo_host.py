import json
import time
import zmq

from .config_rigbo import RigboConfig, RigboHostConfig
from .rigbo import Rigbo


def main():
    robot = Rigbo(RigboConfig())
    robot.connect()

    host_cfg = RigboHostConfig()

    ctx = zmq.Context()
    cmd = ctx.socket(zmq.PULL)
    cmd.bind(f"tcp://*:{host_cfg.port_zmq_cmd}")

    obs = ctx.socket(zmq.PUSH)
    obs.bind(f"tcp://*:{host_cfg.port_zmq_observations}")

    try:
        while True:
            try:
                msg = cmd.recv_string(zmq.NOBLOCK)
                robot.send_action(json.loads(msg))
            except zmq.Again:
                pass

            observation = robot.get_observation()
            obs.send_string(json.dumps(observation))
            time.sleep(1 / host_cfg.max_loop_freq_hz)

    finally:
        robot.disconnect()
        cmd.close()
        obs.close()
        ctx.term()


if __name__ == "__main__":
    main()