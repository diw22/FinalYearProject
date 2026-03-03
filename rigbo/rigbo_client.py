import json
import zmq
from lerobot.robots.robot import Robot
from .config_rigbo import RigboClientConfig


class RigboClient(Robot):
    config_class = RigboClientConfig
    name = "rigbo_client"

    def __init__(self, config: RigboClientConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

    def connect(self):
        self.ctx = zmq.Context()
        self.cmd = self.ctx.socket(zmq.PUSH)
        self.cmd.connect(f"tcp://{self.config.remote_ip}:{self.config.port_zmq_cmd}")

        self.obs = self.ctx.socket(zmq.PULL)
        self.obs.connect(f"tcp://{self.config.remote_ip}:{self.config.port_zmq_observations}")

        self._connected = True

    def get_observation(self):
        return json.loads(self.obs.recv_string())

    def send_action(self, action):
        self.cmd.send_string(json.dumps(action))
        return action

    def disconnect(self):
        self.cmd.close()
        self.obs.close()
        self.ctx.term()
        self._connected = False