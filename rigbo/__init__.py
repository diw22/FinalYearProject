# lerobot/robots/rigbo/__init__.py

from .config_rigbo import RigboConfig, RigboHostConfig, RigboClientConfig
from .rigbo import Rigbo
from .rigbo_client import RigboClient

__all__ = [
    "RigboConfig",
    "RigboHostConfig",
    "RigboClientConfig",
    "Rigbo",
    "RigboClient",
]