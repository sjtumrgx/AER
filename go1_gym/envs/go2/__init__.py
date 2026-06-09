"""Unitree Go2 environment configuration package.

This package intentionally avoids importing Isaac Gym-dependent environment
classes at module import time so configuration modules can be inspected on
machines where Isaac Gym is not installed.
"""

from go1_gym.envs.go2.go2_config import Go2Config
from go1_gym.envs.go2.go2_config_adaptive import AdaptiveGo2Config
from go1_gym.envs.go2.go2_config_adaptive_terrain import AdaptiveGo2ConfigTerrain

__all__ = [
    "Go2Config",
    "AdaptiveGo2Config",
    "AdaptiveGo2ConfigTerrain",
]
