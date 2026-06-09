from go1_gym.envs.go1.go1_config_adaptive_terrain import AdaptiveGo1ConfigTerrain
from go1_gym.envs.go2.go2_config import Go2Config


class AdaptiveGo2ConfigTerrain(AdaptiveGo1ConfigTerrain):
    """Terrain adaptive-energy Go2 config.

    Reuses the Go1 terrain curriculum and adaptive-energy reward definitions,
    while pointing Isaac Gym at the Go2 asset folder.
    """

    class init_state(Go2Config.init_state):
        pass

    class env(AdaptiveGo1ConfigTerrain.env):
        pass

    class rewards(AdaptiveGo1ConfigTerrain.rewards):
        base_height_target = Go2Config.rewards.base_height_target
        terminal_body_height = Go2Config.rewards.terminal_body_height

        class scales(AdaptiveGo1ConfigTerrain.rewards.scales):
            pass

    class terrain(AdaptiveGo1ConfigTerrain.terrain):
        pass

    class asset(Go2Config.asset):
        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = ["base", "Head", "hip", "thigh"]

    class commands(AdaptiveGo1ConfigTerrain.commands):
        limit_vel_x = [-1.5, 1.5]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-1.5, 1.5]

    class domain_rand(AdaptiveGo1ConfigTerrain.domain_rand):
        pass
