from go1_gym.envs.go1.go1_config_adaptive import AdaptiveGo1Config
from go1_gym.envs.go2.go2_config import Go2Config


class AdaptiveGo2Config(AdaptiveGo1Config):
    """Flat-ground adaptive-energy Go2 config.

    Inherits the adaptive energy reward scales from `AdaptiveGo1Config` and the
    Go2 asset/morphology overrides from `Go2Config`.
    """

    class rewards(AdaptiveGo1Config.rewards):
        base_height_target = Go2Config.rewards.base_height_target
        terminal_body_height = Go2Config.rewards.terminal_body_height

        class scales(AdaptiveGo1Config.rewards.scales):
            pass

    class init_state(Go2Config.init_state):
        pass

    class commands(Go2Config.commands):
        pass

    class asset(Go2Config.asset):
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "Head", "hip"]
