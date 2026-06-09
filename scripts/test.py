import isaacgym
assert isaacgym
import torch

from gym.envs.go2.go2_config import Go2Config
from gym.envs.velocity_tracking import VelocityTrackingEasyEnv


def run_env(render=False, headless=True, device="cuda:0"):
    cfg = Go2Config()
    cfg.env.num_envs = 1
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 1
    env = VelocityTrackingEasyEnv(sim_device=device, headless=headless, cfg=cfg)
    obs = env.reset()
    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(10):
        obs, rew, done, info = env.step(action)
    return env, obs


if __name__ == '__main__':
    run_env()
