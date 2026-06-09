from argparse import ArgumentParser
import sys


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--headless', action="store_true")
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cfg', default="adaptive_en", choices=["original", "adaptive_en", "adaen_terrain"], type=str)
    parser.add_argument('--en_new_actual', default=0.0, type=float)
    parser.add_argument('--en_new_cmd', default=0.0, type=float)
    parser.add_argument('--iterations', default=5000, type=int, help="Number of PPO learning iterations.")
    parser.add_argument('--num_envs', default=None, type=int, help="Override the Go2 environment count for smoke tests or smaller GPUs.")
    parser.add_argument('--num_steps_per_env', default=None, type=int, help="Override PPO rollout steps per environment for smoke tests.")
    return parser


if __name__ == '__main__' and any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
    build_parser().parse_args()


import isaacgym
assert isaacgym
import torch
import numpy as np
import random

from gym import MINI_GYM_ROOT_DIR
from gym.envs.base.legged_robot_config import Cfg
from gym.envs.go2.go2_config import Go2Config
from gym.envs.go2.go2_config_adaptive import AdaptiveGo2Config
from gym.envs.go2.go2_config_adaptive_terrain import AdaptiveGo2ConfigTerrain
from gym.envs.velocity_tracking import VelocityTrackingEasyEnv
from gym.envs.wrappers.history_wrapper import HistoryWrapper
from gym.utils.helpers import class_to_dict

from ml_logger import logger

from gym_learn.ppo_cse import Runner
from gym_learn.ppo_cse.actor_critic import AC_Args
from gym_learn.ppo_cse.ppo import PPO_Args
from gym_learn.ppo_cse import RunnerArgs

from pathlib import Path
import yaml


CFG_MAPPINGS = {
    "original": Go2Config,
    "adaptive_en": AdaptiveGo2Config,
    "adaen_terrain": AdaptiveGo2ConfigTerrain,
}


def train_go2(args, logdir):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg: Cfg = CFG_MAPPINGS[args.cfg]()
    if args.num_envs is not None:
        cfg.env.num_envs = args.num_envs
    if args.num_steps_per_env is not None:
        RunnerArgs.num_steps_per_env = args.num_steps_per_env
    cfg.rewards.scales.energy_new_actual = args.en_new_actual
    cfg.rewards.scales.energy_new_cmd = args.en_new_cmd
    Path(logdir).mkdir(parents=True, exist_ok=True)
    env_dict = class_to_dict(cfg)
    with open(f"{logdir}/env_cfg.yaml", "w") as file:
        yaml.dump(env_dict, file)

    env = VelocityTrackingEasyEnv(sim_device=f'cuda:{args.device}', headless=args.headless, cfg=cfg)

    logger.log_params(
        AC_Args=vars(AC_Args),
        PPO_Args=vars(PPO_Args),
        RunnerArgs=vars(RunnerArgs),
        Cfg=vars(Cfg)
    )

    env = HistoryWrapper(env)
    runner = Runner(env, device=f"cuda:{args.device}")
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    stem = Path(__file__).stem
    if args.cfg == "adaen_terrain":
        run_name = f"terrain-seed-{args.seed}-ennewa-{args.en_new_actual:.1f}-ennewc-{args.en_new_cmd:.1f}"
    else:
        run_name = f"seed-{args.seed}-ennewa-{args.en_new_actual:.1f}-ennewc-{args.en_new_cmd:.1f}"
    logdir = f"{MINI_GYM_ROOT_DIR}/checkpoints/{stem}/{run_name}"
    logger.configure(
        logger.utcnow(f'{stem}/{run_name}'),
        root=Path(f"{MINI_GYM_ROOT_DIR}/checkpoints").resolve()
    )
    logger.log_text(
        """
        charts: 
        - yKey: train/episode/rew_total/mean
          xKey: iterations
        - yKey: train/episode/rew_tracking_lin_vel/mean
          xKey: iterations
        - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
          xKey: iterations
        - yKey: train/episode/rew_action_smoothness_1/mean
          xKey: iterations
        - yKey: train/episode/rew_action_smoothness_2/mean
          xKey: iterations
        - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
          xKey: iterations
        - yKey: train/episode/rew_orientation_control/mean
          xKey: iterations
        - yKey: train/episode/rew_dof_pos/mean
          xKey: iterations
        - yKey: train/episode/command_area_trot/mean
          xKey: iterations
        - yKey: train/episode/max_terrain_height/mean
          xKey: iterations
        - type: video
          glob: "videos/*.mp4"
        - yKey: adaptation_loss/mean
          xKey: iterations
        """,
        filename=".charts.yml",
        dedent=True
    )


    # to see the environment rendering, set headless=False
    train_go2(args=args, logdir=logdir)
