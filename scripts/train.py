from argparse import ArgumentParser
import sys


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--headless', action="store_true")
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cfg', default="adaptive_en", choices=["original", "adaptive_en", "adaen_terrain", "load_carry"], type=str)
    parser.add_argument(
        '--training_stage',
        default="teacher_ppo",
        choices=["teacher_ppo", "latent_pretrain", "student_distill", "student_ppo", "full"],
        type=str,
        help="Load-carry training phase: teacher PPO, latent pretraining, student distillation, or student PPO fine-tuning.",
    )
    parser.add_argument(
        '--baseline',
        default="adaptive_energy",
        choices=["adaptive_energy", "no_energy", "fixed_energy", "domain_rand_policy"],
        type=str,
        help="Load-carry ablation baseline. domain_rand_policy trains student PPO directly without teacher-student distillation.",
    )
    parser.add_argument('--adaptation_encoder', default="cnn", choices=["cnn", "gru", "mlp"], type=str, help="Temporal encoder for z_t = E(o_{t-H:t}).")
    parser.add_argument('--latent_dim', default=16, choices=[8, 16], type=int, help="Dimension of load-adaptation latent z_t.")
    parser.add_argument('--history_steps', default=None, type=int, help="Override proprioceptive history length H; default comes from config.")
    parser.add_argument('--distillation_coef', default=1.0, type=float, help="Student behavior-cloning/DAgger loss coefficient.")
    parser.add_argument('--latent_supervised_coef', default=1.0, type=float, help="Latent supervised pretraining loss coefficient.")
    parser.add_argument('--dagger_teacher_prob', default=0.0, type=float, help="Probability of executing teacher action during student distillation rollouts.")
    parser.add_argument('--teacher_checkpoint', default=None, type=str, help="Local ac_weights checkpoint used to initialize/freeze the teacher for latent pretrain, distillation, or student PPO.")
    parser.add_argument('--en_new_actual', default=0.0, type=float)
    parser.add_argument('--en_new_cmd', default=0.0, type=float)
    parser.add_argument('--iterations', default=5000, type=int, help="Number of PPO learning iterations.")
    parser.add_argument('--num_envs', default=None, type=int, help="Override the Go2 environment count for smoke tests or smaller GPUs.")
    parser.add_argument('--num_steps_per_env', default=None, type=int, help="Override PPO rollout steps per environment for smoke tests.")
    parser.add_argument('--wandb', action='store_true', help="Enable Weights & Biases logging for reward, loss, curriculum, and timing curves.")
    parser.add_argument('--wandb_project', default='aer-go2', type=str, help="Weights & Biases project name used when --wandb is set.")
    parser.add_argument('--wandb_entity', default=None, type=str, help="Optional Weights & Biases entity/team.")
    parser.add_argument('--wandb_name', default=None, type=str, help="Optional Weights & Biases run name. Defaults to the checkpoint run name.")
    parser.add_argument('--wandb_group', default=None, type=str, help="Optional Weights & Biases group for multi-run comparisons.")
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], type=str, help="Weights & Biases mode used when --wandb is set.")
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
from gym.envs.go2.go2_config_load_carry import (
    LoadCarryGo2Config,
    LoadCarryGo2DomainRandPolicyConfig,
    LoadCarryGo2FixedEnergyConfig,
    LoadCarryGo2NoEnergyConfig,
)
from gym.envs.velocity_tracking import VelocityTrackingEasyEnv
from gym.envs.wrappers.history_wrapper import HistoryWrapper
from gym.utils.helpers import class_to_dict

from ml_logger import logger

from gym_learn.ppo_cse import Runner
from gym_learn.ppo_cse.actor_critic import AC_Args
from gym_learn.ppo_cse.ppo import PPO_Args
from gym_learn.ppo_cse import RunnerArgs
from gym_learn.utils.wandb_logging import WandbLogger

from pathlib import Path
import yaml


CFG_MAPPINGS = {
    "original": Go2Config,
    "adaptive_en": AdaptiveGo2Config,
    "adaen_terrain": AdaptiveGo2ConfigTerrain,
    "load_carry": LoadCarryGo2Config,
}

LOAD_CARRY_BASELINES = {
    "adaptive_energy": LoadCarryGo2Config,
    "no_energy": LoadCarryGo2NoEnergyConfig,
    "fixed_energy": LoadCarryGo2FixedEnergyConfig,
    "domain_rand_policy": LoadCarryGo2DomainRandPolicyConfig,
}


def train_go2(args, logdir):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg_cls = CFG_MAPPINGS[args.cfg]
    if args.cfg == "load_carry":
        cfg_cls = LOAD_CARRY_BASELINES[args.baseline]
    cfg: Cfg = cfg_cls()
    if args.num_envs is not None:
        cfg.env.num_envs = args.num_envs
    if args.num_steps_per_env is not None:
        RunnerArgs.num_steps_per_env = args.num_steps_per_env
    if args.cfg == "load_carry":
        if args.history_steps is not None:
            cfg.env.num_observation_history = args.history_steps
        AC_Args.adaptation_encoder_type = args.adaptation_encoder
        AC_Args.latent_dim = args.latent_dim
        AC_Args.history_steps = cfg.env.num_observation_history
        AC_Args.default_policy_mode = "student" if args.baseline == "domain_rand_policy" else "teacher"
        PPO_Args.training_stage = "student_ppo" if args.baseline == "domain_rand_policy" else args.training_stage
        PPO_Args.baseline = args.baseline
        if args.baseline == "domain_rand_policy":
            PPO_Args.distillation_loss_coef = 0.0
            PPO_Args.latent_supervised_loss_coef = 0.0
        else:
            PPO_Args.distillation_loss_coef = args.distillation_coef
            PPO_Args.latent_supervised_loss_coef = args.latent_supervised_coef
        RunnerArgs.training_stage = PPO_Args.training_stage
        RunnerArgs.baseline = args.baseline
        RunnerArgs.dagger_teacher_prob = args.dagger_teacher_prob
        RunnerArgs.teacher_checkpoint = args.teacher_checkpoint
    cfg.rewards.scales.energy_new_actual = args.en_new_actual
    cfg.rewards.scales.energy_new_cmd = args.en_new_cmd
    Path(logdir).mkdir(parents=True, exist_ok=True)
    env_dict = class_to_dict(cfg)
    with open(f"{logdir}/env_cfg.yaml", "w") as file:
        yaml.dump(env_dict, file)

    run_name = args.wandb_name or Path(logdir).name
    wandb_logger = WandbLogger(enabled=args.wandb)
    wandb_config = {
        "args": vars(args),
        "env": env_dict,
        "AC_Args": vars(AC_Args),
        "PPO_Args": vars(PPO_Args),
        "RunnerArgs": vars(RunnerArgs),
    }

    try:
        wandb_logger.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            config=wandb_config,
            dir=logdir,
        )

        env = VelocityTrackingEasyEnv(sim_device=f'cuda:{args.device}', headless=args.headless, cfg=cfg)

        logger.log_params(
            AC_Args=vars(AC_Args),
            PPO_Args=vars(PPO_Args),
            RunnerArgs=vars(RunnerArgs),
            Cfg=vars(Cfg)
        )

        env = HistoryWrapper(env)
        runner = Runner(env, device=f"cuda:{args.device}", metrics_logger=wandb_logger)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True, eval_freq=100)
    finally:
        wandb_logger.finish()


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    stem = Path(__file__).stem
    if args.cfg == "load_carry":
        stage = "student_ppo" if args.baseline == "domain_rand_policy" else args.training_stage
        run_name = f"load-carry-{args.baseline}-{stage}-{args.adaptation_encoder}-z{args.latent_dim}-seed-{args.seed}"
    elif args.cfg == "adaen_terrain":
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
        - yKey: distillation_loss/mean
          xKey: iterations
        - yKey: train/episode/rew_load_normalized_transport/mean
          xKey: iterations
        """,
        filename=".charts.yml",
        dedent=True
    )


    # to see the environment rendering, set headless=False
    train_go2(args=args, logdir=logdir)
