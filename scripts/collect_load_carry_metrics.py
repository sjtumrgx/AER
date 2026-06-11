"""Run load-carry policy rollouts and write metrics CSV for plotting."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
from pathlib import Path
import sys


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="Checkpoint dir relative to repo root or absolute path.")
    parser.add_argument("--output_csv", default=None, type=str, help="Output CSV path. Defaults to <model_dir>/analysis/load_carry_rollouts.csv")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_steps", default=200, type=int)
    parser.add_argument("--lin_speed", default=0.5, type=float)
    parser.add_argument("--ang_speed", default=0.0, type=float)
    parser.add_argument("--payload_masses", default="0,2,4,6,8", type=str)
    parser.add_argument("--com_offsets", default="0.0,0.04,-0.04", type=str)
    parser.add_argument("--dynamic_payload", action="store_true")
    parser.add_argument("--max_conditions", default=None, type=int, help="Optional cap for smoke runs.")
    return parser


if __name__ == '__main__' and any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
    build_parser().parse_args()
    raise SystemExit(0)

import isaacgym
assert isaacgym
import torch
import yaml

from gym import MINI_GYM_ROOT_DIR
from gym.envs.go2.go2_config_load_carry import LoadCarryGo2Config
from gym.envs.velocity_tracking import VelocityTrackingEasyEnv
from gym.envs.wrappers.history_wrapper import HistoryWrapper
from gym.utils.helpers import dict_to_env_cfg
from scripts.play_command_defaults import assign_nominal_go2_commands


def parse_floats(text: str):
    return [float(item) for item in text.split(',') if item.strip()]


def resolve_model_dir(model_dir: str) -> Path:
    path = Path(model_dir)
    if not path.is_absolute():
        path = Path(MINI_GYM_ROOT_DIR) / path
    return path


def load_policy(logdir: Path):
    body = torch.jit.load(str(logdir / 'checkpoints/body_latest.jit'), map_location='cpu')
    adaptation_module = torch.jit.load(str(logdir / 'checkpoints/adaptation_module_latest.jit'), map_location='cpu')

    def policy(obs, info=None):
        if info is None:
            info = {}
        obs_history = obs['obs_history'].to('cpu')
        current_obs = obs['obs'].to('cpu')
        latent = adaptation_module.forward(obs_history)
        try:
            action = body.forward(torch.cat((current_obs, latent), dim=-1))
        except RuntimeError:
            action = body.forward(torch.cat((obs_history, latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def make_condition_tensors(num_envs: int, mass: float, com_x: float, dynamic: bool, device: str):
    tensors = {
        'payloads': torch.full((num_envs,), mass, dtype=torch.float, device=device),
        'com_displacements': torch.tensor([[com_x, 0.0, 0.0]], dtype=torch.float, device=device).repeat(num_envs, 1),
        'payload_inertias': torch.full((num_envs, 6), 0.02 + 0.002 * mass, dtype=torch.float, device=device),
        'payload_relative_pos': torch.zeros(num_envs, 3, dtype=torch.float, device=device),
        'payload_relative_quat': torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float, device=device).repeat(num_envs, 1),
        'payload_relative_velocity': torch.zeros(num_envs, 6, dtype=torch.float, device=device),
    }
    if dynamic:
        tensors['payload_relative_velocity'][:, 0] = 0.15
        tensors['payload_relative_velocity'][:, 5] = 0.35
    return tensors


def load_env(logdir: Path, device_id: int, headless: bool, mass: float, com_x: float, dynamic: bool):
    with (logdir / 'env_cfg.yaml').open('r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg = dict_to_env_cfg(LoadCarryGo2Config(), yaml_cfg)
    cfg.env.num_envs = 1
    cfg.env.num_recording_envs = 0
    cfg.terrain.curriculum = False
    cfg.terrain.selected = False
    cfg.terrain.num_rows = 3
    cfg.terrain.num_cols = 3
    cfg.terrain.border_size = 0
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 1
    cfg.terrain.teleport_robots = True
    cfg.asset.flip_visual_attachments = True

    # Deterministic evaluation condition.
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_ground_friction = False
    cfg.domain_rand.randomize_gravity = False
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_motor_offset = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_com_displacement = False
    cfg.domain_rand.randomize_payload_inertia = False
    cfg.domain_rand.randomize_payload_relative_pose = False
    cfg.domain_rand.randomize_payload_relative_velocity = False

    sim_device = f'cuda:{device_id}'
    initial = make_condition_tensors(cfg.env.num_envs, mass, com_x, dynamic, sim_device)
    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=cfg, enable_camera_sensor=False, temp_cap_dir=str(logdir / 'temp_cap_dir_eval'), initial_dynamics_dict=initial)
    return HistoryWrapper(env)


def collect_condition(logdir: Path, policy, args, mass: float, com_x: float, dynamic: bool):
    env = load_env(logdir, args.device, args.headless, mass, com_x, dynamic)
    obs = env.reset()
    rows = []
    condition = f"mass_{mass:g}_comx_{com_x:g}_{'dynamic' if dynamic else 'static'}"
    for step in range(args.num_steps):
        assign_nominal_go2_commands(env.commands, args.lin_speed, 0.0, args.ang_speed)
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        base_lin = env.env.base_lin_vel[0]
        base_ang = env.env.base_ang_vel[0]
        speed = torch.clamp(torch.norm(base_lin[:2]), min=0.05)
        joint_power = torch.sum(torch.abs(env.env.torques[0] * env.env.dof_vel[0]))
        total_mass = env.env.default_body_mass + env.env.payloads[0]
        cot = joint_power / (total_mass * 9.81 * speed)
        tracking_error = torch.norm(base_lin[:2] - env.env.commands[0, :2]) + torch.abs(base_ang[2] - env.env.commands[0, 2])
        contacts = (env.env.contact_forces[0, env.env.feet_indices, 2] > 1.).float()
        desired = env.env.desired_contact_states[0].float()
        rows.append({
            'condition': condition,
            'step': step,
            'payload_mass': float(mass),
            'com_offset_x': float(com_x),
            'dynamic_payload': int(dynamic),
            'CoT': float(cot.detach().cpu()),
            'tracking_error': float(tracking_error.detach().cpu()),
            'fall_rate': float(done[0].detach().cpu()),
            'joint_power': float(joint_power.detach().cpu()),
            'foot_contact_schedule': float(torch.mean(contacts).detach().cpu()),
            'stance_duration': float(torch.mean(contacts).detach().cpu()) * float(env.env.dt),
            'body_height': float(env.env.root_states[0, 2].detach().cpu()),
            'gait_transition': float(torch.mean(torch.abs(contacts - desired)).detach().cpu()),
        })
    env.close()
    return rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    logdir = resolve_model_dir(args.model_dir)
    out_csv = Path(args.output_csv) if args.output_csv else logdir / 'analysis/load_carry_rollouts.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    policy = load_policy(logdir)
    rows = []
    conditions = []
    for mass in parse_floats(args.payload_masses):
        for com_x in parse_floats(args.com_offsets):
            conditions.append((mass, com_x, False))
            if args.dynamic_payload:
                conditions.append((mass, com_x, True))
    if args.max_conditions is not None:
        conditions = conditions[:args.max_conditions]
    for mass, com_x, dynamic in conditions:
        rows.extend(collect_condition(logdir, policy, args, mass, com_x, dynamic))
    fieldnames = ['condition', 'step', 'payload_mass', 'com_offset_x', 'dynamic_payload', *list(rows[0].keys())[5:]] if rows else []
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} load-carry metric rows to {out_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
