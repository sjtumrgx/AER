"""Load-carry config, reward, CLI, and metric surface tests."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]


class LoadCarryConfigRewardMetricsTest(unittest.TestCase):
    def test_load_carry_config_exposes_requested_privileged_state(self):
        from gym.envs.go2.go2_config_load_carry import LoadCarryGo2Config

        cfg = LoadCarryGo2Config()
        self.assertGreaterEqual(cfg.env.num_observation_history, 20)
        self.assertLessEqual(cfg.env.num_observation_history, 50)
        self.assertEqual(40, cfg.env.num_privileged_obs)
        for flag in [
            "priv_observe_robot_base_velocity",
            "priv_observe_payload_mass",
            "priv_observe_payload_com",
            "priv_observe_payload_inertia",
            "priv_observe_payload_relative_pose",
            "priv_observe_payload_relative_velocity",
            "priv_observe_ground_friction",
            "priv_observe_terrain_parameters",
            "priv_observe_external_disturbance",
        ]:
            self.assertTrue(getattr(cfg.env, flag), flag)
        self.assertEqual("adaptive", cfg.rewards.energy_regularization_mode)
        self.assertTrue(hasattr(cfg.rewards.scales, "load_normalized_transport"))

    def test_train_help_exposes_load_carry_training_controls_without_running_isaacgym(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train.py"), "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        for token in ["load_carry", "--training_stage", "--baseline", "--adaptation_encoder", "--teacher_checkpoint"]:
            self.assertIn(token, result.stdout)

    def test_domain_rand_policy_baseline_disables_teacher_student_losses(self):
        train_source = (ROOT / "scripts" / "train.py").read_text()
        ppo_source = (ROOT / "gym_learn" / "ppo_cse" / "ppo.py").read_text()
        self.assertIn('if args.baseline == "domain_rand_policy":', train_source)
        self.assertIn("PPO_Args.distillation_loss_coef = 0.0", train_source)
        self.assertIn("PPO_Args.latent_supervised_loss_coef = 0.0", train_source)
        self.assertIn("and PPO_Args.distillation_loss_coef != 0.0", ppo_source)
        self.assertIn("and PPO_Args.latent_supervised_loss_coef != 0.0", ppo_source)

    def test_student_stages_can_load_and_freeze_teacher_checkpoint(self):
        train_source = (ROOT / "scripts" / "train.py").read_text()
        runner_source = (ROOT / "gym_learn" / "ppo_cse" / "__init__.py").read_text()
        self.assertIn("--teacher_checkpoint", train_source)
        self.assertIn("RunnerArgs.teacher_checkpoint = args.teacher_checkpoint", train_source)
        self.assertIn("actor_critic.load_state_dict(weights, strict=False)", runner_source)
        self.assertIn('RunnerArgs.training_stage in {"latent_pretrain", "student_distill", "student_ppo", "full"}', runner_source)
        self.assertIn("param.requires_grad_(False)", runner_source)

    def test_load_normalized_transport_reward_modes(self):
        from gym.envs.rewards.corl_rewards import CoRLRewards

        def make_env(mode: str, alpha=None):
            env = SimpleNamespace()
            env.device = torch.device("cpu")
            env.dt = 0.1
            env.default_body_mass = 12.0
            env.payloads = torch.tensor([3.0, 0.0])
            env.torques = torch.tensor([[2.0, -1.0, 0.0], [1.0, 1.0, -2.0]])
            env.dof_vel = torch.tensor([[0.5, -0.5, 2.0], [1.0, -1.0, 0.5]])
            env.base_lin_vel = torch.tensor([[0.5, 0.0, 0.0], [0.1, 0.0, 0.0]])
            env.commands = torch.zeros(2, 3)
            env.cfg = SimpleNamespace(
                rewards=SimpleNamespace(
                    energy_regularization_mode=mode,
                    fixed_energy_alpha=2.0,
                    transport_speed_floor=0.05,
                    transport_cost_clip=100.0,
                )
            )
            if alpha is not None:
                env.policy_energy_alpha = alpha
            return env

        none_cost = CoRLRewards(make_env("none"))._reward_load_normalized_transport()
        fixed_cost = CoRLRewards(make_env("fixed"))._reward_load_normalized_transport()
        adaptive_cost = CoRLRewards(make_env("adaptive", torch.tensor([[0.5], [3.0]])))._reward_load_normalized_transport()

        self.assertTrue(torch.allclose(none_cost, torch.zeros_like(none_cost)))
        self.assertTrue(torch.all(fixed_cost > 0.0))
        self.assertAlmostEqual(2.0, float(fixed_cost[0] / adaptive_cost[0] * 0.5), places=4)
        self.assertGreater(float(adaptive_cost[1]), float(fixed_cost[1]))

    def test_eval_metric_registry_covers_required_payload_plots(self):
        from gym_learn.eval_metrics.metrics import LOAD_CARRY_PLOT_SPECS, METRICS_FNS

        expected_metrics = {
            "CoT",
            "tracking_error",
            "fall_rate",
            "joint_power",
            "foot_contact_schedule",
            "stance_duration",
            "body_height",
            "gait_transition",
        }
        self.assertTrue(expected_metrics.issubset(METRICS_FNS.keys()))
        for name in expected_metrics:
            self.assertIn(name, LOAD_CARRY_PLOT_SPECS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
