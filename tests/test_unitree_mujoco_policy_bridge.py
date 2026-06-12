"""Unitree MuJoCo AER policy bridge contract tests."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "unitree_mujoco" / "policies" / "go2_load_carry_student"
GO2_DIR = ROOT / "unitree_mujoco" / "unitree_robots" / "go2"


class UnitreeMujocoPolicyBridgeTest(unittest.TestCase):
    def test_policy_bundle_contains_migrated_weights_and_manifest(self):
        for rel in [
            "body_latest.jit",
            "adaptation_module_latest.jit",
            "unitree_go2_actuator.pt",
            "env_cfg.yaml",
            "manifest.json",
        ]:
            self.assertTrue((BUNDLE / rel).exists(), rel)
        manifest = (BUNDLE / "manifest.json").read_text()
        self.assertIn("load-carry-adaptive_energy-student_ppo-cnn-z16-seed-200", manifest)
        self.assertIn("num_observations", manifest)

    def test_joint_order_and_observation_contract_are_explicit(self):
        from unitree_mujoco.aer_policy.policy import Go2AERPolicyConfig, Go2JointOrder, UnitreeAERPolicy

        cfg = Go2AERPolicyConfig.from_bundle(BUNDLE)
        self.assertEqual(70, cfg.num_observations)
        self.assertEqual(30, cfg.num_observation_history)
        self.assertEqual(2100, cfg.history_dim)
        self.assertEqual(12, cfg.num_actions)
        self.assertEqual(15, cfg.num_commands)
        self.assertEqual(6, cfg.lag_timesteps)
        self.assertEqual([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], Go2JointOrder.policy_to_unitree.tolist())
        self.assertEqual([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], Go2JointOrder.unitree_to_policy.tolist())

        policy = UnitreeAERPolicy(BUNDLE)
        obs = policy.build_observation(
            q_unitree=policy.default_dof_pos_unitree,
            dq_unitree=[0.0] * 12,
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            gyro_xyz=[0.0, 0.0, 0.0],
            lin_vel_xyz=[0.0, 0.0, 0.0],
            command_xyz=[0.5, 0.0, 0.0],
        )
        self.assertEqual((1, 70), tuple(obs.shape))
        action = policy.act_from_observation(obs)
        self.assertEqual((1, 12), tuple(action.shape))
        q_target_unitree = policy.action_to_unitree_position_target(action)
        self.assertEqual((12,), tuple(q_target_unitree.shape))

        lagged_first = policy.lagged_action_to_unitree_position_target(action)
        self.assertTrue((abs(lagged_first - policy.default_dof_pos_unitree) < 1e-6).all())

    def test_observation_postprocess_matches_isaacgym_zero_mask(self):
        from unitree_mujoco.aer_policy.policy import UnitreeAERPolicy

        policy = UnitreeAERPolicy(BUNDLE)
        policy.update_clock()
        obs = policy.build_observation(
            q_unitree=policy.default_dof_pos_unitree,
            dq_unitree=[0.0] * 12,
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            gyro_xyz=[0.0, 0.0, 0.0],
            command_xyz=[0.5, 0.0, 0.0],
        )
        obs_np = obs.detach().cpu().numpy()[0]

        # gym/envs/base/legged_robot.py zeroes these slots after noise:
        # obs[6:18] = non-velocity command tail, obs[66:70] = clock inputs.
        self.assertAlmostEqual(1.0, float(obs_np[3]), places=6)
        self.assertTrue((obs_np[6:18] == 0.0).all(), obs_np[6:18])
        self.assertTrue((obs_np[66:70] == 0.0).all(), obs_np[66:70])

    def test_sim2sim_and_sim2real_cli_are_help_safe_and_network_configurable(self):
        for rel in ["sim2sim_aer.py", "sim2real_aer.py"]:
            result = subprocess.run(
                [sys.executable, str(ROOT / "unitree_mujoco" / "aer_policy" / rel), "--help"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("--policy_dir", result.stdout)
            self.assertNotIn("ros", result.stdout.lower())
        sim2real_help = subprocess.run(
            [sys.executable, str(ROOT / "unitree_mujoco" / "aer_policy" / "sim2real_aer.py"), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        ).stdout
        self.assertIn("--net", sim2real_help)
        self.assertIn("--dry_run", sim2real_help)

    def test_aer_flat_scene_contains_visible_payload_mesh(self):
        scene = GO2_DIR / "scene_aer_flat.xml"
        robot = GO2_DIR / "go2_aer_load_carry.xml"
        self.assertTrue(scene.exists())
        self.assertTrue(robot.exists())
        self.assertIn("go2_aer_load_carry.xml", scene.read_text())
        self.assertIn("aer_payload_visual", robot.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
