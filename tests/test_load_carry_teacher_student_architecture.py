"""Architecture tests for the load-carry asymmetric teacher-student policy."""

from __future__ import annotations

import contextlib
import unittest

import torch

from gym_learn.ppo_cse.actor_critic import AC_Args, ActorCritic


@contextlib.contextmanager
def patched_ac_args(**updates):
    original = {key: getattr(AC_Args, key, None) for key in updates}
    missing = {key for key in updates if not hasattr(AC_Args, key)}
    try:
        for key, value in updates.items():
            setattr(AC_Args, key, value)
        yield
    finally:
        for key, value in original.items():
            if key in missing:
                delattr(AC_Args, key)
            else:
                setattr(AC_Args, key, value)


class LoadCarryTeacherStudentArchitectureTest(unittest.TestCase):
    def build_policy(self, encoder_type="cnn"):
        self.num_obs = 70
        self.num_privileged = 40
        self.history_steps = 30
        self.num_history = self.num_obs * self.history_steps
        self.num_actions = 12
        with patched_ac_args(
            use_asymmetric_teacher_student=True,
            latent_dim=16,
            adaptation_encoder_type=encoder_type,
            history_steps=self.history_steps,
            teacher_actor_hidden_dims=[512, 256, 128],
            student_actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            alpha_hidden_dims=[64, 32],
        ):
            return ActorCritic(
                self.num_obs,
                self.num_privileged,
                self.num_history,
                self.num_actions,
            )

    def test_teacher_student_have_requested_inputs_and_outputs(self):
        policy = self.build_policy("cnn")
        obs = torch.randn(5, self.num_obs)
        priv = torch.randn(5, self.num_privileged)
        history = torch.randn(5, self.num_history)

        teacher = policy.act_teacher(obs, priv)
        student = policy.act_student(obs, history)
        value = policy.evaluate(obs, priv)
        latent = policy.get_student_latent(history)

        self.assertEqual((5, self.num_actions), tuple(teacher.shape))
        self.assertEqual((5, self.num_actions), tuple(student.shape))
        self.assertEqual((5, 1), tuple(value.shape))
        self.assertEqual((5, 16), tuple(latent.shape))

        self.assertEqual(self.num_obs + self.num_privileged, policy.teacher_actor_body[0].in_features)
        self.assertEqual(self.num_obs + 16, policy.student_actor_body[0].in_features)
        self.assertEqual(self.num_obs + self.num_privileged, policy.critic_body[0].in_features)

    def test_gru_encoder_matches_cnn_encoder_contract(self):
        policy = self.build_policy("gru")
        history = torch.randn(3, self.num_history)
        latent = policy.get_student_latent(history)
        self.assertEqual((3, 16), tuple(latent.shape))

    def test_adaptive_alpha_head_is_positive_and_policy_info_visible(self):
        policy = self.build_policy("cnn")
        obs = torch.randn(4, self.num_obs)
        history = torch.randn(4, self.num_history)
        command = torch.tensor([[0.7, 0.0], [0.3, 0.2], [-0.5, -0.1], [0.0, 0.0]])
        terrain = torch.tensor([[0.0], [0.2], [0.6], [1.0]])

        policy_info = {}
        _ = policy.act_student(obs, history, command=command, terrain_difficulty=terrain, policy_info=policy_info)
        alpha = policy_info["energy_alpha"]

        self.assertEqual((4, 1), tuple(alpha.shape))
        self.assertTrue(torch.all(alpha > 0.0))
        self.assertIn("latents", policy_info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
