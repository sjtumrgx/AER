"""Asymmetric teacher-student actor critic for load-adaptive locomotion.

The default architecture implements the requested load-carrying setup:

* teacher actor: current proprioceptive observation + privileged state
* student actor: current proprioceptive observation + z_t from observation history
* privileged critic: current observation + privileged state
* adaptation encoder: temporal CNN/GRU/MLP over flattened H x obs_dim history
* adaptive energy alpha head: f(z_t, v_cmd, omega_cmd, terrain_difficulty)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    # Load-carry asymmetric teacher-student architecture.
    use_asymmetric_teacher_student = True
    latent_dim = 16
    adaptation_encoder_type = "cnn"  # cnn, gru, mlp
    adaptation_hidden_dim = 128
    history_steps = None  # inferred from num_obs_history // num_obs when None
    teacher_actor_hidden_dims = [512, 256, 128]
    student_actor_hidden_dims = [512, 256, 128]
    privileged_encoder_hidden_dims = [128]
    alpha_hidden_dims = [64, 32]
    alpha_min = 1e-4
    alpha_max = 10.0
    default_policy_mode = "teacher"  # teacher for phase-1 PPO; student for deployment/fine-tune
    print_networks = False

    # Legacy names retained so older config logging does not break.
    adaptation_module_branch_hidden_dims = [256, 128]
    use_decoder = False


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"invalid activation function: {act_name}")


def _activation_factory(act_name: str):
    def make():
        return get_activation(act_name)

    return make


def build_mlp(input_dim: int, hidden_dims, output_dim: int, activation_name: str) -> nn.Sequential:
    make_activation = _activation_factory(activation_name)
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(make_activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class TemporalAdaptationEncoder(nn.Module):
    """Encodes flattened H x obs_dim proprioceptive history into z_t."""

    def __init__(
        self,
        num_obs: int,
        num_obs_history: int,
        latent_dim: int,
        encoder_type: str = "cnn",
        hidden_dim: int = 128,
        activation_name: str = "elu",
        history_steps: Optional[int] = None,
    ):
        super().__init__()
        if num_obs <= 0:
            raise ValueError("num_obs must be positive")
        if num_obs_history % num_obs != 0:
            raise ValueError(f"num_obs_history={num_obs_history} must be divisible by num_obs={num_obs}")
        inferred_steps = num_obs_history // num_obs
        if history_steps is not None and int(history_steps) != inferred_steps:
            raise ValueError(
                f"history_steps={history_steps} does not match num_obs_history // num_obs={inferred_steps}"
            )
        self.num_obs = int(num_obs)
        self.num_obs_history = int(num_obs_history)
        self.history_steps = int(inferred_steps)
        self.latent_dim = int(latent_dim)
        self.encoder_type = encoder_type.lower()
        self.hidden_dim = int(hidden_dim)

        if self.encoder_type not in {"cnn", "gru", "mlp"}:
            raise ValueError(f"Unsupported adaptation_encoder_type: {encoder_type}")
        self.cnn = nn.Sequential(
            nn.Conv1d(self.num_obs, self.hidden_dim, kernel_size=5, padding=2),
            get_activation(activation_name),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            get_activation(activation_name),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.gru = nn.GRU(input_size=self.num_obs, hidden_size=self.hidden_dim, batch_first=True)
        self.output = nn.Linear(self.hidden_dim, self.latent_dim)
        self.mlp = build_mlp(self.num_obs_history, [self.hidden_dim, self.hidden_dim], self.latent_dim, activation_name)

    def _as_sequence(self, observation_history: torch.Tensor) -> torch.Tensor:
        if observation_history.dim() != 2 or observation_history.shape[-1] != self.num_obs_history:
            raise RuntimeError("invalid observation_history shape")
        return observation_history.view(observation_history.shape[0], self.history_steps, self.num_obs)

    def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "cnn":
            sequence = self._as_sequence(observation_history).transpose(1, 2)
            return self.cnn(sequence)
        if self.encoder_type == "gru":
            sequence = self._as_sequence(observation_history)
            _, h_n = self.gru(sequence)
            return self.output(h_n[-1])
        return self.mlp(observation_history)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs, num_privileged_obs, num_obs_history, num_actions, **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super().__init__()

        self.decoder = AC_Args.use_decoder
        self.num_obs = int(num_obs)
        self.num_obs_history = int(num_obs_history)
        self.num_privileged_obs = int(num_privileged_obs)
        self.num_actions = int(num_actions)
        self.latent_dim = int(AC_Args.latent_dim)
        self.policy_mode = AC_Args.default_policy_mode

        activation_name = AC_Args.activation
        teacher_hidden_dims = getattr(AC_Args, "teacher_actor_hidden_dims", AC_Args.actor_hidden_dims)
        student_hidden_dims = getattr(AC_Args, "student_actor_hidden_dims", AC_Args.actor_hidden_dims)

        self.adaptation_module = TemporalAdaptationEncoder(
            num_obs=self.num_obs,
            num_obs_history=self.num_obs_history,
            latent_dim=self.latent_dim,
            encoder_type=AC_Args.adaptation_encoder_type,
            hidden_dim=AC_Args.adaptation_hidden_dim,
            activation_name=activation_name,
            history_steps=AC_Args.history_steps,
        )

        # Teacher actor sees privileged state directly, not student latent.
        self.teacher_actor_body = build_mlp(
            self.num_obs + self.num_privileged_obs,
            teacher_hidden_dims,
            self.num_actions,
            activation_name,
        )

        # Student actor is the deployable body. Keep actor_body alias for legacy export code.
        self.student_actor_body = build_mlp(
            self.num_obs + self.latent_dim,
            student_hidden_dims,
            self.num_actions,
            activation_name,
        )
        self.actor_body = self.student_actor_body

        self.critic_body = build_mlp(
            self.num_obs + self.num_privileged_obs,
            AC_Args.critic_hidden_dims,
            1,
            activation_name,
        )

        # Privileged-to-latent target encoder supports latent supervised pretraining / teacher-latent distillation.
        self.teacher_latent_encoder = build_mlp(
            self.num_privileged_obs,
            AC_Args.privileged_encoder_hidden_dims,
            self.latent_dim,
            activation_name,
        )
        self.env_factor_encoder = self.teacher_latent_encoder

        self.energy_alpha_head = build_mlp(
            self.latent_dim + 3,  # z_t, v_cmd, omega_cmd, terrain difficulty
            AC_Args.alpha_hidden_dims,
            1,
            activation_name,
        )

        if AC_Args.print_networks:
            print(f"Adaptation Encoder ({AC_Args.adaptation_encoder_type}): {self.adaptation_module}")
            print(f"Teacher Actor MLP: {self.teacher_actor_body}")
            print(f"Student Actor MLP: {self.student_actor_body}")
            print(f"Privileged Critic MLP: {self.critic_body}")
            print(f"Energy Alpha Head: {self.energy_alpha_head}")

        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.last_energy_alpha = None
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _command_features(self, batch_size: int, device, command: Optional[torch.Tensor]) -> torch.Tensor:
        if command is None:
            return torch.zeros(batch_size, 2, device=device)
        if command.dim() == 1:
            command = command.unsqueeze(1)
        command = command.to(device)
        if command.shape[1] >= 3:
            return torch.stack((command[:, 0], command[:, 2]), dim=1)
        if command.shape[1] == 1:
            return torch.cat((command, torch.zeros(batch_size, 1, device=device)), dim=1)
        return command[:, :2]

    def _terrain_feature(self, batch_size: int, device, terrain_difficulty: Optional[torch.Tensor]) -> torch.Tensor:
        if terrain_difficulty is None:
            return torch.zeros(batch_size, 1, device=device)
        terrain_difficulty = terrain_difficulty.to(device)
        if terrain_difficulty.dim() == 1:
            terrain_difficulty = terrain_difficulty.unsqueeze(1)
        return terrain_difficulty[:, :1]

    def predict_energy_alpha(self, latent: torch.Tensor, command: Optional[torch.Tensor] = None,
                             terrain_difficulty: Optional[torch.Tensor] = None) -> torch.Tensor:
        command_features = self._command_features(latent.shape[0], latent.device, command)
        terrain_feature = self._terrain_feature(latent.shape[0], latent.device, terrain_difficulty)
        raw_alpha = self.energy_alpha_head(torch.cat((latent, command_features, terrain_feature), dim=-1))
        alpha = F.softplus(raw_alpha) + AC_Args.alpha_min
        if AC_Args.alpha_max is not None:
            alpha = torch.clamp(alpha, max=float(AC_Args.alpha_max))
        return alpha

    def get_student_latent(self, observation_history: torch.Tensor) -> torch.Tensor:
        return self.adaptation_module(observation_history)

    def get_teacher_latent(self, privileged_observations: torch.Tensor) -> torch.Tensor:
        return self.teacher_latent_encoder(privileged_observations)

    def _teacher_mean(self, observations: torch.Tensor, privileged_info: torch.Tensor) -> torch.Tensor:
        return self.teacher_actor_body(torch.cat((observations, privileged_info), dim=-1))

    def _student_mean(self, observations: torch.Tensor, observation_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.get_student_latent(observation_history)
        return self.student_actor_body(torch.cat((observations, latent), dim=-1)), latent

    def update_distribution(self, observations: torch.Tensor, privileged_observations: Optional[torch.Tensor] = None,
                            observation_history: Optional[torch.Tensor] = None, policy_mode: Optional[str] = None,
                            command: Optional[torch.Tensor] = None,
                            terrain_difficulty: Optional[torch.Tensor] = None):
        mode = (policy_mode or self.policy_mode or "teacher").lower()
        if mode in {"teacher", "teacher_ppo", "expert"}:
            if privileged_observations is None:
                raise ValueError("privileged_observations are required for teacher policy")
            mean = self._teacher_mean(observations, privileged_observations)
            latent_for_alpha = self.get_teacher_latent(privileged_observations)
        elif mode in {"student", "student_ppo", "student_distill", "distill", "inference"}:
            if observation_history is None:
                raise ValueError("observation_history is required for student policy")
            mean, latent_for_alpha = self._student_mean(observations, observation_history)
        else:
            raise ValueError(f"Unsupported policy_mode: {policy_mode}")
        self.last_energy_alpha = self.predict_energy_alpha(latent_for_alpha, command, terrain_difficulty).detach()
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, privileged_observations=None, observation_history=None, policy_mode=None, **kwargs):
        self.update_distribution(observations, privileged_observations, observation_history, policy_mode, **kwargs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs"], ob["privileged_obs"], policy_info=policy_info)

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs"], ob["obs_history"], policy_info=policy_info)

    def act_student(self, observations, observation_history, command=None, terrain_difficulty=None, policy_info={}):
        actions_mean, latent = self._student_mean(observations, observation_history)
        energy_alpha = self.predict_energy_alpha(latent, command, terrain_difficulty)
        self.last_energy_alpha = energy_alpha.detach()
        policy_info["latents"] = latent.detach().cpu().numpy()
        policy_info["energy_alpha"] = energy_alpha.detach()
        return actions_mean

    def act_teacher(self, observations, privileged_info, command=None, terrain_difficulty=None, policy_info={}):
        actions_mean = self._teacher_mean(observations, privileged_info)
        teacher_latent = self.get_teacher_latent(privileged_info)
        energy_alpha = self.predict_energy_alpha(teacher_latent, command, terrain_difficulty)
        self.last_energy_alpha = energy_alpha.detach()
        policy_info["latents"] = teacher_latent.detach().cpu().numpy()
        policy_info["privileged_obs"] = privileged_info.detach().cpu().numpy()
        policy_info["energy_alpha"] = energy_alpha.detach()
        return actions_mean

    def evaluate(self, critic_observations, privileged_observations, **kwargs):
        return self.critic_body(torch.cat((critic_observations, privileged_observations), dim=-1))

    def latent_supervised_loss(self, observation_history: torch.Tensor, privileged_observations: torch.Tensor) -> torch.Tensor:
        student_latent = self.get_student_latent(observation_history)
        with torch.no_grad():
            teacher_latent = self.get_teacher_latent(privileged_observations)
        return F.mse_loss(student_latent, teacher_latent)

    def distillation_loss(self, observations: torch.Tensor, observation_history: torch.Tensor,
                          privileged_observations: torch.Tensor) -> torch.Tensor:
        student_mean, _ = self._student_mean(observations, observation_history)
        with torch.no_grad():
            teacher_mean = self._teacher_mean(observations, privileged_observations)
        return F.mse_loss(student_mean, teacher_mean)
