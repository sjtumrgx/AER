"""AER TorchScript policy bridge for Unitree Go2 MuJoCo and SDK2 deployment.

The training policy uses Isaac Gym's Go2 joint order:
FL, FR, RL, RR. Unitree MuJoCo and SDK2 low-level messages use:
FR, FL, RR, RL. This module keeps that mapping explicit and centralizes the
70-dimensional observation construction used by the exported AER student.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import json

import numpy as np
import torch
import yaml


POLICY_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

UNITREE_MUJOCO_JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


class Go2JointOrder:
    """Joint-order permutations between trained policy and Unitree SDK/MuJoCo."""

    policy = POLICY_JOINT_ORDER
    unitree_mujoco = UNITREE_MUJOCO_JOINT_ORDER
    policy_to_unitree = np.array([UNITREE_MUJOCO_JOINT_ORDER.index(name) for name in POLICY_JOINT_ORDER], dtype=np.int64)
    unitree_to_policy = np.array([POLICY_JOINT_ORDER.index(name) for name in UNITREE_MUJOCO_JOINT_ORDER], dtype=np.int64)


@dataclass(frozen=True)
class Go2CommandDefaults:
    body_height: float = 0.0
    gait_frequency: float = 3.0
    gait_phase: float = 0.5
    gait_offset: float = 0.0
    gait_bound: float = 0.0
    gait_duration: float = 0.5
    footswing_height: float = 0.08
    body_pitch: float = 0.0
    body_roll: float = 0.0
    stance_width: float = 0.25
    stance_length: float = 0.40
    aux_reward_coef: float = 0.0

    def as_vector(self, x_vel: float, y_vel: float = 0.0, yaw_vel: float = 0.0, num_commands: int = 15) -> np.ndarray:
        command = np.zeros(num_commands, dtype=np.float32)
        values = [
            x_vel,
            y_vel,
            yaw_vel,
            self.body_height,
            self.gait_frequency,
            self.gait_phase,
            self.gait_offset,
            self.gait_bound,
            self.gait_duration,
            self.footswing_height,
            self.body_pitch,
            self.body_roll,
            self.stance_width,
            self.stance_length,
            self.aux_reward_coef,
        ]
        command[: min(num_commands, len(values))] = values[:num_commands]
        return command


@dataclass
class Go2AERPolicyConfig:
    bundle_dir: Path
    env_cfg: dict
    manifest: dict

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path) -> "Go2AERPolicyConfig":
        bundle = Path(bundle_dir).expanduser().resolve()
        with (bundle / "env_cfg.yaml").open("r") as f:
            env_cfg = yaml.safe_load(f)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return cls(bundle, env_cfg, manifest)

    @property
    def num_observations(self) -> int:
        return int(self.env_cfg["env"]["num_observations"])

    @property
    def num_observation_history(self) -> int:
        return int(self.env_cfg["env"]["num_observation_history"])

    @property
    def history_dim(self) -> int:
        return self.num_observations * self.num_observation_history

    @property
    def num_actions(self) -> int:
        return int(self.env_cfg["env"]["num_actions"])

    @property
    def num_commands(self) -> int:
        return int(self.env_cfg["commands"]["num_commands"])

    @property
    def obs_scales(self) -> dict:
        return self.env_cfg["normalization"]["obs_scales"]

    @property
    def clip_actions(self) -> float:
        return float(self.env_cfg["normalization"]["clip_actions"])

    @property
    def action_scale(self) -> float:
        return float(self.env_cfg["control"]["action_scale"])

    @property
    def hip_scale_reduction(self) -> float:
        return float(self.env_cfg["control"].get("hip_scale_reduction", 1.0))

    @property
    def control_dt(self) -> float:
        return float(self.env_cfg["control"]["decimation"]) * float(self.env_cfg["sim"]["dt"])

    @property
    def lag_timesteps(self) -> int:
        if not self.env_cfg.get("domain_rand", {}).get("randomize_lag_timesteps", False):
            return 0
        return int(self.env_cfg.get("domain_rand", {}).get("lag_timesteps", 0))

    @property
    def stiffness(self) -> float:
        return float(self.env_cfg["control"]["stiffness"].get("joint", 20.0))

    @property
    def damping(self) -> float:
        return float(self.env_cfg["control"]["damping"].get("joint", 0.5))

    @property
    def default_dof_pos_policy(self) -> np.ndarray:
        angles = self.env_cfg["init_state"]["default_joint_angles"]
        return np.array([angles[name] for name in POLICY_JOINT_ORDER], dtype=np.float32)

    @property
    def default_dof_pos_unitree(self) -> np.ndarray:
        return self.default_dof_pos_policy[Go2JointOrder.unitree_to_policy]

    @property
    def command_scale(self) -> np.ndarray:
        scales = self.obs_scales
        values = [
            scales["lin_vel"],
            scales["lin_vel"],
            scales["ang_vel"],
            scales["body_height_cmd"],
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            scales["footswing_height_cmd"],
            scales["body_pitch_cmd"],
            scales["body_roll_cmd"],
            scales["stance_width_cmd"],
            scales["stance_length_cmd"],
            scales.get("aux_reward_cmd", 1.0),
        ]
        return np.array(values[: self.num_commands], dtype=np.float32)


def _as_np(values: Sequence[float] | np.ndarray | torch.Tensor, length: int, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    arr = arr.astype(np.float32).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {arr.shape[0]}")
    return arr


def gravity_in_body_from_quat_wxyz(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Return Isaac-style projected gravity from a world->body quaternion.

    MuJoCo framequat sensors and Unitree SDK2 lowstate expose quaternions in
    wxyz order. For the identity orientation this returns [0, 0, -1].
    """

    w, x, y, z = _as_np(quat_wxyz, 4, "quat_wxyz")
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm <= 1e-9:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # Rotation matrix maps body -> world. Project world gravity into body by R^T g.
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)


class UnitreeAERPolicy:
    """TorchScript AER policy wrapper with Unitree/MuJoCo observation mapping."""

    def __init__(self, bundle_dir: str | Path, device: str = "cpu"):
        self.cfg = Go2AERPolicyConfig.from_bundle(bundle_dir)
        self.device = torch.device(device)
        self.body = torch.jit.load(str(self.cfg.bundle_dir / "body_latest.jit"), map_location=self.device)
        self.adaptation_module = torch.jit.load(str(self.cfg.bundle_dir / "adaptation_module_latest.jit"), map_location=self.device)
        actuator_name = self.cfg.manifest.get("actuator_net", "unitree_go2_actuator.pt")
        actuator_path = self.cfg.bundle_dir / actuator_name
        self.actuator_network = (
            torch.jit.load(str(actuator_path), map_location=self.device).eval()
            if actuator_path.exists()
            else None
        )
        self.command_defaults = Go2CommandDefaults()
        self.default_dof_pos_policy = self.cfg.default_dof_pos_policy
        self.default_dof_pos_unitree = self.cfg.default_dof_pos_unitree
        self.commands = self.command_defaults.as_vector(0.5, 0.0, 0.0, self.cfg.num_commands)
        self.actions = torch.zeros(1, self.cfg.num_actions, dtype=torch.float32, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.clock_inputs = np.zeros(4, dtype=np.float32)
        self.gait_index = 0.0
        self.obs_history = torch.zeros(1, self.cfg.history_dim, dtype=torch.float32, device=self.device)
        self.joint_pos_err_last = torch.zeros(1, 12, dtype=torch.float32, device=self.device)
        self.joint_pos_err_last_last = torch.zeros(1, 12, dtype=torch.float32, device=self.device)
        self.joint_vel_last = torch.zeros(1, 12, dtype=torch.float32, device=self.device)
        self.joint_vel_last_last = torch.zeros(1, 12, dtype=torch.float32, device=self.device)
        self.action_lag_buffer = [
            np.zeros(12, dtype=np.float32)
            for _ in range(max(0, self.cfg.lag_timesteps) + 1)
        ]

    def reset(self) -> None:
        self.actions.zero_()
        self.last_actions.zero_()
        self.clock_inputs[:] = 0.0
        self.gait_index = 0.0
        self.obs_history.zero_()
        self.joint_pos_err_last.zero_()
        self.joint_pos_err_last_last.zero_()
        self.joint_vel_last.zero_()
        self.joint_vel_last_last.zero_()
        for item in self.action_lag_buffer:
            item[:] = 0.0

    def set_command(self, x_vel: float, y_vel: float = 0.0, yaw_vel: float = 0.0) -> None:
        self.commands = self.command_defaults.as_vector(x_vel, y_vel, yaw_vel, self.cfg.num_commands)

    def update_clock(self) -> None:
        frequency = float(self.commands[4]) if self.cfg.num_commands > 4 else 0.0
        phase = float(self.commands[5]) if self.cfg.num_commands > 5 else 0.0
        offset = float(self.commands[6]) if self.cfg.num_commands > 6 else 0.0
        bound = float(self.commands[7]) if self.cfg.num_commands > 7 else 0.0
        duration = float(self.commands[8]) if self.cfg.num_commands > 8 else 0.5
        self.gait_index = (self.gait_index + self.cfg.control_dt * frequency) % 1.0
        foot_indices = [
            self.gait_index + phase + offset + bound,
            self.gait_index + offset,
            self.gait_index + bound,
            self.gait_index + phase,
        ]
        warped = []
        duration = min(max(duration, 1e-3), 1.0 - 1e-3)
        for idx in foot_indices:
            idx = idx % 1.0
            if idx < duration:
                idx = idx * (0.5 / duration)
            else:
                idx = 0.5 + (idx - duration) * (0.5 / (1.0 - duration))
            warped.append(idx)
        self.clock_inputs[:] = np.sin(2.0 * np.pi * np.array(warped, dtype=np.float32))

    def build_observation(
        self,
        q_unitree: Sequence[float],
        dq_unitree: Sequence[float],
        quat_wxyz: Sequence[float],
        gyro_xyz: Sequence[float],
        lin_vel_xyz: Sequence[float] | None = None,
        command_xyz: Sequence[float] | None = None,
    ) -> torch.Tensor:
        if command_xyz is not None:
            self.set_command(*_as_np(command_xyz, 3, "command_xyz"))
        q_unitree_np = _as_np(q_unitree, 12, "q_unitree")
        dq_unitree_np = _as_np(dq_unitree, 12, "dq_unitree")
        gyro_np = _as_np(gyro_xyz, 3, "gyro_xyz")
        q_policy = q_unitree_np[Go2JointOrder.policy_to_unitree]
        dq_policy = dq_unitree_np[Go2JointOrder.policy_to_unitree]
        gravity = gravity_in_body_from_quat_wxyz(quat_wxyz)
        scaled_command = self.commands * self.cfg.command_scale
        obs_parts = [
            gravity,
            scaled_command,
            (q_policy - self.default_dof_pos_policy) * float(self.cfg.obs_scales["dof_pos"]),
            dq_policy * float(self.cfg.obs_scales["dof_vel"]),
            torch.clamp(self.actions, -self.cfg.clip_actions, self.cfg.clip_actions).detach().cpu().numpy().reshape(-1),
        ]
        if self.cfg.env_cfg["env"].get("observe_two_prev_actions", False):
            obs_parts.append(self.last_actions.detach().cpu().numpy().reshape(-1))
        if self.cfg.env_cfg["env"].get("observe_clock_inputs", False):
            obs_parts.append(self.clock_inputs)
        if self.cfg.env_cfg["env"].get("observe_vel", False):
            lin = _as_np(lin_vel_xyz if lin_vel_xyz is not None else [0.0, 0.0, 0.0], 3, "lin_vel_xyz")
            obs_parts = [lin * float(self.cfg.obs_scales["lin_vel"]), gyro_np * float(self.cfg.obs_scales["ang_vel"])] + obs_parts
        obs = np.concatenate(obs_parts).astype(np.float32).reshape(1, -1)
        if obs.shape[1] != self.cfg.num_observations:
            raise RuntimeError(f"observation dimension mismatch: {obs.shape[1]} != {self.cfg.num_observations}")
        # Match the final IsaacGym post-processing in
        # gym/envs/base/legged_robot.py::compute_observations.  The exported
        # student was trained after these fields were zeroed, so keeping the
        # raw gait-command tail or clock inputs here makes sim2sim
        # out-of-distribution even though the nominal observation size matches.
        obs[:, 6:18] = 0.0
        obs[:, 66:70] = 0.0
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def act_from_observation(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        self.obs_history = torch.cat((self.obs_history[:, self.cfg.num_observations :], obs), dim=-1)
        latent = self.adaptation_module.forward(self.obs_history)
        try:
            action = self.body.forward(torch.cat((obs, latent), dim=-1))
        except RuntimeError:
            action = self.body.forward(torch.cat((self.obs_history, latent), dim=-1))
        self.last_actions = self.actions.detach().clone()
        self.actions = torch.clamp(action[:, : self.cfg.num_actions], -self.cfg.clip_actions, self.cfg.clip_actions).detach()
        return self.actions

    def action_to_policy_delta(self, action: torch.Tensor | np.ndarray | None = None) -> np.ndarray:
        if action is None:
            action_np = self.actions.detach().cpu().numpy().reshape(-1)
        elif isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy().reshape(-1)
        else:
            action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_np.shape[0] != self.cfg.num_actions:
            raise ValueError(f"action must have length {self.cfg.num_actions}, got {action_np.shape[0]}")
        delta = action_np[:12] * self.cfg.action_scale
        delta[[0, 3, 6, 9]] *= self.cfg.hip_scale_reduction
        return delta.astype(np.float32)

    def action_to_policy_position_target(self, action: torch.Tensor | np.ndarray | None = None) -> np.ndarray:
        return self.action_to_policy_delta(action) + self.default_dof_pos_policy

    def action_to_unitree_position_target(self, action: torch.Tensor | np.ndarray | None = None) -> np.ndarray:
        return self.action_to_policy_position_target(action)[Go2JointOrder.unitree_to_policy]

    def lagged_action_to_unitree_position_target(self, action: torch.Tensor | np.ndarray | None = None) -> np.ndarray:
        """Return the policy target after IsaacGym's action-lag buffer."""

        delta_policy = self.action_to_policy_delta(action)
        if not self.action_lag_buffer:
            return (delta_policy + self.default_dof_pos_policy)[Go2JointOrder.unitree_to_policy]
        self.action_lag_buffer = self.action_lag_buffer[1:] + [delta_policy.copy()]
        lagged_delta = self.action_lag_buffer[0]
        return (lagged_delta + self.default_dof_pos_policy)[Go2JointOrder.unitree_to_policy]

    def pd_torque_unitree(self, q_target_unitree: Sequence[float], q_unitree: Sequence[float], dq_unitree: Sequence[float]) -> np.ndarray:
        q_target = _as_np(q_target_unitree, 12, "q_target_unitree")
        q = _as_np(q_unitree, 12, "q_unitree")
        dq = _as_np(dq_unitree, 12, "dq_unitree")
        return self.cfg.stiffness * (q_target - q) - self.cfg.damping * dq

    def actuator_net_torque_unitree(
        self,
        q_target_unitree: Sequence[float],
        q_unitree: Sequence[float],
        dq_unitree: Sequence[float],
        torque_limits_unitree: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Evaluate the same Go2 actuator network used during IsaacGym training.

        The network consumes policy-order joint position error (`q - q_target`)
        and joint velocity histories every physics step.  Returned torques are
        mapped back to Unitree/MuJoCo actuator order.
        """

        if self.actuator_network is None:
            return self.pd_torque_unitree(q_target_unitree, q_unitree, dq_unitree)

        q_target_unitree_np = _as_np(q_target_unitree, 12, "q_target_unitree")
        q_unitree_np = _as_np(q_unitree, 12, "q_unitree")
        dq_unitree_np = _as_np(dq_unitree, 12, "dq_unitree")
        q_target_policy = q_target_unitree_np[Go2JointOrder.policy_to_unitree]
        q_policy = q_unitree_np[Go2JointOrder.policy_to_unitree]
        dq_policy = dq_unitree_np[Go2JointOrder.policy_to_unitree]

        joint_pos_err = torch.tensor((q_policy - q_target_policy).reshape(1, 12), dtype=torch.float32, device=self.device)
        joint_vel = torch.tensor(dq_policy.reshape(1, 12), dtype=torch.float32, device=self.device)
        xs = torch.cat(
            (
                joint_pos_err.unsqueeze(-1),
                self.joint_pos_err_last.unsqueeze(-1),
                self.joint_pos_err_last_last.unsqueeze(-1),
                joint_vel.unsqueeze(-1),
                self.joint_vel_last.unsqueeze(-1),
                self.joint_vel_last_last.unsqueeze(-1),
            ),
            dim=-1,
        )
        with torch.no_grad():
            torque_policy = self.actuator_network(xs.reshape(12, 6)).reshape(12)

        self.joint_pos_err_last_last = self.joint_pos_err_last.detach().clone()
        self.joint_pos_err_last = joint_pos_err.detach().clone()
        self.joint_vel_last_last = self.joint_vel_last.detach().clone()
        self.joint_vel_last = joint_vel.detach().clone()

        torque_unitree = torque_policy.detach().cpu().numpy().astype(np.float32)[Go2JointOrder.unitree_to_policy]
        if torque_limits_unitree is not None:
            limits = np.abs(_as_np(torque_limits_unitree, 12, "torque_limits_unitree"))
            torque_unitree = np.clip(torque_unitree, -limits, limits)
        return torque_unitree.astype(np.float32)
