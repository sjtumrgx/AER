"""Shared Go2 play-time command defaults.

The policy was trained with a 15-dimensional command vector.  Keeping these
conditioning values inside the training ranges is important even for visual
playback: invalid zeros for gait frequency, gait duration, foot swing height,
or stance width make the policy produce folded/disconnected-looking legs while
the simulated robot can still translate forward.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Go2PlayCommandDefaults:
    """Nominal commands matching the Go2 training command distribution."""

    body_height: float = 0.0
    gait_frequency: float = 3.0
    gait_phase: float = 0.5  # trot: diagonal legs out of phase
    gait_offset: float = 0.0
    gait_bound: float = 0.0
    gait_duration: float = 0.5
    footswing_height: float = 0.08
    body_pitch: float = 0.0
    body_roll: float = 0.0
    stance_width: float = 0.25
    stance_length: float = 0.40
    aux_reward_coef: float = 0.0


NOMINAL_GO2_PLAY_COMMANDS = Go2PlayCommandDefaults()


def assign_nominal_go2_commands(commands, x_vel: float, y_vel: float = 0.0, yaw_vel: float = 0.0) -> None:
    """Fill an Isaac Gym command tensor with valid nominal Go2 play commands.

    ``commands`` is intentionally duck-typed so this helper stays lightweight and
    can be imported by ``scripts/* --help`` tests without importing Isaac Gym or
    torch. It supports torch tensors and numpy arrays.
    """

    defaults = NOMINAL_GO2_PLAY_COMMANDS
    commands[:, 0] = x_vel
    commands[:, 1] = y_vel
    commands[:, 2] = yaw_vel
    commands[:, 3] = defaults.body_height
    commands[:, 4] = defaults.gait_frequency
    commands[:, 5] = defaults.gait_phase
    commands[:, 6] = defaults.gait_offset
    commands[:, 7] = defaults.gait_bound
    commands[:, 8] = defaults.gait_duration
    commands[:, 9] = defaults.footswing_height
    commands[:, 10] = defaults.body_pitch
    commands[:, 11] = defaults.body_roll
    commands[:, 12] = defaults.stance_width
    if commands.shape[1] > 13:
        commands[:, 13] = defaults.stance_length
    if commands.shape[1] > 14:
        commands[:, 14] = defaults.aux_reward_coef
