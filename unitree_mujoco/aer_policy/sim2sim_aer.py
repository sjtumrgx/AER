"""Run AER Go2 TorchScript policy in Unitree MuJoCo (sim2sim).

This script does not use ROS. It can optionally initialize Unitree SDK2 Python's
DDS bridge (`--sdk_bridge`) while the policy loop controls MuJoCo directly with
the same low-level PD targets used by sim2real.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from unitree_mujoco.aer_policy.policy import UnitreeAERPolicy
else:
    from .policy import UnitreeAERPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "policies" / "go2_load_carry_student"
DEFAULT_SCENE = ROOT / "unitree_robots" / "go2" / "scene_aer_flat.xml"
DEFAULT_METRICS = ROOT / "logs" / "sim2sim_aer_metrics.csv"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="AER Go2 policy sim2sim in Unitree MuJoCo using SDK2 Python only.")
    parser.add_argument("--policy_dir", default=str(DEFAULT_POLICY), help="Migrated AER policy bundle directory.")
    parser.add_argument("--scene", default=str(DEFAULT_SCENE), help="Unitree MuJoCo Go2 scene XML.")
    parser.add_argument("--steps", type=int, default=300, help="Number of 50Hz policy steps to simulate.")
    parser.add_argument("--lin_speed", type=float, default=0.5, help="Forward velocity command in m/s.")
    parser.add_argument("--yaw_speed", type=float, default=0.0, help="Yaw velocity command in rad/s.")
    parser.add_argument("--viewer", action="store_true", help="Open a MuJoCo passive viewer window.")
    parser.add_argument("--sdk_bridge", action="store_true", help="Initialize Unitree SDK2 DDS bridge in the simulator.")
    parser.add_argument("--net", default="lo", help="Network interface for SDK2 DDS bridge; use real NIC for robot.")
    parser.add_argument("--domain", type=int, default=1, help="DDS domain id for SDK2 bridge.")
    parser.add_argument(
        "--control",
        choices=["actuator_net", "pd"],
        default="pd",
        help="Low-level MuJoCo torque source. Default matches sim2real LowCmd position-PD deployment.",
    )
    parser.add_argument("--kp", type=float, default=None, help="Override low-level joint kp; default comes from policy cfg.")
    parser.add_argument("--kd", type=float, default=None, help="Override low-level joint kd; default comes from policy cfg.")
    parser.add_argument("--disable_action_lag", action="store_true", default=True, help="Disable IsaacGym action-lag emulation.")
    parser.add_argument("--match_isaacgym_action_lag", dest="disable_action_lag", action="store_false", help="Use env_cfg domain_rand.lag_timesteps.")
    parser.add_argument("--friction", type=float, default=1.0, help="Override MuJoCo primary geom friction for flat-ground gait checks.")
    parser.add_argument("--joint_damping", type=float, default=0.0, help="Override MuJoCo joint damping to match IsaacGym asset damping.")
    parser.add_argument("--joint_armature", type=float, default=0.0, help="Override MuJoCo joint armature to match IsaacGym asset armature.")
    parser.add_argument("--joint_frictionloss", type=float, default=0.0, help="Override MuJoCo joint friction loss.")
    parser.add_argument("--payload_mass", type=float, default=0.0, help="Extra payload mass applied to base_link for load-carry checks.")
    parser.add_argument("--payload_com_x", type=float, default=0.0, help="Physical payload COM x offset in base_link frame.")
    parser.add_argument("--payload_com_z", type=float, default=0.0, help="Physical payload COM z offset in base_link frame.")
    parser.add_argument("--output_csv", default=str(DEFAULT_METRICS), help="CSV metrics path.")
    parser.add_argument("--json", action="store_true", help="Print summary JSON to stdout.")
    return parser


def _lazy_import_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install MuJoCo Python with `pip install mujoco`.") from exc
    return mujoco


def _sensor_block(data, num_motor: int):
    q = np.asarray(data.sensordata[:num_motor], dtype=np.float32).copy()
    dq = np.asarray(data.sensordata[num_motor : 2 * num_motor], dtype=np.float32).copy()
    quat = np.asarray(data.sensordata[3 * num_motor : 3 * num_motor + 4], dtype=np.float32).copy()
    gyro = np.asarray(data.sensordata[3 * num_motor + 4 : 3 * num_motor + 7], dtype=np.float32).copy()
    frame_pos = np.asarray(data.sensordata[3 * num_motor + 10 : 3 * num_motor + 13], dtype=np.float32).copy()
    frame_vel = np.asarray(data.sensordata[3 * num_motor + 13 : 3 * num_motor + 16], dtype=np.float32).copy()
    return q, dq, quat, gyro, frame_pos, frame_vel


def _maybe_start_sdk_bridge(model, data, net: str, domain: int):
    sim_py = ROOT / "simulate_python"
    sys.path.insert(0, str(sim_py))
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py_bridge import UnitreeSdk2Bridge
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Unitree SDK2 Python dependency. Install unitree_sdk2_python, or rerun without --sdk_bridge."
        ) from exc
    ChannelFactoryInitialize(domain, net)
    return UnitreeSdk2Bridge(model, data)


def run_sim2sim(args) -> dict:
    mujoco = _lazy_import_mujoco()
    viewer = None
    if args.viewer:
        import mujoco.viewer

    scene = Path(args.scene).expanduser().resolve()
    policy = UnitreeAERPolicy(args.policy_dir)
    policy.set_command(args.lin_speed, 0.0, args.yaw_speed)

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005
    if args.friction is not None:
        model.geom_friction[:, 0] = float(args.friction)
    model.dof_damping[:] = float(args.joint_damping)
    model.dof_armature[:] = float(args.joint_armature)
    model.dof_frictionloss[:] = float(args.joint_frictionloss)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if args.payload_mass > 0.0 and base_body_id >= 0:
        base_mass = float(model.body_mass[base_body_id])
        payload_com = np.array([args.payload_com_x, 0.0, args.payload_com_z], dtype=np.float64)
        combined_mass = base_mass + float(args.payload_mass)
        model.body_ipos[base_body_id] = (
            model.body_ipos[base_body_id] * base_mass + payload_com * float(args.payload_mass)
        ) / combined_mass
        model.body_mass[base_body_id] = combined_mass
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    # Align MuJoCo initial state to the Isaac Gym training reset instead of the
    # upstream MuJoCo "home" keyframe (0.27m, all thighs/calf mirrored), which
    # makes the exported policy start from an out-of-distribution crouch.
    data.qpos[0:3] = np.array(policy.cfg.env_cfg["init_state"]["pos"], dtype=np.float32)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # MuJoCo free-joint quat is wxyz.
    data.qpos[7:19] = policy.default_dof_pos_policy
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    num_motor = model.nu
    if num_motor != 12:
        raise RuntimeError(f"Go2 policy expects 12 actuators, scene has {num_motor}")

    sdk_bridge = _maybe_start_sdk_bridge(model, data, args.net, args.domain) if args.sdk_bridge else None
    if args.viewer:
        viewer = mujoco.viewer.launch_passive(model, data)

    policy_steps_per_control = max(1, round(policy.cfg.control_dt / model.opt.timestep))
    torque_limits = np.max(np.abs(np.asarray(model.actuator_ctrlrange, dtype=np.float32)), axis=1)
    rows = []
    start_x = float(data.qpos[0])
    start_time = time.perf_counter()
    try:
        for step in range(args.steps):
            q, dq, quat, gyro, frame_pos, frame_vel = _sensor_block(data, num_motor)
            obs = policy.build_observation(q, dq, quat, gyro, frame_vel, [args.lin_speed, 0.0, args.yaw_speed])
            action = policy.act_from_observation(obs)
            q_target = policy.action_to_unitree_position_target(action)
            kp = policy.cfg.stiffness if args.kp is None else args.kp
            kd = policy.cfg.damping if args.kd is None else args.kd
            torque = np.zeros(num_motor, dtype=np.float32)
            for _ in range(policy_steps_per_control):
                q_sub, dq_sub, *_ = _sensor_block(data, num_motor)
                if not args.disable_action_lag:
                    q_target = policy.lagged_action_to_unitree_position_target(action)
                if args.control == "actuator_net":
                    torque = policy.actuator_net_torque_unitree(q_target, q_sub, dq_sub, torque_limits)
                else:
                    torque = np.clip(kp * (q_target - q_sub) - kd * dq_sub, -torque_limits, torque_limits)
                data.ctrl[:] = torque
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
            policy.update_clock()
            base_x = float(data.qpos[0])
            base_z = float(data.qpos[2])
            rows.append(
                {
                    "step": step,
                    "base_x": base_x,
                    "base_z": base_z,
                    "base_vx": float(frame_vel[0]) if frame_vel.size >= 1 else 0.0,
                    "action_rms": float(np.sqrt(np.mean(np.square(action.detach().cpu().numpy())))),
                    "torque_rms": float(np.sqrt(np.mean(np.square(torque)))),
                }
            )
    finally:
        if viewer is not None:
            viewer.close()

    elapsed = time.perf_counter() - start_time
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    sim_time = args.steps * policy.cfg.control_dt
    summary = {
        "steps": args.steps,
        "sim_time_s": sim_time,
        "wall_time_s": elapsed,
        "mean_forward_velocity": (float(data.qpos[0]) - start_x) / max(sim_time, 1e-6),
        "mean_base_height": float(np.mean([r["base_z"] for r in rows])) if rows else 0.0,
        "min_base_height": float(np.min([r["base_z"] for r in rows])) if rows else 0.0,
        "output_csv": str(output_csv),
        "scene": str(scene),
        "policy_dir": str(Path(args.policy_dir).expanduser().resolve()),
        "sdk_bridge": bool(sdk_bridge),
        "control": args.control,
        "action_lag_timesteps": 0 if args.disable_action_lag else policy.cfg.lag_timesteps,
        "friction": args.friction,
        "joint_damping": args.joint_damping,
        "joint_armature": args.joint_armature,
        "joint_frictionloss": args.joint_frictionloss,
        "payload_mass": args.payload_mass,
        "payload_com_x": args.payload_com_x,
        "payload_com_z": args.payload_com_z,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_sim2sim(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("sim2sim summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
