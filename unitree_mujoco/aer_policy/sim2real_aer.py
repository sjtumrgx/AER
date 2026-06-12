"""Deploy the migrated AER Go2 policy through Unitree SDK2 Python.

No ROS is used. After sim2sim succeeds, switching from simulator loopback to a
real robot is intentionally a network-interface change, for example:

    python unitree_mujoco/aer_policy/sim2real_aer.py --net eth0 --max_steps 5000

Use --dry_run for local validation without publishing LowCmd messages.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import time
from threading import Event, Lock

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from unitree_mujoco.aer_policy.policy import UnitreeAERPolicy
else:
    from .policy import UnitreeAERPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "policies" / "go2_load_carry_student"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="AER Go2 sim2real through Unitree SDK2 Python only.")
    parser.add_argument("--policy_dir", default=str(DEFAULT_POLICY), help="Migrated AER policy bundle directory.")
    parser.add_argument("--net", default="lo", help="DDS network interface: lo for sim, robot NIC for sim2real.")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id. Real Go2 commonly uses 0.")
    parser.add_argument("--max_steps", type=int, default=1000000, help="Maximum 50Hz policy steps.")
    parser.add_argument("--lin_speed", type=float, default=0.5, help="Forward velocity command in m/s.")
    parser.add_argument("--yaw_speed", type=float, default=0.0, help="Yaw velocity command in rad/s.")
    parser.add_argument("--kp", type=float, default=None, help="Override low-level joint kp; default comes from policy cfg.")
    parser.add_argument("--kd", type=float, default=None, help="Override low-level joint kd; default comes from policy cfg.")
    parser.add_argument("--dry_run", action="store_true", help="Load policy and compute targets, but do not publish LowCmd.")
    parser.add_argument("--json", action="store_true", help="Print dry-run summary JSON.")
    return parser


def _lazy_import_sdk2():
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Unitree SDK2 Python dependency. Install https://github.com/unitreerobotics/unitree_sdk2_python."
        ) from exc
    return ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber, unitree_go_msg_dds__LowCmd_, LowCmd_, LowState_, CRC


class LowStateBuffer:
    def __init__(self):
        self._lock = Lock()
        self._event = Event()
        self.latest = None

    def update(self, msg):
        with self._lock:
            self.latest = msg
            self._event.set()

    def wait(self, timeout: float = 2.0):
        if not self._event.wait(timeout):
            return None
        with self._lock:
            return self.latest


def _extract_lowstate(msg):
    q = np.array([msg.motor_state[i].q for i in range(12)], dtype=np.float32)
    dq = np.array([msg.motor_state[i].dq for i in range(12)], dtype=np.float32)
    quat = np.array([msg.imu_state.quaternion[i] for i in range(4)], dtype=np.float32)
    gyro = np.array([msg.imu_state.gyroscope[i] for i in range(3)], dtype=np.float32)
    return q, dq, quat, gyro


def _make_lowcmd(default_cmd_type):
    cmd = default_cmd_type()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0
    return cmd


def _fill_lowcmd(cmd, crc, q_target, kp: float, kd: float):
    for i in range(12):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q = float(q_target[i])
        cmd.motor_cmd[i].kp = float(kp)
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = float(kd)
        cmd.motor_cmd[i].tau = 0.0
    cmd.crc = crc.Crc(cmd)


def dry_run(args) -> dict:
    policy = UnitreeAERPolicy(args.policy_dir)
    policy.set_command(args.lin_speed, 0.0, args.yaw_speed)
    obs = policy.build_observation(
        q_unitree=policy.default_dof_pos_unitree,
        dq_unitree=[0.0] * 12,
        quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        gyro_xyz=[0.0, 0.0, 0.0],
        command_xyz=[args.lin_speed, 0.0, args.yaw_speed],
    )
    action = policy.act_from_observation(obs)
    q_target = policy.action_to_unitree_position_target(action)
    return {
        "policy_dir": str(Path(args.policy_dir).expanduser().resolve()),
        "net": args.net,
        "domain": args.domain,
        "dry_run": True,
        "obs_dim": int(obs.shape[1]),
        "action_dim": int(action.shape[1]),
        "q_target_first3": [float(x) for x in q_target[:3]],
    }


def run_real(args) -> dict:
    if args.dry_run:
        return dry_run(args)
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber, default_cmd_type, LowCmd_, LowState_, CRC = _lazy_import_sdk2()
    ChannelFactoryInitialize(args.domain, args.net)
    policy = UnitreeAERPolicy(args.policy_dir)
    policy.set_command(args.lin_speed, 0.0, args.yaw_speed)
    state_buffer = LowStateBuffer()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(state_buffer.update, 10)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()
    cmd = _make_lowcmd(default_cmd_type)

    next_time = time.perf_counter()
    for step in range(args.max_steps):
        msg = state_buffer.wait(timeout=2.0)
        if msg is None:
            raise RuntimeError("No rt/lowstate received. Check --net, robot power, and SDK2 domain.")
        q, dq, quat, gyro = _extract_lowstate(msg)
        obs = policy.build_observation(q, dq, quat, gyro, command_xyz=[args.lin_speed, 0.0, args.yaw_speed])
        action = policy.act_from_observation(obs)
        q_target = policy.action_to_unitree_position_target(action)
        kp = policy.cfg.stiffness if args.kp is None else args.kp
        kd = policy.cfg.damping if args.kd is None else args.kd
        _fill_lowcmd(cmd, crc, q_target, kp, kd)
        pub.Write(cmd)
        policy.update_clock()
        next_time += policy.cfg.control_dt
        time.sleep(max(0.0, next_time - time.perf_counter()))
    return {"dry_run": False, "steps": args.max_steps, "net": args.net, "domain": args.domain}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_real(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("sim2real summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
