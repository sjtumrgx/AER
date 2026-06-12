# AER Unitree MuJoCo 桥接

该目录是本 AER 仓库的 Go2 no-ROS sim2sim/sim2real 入口。当前只保留 Go2
MuJoCo 资产、Unitree SDK2 Python 桥接辅助代码，以及部署验证所需的负重行走
策略 bundle。

## 目录结构

```text
unitree_mujoco/
├── aer_policy/                         # AER 策略桥接、sim2sim、sim2real
├── policies/go2_load_carry_student/    # 已迁移的 TorchScript 策略 bundle
├── simulate_python/                    # Unitree SDK2 Python MuJoCo 桥接辅助代码
└── unitree_robots/go2/                 # Go2 MJCF、mesh、AER 平地场景
```

## 策略契约

策略 bundle 位于 `policies/go2_load_carry_student/`，包含：

- `body_latest.jit`
- `adaptation_module_latest.jit`
- `unitree_go2_actuator.pt`
- `env_cfg.yaml`
- `manifest.json`

`aer_policy/policy.py` 负责把 IsaacGym 导出的策略与 Unitree MuJoCo / SDK2 对齐：

- 观测维度：`70`，历史长度：`30`
- 动作维度：`12`
- IsaacGym 关节顺序：`FL, FR, RL, RR`
- Unitree SDK/MuJoCo 关节顺序：`FR, FL, RR, RL`
- 复现 IsaacGym 观测后处理：`obs[6:18] = 0`，`obs[66:70] = 0`
- sim2sim 默认使用位置 PD target，与实物 `LowCmd` 路径一致
- 可选 `actuator_net` torque 模式，用于调试训练时的电机模型

## Sim2sim

无窗口 smoke test：

```bash
MUJOCO_GL=egl python unitree_mujoco/aer_policy/sim2sim_aer.py \
  --steps 300 \
  --lin_speed 0.5 \
  --json
```

打开 MuJoCo 窗口并显示橙色 payload 货物：

```bash
python unitree_mujoco/aer_policy/sim2sim_aer.py \
  --viewer \
  --lin_speed 0.5 \
  --payload_mass 2.0
```

`--payload_mass` 会在 MuJoCo 中施加物理 payload 质量/质心。橙色货物 box
位于 `unitree_robots/go2/go2_aer_load_carry.xml`，默认只是可视化 mesh，不会
重复计算质量；只有设置 `--payload_mass` 时才施加物理负重。

调试 IsaacGym actuator-network 路径：

```bash
MUJOCO_GL=egl python unitree_mujoco/aer_policy/sim2sim_aer.py \
  --control actuator_net \
  --match_isaacgym_action_lag \
  --json
```

## Sim2real

先 dry-run SDK2 loop，不发布电机指令：

```bash
python unitree_mujoco/aer_policy/sim2real_aer.py --dry_run --json
```

sim2sim 通过后，上实物只需要把 DDS 网卡从 loopback 换成机器人网口：

```bash
python unitree_mujoco/aer_policy/sim2real_aer.py \
  --net eth0 \
  --domain 0 \
  --lin_speed 0.2 \
  --max_steps 5000
```

将 `eth0` 替换为实际连接 Unitree 的网口。上实物时必须保留急停人员，并从保守
速度开始。

## 依赖

- sim2sim 需要 MuJoCo Python（`pip install mujoco`）。
- `--sdk_bridge` 和实物 sim2real 需要 Unitree SDK2 Python。
- AER 桥接不使用 ROS。
