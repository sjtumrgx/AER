# AER Unitree MuJoCo bridge

This directory is the no-ROS Go2 sim2sim/sim2real surface for this AER checkout.
It keeps only the Go2 MuJoCo assets, Unitree SDK2 Python bridge helpers, and the
migrated load-carry policy bundle needed for deployment validation.

## Layout

```text
unitree_mujoco/
├── aer_policy/                         # AER policy bridge, sim2sim, sim2real
├── policies/go2_load_carry_student/    # migrated TorchScript policy bundle
├── simulate_python/                    # Unitree SDK2 Python MuJoCo bridge helpers
└── unitree_robots/go2/                 # Go2 MJCF, meshes, flat AER scene
```

## Policy contract

The migrated policy bundle is `policies/go2_load_carry_student/` and contains:

- `body_latest.jit`
- `adaptation_module_latest.jit`
- `unitree_go2_actuator.pt`
- `env_cfg.yaml`
- `manifest.json`

`aer_policy/policy.py` aligns the IsaacGym policy with Unitree MuJoCo / SDK2:

- observation shape: `70`, history length: `30`
- action shape: `12`
- IsaacGym joint order: `FL, FR, RL, RR`
- Unitree SDK/MuJoCo joint order: `FR, FL, RR, RL`
- IsaacGym observation post-processing: `obs[6:18] = 0`, `obs[66:70] = 0`
- default sim2sim control: position-PD targets, matching real-robot `LowCmd`
- optional `actuator_net` torque mode for debugging the training motor model

## Sim2sim

Headless smoke test:

```bash
MUJOCO_GL=egl python unitree_mujoco/aer_policy/sim2sim_aer.py \
  --steps 300 \
  --lin_speed 0.5 \
  --json
```

Open a MuJoCo viewer with the visible orange payload mesh:

```bash
python unitree_mujoco/aer_policy/sim2sim_aer.py \
  --viewer \
  --lin_speed 0.5 \
  --payload_mass 2.0
```

`--payload_mass` applies physical base payload mass/COM in MuJoCo. The orange
payload box in `unitree_robots/go2/go2_aer_load_carry.xml` is visual-only, so it
is visible for review without double-counting mass unless `--payload_mass` is
set.

For IsaacGym actuator-network debugging:

```bash
MUJOCO_GL=egl python unitree_mujoco/aer_policy/sim2sim_aer.py \
  --control actuator_net \
  --match_isaacgym_action_lag \
  --json
```

## Sim2real

Dry-run the SDK2 loop without publishing motor commands:

```bash
python unitree_mujoco/aer_policy/sim2real_aer.py --dry_run --json
```

After sim2sim is acceptable, deploy by changing only the DDS network interface
from loopback to the robot-facing NIC:

```bash
python unitree_mujoco/aer_policy/sim2real_aer.py \
  --net eth0 \
  --domain 0 \
  --lin_speed 0.2 \
  --max_steps 5000
```

Replace `eth0` with the actual Unitree network interface. Keep an emergency-stop
operator present and start with conservative commands.

## Dependencies

- MuJoCo Python (`pip install mujoco`) for sim2sim.
- Unitree SDK2 Python for `--sdk_bridge` and real-robot sim2real.
- No ROS is used by the AER bridge.
