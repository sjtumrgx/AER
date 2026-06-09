# Go2 robot assets

This folder mirrors the existing `resources/robots/go1` layout for Unitree Go2.

- `urdf/go2.urdf`: Isaac Gym friendly URDF generated from `Go2_URDF/GO2_URDF/urdf/go2_description.urdf` with `package://go2_description/...` mesh URIs rewritten to relative paths.
- `urdf/go2_description.urdf`: Original ROS-style URDF kept for reference.
- `dae/`, `meshes/`, `config/`, `launch/`, `xacro/`, `package.xml`, `CMakeLists.txt`: Migrated from the provided Go2 URDF package.
- `xml/go2.xml`, `xml/scene_go2.xml`, `xml/assets/`: Downloaded from `unitreerobotics/unitree_rl_mjlab/src/assets/robots/unitree_go2`.

Use `go1_gym.envs.go2.Go2Config` or `scripts/train.py --robot go2 ...` for simulation training.
