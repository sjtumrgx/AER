# Go2 robot assets

This folder contains the Unitree Go2 assets used by the Isaac Gym training and play scripts.

- `urdf/go2.urdf`: Isaac Gym friendly URDF generated from the supplied Go2 URDF package with ROS package mesh URIs rewritten to relative paths.
- `urdf/go2_description.urdf`: Original ROS-style URDF kept for reference.
- `dae/`, `meshes/`, `config/`, `launch/`, `xacro/`, `package.xml`, `CMakeLists.txt`: Migrated from the supplied Go2 URDF package.
- `xml/go2.xml`, `xml/scene_go2.xml`, `xml/assets/`: Downloaded from `unitreerobotics/unitree_rl_mjlab/src/assets/robots/unitree_go2`.

Use `gym.envs.go2.Go2Config` or `scripts/train.py --cfg adaptive_en ...` for simulation training.
