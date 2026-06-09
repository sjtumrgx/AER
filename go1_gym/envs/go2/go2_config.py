from go1_gym.envs.go1.go1_config import Go1Config


class Go2Config(Go1Config):
    """Unitree Go2 simulation config matching the existing Go1 task interface.

    Go2 keeps the same 12-DOF joint naming convention as Go1, so the existing
    velocity-tracking environment, history wrapper, commands, and reward code can
    be reused while swapping the robot asset and first-pass morphology constants.
    """

    class init_state(Go1Config.init_state):
        # The Go2 MJCF source places the floating base at z=0.445. The URDF is
        # loaded by Isaac Gym with the configured base pose, so start slightly
        # above nominal standing height to avoid initial ground penetration.
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,

            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class asset(Go1Config.asset):
        file = '{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "Head", "hip", "thigh", "calf"]
        self_collisions = 0
        flip_visual_attachments = False
        fix_base_link = False

    class rewards(Go1Config.rewards):
        # Go2's nominal standing height is higher than Go1 because of the longer
        # body/leg geometry in the downloaded model.
        base_height_target = 0.32
        terminal_body_height = 0.08

    class commands(Go1Config.commands):
        # Conservative first-pass command bounds for Go2 adaptation. Expand only
        # after policy validation in simulation.
        lin_vel_x = [-1.0, 1.0]
        lin_vel_y = [-0.6, 0.6]
        ang_vel_yaw = [-1.0, 1.0]
        limit_vel_x = [-2.0, 2.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-2.0, 2.0]
