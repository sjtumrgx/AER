"""Go2 load-carrying asymmetric teacher-student configuration."""

from gym.envs.go2.go2_config_adaptive_terrain import AdaptiveGo2ConfigTerrain


class LoadCarryGo2Config(AdaptiveGo2ConfigTerrain):
    """Payload domain-randomized Go2 config for asymmetric teacher-student RL.

    Privileged observation schema (40 dims):
    - robot base linear/angular velocity: 6
    - payload mass: 1
    - payload CoM offset: 3
    - payload inertia diagonal/off-diagonal surrogate: 6
    - payload relative position + quaternion: 7
    - payload relative linear/angular velocity: 6
    - ground friction: 1
    - terrain parameters: 4
    - external disturbance linear/angular velocity impulse: 6
    """

    class env(AdaptiveGo2ConfigTerrain.env):
        num_observation_history = 30  # H in [20, 50]
        num_privileged_obs = 40
        privileged_load_carry_schema = True

        # Disable legacy privileged schema terms so the 40-dim load schema is exact.
        priv_observe_friction = False
        priv_observe_restitution = False
        priv_observe_base_mass = False
        priv_observe_com_displacement = False
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_Kp_factor = False
        priv_observe_Kd_factor = False
        priv_observe_body_velocity = False
        priv_observe_body_height = False
        priv_observe_gravity = False
        priv_observe_clock_inputs = False
        priv_observe_desired_contact_states = False

        # Requested privileged state channels.
        priv_observe_robot_base_velocity = True
        priv_observe_payload_mass = True
        priv_observe_payload_com = True
        priv_observe_payload_inertia = True
        priv_observe_payload_relative_pose = True
        priv_observe_payload_relative_velocity = True
        priv_observe_ground_friction = True
        priv_observe_terrain_parameters = True
        priv_observe_external_disturbance = True

    class asset(AdaptiveGo2ConfigTerrain.asset):
        # Dedicated load-carry URDF adds a visual-only cargo mesh so review videos
        # show the payload. Physical payload mass/CoM is still applied in
        # LeggedRobot._process_rigid_body_props to avoid double-counting mass.
        file = '{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_load_carry_visual.urdf'

    class domain_rand(AdaptiveGo2ConfigTerrain.domain_rand):
        randomize_base_mass = True
        added_mass_range = [0.0, 8.0]
        randomize_com_displacement = True
        com_displacement_range = [-0.12, 0.12]
        randomize_payload_inertia = True
        payload_inertia_range = [0.005, 0.08]
        randomize_payload_relative_pose = True
        payload_relative_pos_range = [-0.08, 0.08]
        payload_relative_angle_range = [-0.25, 0.25]
        randomize_payload_relative_velocity = True
        payload_relative_lin_vel_range = [-0.3, 0.3]
        payload_relative_ang_vel_range = [-0.8, 0.8]
        push_robots = True
        max_push_vel_xy = 1.5
        max_push_vel_z = 0.5
        max_push_ang_rpy = 1.0
        randomize_ground_friction = True
        ground_friction_range = [0.2, 2.5]
        friction_range = [0.2, 2.5]

    class normalization(AdaptiveGo2ConfigTerrain.normalization):
        added_mass_range = [0.0, 8.0]
        com_displacement_range = [-0.12, 0.12]
        payload_inertia_range = [0.005, 0.08]
        payload_relative_pos_range = [-0.08, 0.08]
        payload_relative_lin_vel_range = [-0.3, 0.3]
        payload_relative_ang_vel_range = [-0.8, 0.8]
        terrain_parameter_range = [0.0, 1.0]
        external_disturbance_range = [-1.5, 1.5]
        ground_friction_range = [0.2, 2.5]
        friction_range = [0.2, 2.5]

    class rewards(AdaptiveGo2ConfigTerrain.rewards):
        energy_regularization_mode = "adaptive"
        fixed_energy_alpha = 1.0
        transport_speed_floor = 0.05
        transport_cost_clip = 50.0

        class scales(AdaptiveGo2ConfigTerrain.rewards.scales):
            energy = 0.0
            energy_dep = 0.0
            energy_new_actual = 0.0
            energy_new_cmd = 0.0
            base_height = 0.0
            # Cost term: reward function returns alpha * CoT, negative scale penalizes it.
            load_normalized_transport = -0.02


class LoadCarryGo2NoEnergyConfig(LoadCarryGo2Config):
    """Baseline: payload domain randomization with no energy regularization."""

    class rewards(LoadCarryGo2Config.rewards):
        energy_regularization_mode = "none"

        class scales(LoadCarryGo2Config.rewards.scales):
            load_normalized_transport = 0.0


class LoadCarryGo2FixedEnergyConfig(LoadCarryGo2Config):
    """Baseline: fixed alpha_E instead of latent-adaptive alpha_E."""

    class rewards(LoadCarryGo2Config.rewards):
        energy_regularization_mode = "fixed"
        fixed_energy_alpha = 1.0


class LoadCarryGo2DomainRandPolicyConfig(LoadCarryGo2NoEnergyConfig):
    """Baseline: ordinary domain-randomized policy without teacher-student distillation."""

    class env(LoadCarryGo2NoEnergyConfig.env):
        # Still provide privileged critic observations during training, but train script
        # selects student_ppo/domain_rand_policy so no frozen teacher is used.
        privileged_load_carry_schema = True
