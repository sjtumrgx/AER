def to_numpy(fn):
    def thunk(*args, **kwargs):
        return fn(*args, **kwargs).cpu().numpy()

    return thunk


def lin_vel_rmsd(env, actor_critic, obs):
    return ((env.base_lin_vel[:, 0] - env.commands[:, 0]) ** 2).cpu() ** 0.5


def ang_vel_rmsd(env, actor_critic, obs):
    return ((env.base_ang_vel[:, 2] - env.commands[:, 2]) ** 2).cpu() ** 0.5


def lin_vel_x(env, actor_critic, obs):
    return env.base_lin_vel[:, 0].cpu()


def ang_vel_yaw(env, actor_critic, obs):
    return env.base_ang_vel[:, 2].cpu()


def base_height(env, actor_critic, obs):
    import torch
    return torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1).cpu()


def body_height(env, actor_critic, obs):
    return base_height(env, actor_critic, obs)


def max_torques(env, actor_critic, obs):
    import torch
    max_torque, max_torque_indices = torch.max(torch.abs(env.torques), dim=1)
    return max_torque.cpu()


def power_consumption(env, actor_critic, obs):
    import torch
    return torch.sum(torch.multiply(env.torques, env.dof_vel), dim=1).cpu()


def CoT(env, actor_critic, obs):
    # P / (mgv)
    import torch
    P = torch.abs(power_consumption(env, actor_critic, obs))
    m = (env.default_body_mass + env.payloads).cpu()
    g = 9.8  # m/s^2
    v = torch.clamp(torch.norm(env.base_lin_vel[:, 0:2], dim=1).cpu(), min=0.05)
    return P / (m * g * v)


def tracking_error(env, actor_critic, obs):
    import torch
    lin_err = torch.norm(env.base_lin_vel[:, :2] - env.commands[:, :2], dim=1)
    yaw_err = torch.abs(env.base_ang_vel[:, 2] - env.commands[:, 2])
    return torch.stack((lin_err, yaw_err), dim=1).cpu()


def fall_rate(env, actor_critic, obs):
    import torch
    return env.reset_buf.float().cpu()


def joint_power(env, actor_critic, obs):
    import torch
    return torch.abs(env.torques * env.dof_vel).cpu()


def foot_contact_schedule(env, actor_critic, obs):
    return (env.contact_forces[:, env.feet_indices, 2] > 1.).float().cpu()


def stance_duration(env, actor_critic, obs):
    import torch
    contacts = foot_contact_schedule(env, actor_critic, obs)
    if hasattr(env, "dt"):
        return contacts * float(env.dt)
    return contacts


def gait_transition(env, actor_critic, obs):
    import torch
    if hasattr(env, "desired_contact_states"):
        desired = env.desired_contact_states.float()
    else:
        desired = torch.zeros_like(env.contact_forces[:, env.feet_indices, 2])
    actual = foot_contact_schedule(env, actor_critic, obs).to(desired.device)
    return torch.abs(actual - desired).cpu()


def froude_number(env, actor_critic, obs):
    # v^2 / (gh)
    v = lin_vel_x(env, actor_critic, obs)
    g = 9.8
    h = 0.30
    return v ** 2 / (g * h)


def adaptation_loss(env, actor_critic, obs):
    import torch
    if hasattr(actor_critic, "adaptation_module"):
        pred = actor_critic.adaptation_module(obs["obs_history"])
        target = actor_critic.env_factor_encoder(obs["privileged_obs"])
        return torch.mean((pred.cpu().detach() - target.cpu().detach()) ** 2, dim=1)


def auxiliary_rewards(env, actor_critic, obs):
    rewards = {}
    for i in range(len(env.reward_functions)):
        name = env.reward_names[i]
        rew = env.reward_functions[i]() * env.reward_scales[name]
        rewards[name] = rew.cpu().detach()
        return rewards


def termination(env, actor_critic, obs):
    return env.reset_buf.cpu().detach()


def privileged_obs(env, actor_critic, obs):
    return obs["privileged_obs"].cpu().numpy()


def latents(env, actor_critic, obs):
    if hasattr(actor_critic, "get_student_latent") and "obs_history" in obs:
        return actor_critic.get_student_latent(obs["obs_history"]).detach().cpu().numpy()
    return actor_critic.env_factor_encoder(obs["privileged_obs"]).detach().cpu().numpy()


METRICS_FNS = {name: fn for name, fn in locals().items() if name not in ['to_numpy'] and "__" not in name}

LOAD_CARRY_PLOT_SPECS = {
    "CoT": {"ylabel": "Cost of Transport", "kind": "line"},
    "tracking_error": {"ylabel": "Tracking error [lin,yaw]", "kind": "line"},
    "fall_rate": {"ylabel": "Fall rate", "kind": "bar"},
    "joint_power": {"ylabel": "Joint power |tau*qdot|", "kind": "heatmap"},
    "foot_contact_schedule": {"ylabel": "Foot contact", "kind": "raster"},
    "stance_duration": {"ylabel": "Stance duration [s]", "kind": "bar"},
    "body_height": {"ylabel": "Body height [m]", "kind": "line"},
    "gait_transition": {"ylabel": "Gait transition/contact mismatch", "kind": "raster"},
}

if __name__ == '__main__':
    print(*METRICS_FNS.items(), sep="\n")

    import torch

    env = lambda: None
    env.base_lin_vel = torch.rand(10, 3)
    env.commands = torch.rand(10, 3)

    metric = lin_vel_rmsd(env, None, None)
    print(metric)
