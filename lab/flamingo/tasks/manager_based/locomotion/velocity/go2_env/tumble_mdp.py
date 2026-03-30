from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _robot(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


def _sensor_body_ids(sensor_cfg: SceneEntityCfg):
    body_ids = getattr(sensor_cfg, "body_ids", None)
    if body_ids is None:
        return slice(None)
    return body_ids


def _contact_sensor(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    return env.scene.sensors[sensor_cfg.name]


def _sim_dt(env: ManagerBasedRLEnv) -> float:
    dt = getattr(env, "step_dt", None)
    if dt is None:
        dt = getattr(getattr(env, "cfg", None), "sim", None)
        dt = getattr(dt, "dt", 1.0 / 60.0)
    return float(dt)


def _ensure_jump_buffers(env: ManagerBasedRLEnv, ref: torch.Tensor):
    num_envs = ref.shape[0]
    device = ref.device
    dtype = ref.dtype

    if (not hasattr(env, "_jump_takeoff_awarded")) or (env._jump_takeoff_awarded.shape[0] != num_envs):
        env._jump_takeoff_awarded = torch.zeros(num_envs, device=device, dtype=torch.bool)
        env._jump_max_air_time = torch.zeros(num_envs, device=device, dtype=dtype)
        env._jump_peak_height = ref.clone()
        env._flip_pitch_progress = torch.zeros(num_envs, device=device, dtype=dtype)
        env._flip_peak_pitch_rate = torch.zeros(num_envs, device=device, dtype=dtype)

    reset_mask = env.episode_length_buf <= 1
    env._jump_takeoff_awarded[reset_mask] = False
    env._jump_max_air_time[reset_mask] = 0.0
    env._jump_peak_height[reset_mask] = ref[reset_mask]
    env._flip_pitch_progress[reset_mask] = 0.0
    env._flip_peak_pitch_rate[reset_mask] = 0.0


def _feet_air_times(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = _contact_sensor(env, sensor_cfg)
    air_t = sensor.data.current_air_time
    body_ids = _sensor_body_ids(sensor_cfg)
    air_t = air_t[:, body_ids]
    if air_t.ndim == 1:
        air_t = air_t.unsqueeze(-1)
    return air_t


def _feet_contact_times(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = _contact_sensor(env, sensor_cfg)
    contact_t = sensor.data.current_contact_time
    body_ids = _sensor_body_ids(sensor_cfg)
    contact_t = contact_t[:, body_ids]
    if contact_t.ndim == 1:
        contact_t = contact_t.unsqueeze(-1)
    return contact_t


def _all_feet_airborne_mask(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_air_time: float = 0.02,
) -> torch.Tensor:
    feet_air_t = _feet_air_times(env, sensor_cfg)
    return torch.all(feet_air_t > min_air_time, dim=1)


def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _robot(env, asset_cfg).data.root_pos_w[:, 2]


def projected_gravity_xy_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return torch.norm(robot.data.projected_gravity_b[:, :2], dim=1)


def lateral_tilt_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return torch.abs(robot.data.projected_gravity_b[:, 1])


def roll_rate_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, axis: int = 0) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return torch.abs(robot.data.root_ang_vel_b[:, axis])


def yaw_rate_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, axis: int = 2) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return torch.abs(robot.data.root_ang_vel_b[:, axis])


def lateral_velocity_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, axis: int = 1) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return torch.abs(robot.data.root_lin_vel_b[:, axis])


def takeoff_event_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_air_time: float = 0.02,
    min_vz: float = 0.35,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    z = robot.data.root_pos_w[:, 2]
    vz = robot.data.root_lin_vel_w[:, 2]
    _ensure_jump_buffers(env, z)

    all_off = _all_feet_airborne_mask(env, sensor_cfg, min_air_time=min_air_time)
    takeoff_now = all_off & (~env._jump_takeoff_awarded) & (vz > min_vz)
    env._jump_takeoff_awarded[takeoff_now] = True
    return takeoff_now.float()


def airtime_progress_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_air_time: float = 0.10,
) -> torch.Tensor:
    z = base_height(env, asset_cfg)
    _ensure_jump_buffers(env, z)

    feet_air_t = _feet_air_times(env, sensor_cfg)
    min_air_t = torch.min(feet_air_t, dim=1).values

    prev_max = env._jump_max_air_time.clone()
    env._jump_max_air_time = torch.maximum(env._jump_max_air_time, min_air_t)
    delta = torch.clamp(env._jump_max_air_time - prev_max, min=0.0)
    return torch.clamp(delta / max(target_air_time, 1.0e-6), min=0.0, max=1.0)


def airborne_height_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal_height: float = 0.33,
    target_bonus_height: float = 0.10,
    min_air_time: float = 0.02,
) -> torch.Tensor:
    h = base_height(env, asset_cfg)
    _ensure_jump_buffers(env, h)

    airborne = _all_feet_airborne_mask(env, sensor_cfg, min_air_time=min_air_time)
    prev_peak = env._jump_peak_height.clone()
    candidate = torch.where(airborne, h, env._jump_peak_height)
    env._jump_peak_height = torch.maximum(env._jump_peak_height, candidate)
    new_peak = torch.clamp(env._jump_peak_height - prev_peak, min=0.0)
    return torch.clamp(new_peak / max(target_bonus_height, 1.0e-6), min=0.0, max=1.0)


def upward_velocity_hint(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.20,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    h = robot.data.root_pos_w[:, 2]
    vz = robot.data.root_lin_vel_w[:, 2]
    _ensure_jump_buffers(env, h)
    pre_takeoff = (~env._jump_takeoff_awarded).float()
    return torch.clamp(vz - threshold, min=0.0) * pre_takeoff


def pre_takeoff_upright_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    h = base_height(env, asset_cfg)
    _ensure_jump_buffers(env, h)
    pre_takeoff = (~env._jump_takeoff_awarded).float()
    tilt = projected_gravity_xy_norm(env, asset_cfg)
    return torch.exp(-4.0 * tilt) * pre_takeoff


def pre_takeoff_forward_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    h = robot.data.root_pos_w[:, 2]
    _ensure_jump_buffers(env, h)
    pre_takeoff = (~env._jump_takeoff_awarded).float()
    return torch.abs(robot.data.root_lin_vel_b[:, 0]) * pre_takeoff


def post_takeoff_backward_pitch_rate_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    backward_sign: float = -1.0,
    threshold: float = 0.8,
    min_air_time: float = 0.02,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    h = robot.data.root_pos_w[:, 2]
    _ensure_jump_buffers(env, h)
    airborne = _all_feet_airborne_mask(env, sensor_cfg, min_air_time=min_air_time).float()
    pitch_rate = backward_sign * robot.data.root_ang_vel_b[:, 1]
    return torch.clamp(pitch_rate - threshold, min=0.0) * airborne


def airborne_flip_progress_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    backward_sign: float = -1.0,
    min_air_time: float = 0.02,
    target_angle_rad: float = 1.57079632679,
    rate_clip: float = 10.0,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    h = robot.data.root_pos_w[:, 2]
    _ensure_jump_buffers(env, h)

    dt = _sim_dt(env)
    airborne = _all_feet_airborne_mask(env, sensor_cfg, min_air_time=min_air_time)
    signed_pitch_rate = backward_sign * robot.data.root_ang_vel_b[:, 1]
    positive_rate = torch.clamp(signed_pitch_rate, min=0.0, max=rate_clip)

    prev_progress = env._flip_pitch_progress.clone()
    env._flip_pitch_progress = torch.where(
        airborne,
        env._flip_pitch_progress + positive_rate * dt,
        env._flip_pitch_progress,
    )
    env._flip_peak_pitch_rate = torch.maximum(env._flip_peak_pitch_rate, positive_rate)

    delta = torch.clamp(env._flip_pitch_progress - prev_progress, min=0.0)
    return torch.clamp(delta / max(target_angle_rad, 1.0e-6), min=0.0, max=1.0)


def low_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    low_height: float = 0.20,
) -> torch.Tensor:
    h = base_height(env, asset_cfg)
    return (h < low_height).float()


def body_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_contact_time: float = 0.0,
) -> torch.Tensor:
    contact_t = _feet_contact_times(env, sensor_cfg)
    return torch.any(contact_t > min_contact_time, dim=1).float()
