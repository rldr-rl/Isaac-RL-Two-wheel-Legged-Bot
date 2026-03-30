from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from . import tumble_phase


# --------------------------------------------------------------------------------------
# basic robot handles
# --------------------------------------------------------------------------------------

def _robot(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> Articulation:
    return env.scene[asset_cfg.name]


# --------------------------------------------------------------------------------------
# raw kinematics
# --------------------------------------------------------------------------------------

def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return robot.data.root_pos_w[:, 2]


def base_lin_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return robot.data.root_lin_vel_w


def base_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return robot.data.root_lin_vel_b


def base_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return robot.data.root_ang_vel_b


# --------------------------------------------------------------------------------------
# scalar metrics
# --------------------------------------------------------------------------------------

def vertical_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return base_lin_vel_w(env, asset_cfg)[:, 2]


def forward_velocity_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.abs(base_lin_vel_b(env, asset_cfg)[:, 0])


def lateral_velocity_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.abs(base_lin_vel_b(env, asset_cfg)[:, 1])


def pitch_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return base_ang_vel_b(env, asset_cfg)[:, 1]


def pitch_rate_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.abs(pitch_rate(env, asset_cfg))


def roll_rate_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.abs(base_ang_vel_b(env, asset_cfg)[:, 0])


def yaw_rate_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.abs(base_ang_vel_b(env, asset_cfg)[:, 2])


def projected_gravity_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return robot.data.projected_gravity_b


def sagittal_pitch_angle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    g_b = projected_gravity_b(env, asset_cfg)
    return torch.atan2(g_b[:, 0], -g_b[:, 2])


def lateral_tilt_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.abs(projected_gravity_b(env, asset_cfg)[:, 1])


def upright_score(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    g_b = projected_gravity_b(env, asset_cfg)
    score = 0.5 * (1.0 - g_b[:, 2])
    return torch.clamp(score, 0.0, 1.0)


def inverted_score(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    g_b = projected_gravity_b(env, asset_cfg)
    score = 0.5 * (1.0 + g_b[:, 2])
    return torch.clamp(score, 0.0, 1.0)


def sagittal_gate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    g_b = projected_gravity_b(env, asset_cfg)
    gate = torch.exp(
        -8.0 * torch.abs(g_b[:, 1])
        -0.30 * roll_rate_abs(env, asset_cfg)
        -0.15 * yaw_rate_abs(env, asset_cfg)
        -1.60 * lateral_velocity_abs(env, asset_cfg)
    )
    return torch.clamp(gate, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# phase-aware metrics
# --------------------------------------------------------------------------------------

def phase_id(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return tumble_phase.get_phase(env, asset_cfg=asset_cfg)


def pre_takeoff_mask(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return tumble_phase.is_pre_takeoff(env, asset_cfg=asset_cfg)


def in_air_mask(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return tumble_phase.is_in_air(env, asset_cfg=asset_cfg)


def landed_mask(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return tumble_phase.has_landed(env, asset_cfg=asset_cfg)


def recovered_mask(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return tumble_phase.has_recovered(env, asset_cfg=asset_cfg)


def cumulative_pitch_rotation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return tumble_phase.cumulative_pitch_rotation(env, asset_cfg=asset_cfg)


def peak_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return tumble_phase.peak_height(env, asset_cfg=asset_cfg)


def air_time(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return tumble_phase.air_time(env, asset_cfg=asset_cfg)


# --------------------------------------------------------------------------------------
# convenience metrics for reward design
# --------------------------------------------------------------------------------------

def jump_height_above_nominal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    nominal_height: float = 0.33,
) -> torch.Tensor:
    return torch.clamp(base_height(env, asset_cfg) - nominal_height, min=0.0)


def takeoff_quality(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vertical_vel_threshold: float = 0.35,
) -> torch.Tensor:
    vz = vertical_velocity(env, asset_cfg)
    gate = sagittal_gate(env, asset_cfg)
    pre = pre_takeoff_mask(env, asset_cfg).float()
    return torch.clamp(vz - vertical_vel_threshold, min=0.0) * gate * pre


def in_air_pitch_drive(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.50,
) -> torch.Tensor:
    pr = pitch_rate(env, asset_cfg)
    gate = sagittal_gate(env, asset_cfg)
    air = in_air_mask(env, asset_cfg).float()
    return torch.clamp(pr - threshold, min=0.0) * gate * air


def landing_stability_score(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    nominal_height: float = 0.33,
) -> torch.Tensor:
    landed = landed_mask(env, asset_cfg).float()
    upright = upright_score(env, asset_cfg)
    low_vz = torch.exp(-3.0 * torch.abs(vertical_velocity(env, asset_cfg)))
    low_xy = torch.exp(-2.0 * (forward_velocity_abs(env, asset_cfg) + lateral_velocity_abs(env, asset_cfg)))
    near_nominal = torch.exp(-20.0 * torch.abs(base_height(env, asset_cfg) - nominal_height))
    return landed * upright * low_vz * low_xy * near_nominal
