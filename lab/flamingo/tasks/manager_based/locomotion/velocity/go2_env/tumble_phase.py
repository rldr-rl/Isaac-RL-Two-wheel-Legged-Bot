from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# phase ids
STANCE = 0
CROUCH = 1
TAKEOFF = 2
AERIAL = 3
LANDING = 4
RECOVERY = 5


DEFAULT_CONTACT_SENSOR_NAMES = (
    "contact_forces",
    "contact_sensor",
    "foot_contact",
    "feet_contact",
)


class TumblePhase:
    STANCE = STANCE
    CROUCH = CROUCH
    TAKEOFF = TAKEOFF
    AERIAL = AERIAL
    LANDING = LANDING
    RECOVERY = RECOVERY


# --------------------------------------------------------------------------------------
# buffer helpers
# --------------------------------------------------------------------------------------

def initialize_buffers(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Create per-env buffers used by the tumble task state machine.

    Call this from reward/observation/termination functions before using phase data.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w[:, 2]
    num_envs = height.shape[0]
    device = height.device
    dtype = height.dtype

    if not hasattr(env, "_tumble_phase") or env._tumble_phase.shape[0] != num_envs:
        env._tumble_phase = torch.zeros(num_envs, device=device, dtype=torch.long)
        env._tumble_time_in_phase = torch.zeros(num_envs, device=device, dtype=dtype)
        env._tumble_air_time = torch.zeros(num_envs, device=device, dtype=dtype)
        env._tumble_takeoff_step = torch.zeros(num_envs, device=device, dtype=torch.long)
        env._tumble_takeoff_height = torch.zeros(num_envs, device=device, dtype=dtype)
        env._tumble_peak_height = height.clone()
        env._tumble_prev_pitch = torch.zeros(num_envs, device=device, dtype=dtype)
        env._tumble_cum_pitch = torch.zeros(num_envs, device=device, dtype=dtype)
        env._tumble_full_rotation = torch.zeros(num_envs, device=device, dtype=torch.bool)
        env._tumble_landed = torch.zeros(num_envs, device=device, dtype=torch.bool)
        env._tumble_recovered = torch.zeros(num_envs, device=device, dtype=torch.bool)
        env._tumble_takeoff_seen = torch.zeros(num_envs, device=device, dtype=torch.bool)

    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
        reset_idx = reset_mask.nonzero(as_tuple=False).squeeze(-1)
        reset_buffers(env, reset_idx, asset_cfg=asset_cfg)


def reset_buffers(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    if env_ids.numel() == 0:
        return
    robot: Articulation = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w[env_ids, 2]
    pitch = _pitch_from_projected_gravity(robot.data.projected_gravity_b[env_ids])

    env._tumble_phase[env_ids] = STANCE
    env._tumble_time_in_phase[env_ids] = 0.0
    env._tumble_air_time[env_ids] = 0.0
    env._tumble_takeoff_step[env_ids] = 0
    env._tumble_takeoff_height[env_ids] = height
    env._tumble_peak_height[env_ids] = height
    env._tumble_prev_pitch[env_ids] = pitch
    env._tumble_cum_pitch[env_ids] = 0.0
    env._tumble_full_rotation[env_ids] = False
    env._tumble_landed[env_ids] = False
    env._tumble_recovered[env_ids] = False
    env._tumble_takeoff_seen[env_ids] = False


# --------------------------------------------------------------------------------------
# sensor helpers
# --------------------------------------------------------------------------------------

def _try_get_contact_sensor(env: ManagerBasedRLEnv):
    for name in DEFAULT_CONTACT_SENSOR_NAMES:
        try:
            return env.scene[name]
        except Exception:
            continue
    return None


def _feet_contact_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Best-effort feet contact estimation.

    1) If a contact sensor exists, uses a force threshold.
    2) Otherwise falls back to height + vertical velocity heuristic.
    """
    robot = env.scene["robot"]
    sensor = _try_get_contact_sensor(env)

    if sensor is not None:
        data = getattr(sensor, "data", None)
        if data is not None:
            force = None
            for attr in ("net_forces_w", "net_forces_b", "force_matrix_w"):
                if hasattr(data, attr):
                    force = getattr(data, attr)
                    break
            if force is not None:
                force_mag = torch.linalg.norm(force[..., :3], dim=-1)
                return torch.any(force_mag > 1.0, dim=-1)

    height = robot.data.root_pos_w[:, 2]
    z_vel = torch.abs(robot.data.root_lin_vel_w[:, 2])
    return (height < 0.36) & (z_vel < 0.50)


# --------------------------------------------------------------------------------------
# kinematics helpers
# --------------------------------------------------------------------------------------

def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _pitch_from_projected_gravity(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """Return signed sagittal pitch angle in radians.

    upright ~= 0, forward rotation positive.
    backward rotation negative.
    """
    return torch.atan2(projected_gravity_b[:, 0], -projected_gravity_b[:, 2])


def _upright_mask(robot: Articulation) -> torch.Tensor:
    g_b = robot.data.projected_gravity_b
    return (
        (-g_b[:, 2] > 0.90)
        & (torch.abs(g_b[:, 0]) < 0.28)
        & (torch.abs(g_b[:, 1]) < 0.22)
    )


def _advance_phase_if(mask: torch.Tensor, env: ManagerBasedRLEnv, new_phase: int) -> None:
    if torch.any(mask):
        env._tumble_phase[mask] = new_phase
        env._tumble_time_in_phase[mask] = 0.0


# --------------------------------------------------------------------------------------
# main update
# --------------------------------------------------------------------------------------

def update(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    crouch_height_offset: float = -0.03,
    takeoff_height: float = 0.372,
    takeoff_vertical_vel: float = 0.32,
    takeoff_backward_pitch_rate: float = 0.60,
    early_takeoff_height_slack: float = 0.025,
    landing_height: float = 0.375,
    recovery_horizon_s: float = 0.22,
) -> None:
    """Update tumble task phase machine once per simulation step.

    Backflip-tuned changes versus the baseline:
    - crouch is detected a bit earlier
    - takeoff can trigger from a lower/faster rotating launch
    - aerial can begin slightly earlier once contact breaks
    """
    initialize_buffers(env, asset_cfg=asset_cfg)

    robot: Articulation = env.scene[asset_cfg.name]
    dt = float(env.step_dt)

    height = robot.data.root_pos_w[:, 2]
    z_vel = robot.data.root_lin_vel_w[:, 2]
    pitch = _pitch_from_projected_gravity(robot.data.projected_gravity_b)
    pitch_rate = robot.data.root_ang_vel_b[:, 1]
    feet_contact = _feet_contact_mask(env)
    upright = _upright_mask(robot)

    env._tumble_time_in_phase += dt
    env._tumble_peak_height = torch.maximum(env._tumble_peak_height, height)

    prev_pitch = env._tumble_prev_pitch.clone()
    delta_pitch = _wrap_to_pi(pitch - prev_pitch)

    aerial_like = (env._tumble_phase >= TAKEOFF) & (env._tumble_phase <= LANDING)
    env._tumble_cum_pitch[aerial_like] += delta_pitch[aerial_like]
    env._tumble_prev_pitch.copy_(pitch)

    env._tumble_air_time[aerial_like & (~feet_contact)] += dt
    env._tumble_air_time[~aerial_like] = 0.0

    nominal_height = env._tumble_takeoff_height.new_full(height.shape, 0.33)
    crouched = height < (nominal_height + crouch_height_offset)

    # A light preload or backward drive can count as crouch onset.
    crouch_drive = (z_vel < -0.08) | (pitch_rate < -0.30)

    # STANCE -> CROUCH
    mask = (env._tumble_phase == STANCE) & feet_contact & (crouched | crouch_drive)
    _advance_phase_if(mask, env, CROUCH)

    # CROUCH/STANCE -> TAKEOFF
    # Standard upward launch.
    standard_takeoff = (height > takeoff_height) & (z_vel > takeoff_vertical_vel)
    # Earlier backflip-oriented launch: allow slightly lower height if there is clear backward rotation.
    rotating_takeoff = (
        (height > (takeoff_height - early_takeoff_height_slack))
        & (z_vel > (takeoff_vertical_vel - 0.06))
        & (pitch_rate < -takeoff_backward_pitch_rate)
    )
    # Broken-contact launch: once feet leave the ground while rising and already pitching backward.
    contact_break_takeoff = (
        (~feet_contact)
        & (z_vel > 0.18)
        & (pitch_rate < -(takeoff_backward_pitch_rate * 0.85))
    )

    mask = (env._tumble_phase <= CROUCH) & (standard_takeoff | rotating_takeoff | contact_break_takeoff)
    if torch.any(mask):
        env._tumble_takeoff_seen[mask] = True
        env._tumble_takeoff_step[mask] = env.episode_length_buf[mask]
        env._tumble_takeoff_height[mask] = height[mask]
    _advance_phase_if(mask, env, TAKEOFF)

    # TAKEOFF -> AERIAL
    # Once contact is broken or the robot is clearly airborne, enter aerial quickly.
    mask = (env._tumble_phase == TAKEOFF) & (
        (~feet_contact)
        | (height > takeoff_height + 0.006)
        | ((env._tumble_time_in_phase > 0.03) & (z_vel > 0.10))
    )
    _advance_phase_if(mask, env, AERIAL)

    # detect full rotation during aerial phase (backflip direction -> negative pitch accumulation)
    full_rot_mask = (
        (env._tumble_phase == AERIAL)
        & (~env._tumble_full_rotation)
        & (env._tumble_cum_pitch < -(2.0 * torch.pi * 0.90))
    )
    env._tumble_full_rotation[full_rot_mask] = True

    # AERIAL -> LANDING
    mask = (
        (env._tumble_phase == AERIAL)
        & feet_contact
        & (height < landing_height)
        & (env._tumble_time_in_phase > 0.06)
    )
    if torch.any(mask):
        env._tumble_landed[mask] = True
    _advance_phase_if(mask, env, LANDING)

    # LANDING -> RECOVERY
    mask = (
        (env._tumble_phase == LANDING)
        & upright
        & feet_contact
        & (torch.abs(z_vel) < 0.55)
        & (env._tumble_time_in_phase > recovery_horizon_s)
    )
    if torch.any(mask):
        env._tumble_recovered[mask] = True
    _advance_phase_if(mask, env, RECOVERY)
