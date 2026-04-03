from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation

# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------

def _get_robot(env, asset_cfg) -> Articulation:
    return env.scene[asset_cfg.name]


def _resolve_entity_ids(ids, names, resolver_name: str, obj, *, entity_label: str):
    """Resolve SceneEntityCfg ids robustly across IsaacLab revisions."""
    if names is not None:
        if ids is not None and not isinstance(ids, slice):
            try:
                if len(ids) > 0:
                    return ids
            except TypeError:
                return ids

        resolver = getattr(obj, resolver_name, None)
        if resolver is None:
            raise AttributeError(
                f"Could not resolve {entity_label} ids from names {names!r}: "
                f"object of type {type(obj).__name__} has no {resolver_name}()."
            )
        resolved_ids, _ = resolver(names)
        return resolved_ids

    if ids is None:
        return slice(None)
    return ids


def _resolve_joint_ids(robot: Articulation, asset_cfg):
    return _resolve_entity_ids(
        getattr(asset_cfg, "joint_ids", None),
        getattr(asset_cfg, "joint_names", None),
        "find_joints",
        robot,
        entity_label="joint",
    )


def _resolve_body_ids(obj, cfg):
    return _resolve_entity_ids(
        getattr(cfg, "body_ids", None),
        getattr(cfg, "body_names", None),
        "find_bodies",
        obj,
        entity_label="body",
    )


def _get_body_pos_w(robot: Articulation) -> torch.Tensor:
    data = robot.data
    for attr_name in ("body_link_pos_w", "body_com_pos_w", "body_pos_w"):
        value = getattr(data, attr_name, None)
        if value is not None:
            return value
    body_state_w = getattr(data, "body_state_w", None)
    if body_state_w is not None:
        return body_state_w[..., :3]
    body_link_state_w = getattr(data, "body_link_state_w", None)
    if body_link_state_w is not None:
        return body_link_state_w[..., :3]
    raise AttributeError("Robot articulation data does not expose body world positions.")


def _get_body_lin_vel_w(robot: Articulation) -> torch.Tensor:
    data = robot.data
    for attr_name in ("body_link_lin_vel_w", "body_com_lin_vel_w", "body_lin_vel_w"):
        value = getattr(data, attr_name, None)
        if value is not None:
            return value
    for attr_name in ("body_state_w", "body_link_state_w"):
        value = getattr(data, attr_name, None)
        if value is not None and value.shape[-1] >= 10:
            return value[..., 7:10]
    raise AttributeError("Robot articulation data does not expose body world linear velocities.")


def _get_root_height_w(robot: Articulation) -> torch.Tensor:
    data = robot.data
    for attr_name in ("root_pos_w", "root_link_pos_w", "root_com_pos_w"):
        value = getattr(data, attr_name, None)
        if value is not None:
            return value[:, 2]
    raise AttributeError("Robot articulation data does not expose root world position.")


def _get_env_origin_z(env) -> torch.Tensor | None:
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        return None
    return env_origins[:, 2]


def _get_sensor(env, sensor_cfg):
    scene = env.scene
    if hasattr(scene, "sensors") and sensor_cfg.name in scene.sensors:
        return scene.sensors[sensor_cfg.name]
    return scene[sensor_cfg.name]


def _get_contact_force_magnitude_per_body(env, sensor_cfg) -> torch.Tensor:
    sensor = _get_sensor(env, sensor_cfg)
    data = sensor.data
    body_ids = _resolve_body_ids(sensor, sensor_cfg)

    net_forces_w = getattr(data, "net_forces_w", None)
    if net_forces_w is not None:
        forces = net_forces_w[:, body_ids, :]
    else:
        force_matrix_w = getattr(data, "force_matrix_w", None)
        if force_matrix_w is None:
            raise AttributeError("Contact sensor data does not expose net_forces_w or force_matrix_w.")
        forces = force_matrix_w[:, body_ids, ...]
        while forces.ndim > 3:
            forces = torch.sum(forces, dim=-2)

    return torch.linalg.vector_norm(forces, dim=-1)


def _contact_flags(env, sensor_cfg, contact_force_threshold: float = 1.0) -> torch.Tensor:
    contact_force_mag = _get_contact_force_magnitude_per_body(env, sensor_cfg)
    contact = (contact_force_mag >= contact_force_threshold).float()
    if contact.ndim == 1:
        contact = contact.unsqueeze(-1)
    return contact


def _after_settle_mask(env, settle_time_s: float):
    if settle_time_s <= 0.0:
        return 1.0
    if hasattr(env, "episode_length_buf") and hasattr(env, "step_dt"):
        elapsed_s = env.episode_length_buf.float() * float(env.step_dt)
        return (elapsed_s >= settle_time_s).float()
    return 1.0


def _validate_positive(name: str, value: float):
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0. Received {value}.")


def _validate_positive_int(name: str, value: int):
    if value <= 0:
        raise ValueError(f"{name} must be >= 1. Received {value}.")


def _validate_target_delta(target_delta: Sequence[float], expected_dim: int):
    if len(target_delta) != expected_dim:
        raise ValueError(
            f"target_delta length mismatch: expected {expected_dim}, got {len(target_delta)}."
        )


# -----------------------------------------------------------------------------
# Reward primitives
# -----------------------------------------------------------------------------

def selected_joint_deviation_l2_exp(env, asset_cfg, target_delta, sigma: float = 0.35):
    _validate_positive("sigma", sigma)

    robot = _get_robot(env, asset_cfg)
    joint_ids = _resolve_joint_ids(robot, asset_cfg)
    joint_pos = robot.data.joint_pos[:, joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, joint_ids]
    _validate_target_delta(target_delta, joint_pos.shape[1])

    target = default_joint_pos + joint_pos.new_tensor(target_delta).view(1, -1)
    err = torch.sum((joint_pos - target) ** 2, dim=1)
    return torch.exp(-err / (sigma**2))


def selected_joint_default_l2_exp(env, asset_cfg, sigma: float = 0.25):
    _validate_positive("sigma", sigma)

    robot = _get_robot(env, asset_cfg)
    joint_ids = _resolve_joint_ids(robot, asset_cfg)
    joint_pos = robot.data.joint_pos[:, joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, joint_ids]
    err = torch.sum((joint_pos - default_joint_pos) ** 2, dim=1)
    return torch.exp(-err / (sigma**2))


def selected_joint_vel_l2_when_no_contact(
    env,
    asset_cfg,
    sensor_cfg,
    contact_force_threshold: float = 1.0,
    settle_time_s: float = 0.0,
):

    robot = _get_robot(env, asset_cfg)
    joint_ids = _resolve_joint_ids(robot, asset_cfg)
    joint_vel = robot.data.joint_vel[:, joint_ids]
    value = torch.mean(joint_vel**2, dim=1)

    contact = _contact_flags(env, sensor_cfg, contact_force_threshold=contact_force_threshold)
    has_contact = contact[:, 0] if contact.shape[1] == 1 else torch.any(contact > 0.5, dim=1).float()
    no_contact = 1.0 - has_contact
    return value * no_contact * _after_settle_mask(env, settle_time_s)


def base_height_l2_exp(env, asset_cfg, target_height: float = 0.33, sigma: float = 0.04):
    _validate_positive("sigma", sigma)

    robot = _get_robot(env, asset_cfg)
    base_height = _get_root_height_w(robot)
    env_origin_z = _get_env_origin_z(env)
    if env_origin_z is not None:
        base_height = base_height - env_origin_z

    err = (base_height - target_height) ** 2
    return torch.exp(-err / (sigma**2))


def selected_body_contact_indicator(
    env,
    sensor_cfg,
    contact_force_threshold: float = 1.0,
    settle_time_s: float = 0.0,
):
    contact = _contact_flags(env, sensor_cfg, contact_force_threshold=contact_force_threshold)
    value = contact[:, 0] if contact.shape[1] == 1 else torch.any(contact > 0.5, dim=1).float()
    return value * _after_settle_mask(env, settle_time_s)


def selected_bodies_contact_fraction(
    env,
    sensor_cfg,
    contact_force_threshold: float = 1.0,
    settle_time_s: float = 0.0,
):
    contact = _contact_flags(env, sensor_cfg, contact_force_threshold=contact_force_threshold)
    value = torch.mean(contact, dim=1)
    return value * _after_settle_mask(env, settle_time_s)


def selected_bodies_min_contact_fraction(
    env,
    sensor_cfg,
    contact_force_threshold: float = 1.0,
    min_contact_count: int = 2,
    settle_time_s: float = 0.0,
):
    _validate_positive_int("min_contact_count", min_contact_count)

    contact = _contact_flags(env, sensor_cfg, contact_force_threshold=contact_force_threshold)
    contact_count = torch.sum(contact, dim=1)
    denom = float(min_contact_count)
    value = torch.clamp(contact_count / denom, max=1.0)
    return value * _after_settle_mask(env, settle_time_s)


def selected_bodies_contact_xy_vel_l2(
    env,
    asset_cfg,
    sensor_cfg,
    contact_force_threshold: float = 1.0,
    settle_time_s: float = 0.0,
):
    robot: Articulation = _get_robot(env, asset_cfg)
    body_ids = _resolve_body_ids(robot, asset_cfg)
    body_lin_vel_w = _get_body_lin_vel_w(robot)[:, body_ids, :2]
    if body_lin_vel_w.ndim == 2:
        body_lin_vel_w = body_lin_vel_w.unsqueeze(1)

    slip_sq = torch.sum(body_lin_vel_w**2, dim=-1)
    contact = _contact_flags(env, sensor_cfg, contact_force_threshold=contact_force_threshold)

    if slip_sq.shape[1] != contact.shape[1]:
        num_bodies = min(slip_sq.shape[1], contact.shape[1])
        slip_sq = slip_sq[:, :num_bodies]
        contact = contact[:, :num_bodies]

    contact_count = torch.sum(contact, dim=1)
    denom = torch.clamp(contact_count, min=1.0)
    value = torch.sum(slip_sq * contact, dim=1) / denom
    return value * _after_settle_mask(env, settle_time_s)


def gated_selected_joint_deviation_l2_exp(
    env,
    asset_cfg,
    target_delta,
    sigma: float,
    settle_time_s: float = 0.0,
):
    value = selected_joint_deviation_l2_exp(
        env,
        asset_cfg=asset_cfg,
        target_delta=target_delta,
        sigma=sigma,
    )
    return value * _after_settle_mask(env, settle_time_s)


def selected_body_height_above_min_no_contact_with_min_support_exp(
    env,
    asset_cfg,
    lifted_sensor_cfg,
    support_sensor_cfg,
    min_height: float = 0.08,
    sigma: float = 0.03,
    contact_force_threshold: float = 1.0,
    min_contact_count: int = 2,
    settle_time_s: float = 0.0,
):
    _validate_positive("sigma", sigma)
    _validate_positive_int("min_contact_count", min_contact_count)

    robot: Articulation = _get_robot(env, asset_cfg)
    body_ids = _resolve_body_ids(robot, asset_cfg)
    body_pos_w = _get_body_pos_w(robot)
    body_height = body_pos_w[:, body_ids, 2]

    env_origin_z = _get_env_origin_z(env)
    if env_origin_z is not None:
        if body_height.ndim == 2:
            body_height = body_height - env_origin_z.unsqueeze(-1)
        else:
            body_height = body_height - env_origin_z

    if body_height.ndim == 2 and body_height.shape[1] == 1:
        body_height = body_height[:, 0]
    elif body_height.ndim == 2:
        body_height = torch.mean(body_height, dim=1)

    deficit = torch.clamp(min_height - body_height, min=0.0)
    clearance_reward = torch.exp(-(deficit**2) / (sigma**2))

    contact = _contact_flags(env, lifted_sensor_cfg, contact_force_threshold=contact_force_threshold)
    no_contact = 1.0 - (contact[:, 0] if contact.shape[1] == 1 else torch.any(contact > 0.5, dim=1).float())

    support_ok = selected_bodies_min_contact_fraction(
        env,
        sensor_cfg=support_sensor_cfg,
        contact_force_threshold=contact_force_threshold,
        min_contact_count=min_contact_count,
        settle_time_s=0.0,
    )

    return clearance_reward * no_contact * support_ok * _after_settle_mask(env, settle_time_s)


def selected_body_no_contact_with_min_support_bonus(
    env,
    sensor_cfg,
    support_sensor_cfg,
    contact_force_threshold: float = 1.0,
    min_contact_count: int = 2,
    settle_time_s: float = 0.0,
):
    _validate_positive_int("min_contact_count", min_contact_count)

    contact = _contact_flags(env, sensor_cfg, contact_force_threshold=contact_force_threshold)
    no_contact = 1.0 - (contact[:, 0] if contact.shape[1] == 1 else torch.any(contact > 0.5, dim=1).float())

    support_ok = selected_bodies_min_contact_fraction(
        env,
        sensor_cfg=support_sensor_cfg,
        contact_force_threshold=contact_force_threshold,
        min_contact_count=min_contact_count,
        settle_time_s=0.0,
    )

    return no_contact * support_ok * _after_settle_mask(env, settle_time_s)
