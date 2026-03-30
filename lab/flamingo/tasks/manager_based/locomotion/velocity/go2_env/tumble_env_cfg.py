import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from . import tumble_mdp


FOOT_SENSOR = SceneEntityCfg("contact_forces", body_names=".*_foot")
BAD_BODY_SENSOR = SceneEntityCfg("contact_forces", body_names=["base", ".*_thigh", ".*_calf"])
ROBOT_ENTITY = SceneEntityCfg("robot")


@configclass
class UnitreeGo2TumbleRewardsCfg(RewardsCfg):
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    feet_air_time = None

    # Boost takeoff and flight height first; rotation stays present but secondary.
    takeoff_event_bonus = RewTerm(
        func=tumble_mdp.takeoff_event_bonus,
        weight=18.0,
        params={
            "asset_cfg": ROBOT_ENTITY,
            "sensor_cfg": FOOT_SENSOR,
            "min_air_time": 0.02,
            "min_vz": 0.35,
        },
    )

    airtime_progress = RewTerm(
        func=tumble_mdp.airtime_progress_reward,
        weight=22.0,
        params={
            "asset_cfg": ROBOT_ENTITY,
            "sensor_cfg": FOOT_SENSOR,
            "target_air_time": 0.11,
        },
    )

    airborne_height_progress = RewTerm(
        func=tumble_mdp.airborne_height_progress,
        weight=11.0,
        params={
            "asset_cfg": ROBOT_ENTITY,
            "sensor_cfg": FOOT_SENSOR,
            "nominal_height": 0.33,
            "target_bonus_height": 0.12,
            "min_air_time": 0.02,
        },
    )

    takeoff_upward_velocity = RewTerm(
        func=tumble_mdp.upward_velocity_hint,
        weight=5.5,
        params={
            "asset_cfg": ROBOT_ENTITY,
            "threshold": 0.18,
        },
    )

    pre_takeoff_upright_reward = RewTerm(
        func=tumble_mdp.pre_takeoff_upright_reward,
        weight=0.8,
        params={"asset_cfg": ROBOT_ENTITY},
    )

    pre_takeoff_forward_velocity_penalty = RewTerm(
        func=tumble_mdp.pre_takeoff_forward_velocity_penalty,
        weight=-1.2,
        params={"asset_cfg": ROBOT_ENTITY},
    )

    post_takeoff_backward_pitch_rate_reward = RewTerm(
        func=tumble_mdp.post_takeoff_backward_pitch_rate_reward,
        weight=4.0,
        params={
            "asset_cfg": ROBOT_ENTITY,
            "sensor_cfg": FOOT_SENSOR,
            "backward_sign": -1.0,
            "threshold": 0.75,
            "min_air_time": 0.02,
        },
    )

    airborne_flip_progress = RewTerm(
        func=tumble_mdp.airborne_flip_progress_reward,
        weight=6.5,
        params={
            "asset_cfg": ROBOT_ENTITY,
            "sensor_cfg": FOOT_SENSOR,
            "backward_sign": -1.0,
            "min_air_time": 0.02,
            "target_angle_rad": 1.57079632679,
            "rate_clip": 10.0,
        },
    )

    body_contact_penalty = RewTerm(
        func=tumble_mdp.body_contact_penalty,
        weight=-4.0,
        params={
            "sensor_cfg": BAD_BODY_SENSOR,
            "min_contact_time": 0.0,
        },
    )

    low_height_penalty = RewTerm(
        func=tumble_mdp.low_height_penalty,
        weight=-3.0,
        params={
            "asset_cfg": ROBOT_ENTITY,
            "low_height": 0.20,
        },
    )

    lateral_tilt_penalty = RewTerm(
        func=tumble_mdp.lateral_tilt_abs,
        weight=-2.0,
        params={"asset_cfg": ROBOT_ENTITY},
    )

    roll_rate_penalty = RewTerm(
        func=tumble_mdp.roll_rate_abs,
        weight=-0.25,
        params={"asset_cfg": ROBOT_ENTITY, "axis": 0},
    )

    yaw_rate_penalty = RewTerm(
        func=tumble_mdp.yaw_rate_abs,
        weight=-0.15,
        params={"asset_cfg": ROBOT_ENTITY, "axis": 2},
    )

    lateral_velocity_penalty = RewTerm(
        func=tumble_mdp.lateral_velocity_abs,
        weight=-1.0,
        params={"asset_cfg": ROBOT_ENTITY, "axis": 1},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-12.0)

    undesired_contacts = None
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = None
    flat_orientation_l2 = None

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-7)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-6)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.08)


@configclass
class UnitreeGo2TumbleEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: UnitreeGo2TumbleRewardsCfg = UnitreeGo2TumbleRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.observations.policy.enable_corruption = False
        self.episode_length_s = 1.2

        self.actions.joint_pos.scale = 0.20

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.debug_vis = False

        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
        if hasattr(self.events, "add_base_mass"):
            self.events.add_base_mass = None
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "base_com"):
            self.events.base_com = None

        if hasattr(self.events, "reset_robot_joints"):
            self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        if hasattr(self.events, "reset_base"):
            self.events.reset_base.params = {
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "yaw": (-0.02, 0.02),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            }

        if hasattr(self.terminations, "base_contact"):
            self.terminations.base_contact.params["sensor_cfg"].body_names = ["base", ".*_thigh", ".*_calf"]


@configclass
class UnitreeGo2TumbleEnvCfg_PLAY(UnitreeGo2TumbleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False

        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


# Optional aliases for any leftover references.
UnitreeGo2BackflipStage1EnvCfg = UnitreeGo2TumbleEnvCfg
UnitreeGo2BackflipStage1EnvCfg_PLAY = UnitreeGo2TumbleEnvCfg_PLAY
