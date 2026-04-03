from __future__ import annotations

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import Leglift_walk_mdp as walk_mdp


# -----------------------------------------------------------------------------
# Task entities
# -----------------------------------------------------------------------------

ROBOT_ENTITY = SceneEntityCfg("robot")

RL_FOOT = SceneEntityCfg("robot", body_names=["RL_foot"])
RL_CONTACT_SENSOR = SceneEntityCfg("contact_forces", body_names=["RL_foot"])

LIFTED_RL_LEG = SceneEntityCfg(
    "robot",
    joint_names=["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
)

SUPPORT_FEET = SceneEntityCfg(
    "robot",
    body_names=["FL_foot", "FR_foot", "RR_foot"],
)

SUPPORT_FEET_CONTACT = SceneEntityCfg(
    "contact_forces",
    body_names=["FL_foot", "FR_foot", "RR_foot"],
)

SUPPORT_LEGS = SceneEntityCfg(
    "robot",
    joint_names=[
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ],
)

NON_FOOT_LINK_CONTACT = SceneEntityCfg(
    "contact_forces",
    body_names=[".*_hip", ".*_thigh", ".*_calf"],
)

SETTLE_TIME_S = 0.18
LIFTED_CONTACT_FORCE_THRESHOLD = 2.5
SUPPORT_CONTACT_FORCE_THRESHOLD = 5.0
SUPPORT_MIN_CONTACT_COUNT = 2


@configclass
class UnitreeGo2LegliftWalkRewardsCfg(RewardsCfg):

    undesired_contacts = None

    # ---------------------------------------------------------------------
    # Locomotion terms
    # ---------------------------------------------------------------------

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=8.0,
        params={"command_name": "base_velocity", "std": 0.18},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.50,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.45)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-9.0)

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.5e-3)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.05)

    # ---------------------------------------------------------------------
    # Rear-left lifted-leg shaping
    # ---------------------------------------------------------------------

    lifted_rear_left_leg_pose = RewTerm(
        func=walk_mdp.gated_selected_joint_deviation_l2_exp,
        weight=2.0,
        params={
            "asset_cfg": LIFTED_RL_LEG,
            "target_delta": [0.0, 0.36, -0.52],
            "sigma": 0.32,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    lifted_rear_left_leg_joint_vel_penalty = RewTerm(
        func=walk_mdp.selected_joint_vel_l2_when_no_contact,
        weight=-0.10,
        params={
            "asset_cfg": LIFTED_RL_LEG,
            "sensor_cfg": RL_CONTACT_SENSOR,
            "contact_force_threshold": LIFTED_CONTACT_FORCE_THRESHOLD,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    rear_left_foot_clearance = RewTerm(
        func=walk_mdp.selected_body_height_above_min_no_contact_with_min_support_exp,
        weight=3.0,
        params={
            "asset_cfg": RL_FOOT,
            "lifted_sensor_cfg": RL_CONTACT_SENSOR,
            "support_sensor_cfg": SUPPORT_FEET_CONTACT,
            "min_height": 0.07,
            "sigma": 0.032,
            "contact_force_threshold": LIFTED_CONTACT_FORCE_THRESHOLD,
            "min_contact_count": SUPPORT_MIN_CONTACT_COUNT,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    rear_left_foot_no_contact_bonus = RewTerm(
        func=walk_mdp.selected_body_no_contact_with_min_support_bonus,
        weight=1.0,
        params={
            "sensor_cfg": RL_CONTACT_SENSOR,
            "support_sensor_cfg": SUPPORT_FEET_CONTACT,
            "contact_force_threshold": LIFTED_CONTACT_FORCE_THRESHOLD,
            "min_contact_count": SUPPORT_MIN_CONTACT_COUNT,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    rear_left_foot_contact_penalty = RewTerm(
        func=walk_mdp.selected_body_contact_indicator,
        weight=-8.0,
        params={
            "sensor_cfg": RL_CONTACT_SENSOR,
            "contact_force_threshold": LIFTED_CONTACT_FORCE_THRESHOLD,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    support_min_two_contact = RewTerm(
        func=walk_mdp.selected_bodies_min_contact_fraction,
        weight=1.5,
        params={
            "sensor_cfg": SUPPORT_FEET_CONTACT,
            "contact_force_threshold": SUPPORT_CONTACT_FORCE_THRESHOLD,
            "min_contact_count": SUPPORT_MIN_CONTACT_COUNT,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    support_foot_slip_penalty = RewTerm(
        func=walk_mdp.selected_bodies_contact_xy_vel_l2,
        weight=-10.0,
        params={
            "asset_cfg": SUPPORT_FEET,
            "sensor_cfg": SUPPORT_FEET_CONTACT,
            "contact_force_threshold": SUPPORT_CONTACT_FORCE_THRESHOLD,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    support_leg_default_pose = RewTerm(
        func=walk_mdp.selected_joint_default_l2_exp,
        weight=2.0,
        params={
            "asset_cfg": SUPPORT_LEGS,
            "sigma": 0.55,
        },
    )

    stand_base_height = RewTerm(
        func=walk_mdp.base_height_l2_exp,
        weight=8.0,
        params={
            "asset_cfg": ROBOT_ENTITY,
            "target_height": 0.34,
            "sigma": 0.06,
        },
    )

    non_foot_link_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-6.0,
        params={
            "sensor_cfg": NON_FOOT_LINK_CONTACT,
            "threshold": 1.0,
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)


@configclass
class UnitreeGo2LegliftWalkEnvCfg(LocomotionVelocityRoughEnvCfg):

    rewards: UnitreeGo2LegliftWalkRewardsCfg = UnitreeGo2LegliftWalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # -----------------------------------------------------------------
        # Robot + terrain
        # -----------------------------------------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None

        self.curriculum.terrain_levels = None

        # -----------------------------------------------------------------
        # Commands
        # -----------------------------------------------------------------
        self.episode_length_s = 8.0

        base_velocity = self.commands.base_velocity
        base_velocity.resampling_time_range = (8.0, 8.0)
        base_velocity.ranges.lin_vel_x = (0.18, 0.26)
        base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        base_velocity.ranges.heading = (0.0, 0.0)
        base_velocity.heading_command = False
        base_velocity.rel_standing_envs = 0.0
        base_velocity.rel_heading_envs = 0.0
        base_velocity.debug_vis = False

        # -----------------------------------------------------------------
        # Observations/actions
        # -----------------------------------------------------------------
        if hasattr(self.observations.policy, "enable_corruption"):
            self.observations.policy.enable_corruption = False

        if hasattr(self.actions, "joint_pos") and hasattr(self.actions.joint_pos, "scale"):
            self.actions.joint_pos.scale = 0.16

        if getattr(self.rewards, "feet_air_time", None) is not None:
            self.rewards.feet_air_time.params["sensor_cfg"].body_names = ["FL_foot", "FR_foot", "RR_foot"]
            self.rewards.feet_air_time.weight = 1.0

        # -----------------------------------------------------------------
        # Randomization/events
        # -----------------------------------------------------------------
        for term_name in ("push_robot", "add_base_mass", "base_external_force_torque", "base_com"):
            if hasattr(self.events, term_name):
                setattr(self.events, term_name, None)

        if hasattr(self.events, "reset_robot_joints"):
            self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
            self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        if hasattr(self.events, "reset_base"):
            self.events.reset_base.params = {
                "pose_range": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "yaw": (-0.05, 0.05),
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
            self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2LegliftWalkEnvCfg_PLAY(UnitreeGo2LegliftWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0

        if hasattr(self.observations.policy, "enable_corruption"):
            self.observations.policy.enable_corruption = False

        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
