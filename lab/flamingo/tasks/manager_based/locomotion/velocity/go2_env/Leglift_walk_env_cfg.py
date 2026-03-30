import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import Leglift_walk_mdp

ROBOT_ENTITY = SceneEntityCfg("robot")
RL_FOOT = SceneEntityCfg("robot", body_names=["RL_foot"])
RL_CONTACT_SENSOR = SceneEntityCfg("contact_forces", body_names=["RL_foot"])
LIFTED_RL_LEG = SceneEntityCfg(
    "robot",
    joint_names=["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
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
SETTLE_TIME_S = 0.35


@configclass
class UnitreeGo2LegliftWalkRewardsCfg(RewardsCfg):

    feet_air_time = None

    lifted_rear_left_leg_pose = RewTerm(
        func=Leglift_walk_mdp.gated_selected_joint_deviation_l2_exp,
        weight=4.0,
        params={
            "asset_cfg": LIFTED_RL_LEG,
            "target_delta": [0.0, 0.45, -0.62],
            "sigma": 0.22,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    rear_left_foot_clearance = RewTerm(
        func=Leglift_walk_mdp.selected_body_height_above_min_no_contact_with_all_support_exp,
        weight=6.0,
        params={
            "asset_cfg": RL_FOOT,
            "lifted_sensor_cfg": RL_CONTACT_SENSOR,
            "support_sensor_cfg": SUPPORT_FEET_CONTACT,
            "min_height": 0.07,
            "sigma": 0.025,
            "contact_force_threshold": 1.0,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    rear_left_foot_no_contact_bonus = RewTerm(
        func=Leglift_walk_mdp.selected_body_no_contact_with_all_support_bonus,
        weight=2.0,
        params={
            "sensor_cfg": RL_CONTACT_SENSOR,
            "support_sensor_cfg": SUPPORT_FEET_CONTACT,
            "contact_force_threshold": 1.0,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    rear_left_foot_contact_penalty = RewTerm(
        func=Leglift_walk_mdp.selected_body_contact_indicator,
        weight=-4.0,
        params={
            "sensor_cfg": RL_CONTACT_SENSOR,
            "contact_force_threshold": 1.0,
            "settle_time_s": SETTLE_TIME_S,
        },
    )

    support_tripod_contact = RewTerm(
        func=Leglift_walk_mdp.selected_bodies_all_contact_indicator,
        weight=4.0,
        params={
            "sensor_cfg": SUPPORT_FEET_CONTACT,
            "contact_force_threshold": 1.0,
        },
    )

    non_foot_link_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-4.0,
        params={
            "sensor_cfg": NON_FOOT_LINK_CONTACT,
            "threshold": 1.0,
        },
    )

    support_leg_default_pose = RewTerm(
        func=Leglift_walk_mdp.selected_joint_default_l2_exp,
        weight=3.0,
        params={
            "asset_cfg": SUPPORT_LEGS,
            "sigma": 0.20,
        },
    )

    stand_base_height = RewTerm(
        func=Leglift_walk_mdp.base_height_l2_exp,
        weight=3.0,
        params={
            "asset_cfg": ROBOT_ENTITY,
            "target_height": 0.33,
            "sigma": 0.030,
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2.0e-5)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.10)


@configclass
class UnitreeGo2LegliftWalkEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: UnitreeGo2LegliftWalkRewardsCfg = UnitreeGo2LegliftWalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        if hasattr(self.observations.policy, "enable_corruption"):
            self.observations.policy.enable_corruption = False
        if hasattr(self.actions, "joint_pos") and hasattr(self.actions.joint_pos, "scale"):
            self.actions.joint_pos.scale = 0.04

        self.episode_length_s = 6.0
        base_velocity = self.commands.base_velocity
        base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        base_velocity.ranges.heading = (0.0, 0.0)
        base_velocity.heading_command = False
        base_velocity.rel_standing_envs = 1.0
        base_velocity.rel_heading_envs = 0.0
        base_velocity.debug_vis = False

        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        if self.rewards.flat_orientation_l2 is not None:
            self.rewards.flat_orientation_l2.weight = -12.0
        if self.rewards.lin_vel_z_l2 is not None:
            self.rewards.lin_vel_z_l2.weight = -1.0
        if self.rewards.ang_vel_xy_l2 is not None:
            self.rewards.ang_vel_xy_l2.weight = -0.40
        if self.rewards.dof_torques_l2 is not None:
            self.rewards.dof_torques_l2.weight = -2.0e-4
        if self.rewards.dof_acc_l2 is not None:
            self.rewards.dof_acc_l2.weight = -2.5e-7

        for term_name in ("push_robot", "add_base_mass", "base_external_force_torque", "base_com"):
            if hasattr(self.events, term_name):
                setattr(self.events, term_name, None)

        if hasattr(self.events, "reset_robot_joints"):
            self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
            self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        if hasattr(self.events, "reset_base"):
            self.events.reset_base.params = {
                "pose_range": {
                    "x": (-0.003, 0.003),
                    "y": (-0.003, 0.003),
                    "yaw": (-0.003, 0.003),
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
