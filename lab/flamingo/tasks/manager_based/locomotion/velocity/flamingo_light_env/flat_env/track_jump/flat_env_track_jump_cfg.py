# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_env.flat_env.track_jump.jump_rewards as mdp_jump

# Light 환경의 기본 Cfg 임포트
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_light_env.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    CurriculumCfg,
    CommandsCfg,
)

# Light 로봇 에셋 임포트
from lab.flamingo.assets.flamingo.flamingo_light_v1 import FLAMINGO_LIGHT_CFG  # isort: skip


@configclass
class FlamingocommandsCfg(CommandsCfg):
    event = mdp.EventCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.1,
        event_during_time=1.2,
        debug_vis=True,
    )


@configclass
class FlamingoCurriculumCfg(CurriculumCfg):
    modify_base_velocity_range = CurrTerm(
        func=mdp.modify_base_velocity_range,
        params={
            "term_name": "base_velocity",
            # Light 버전에 맞게 커리큘럼 속도 범위 조정
            "mod_range": {"lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-2.5, 2.5)},
            "num_steps": 50000,
        },
    )


@configclass
class FlamingoLightJumpRewardsCfg():
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- Jump Event (Light 버전에 맞춰 도약 속도 4.0 -> 2.5 로 하향)
    lin_vel_z_event = RewTerm(
        func=mdp_jump.lin_vel_z_event,
        weight=2.5,
        params={"event_command_name": "event",
                "event_time_range": (0.3, 0.8),
                "max_up_vel": 1.2,
                "up_vel_coef": 20.0,
                "down_vel_coef": 0.0,
                "temperature": 2.0,
                }
    )

    push_ground_event = RewTerm(
        func=mdp_jump.reward_push_ground_event,
        weight=0.05,
        params={
            "event_command_name": "event",
            "event_time_range": (0.3, 0.8),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"),
        }
    )

    wheel_action_zero_event = RewTerm(func=mdp_jump.wheel_action_zero_event, weight=-0.01)

    # -- Penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.05)

    # Hip 관련 페널티 제거됨, 어깨(Shoulder) 관절 제한만 유지
    dof_pos_limits_shoulder = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )

    # 비정상 접촉 부위 설정 (Light 파츠 기준)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_shoulder_link", ".*_leg_link"]),
            "threshold": 1.0,
        },
    )

    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )

    shoulder_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )

    # 기체가 가벼워 흔들림이 크므로 오일러 수평 페널티 사용 (가중치 -10.0으로 완화)
    flat_orientation_l2 = RewTerm(func=mdp.flat_euler_angle_l2, weight=-10.0)

    # 높이 타겟을 Light 버전에 맞춰 하향 (0.36288 -> 0.31)
    base_height = RewTerm(
        func=mdp_jump.base_height_adaptive_l2_event,
        weight=-40.0,
        params={
            "target_height": 0.31,
            "event_command_name": "event",
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # 토크 및 가속도 페널티를 관절(Joint)과 바퀴(Wheel)로 분리 (Light 버전 특징)
    dof_torques_joints_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    )
    dof_torques_wheels_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )

    dof_acc_joints_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    )
    dof_acc_wheels_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class FlamingoLightJumpEnvCfg(LocomotionVelocityFlatEnvCfg):
    rewards: FlamingoLightJumpRewardsCfg = FlamingoLightJumpRewardsCfg()
    commands: FlamingocommandsCfg = FlamingocommandsCfg()
    curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # 로봇 에셋 교체
        self.scene.robot = FLAMINGO_LIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ! ************** scene & observations setup - 0 *********** !#
        self.scene.height_scanner = None
        self.scene.base_height_scanner = None
        self.scene.left_wheel_height_scanner = None
        self.scene.right_wheel_height_scanner = None
        self.scene.left_mask_sensor = None
        self.scene.right_mask_sensor = None

        self.observations.none_stack_critic.base_height_scan = None
        self.observations.none_stack_critic.left_wheel_height_scan = None
        self.observations.none_stack_critic.right_wheel_height_scan = None
        self.observations.none_stack_critic.height_scan = None
        self.observations.none_stack_critic.lift_mask = None
        # ! ********************************************************* !#

        # ! ****************** Observations setup - 0 *************** !#
        if hasattr(self.observations.none_stack_policy.base_pos_z, "params"):
            self.observations.none_stack_policy.base_pos_z.params["sensor_cfg"] = None
        if hasattr(self.observations.none_stack_critic.base_pos_z, "params"):
            self.observations.none_stack_critic.base_pos_z.params["sensor_cfg"] = None

        self.observations.none_stack_policy.height_scan = None
        self.observations.none_stack_policy.base_lin_vel = None
        self.observations.none_stack_policy.base_pos_z = None
        self.observations.none_stack_policy.current_reward = None
        self.observations.none_stack_policy.is_contact = None
        self.observations.none_stack_policy.lift_mask = None

        self.observations.none_stack_policy.roll_pitch_commands = None
        self.observations.none_stack_critic.roll_pitch_commands = None
        # ! ********************************************************* !#

        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)

        # 외란(Push) 약화
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        }

        # 기본 추가 질량 범위 축소
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 1.0)

        # 물리 마찰력 세팅
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.0, 0.0),
            },
        }

        # 명령 범위 축소 (Light 버전 속도 한계)
        self.commands.base_velocity.resampling_time_range = (9.0, 13.0)
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # 에피소드 종료 조건 (골반/어깨 제외, 다리 파츠만 포함)
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            # "base_link",
            "left_leg_link",
            "right_leg_link",
        ]


@configclass
class FlamingoLightJumpEnvCfg_PLAY(FlamingoLightJumpEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.debug_vis = True
        self.sim.render_interval = self.decimation

        self.scene.robot = FLAMINGO_LIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ! ****************** Observations setup ******************* !#
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        # ! ********************************************************* !#

        self.events.reset_robot_joints.params["position_range"] = (-0.0, 0.0)
        self.events.push_robot.interval_range_s = (7.5, 8.5)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 1.0)

        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.commands.base_velocity.resampling_time_range = (3.0, 8.0)
        self.commands.base_velocity.rel_standing_envs = 0.5
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            "left_leg_link",
            "right_leg_link",
        ]