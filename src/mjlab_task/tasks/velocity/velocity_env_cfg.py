"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  TerrainHeightSensorCfg,
  ObjRef,
  RingPatternCfg,
  ContactMatch,
  ContactSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.tasks.velocity.mdp import ZPosVelocityCommandCfg
from src.sensor.leg_length_sensor import LegLengthSensorCfg, SiteRef
from src.sensor.leg_theta_sensor import LegThetaSensorCfg, JointRef
from src.tasks.velocity import mdp

##
# Joint name patterns (regex) shared by observation, action, and reward configs.
##

_HIP_JOINTS = (".*hipjoint",)
_WHEEL_JOINTS = (".*wheel_joint",)
_LEG_JOINTS = _HIP_JOINTS + _WHEEL_JOINTS


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Sensors
  ##

  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(
      ObjRef(type="site", name="L_leglength_sensor_site2", entity="robot"),
      ObjRef(type="site", name="R_leglength_sensor_site2", entity="robot"),
    ),  # Set per-robot: frame and pattern.
    pattern=RingPatternCfg.single_ring(radius=0, num_samples=0, include_center=True),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_radius=0,
    ),
  )

  feet_ground_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=("L_wheel_ground_collision", "R_wheel_ground_collision"), entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  leg_length_scan = LegLengthSensorCfg(
    name="leg_length_scan",
    site_pair=(
      SiteRef(site_top="L_leglength_sensor_site1", site_bottom="L_leglength_sensor_site2",),
      SiteRef(site_top="R_leglength_sensor_site1", site_bottom="R_leglength_sensor_site2",),
    ),
    entity="robot",
    ref_body="robot/base_link",
  )

  leg_theta_scan = LegThetaSensorCfg(
    name="leg_theta_scan",
    joint_pair=(
      JointRef(phi4_joint="L_leg1_hipjoint", phi1_joint="L_thigh_hipjoint",),
      JointRef(phi4_joint="R_thigh_hipjoint", phi1_joint="R_leg1_hipjoint",),
    ),
    negate=(False, True),
    entity="robot",
    link_lengths=(0.215, 0.254, 0.254, 0.215, 0.0),
    ref_body="robot/base_link",
  )

  ##
  # Observations
  ##
  """
  base_ang_vel*3
  projected_gravity*3
  joint_pos_rel*4
  joint_vel_rel*6
  last_action*6
  command*3
  = 25
  """
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.02, n_max=0.02),
      scale=0.5,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.02, n_max=0.02),
      scale=1.0,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={
        "biased": True,
        "asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_JOINTS),
      },
      noise=Unoise(n_min=-0.002, n_max=0.002),
      scale=1.0,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
      noise=Unoise(n_min=-0.02, n_max=0.02),
      # Order: L_thigh, L_wheel, L_leg1, R_thigh, R_wheel, R_leg1.
      scale=(0.5, 0.1, 0.5, 0.5, 0.1, 0.5),
    ),
    "last_actions": ObservationTermCfg(
      func=envs_mdp.last_action,
      scale=1.0,
    ),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
      scale=(1.0, 0.5, 5.0),
    ),
  }
  """
  base_lin_vel*3
  joint_acc_rel*6
  prev_actions*6
  base_height*1
  base_mass*1
  base_com*3
  foot_friction*6
  foot_contact*2
  joint_bias_rel*4
  joint_stiffness_rel*4
  joint_damping_rel*6
  act_delay_steps*1
  constant_force*2
  = 45 + 25 = 70
  """
  critic_terms = {
    **actor_terms,
    # Critic sees the true (unbiased) joint positions as privileged information.
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_JOINTS)},
    ),
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "joint_acc_rel": ObservationTermCfg(
      func=mdp.joint_acc_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
    ),
    "prev_actions": ObservationTermCfg(func=mdp.prev_action),
    "base_height": ObservationTermCfg(func=mdp.base_height),
    "base_mass": ObservationTermCfg(
      func=mdp.base_mass,
      params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    ),
    "base_com": ObservationTermCfg(
      func=mdp.base_com,
      params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    ),
    "foot_friction": ObservationTermCfg(
      func=mdp.foot_friction,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=".*wheel_ground_collision")},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "joint_bias_rel": ObservationTermCfg(
      func=mdp.joint_bias_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_JOINTS)},
    ),
    "joint_stiffness_rel": ObservationTermCfg(
      func=mdp.joint_stiffness_rel,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=list(_HIP_JOINTS))},
    ),
    "joint_damping_rel": ObservationTermCfg(
      func=mdp.joint_damping_rel,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=list(_LEG_JOINTS))},
    ),
    "act_delay_steps": ObservationTermCfg(func=mdp.act_delay_steps),
    "constant_force": ObservationTermCfg(
      func=mdp.tendon_force,
      params={
        "asset_cfg": SceneEntityCfg("robot", tendon_names=("Left_force", "Right_force"))},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=envs_mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=_HIP_JOINTS,
      scale=1.0,
      use_default_offset=True,
      clip={
        "L_leg1_hipjoint": (-1.249341, 1.892251),
        "L_thigh_hipjoint": (-1.892251, +1.249341),
        "R_thigh_hipjoint": (-1.249341, 1.892251),
        "R_leg1_hipjoint": (-1.892251, +1.249341)
      },
    ),
    "joint_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=_WHEEL_JOINTS,
      scale=10.0,
      use_default_offset=True,
      clip={".*wheel_joint": (-30.0, 30.0)},
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": ZPosVelocityCommandCfg(
      entity_name="robot",
      rel_standing_envs=0.1,
      rel_forward_envs=0.2,
      ranges=ZPosVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.5, 1.5),
        ang_vel_z=(-6.0, 6.0),
        lin_pos_z=(0.215, 0.465),
      ),
      viz=ZPosVelocityCommandCfg.VizCfg(
        z_offset=1.5
      ),
      resampling_time_range=(3.0, 8.0),
      debug_vis=True,
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0, 0.01),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("robot")
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
        "asset_cfg": SceneEntityCfg("robot")
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "bias_range": (-0.03, 0.03),
        "asset_cfg": SceneEntityCfg("robot")
      },
    ),
    "base_inertia": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "alpha_range": (-0.1, 0.1),
        "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
      },
    ),
    "joint_gains": EventTermCfg(
      mode="startup",
      func=mdp.randomize_joint_gains,
      params={
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
        "asset_cfg": SceneEntityCfg("robot", actuator_names=list(_LEG_JOINTS)),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
        "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        "operation": "add",
      },
    ),
    "foot_friction_slide": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=("L_wheel_ground_collision", "R_wheel_ground_collision")),
        "operation": "abs",
        "axes": [0],
        "ranges": (0.3, 1.5),
        "shared_random": False,
      },
    ),
    "foot_friction_spin": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=("L_wheel_ground_collision", "R_wheel_ground_collision")),
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [1],
        "ranges": (1e-4, 2e-2),
        "shared_random": False,
      },
    ),
    "foot_friction_roll": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=("L_wheel_ground_collision", "R_wheel_ground_collision")),
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [2],
        "ranges": (1e-5, 5e-3),
        "shared_random": False,
      },
    ),
    "constant_force": EventTermCfg(
      mode="reset",
      func=mdp.randomize_constant_force,
      params={
        "force_range": (90.0, 150.0),
        "asset_cfg": SceneEntityCfg("robot", tendon_names=("Left_force", "Right_force")),
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "upright": RewardTermCfg(
      func=mdp.UprightPenalty,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    ),
    "angular_velocity_xy": RewardTermCfg(
      func=mdp.angular_velocity_xy_penalty,
      weight=-1.0,
    ),
    "linear_velocity_z": RewardTermCfg(
      func=mdp.linear_velocity_z_penalty,
      weight=-1.0,
    ),

    "track_linear_velocity_x": RewardTermCfg(
      func=mdp.track_linear_velocity_x,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_base_height_z": RewardTermCfg(
      func=mdp.track_base_height_z,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.01)},
    ),
    "track_angular_velocity_z": RewardTermCfg(
      func=mdp.track_angular_velocity_z,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),

    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits, 
      weight=-1.0
      ),
    "action_rate_l2": RewardTermCfg(
      func=envs_mdp.action_rate_l2, 
      weight=-0.1
      ),
    "joint_vel_l2": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.05,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_JOINTS),
      },
    ),
    "wheel_torques_l2": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-0.01,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=list(_WHEEL_JOINTS)),
      },
    ),
    "leg_length_limits_penalty": RewardTermCfg(
      func=mdp.leg_length_limits_penalty,
      weight=-1.0,
      params={
        "limits": (0.15, 0.40),
        "sensor_name": "leg_length_scan"},
    ),
    "leg_theta_difference_penalty": RewardTermCfg(
      func=mdp.leg_theta_difference_penalty,
      weight=-1.0,
      params={"sensor_name": "leg_theta_scan"},
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(30.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-2.0, 2.0)},
          {"step": 2000 * 48, "lin_vel_x": (-1.2, 1.2), "ang_vel_z": (-4.0, 4.0)},
          {"step": 4000 * 48, "lin_vel_x": (-1.5, 1.5), "ang_vel_z": (-6.0, 6.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    decimation=4,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(
        foot_height_scan,
        feet_ground_contact,
        leg_length_scan,
        leg_theta_scan,
      ),
      num_envs=8192,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      max_extra_envs=64,
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
      entity_name="robot",
      body_name="base_link",
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005,
        impratio=10,
        cone="elliptic",
        iterations=20,
        ls_iterations=20,
      ),
    ),
    episode_length_s=20.0,
    rewards=rewards,
    terminations=terminations,
    commands=commands,
    curriculum=curriculum,
    metrics=metrics,
  )
