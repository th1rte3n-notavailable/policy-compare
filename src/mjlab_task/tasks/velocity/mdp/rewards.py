from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

from .terrain_utils import terrain_normal_from_sensors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity_x(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base forward (x) linear velocity.

  Only the x component is tracked; the lateral (y) velocity is not penalized.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  lin_vel_error = torch.square(command[:, 0] - actual[:, 0])
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity_z(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking of the commanded yaw rate (body z angular velocity).

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 1] - actual[:, 2])
  return torch.exp(-z_error / std**2)


def angular_velocity_xy_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize body roll/pitch rates (xy angular velocity) with an L2 cost.

  The commanded xy angular velocities are assumed to be zero, so this is a
  regularization term rather than a tracking reward: it returns the squared norm
  of the root link's xy angular velocity in the body frame and is meant to be
  used with a negative weight. The z (yaw) rate is left out, as it is tracked by
  ``track_angular_velocity_z``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel_xy = asset.data.root_link_ang_vel_b[:, :2]
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def linear_velocity_z_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base vertical (z) velocity with an L2 cost.

  The commanded z velocity is assumed to be zero, so this is a regularization
  term rather than a tracking reward: it returns the squared z velocity of the
  root link in the body frame and is meant to be used with a negative weight.
  Being in the body frame, climbing a slope at constant speed does not count as
  vertical motion -- only bouncing does.
  """
  asset: Entity = env.scene[asset_cfg.name]
  lin_vel_z = asset.data.root_link_lin_vel_b[:, 2]
  return torch.square(lin_vel_z)


def track_leg_length_z(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  """Reward for tracking the commanded leg length.

  The commanded leg length is the third component of the command vector
  (``command[:, 2]``) and is a single scalar shared by both legs, while the
  leg length sensor reports one length per leg (left first, then right), so the
  squared error is summed over the legs.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  sensor = env.scene[sensor_name]
  actual = sensor.data
  assert isinstance(actual, torch.Tensor), (
    f"Sensor '{sensor_name}' must return a torch.Tensor, got {type(actual).__name__}."
  )
  actual = actual.reshape(env.num_envs, -1)
  target = command[:, 2].unsqueeze(-1)
  length_error = torch.sum(torch.square(target - actual), dim=-1)
  return torch.exp(-length_error / std**2)


def leg_length_limits_penalty(
  env: ManagerBasedRlEnv,
  limits: tuple[float, float],
  sensor_name: str,
) -> torch.Tensor:
  """Penalize leg lengths that leave the ``(min_length, max_length)`` range.

  Both extremes are penalized, with a squared hinge cost that is exactly zero
  inside the range and grows quadratically once a leg is too short or too long::

      sum_l max(min_length - L_l, 0) ** 2 + max(L_l - max_length, 0) ** 2

  The sensor reports one length per leg (left first, then right) and the cost is
  summed over both legs, so the term is meant to be used with a negative weight.
  """
  min_length, max_length = limits
  sensor = env.scene[sensor_name]
  actual = sensor.data
  assert isinstance(actual, torch.Tensor), (
    f"Sensor '{sensor_name}' must return a torch.Tensor, got {type(actual).__name__}."
  )
  actual = actual.reshape(env.num_envs, -1)  # [B, L]: left leg, then right leg.
  too_short = torch.clamp(min_length - actual, min=0.0)
  too_long = torch.clamp(actual - max_length, min=0.0)
  return torch.sum(too_short + too_long, dim=-1)


def leg_theta_difference_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize the left/right virtual leg angle mismatch with an L2 cost.

  The sensor reports one theta per leg (left first, then right), so the cost is
  the squared difference between the two::

      (theta_left - theta_right) ** 2

  Zero when both legs hold the same angle, so the term is meant to be used with
  a negative weight.
  """
  sensor = env.scene[sensor_name]
  actual = sensor.data
  assert isinstance(actual, torch.Tensor), (
    f"Sensor '{sensor_name}' must return a torch.Tensor, got {type(actual).__name__}."
  )
  theta = actual.reshape(env.num_envs, -1)  # [B, L]: left leg, then right leg.
  assert theta.shape[1] >= 2, (
    f"Sensor '{sensor_name}' must report at least two legs, got {theta.shape[1]}."
  )
  return torch.square(theta[:, 0] - theta[:, 1])


def track_base_height_z(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  height_sensor_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base height above the terrain.

  The commanded height is the third component of the command vector
  (``command[:, 2]``). The actual height is measured as the root link's world z
  minus the terrain height under the feet, so the error is invariant to the
  absolute elevation of the terrain the robot stands on.

  Without ``height_sensor_name``, the ground is assumed to be flat at z = 0, so
  the obstacle height is zero and the height is simply the root link's world z.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  base_z = asset.data.root_link_pos_w[:, 2]  # [B]
  if height_sensor_name is None:
    actual = base_z
  else:
    height_sensor = env.scene[height_sensor_name]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"track_base_height_z requires a TerrainHeightSensor, got "
      f"{type(height_sensor).__name__}"
    )
    data = height_sensor.data
    # Ground height under each foot frame: frame z minus the clearance to terrain.
    terrain_z = (data.frame_pos_w[:, :, 2] - data.heights).mean(dim=1)  # [B]
    actual = base_z - terrain_z
  height_error = torch.square(command[:, 2] - actual)
  return torch.exp(-height_error / std**2)


class UprightPenalty:
  """Penalize tilt of the base with an L2 cost.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.

  The cost is ``sin^2(theta)`` in [0, 1], where *theta* is the tilt angle, so it is
  meant to be used with a negative weight.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return xy_squared

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )





