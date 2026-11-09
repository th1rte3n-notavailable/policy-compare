from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class LegLengthVelocityCommand(CommandTerm):
  """Velocity command whose third component is the commanded leg length.

  The tracking metric reads the current leg length from the sensor named by
  ``cfg.lin_pos_z_sensor_name``. See :class:`ZPosVelocityCommand` for the
  variant that reads the base height from the sim instead.
  """

  cfg: LegLengthVelocityCommandCfg

  def __init__(self, cfg: LegLengthVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_forward_env = torch.zeros_like(self.is_standing_env)

    self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    if self._tracks_lin_pos_z:
      self.metrics["error_lin_pos_z"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def _tracks_lin_pos_z(self) -> bool:
    """Whether the lin_pos_z tracking metric is recorded.

    True when a sensor is configured; overridden by subclasses that read the
    quantity from the sim directly.
    """
    return self.cfg.lin_pos_z_sensor_name is not None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _current_lin_pos_z(self) -> torch.Tensor:
    """Current leg length (m), read from the configured sensor."""
    assert self.cfg.lin_pos_z_sensor_name is not None
    sensor = self._env.scene[self.cfg.lin_pos_z_sensor_name]
    data = sensor.data
    if not isinstance(data, torch.Tensor):
      raise TypeError(
        f"Sensor '{self.cfg.lin_pos_z_sensor_name}' returns {type(data).__name__}, "
        "but a lin_pos_z command sensor must return a torch.Tensor (num_envs, 2)."
      )
    
    data = data.reshape(self.num_envs, -1)
    if data.shape[1] != 2:
      raise ValueError(
        f"Sensor '{self.cfg.lin_pos_z_sensor_name}' returned "
        f"{data.shape[1]} values per env, expected 2."
      )
    return data

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_x"] += (
      torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_link_lin_vel_b[:, 0])
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 1] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )
    if self._tracks_lin_pos_z:
      # Normalize to [B, K] so implementations may return either [B] or [B, K].
      current = self._current_lin_pos_z().reshape(self.num_envs, -1)
      target = self.vel_command_b[:, 2].unsqueeze(-1)
      self.metrics["error_lin_pos_z"] += (
          torch.abs(target - current).mean(dim=-1) / max_command_step
      )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.lin_pos_z)
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      self.vel_command_b[fwd_ids, 0] = (
        self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      )
      self.vel_command_b[fwd_ids, 1] = 0.0

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    if self.cfg.init_velocity_prob > 0.0:
      assert isinstance(env_ids, torch.Tensor)
      r = torch.empty(len(env_ids), device=self.device)
      init_ids = env_ids[r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob]
      if len(init_ids) > 0:
        vel_b = torch.zeros(len(init_ids), 6, device=self.device)
        vel_b[:, 0] = self.vel_command_b[init_ids, 0]
        vel_b[:, 5] = self.vel_command_b[init_ids, 1]
        self.robot.write_root_link_velocity_b_to_sim(vel_b, env_ids=init_ids)
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # Pure function of the current state; refreshing all envs is safe.
    del env_ids
    
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :2] = 0.0

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], 0.0, 0.0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[1]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class LegLengthVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  rel_standing_envs: float = 0.0
  rel_forward_envs: float = 0.0
  init_velocity_prob: float = 0.0
  lin_pos_z_sensor_name: str | None = None
  """Name of the sensor that reports the current leg length, used to log the
  lin_pos_z tracking error. When None, no tracking metric is recorded and the
  command term never reads leg length from the sim."""

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    ang_vel_z: tuple[float, float]
    lin_pos_z: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> LegLengthVelocityCommand:
    return LegLengthVelocityCommand(self, env)


class ZPosVelocityCommand(LegLengthVelocityCommand):
  """Velocity command whose third component is the base z position.

  Identical to :class:`LegLengthVelocityCommand` except that the tracking metric
  reads the base height directly from the root link's world z, so no sensor is
  needed. On flat ground that is the height above the ground; on elevated
  terrain it is the height above the z = 0 plane.
  """

  cfg: ZPosVelocityCommandCfg

  @property
  def _tracks_lin_pos_z(self) -> bool:
    return True

  def _current_lin_pos_z(self) -> torch.Tensor:
    """Current base height (m), read directly from the root link's world z."""
    return self.robot.data.root_link_pos_w[:, 2]


@dataclass(kw_only=True)
class ZPosVelocityCommandCfg(LegLengthVelocityCommandCfg):
  """Config for :class:`ZPosVelocityCommand`.

  ``lin_pos_z_sensor_name`` is inherited but unused -- no sensor is read.
  """

  def build(self, env: ManagerBasedRlEnv) -> ZPosVelocityCommand:
    return ZPosVelocityCommand(self, env)

