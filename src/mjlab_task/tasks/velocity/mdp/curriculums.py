from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from .velocity_command import LegLengthVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  """Widen the commanded velocity ranges as training progresses.

  Only ranges that exist in ``LegLengthVelocityCommandCfg.Ranges`` are
  scheduled: the twist command carries forward velocity (``lin_vel_x``), yaw
  rate (``ang_vel_z``) and height (``lin_pos_z``), with no lateral velocity.
  """
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(LegLengthVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if (lin_vel_x := stage.get("lin_vel_x")) is not None:
        cfg.ranges.lin_vel_x = lin_vel_x
      if (ang_vel_z := stage.get("ang_vel_z")) is not None:
        cfg.ranges.ang_vel_z = ang_vel_z
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }
