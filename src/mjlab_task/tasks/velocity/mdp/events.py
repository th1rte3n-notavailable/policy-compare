from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.actuator import BuiltinVelocityActuator
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def randomize_constant_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  force_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  env_ids = resolve_env_ids(env, env_ids)

  asset: Entity = env.scene[asset_cfg.name]
  tendon_ids = asset_cfg.tendon_ids
  num_tendons = asset.num_tendons if isinstance(tendon_ids, slice) else len(tendon_ids)

  forces = sample_uniform(*force_range, (len(env_ids), num_tendons), env.device)
  asset.set_tendon_effort_target(forces, tendon_ids, env_ids)


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_joint_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Scale the gains of the selected joint actuators by per-env factors (startup DR).

  Mirrors ``dr.pd_gains`` but indexes ctrl slots via ``asset.indexing.ctrl_ids``,
  the same spec-actuator space ``SceneEntityCfg`` and the observation terms use.
  Both ``<position>`` and ``<velocity>`` actuators store their stiffness in
  ``gainprm[0]`` and damping in ``-biasprm[2]``. For position actuators these
  are independent parameters (kp and kd), so ``kp_scale`` and ``kd_scale`` are
  applied independently. For velocity actuators both slots are the same kv, so
  they must receive the same scale; otherwise the steady-state speed shifts by
  ``kp_scale / kd_scale``. Only position actuators additionally get the
  position-feedback term ``biasprm[1] = -kp``; velocity actuators keep
  ``biasprm[1]`` at its default. Select explicitly; tendon/motor actuators are
  not supported.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  ctrl_ids = asset.indexing.ctrl_ids[asset_cfg.actuator_ids]

  velocity_slots = {
    int(slot)
    for act in asset.actuators
    if isinstance(act, BuiltinVelocityActuator)
    for slot in act.global_ctrl_ids
  }
  is_position = torch.tensor(
    [int(slot) not in velocity_slots for slot in ctrl_ids],
    dtype=torch.bool,
    device=env.device,
  )

  gainprm_default = env.sim.get_default_field("actuator_gainprm")
  biasprm_default = env.sim.get_default_field("actuator_biasprm")
  kp_default = gainprm_default[ctrl_ids, 0]
  kp_feedback_default = biasprm_default[ctrl_ids, 1]
  kd_default = -biasprm_default[ctrl_ids, 2]

  n = len(ctrl_ids)
  kp_scale = sample_uniform(kp_range[0], kp_range[1], (len(env_ids), n), env.device)
  kd_scale = sample_uniform(kd_range[0], kd_range[1], (len(env_ids), n), env.device)

  model = env.sim.model
  env_grid, ctrl_grid = env_ids[:, None], ctrl_ids
  model.actuator_gainprm[env_grid, ctrl_grid, 0] = kp_default * kp_scale
  model.actuator_biasprm[env_grid, ctrl_grid, 1] = torch.where(
    is_position, -kp_default * kp_scale, kp_feedback_default
  )
  # Position actuators: -biasprm[2] is kd, independent of kp. Velocity
  # actuators: both gainprm[0] and -biasprm[2] are kv, so they must share the
  # same scale or the steady-state speed becomes (kp_scale / kd_scale) * cmd.
  bias_scale = torch.where(is_position, kd_scale, kp_scale)
  model.actuator_biasprm[env_grid, ctrl_grid, 2] = -kd_default * bias_scale
