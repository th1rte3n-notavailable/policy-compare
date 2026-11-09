from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def joint_acc_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint accelerations of the selected joints (from the last physics step)."""
  asset = env.scene[asset_cfg.name]
  return asset.data.joint_acc[:, asset_cfg.joint_ids]


def prev_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Raw policy output from the previous policy step (before scale/offset)."""
  return env.action_manager.prev_action


def base_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Base height: world z of the root link. Shape [B, 1].

  Same quantity as ``track_base_height_z`` without ``height_sensor_name``
  (flat-ground assumption).
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)


def base_mass(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-env mass of the selected bodies, as randomized by DR. Shape [B, n_bodies]."""
  asset = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
  return asset.data.model.body_mass[:, body_ids]


def base_com(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-env COM offset (body frame) of the selected bodies. Shape [B, n_bodies*3].

  Reads ``body_ipos`` from the sim model, so it reflects ``dr.body_com_offset``.
  """
  asset = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
  return asset.data.model.body_ipos[:, body_ids].flatten(start_dim=1)


def foot_friction(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-env foot friction ``(slide, spin, roll)`` per geom. Shape [B, n_geoms*3].

  Reads ``geom_friction`` from the sim model, so it reflects the
  ``dr.geom_friction`` events.
  """
  asset = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
  return asset.data.model.geom_friction[:, geom_ids].flatten(start_dim=1)


def joint_bias_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-env joint encoder bias of the selected joints. Shape [B, n_joints].

  Reads ``data.encoder_bias``, so it reflects ``dr.encoder_bias`` and explains
  the offset between the (biased) ``joint_pos`` observed by the actor and the
  true ``joint_pos``.
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.encoder_bias[:, asset_cfg.joint_ids]


def joint_stiffness_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-env actuator stiffness (kp) at the selected actuators. Shape [B, n].

  Reads ``actuator_gainprm[..., 0]``, the slot mjlab's spec builders use for
  both ``<position>`` (kp) and ``<velocity>`` (kv) actuators. Select via
  ``actuator_names``; for this robot those match the joint names.
  """
  asset = env.scene[asset_cfg.name]
  ctrl_ids = asset.indexing.ctrl_ids[asset_cfg.actuator_ids]
  return env.sim.model.actuator_gainprm[:, ctrl_ids, 0]


def joint_damping_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-env actuator damping (kd) at the selected actuators. Shape [B, n].

  Reads ``-actuator_biasprm[..., 2]``; same slot for ``<position>`` (kd) and
  ``<velocity>`` (kv) actuators.
  """
  asset = env.scene[asset_cfg.name]
  ctrl_ids = asset.indexing.ctrl_ids[asset_cfg.actuator_ids]
  return -env.sim.model.actuator_biasprm[:, ctrl_ids, 2]


def tendon_force(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-env tendon effort target. Shape [B, n_tendons].

  Reads ``data.tendon_effort_target``, which reflects the constant-force DR
  event and is otherwise invisible to the policy. Useful as privileged critic
  information.
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.tendon_effort_target[:, asset_cfg.tendon_ids]


def act_delay_steps(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Current action-command delay per env, in physics steps. Shape [B, 1].

  Reads the lag currently held by each delayed actuator's ``DelayBuffer``.
  Actuator groups that do not share a buffer (e.g. position and velocity) sample
  their lags independently, so the worst case (maximum) is reported. Returns
  zeros when no actuator delay is configured.
  """
  asset = env.scene[asset_cfg.name]
  lags: list[torch.Tensor] = []
  seen_buffers: set[int] = set()
  for actuator in asset.actuators:
    # Private access mirrors how ``Actuator.apply_delay`` and the fused builtin
    # groups hold their (possibly shared) delay buffer.
    buffer = getattr(actuator, "_delay_buffer", None)
    if buffer is None or id(buffer) in seen_buffers:
      continue
    seen_buffers.add(id(buffer))
    lags.append(buffer.current_lags.float())
  if not lags:
    return torch.zeros((env.num_envs, 1), dtype=torch.float, device=env.device)
  return torch.stack(lags, dim=-1).amax(dim=-1, keepdim=True)
