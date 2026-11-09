"""Virtual leg lengths (L0) of a balance robot.

Computes the distance between paired sites, projected onto the xz-plane of the
entity's root body frame. This is useful when leg length should be measured in
the robot's sagittal plane, ignoring lateral (y) offsets.

Each leg is described by a :class:`SiteRef` pairing a top site and a bottom
site. Multiple legs can be configured in one sensor via ``site_pair``.

Output shape is ``[B]`` for a single leg and ``[B, L]`` for multiple legs.

Site names are given relative to ``entity``; if ``entity`` is None, they are
used as-is (no prefix). When ``entity`` is set, its root body is used as the
reference frame; otherwise the world frame is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg


@dataclass
class SiteRef:
  """A single leg: one top site and one bottom site, both relative to ``entity``."""

  site_top: str
  """Name of the top site (e.g. "L_leglength_sensor_site1")."""

  site_bottom: str
  """Name of the bottom site (e.g. "L_leglength_sensor_site2")."""


@dataclass
class LegLengthSensorCfg(SensorCfg):
  """Configuration for :class:`LegLengthSensor`.

  Args:
    site_pair: A single :class:`SiteRef` for one leg, or a tuple of them for
      multiple legs.
    entity: Optional entity name. Used both as a name prefix for the sites and
      as the source of the reference frame (its root body). If None, site names
      are used as-is and the world frame is used.
    ref_body: Optional reference body override. Defaults to
      ``f"{entity}/base_link"`` when ``entity`` is set.
  """

  site_pair: SiteRef | tuple[SiteRef, ...] = ()
  entity: str | None = None
  ref_body: str | None = None

  def build(self) -> LegLengthSensor:
    return LegLengthSensor(self)


class LegLengthSensor(Sensor[torch.Tensor]):
  """Virtual leg lengths L0 measured on the xz-plane of the entity root body.

  ``sensor.data`` shape is ``[B]`` for a single leg and ``[B, L]`` for
  multiple legs.
  """

  def __init__(self, cfg: LegLengthSensorCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self._data: mjwarp.Data | None = None
    self._top_ids: list[int] = []
    self._bottom_ids: list[int] = []
    self._ref_body_id: int = -1  # -1: world frame
    self._num_legs: int = 0

  @staticmethod
  def _as_tuple(x: SiteRef | tuple[SiteRef, ...]) -> tuple[SiteRef, ...]:
    return (x,) if isinstance(x, SiteRef) else tuple(x)

  def _prefixed(self, name: str) -> str:
    return f"{self.cfg.entity}/{name}" if self.cfg.entity else name

  def edit_spec(
    self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]
  ) -> None:
    del entities
    pairs = self._as_tuple(self.cfg.site_pair)
    if not pairs:
      raise ValueError("site_pair must contain at least one SiteRef.")

    existing_sites = {s.name for s in scene_spec.sites}
    for pair in pairs:
      for raw in (pair.site_top, pair.site_bottom):
        full = self._prefixed(raw)
        if full not in existing_sites:
          raise ValueError(
            f"Site '{full}' not found. Available: {sorted(existing_sites)}"
          )

    if self.cfg.entity is not None:
      ref_body_name = self.cfg.ref_body or f"{self.cfg.entity}/base_link"
      existing_bodies = {b.name for b in scene_spec.bodies}
      if ref_body_name not in existing_bodies:
        raise ValueError(
          f"Reference body '{ref_body_name}' not found. "
          f"Available: {sorted(existing_bodies)}"
        )

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    del model, device
    self._data = data

    pairs = self._as_tuple(self.cfg.site_pair)
    self._num_legs = len(pairs)

    self._top_ids = [
      int(mj_model.site(self._prefixed(p.site_top)).id) for p in pairs
    ]
    self._bottom_ids = [
      int(mj_model.site(self._prefixed(p.site_bottom)).id) for p in pairs
    ]

    if self.cfg.entity is None:
      self._ref_body_id = -1
    else:
      ref_body_name = self.cfg.ref_body or f"{self.cfg.entity}/base_link"
      self._ref_body_id = int(mj_model.body(ref_body_name).id)

  def _compute_data(self) -> torch.Tensor:
    assert self._data is not None
    assert self._num_legs > 0
    assert len(self._top_ids) == len(self._bottom_ids) == self._num_legs

    # [B, L, 3] site positions.
    pos_top = torch.stack(
      [self._data.site_xpos[:, sid] for sid in self._top_ids], dim=1
    )
    pos_bot = torch.stack(
      [self._data.site_xpos[:, sid] for sid in self._bottom_ids], dim=1
    )
    delta_w = pos_top - pos_bot  # [B, L, 3]

    if self._ref_body_id < 0:
      delta = delta_w
    else:
      R_bw = self._data.xmat[:, self._ref_body_id].reshape(-1, 3, 3)  # [B, 3, 3]
      # Rotate world-frame delta into body frame: R^T @ delta_w.
      delta = torch.einsum("bji,blj->bli", R_bw, delta_w)  # [B, L, 3]

    delta_xz = delta[..., [0, 2]]  # [B, L, 2]
    lengths = torch.linalg.norm(delta_xz, dim=-1)  # [B, L]

    # Squeeze for single-leg backward compatibility.
    if self._num_legs == 1:
      return lengths[:, 0]
    return lengths
  