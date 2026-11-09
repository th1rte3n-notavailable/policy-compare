"""Virtual leg angles (theta) of a wheeled-legged robot.

Planar forward kinematics of the 5-bar leg: from the two actuated joint angles
of each leg to the angle of its virtual link, in radians. Angular counterpart of
:mod:`src.sensor.leg_length_sensor`. ``sensor.data`` shape is ``[B]`` for a
single leg and ``[B, L]`` for multiple legs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg


@dataclass
class JointRef:
  """A single leg: the two actuated joints whose angles define theta.

  Listed in ``(phi4, phi1)`` order, the order the forward kinematics expects.
  """

  phi4_joint: str
  """Joint whose angle enters the forward kinematics as ``phi4``."""

  phi1_joint: str
  """Joint whose angle enters the forward kinematics as ``phi1``."""


def leg_theta_from_joint_angles(
  joint_angles: torch.Tensor,
  link_lengths: tuple[float, float, float, float, float],
  base_pitch: torch.Tensor,
) -> torch.Tensor:
  """Forward kinematics of the 5-bar leg: joint angles -> virtual leg angle.

  Planar solution in the sagittal plane (``+x`` forward, ``+y`` up): motor 1 at
  the origin drives link 1 to B, motor 2 at ``(l5, 0)`` drives link 4 to D, and
  links 2/3 close the loop to the foot C. Theta is the angle of the hip midpoint
  -> C vector from the vertical.

  Args:
    joint_angles: ``[B, L, 2]`` radians, ordered ``(phi4, phi1)``. These are raw
      MuJoCo ``qpos`` readings, so each joint's ``ref`` and axis sign still have
      to be accounted for. A ``pi`` bias is applied to ``phi1`` here.
    link_lengths: ``(l1, l2, l3, l4, l5)`` in metres.
    base_pitch: ``[B]`` body pitch ``phi``, one value per environment, in the
      same plane convention as ``phi1`` / ``phi4``.

  Returns:
    ``[B, L]`` radians in ``[-pi, pi)``; ``0`` where the loop cannot close.
  """
  l1, l2, l3, l4, l5 = link_lengths
  phi4 = joint_angles[..., 0]
  # phi1 carries a pi bias relative to the FK's +x axis.
  phi1 = joint_angles[..., 1] + math.pi

  # B: end of link 1.
  x_b = l1 * torch.cos(phi1)
  y_b = l1 * torch.sin(phi1)

  # D: end of link 4.
  x_d = l5 + l4 * torch.cos(phi4)
  y_d = l4 * torch.sin(phi4)

  # Solve for phi2 from |B + l2 * dir(phi2) - D| == l3.
  dx = x_d - x_b
  dy = y_d - y_b
  l_bd_sq = dx * dx + dy * dy
  a0 = 2.0 * l2 * dx
  b0 = 2.0 * l2 * dy
  c0 = l2 * l2 + l_bd_sq - l3 * l3

  disc = a0 * a0 + b0 * b0 - c0 * c0
  valid = disc >= 0.0
  sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
  # Positive root; the other one is the mirrored assembly.
  phi2 = 2.0 * torch.atan2(b0 + sqrt_disc, a0 + c0)

  # C: the foot / wheel centre.
  x_c = x_b + l2 * torch.cos(phi2)
  y_c = y_b + l2 * torch.sin(phi2)

  # Virtual leg: hip midpoint -> C.
  phi0 = torch.atan2(y_c, x_c - l5 / 2.0)

  # [B] -> [B, 1]: broadcast the pitch over the legs, not over the batch.
  phi = base_pitch.unsqueeze(-1)
  theta = math.pi / 2.0 - (phi0 + phi)

  # Wrap to [-pi, pi); unreachable configurations fall back to 0.
  theta = torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi
  return torch.where(valid, theta, torch.zeros_like(theta))


@dataclass
class LegThetaSensorCfg(SensorCfg):
  """Configuration for :class:`LegThetaSensor`.

  Args:
    joint_pair: A single :class:`JointRef` for one leg, or a tuple for multiple.
    negate: One flag per leg, matching ``joint_pair``; ``True`` negates that
      leg's joint angles, for legs driven by mirrored joints. Empty or all
      ``False`` means no negation.
    entity: Optional entity name, used as a prefix for the joint names.
    link_lengths: ``(l1, l2, l3, l4, l5)`` in metres; must be set to the real
      geometry.
    ref_body: Optional body whose pitch is the ``phi`` term, written in full and
      used as-is (no entity prefix). Defaults to ``f"{entity}/base_link"``.
  """

  joint_pair: JointRef | tuple[JointRef, ...] = ()
  negate: tuple[bool, ...] = ()
  entity: str | None = None
  link_lengths: tuple[float, float, float, float, float] = (0, 0, 0, 0, 0)
  ref_body: str | None = None

  def build(self) -> LegThetaSensor:
    return LegThetaSensor(self)


class LegThetaSensor(Sensor[torch.Tensor]):
  """Virtual leg angles theta [rad], one per configured leg."""

  def __init__(self, cfg: LegThetaSensorCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self._data: mjwarp.Data | None = None
    self._qpos_adr: list[int] = []  # phi4 then phi1, per leg.
    self._num_legs: int = 0
    self._ref_body_id: int = -1
    self._sign: torch.Tensor | None = None  # Per-leg sign; None when all +1.

  @staticmethod
  def _as_tuple(x: JointRef | tuple[JointRef, ...]) -> tuple[JointRef, ...]:
    return (x,) if isinstance(x, JointRef) else tuple(x)

  def _prefixed(self, name: str) -> str:
    return f"{self.cfg.entity}/{name}" if self.cfg.entity else name

  def _ref_body_name(self) -> str:
    """Body whose pitch is the ``phi`` term; written in full, used as-is."""
    return self.cfg.ref_body or self._prefixed("base_link")

  def edit_spec(
    self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]
  ) -> None:
    del entities
    pairs = self._as_tuple(self.cfg.joint_pair)
    if not pairs:
      raise ValueError("joint_pair must contain at least one JointRef.")

    flags = tuple(self.cfg.negate)
    if flags and len(flags) != len(pairs):
      raise ValueError(
        f"negate must be empty or hold one flag per leg ({len(pairs)}), "
        f"got {len(flags)}."
      )

    existing_joints = {j.name for j in scene_spec.joints}
    for pair in pairs:
      for raw in (pair.phi4_joint, pair.phi1_joint):
        full = self._prefixed(raw)
        if full not in existing_joints:
          raise ValueError(
            f"Joint '{full}' not found. Available: {sorted(existing_joints)}"
          )

    existing_bodies = {b.name for b in scene_spec.bodies}
    ref_body_name = self._ref_body_name()
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
    del model
    self._data = data

    pairs = self._as_tuple(self.cfg.joint_pair)
    self._num_legs = len(pairs)

    flags = tuple(self.cfg.negate) or (False,) * self._num_legs
    self._sign = (
      None
      if not any(flags)
      else torch.tensor([-1.0 if f else 1.0 for f in flags], device=device)
    )

    # phi4 then phi1, as the forward kinematics expects.
    self._qpos_adr = [
      int(mj_model.jnt_qposadr[mj_model.joint(self._prefixed(name)).id])
      for pair in pairs
      for name in (pair.phi4_joint, pair.phi1_joint)
    ]

    self._ref_body_id = int(mj_model.body(self._ref_body_name()).id)

  def _compute_data(self) -> torch.Tensor:
    assert self._data is not None
    assert self._num_legs > 0
    assert self._ref_body_id >= 0  # Otherwise xmat[:, -1] would read another body.
    assert len(self._qpos_adr) == 2 * self._num_legs

    qpos = self._data.qpos  # [B, nq]
    joint_angles = torch.stack(
      [
        torch.stack(
          [qpos[:, self._qpos_adr[2 * i]], qpos[:, self._qpos_adr[2 * i + 1]]],
          dim=-1,
        )
        for i in range(self._num_legs)
      ],
      dim=1,
    )  # [B, L, 2]

    xmat = self._data.xmat[:, self._ref_body_id].reshape(-1, 3, 3)  # [B, 3, 3]
    # Pitch of the body x-axis off the horizontal. Using the full horizontal
    # magnitude keeps it invariant to the robot's yaw, unlike reading the
    # world x/z components directly.
    x_w = xmat[:, :, 0]  # [B, 3] body x-axis in world.
    base_pitch = torch.atan2(
      -x_w[:, 2], torch.sqrt(x_w[:, 0] ** 2 + x_w[:, 1] ** 2)
    )  # [B]

    # Legs flagged in ``negate``: mirrored joints read with the opposite sign.
    if self._sign is not None:
      joint_angles = joint_angles * self._sign.view(1, -1, 1)

    theta = leg_theta_from_joint_angles(
      joint_angles, self.cfg.link_lengths, base_pitch
    )  # [B, L]

    # Squeeze for single-leg backward compatibility.
    if self._num_legs == 1:
      return theta[:, 0]
    return theta
