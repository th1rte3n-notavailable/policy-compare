"""wheeled_legged_hero constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg, BuiltinMotorActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

HERO_XML: Path = (
  SRC_PATH / "assets" / "robots" / "wheeled_legged_hero" / "xmls" / "hero.xml"
)
assert HERO_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(HERO_XML))


##
# Actuator config.
##

def _hip_actuator(
  joint_name: str,
  armature: float,
  viscous_damping: float,
  frictionloss: float,
) -> BuiltinPositionActuatorCfg:
  """Position-controlled hip actuator with identified joint parameters."""
  return BuiltinPositionActuatorCfg(
    target_names_expr=(joint_name,),
    stiffness=30.0,
    damping=1.0,
    effort_limit=40.0,
    armature=armature,
    viscous_damping=viscous_damping,
    frictionloss=frictionloss,
    # Command latency: 5-20 ms (1-4 physics steps at 5 ms), resampled every 1 s.
    delay_min_lag=1,
    delay_max_lag=4,
    delay_update_period=200,
  )


def _wheel_actuator(
  joint_name: str,
  armature: float,
  viscous_damping: float,
  frictionloss: float,
) -> BuiltinVelocityActuatorCfg:
  """Velocity-controlled wheel actuator with identified joint parameters."""
  return BuiltinVelocityActuatorCfg(
    target_names_expr=(joint_name,),
    damping=0.2,  # kv
    effort_limit=6.0,
    armature=armature,
    viscous_damping=viscous_damping,
    frictionloss=frictionloss,
    # Command latency: 5-20 ms (1-4 physics steps at 5 ms), resampled every 1 s.
    delay_min_lag=1,
    delay_max_lag=4,
    delay_update_period=200,
  )


# Identified per-joint parameters: (armature, viscous damping, friction loss).
HERO_ACTUATORS_HIP = (
  _hip_actuator("L_thigh_hipjoint", 0.032077, 0.102576, 0.524101),
  _hip_actuator("L_leg1_hipjoint", 0.020803, 0.221320, 0.635953),
  _hip_actuator("R_thigh_hipjoint", 0.039156, 0.019438, 0.747308),
  _hip_actuator("R_leg1_hipjoint", 0.020687, 0.019569, 0.204171),
)
HERO_ACTUATORS_WHEEL = (
  _wheel_actuator("L_wheel_joint", 0.000853, 0.000932, 0.024860),
  _wheel_actuator("R_wheel_joint", 0.000672, 0.002292, 0.029213),
)
HERO_CONSTANT_FORCE = BuiltinMotorActuatorCfg(
  target_names_expr=(
    "Left_force",
    "Right_force",
  ),
  transmission_type=TransmissionType.TENDON,
  effort_limit=300.0,
  gear=1.0,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32713),
  joint_pos={
    "L_thigh_hipjoint": -0.463933,
    "L_leg5_joint": -1.965552,
    "L_leg1_hipjoint": 0.463943,
    "R_thigh_hipjoint": 0.463933,
    "R_leg5_joint": 1.965552,
    "R_leg1_hipjoint": -0.463933,
    ".*_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_wheel_regex = ".*_ground_collision"
_collision_regex = ".*_collision"


FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_wheel_regex, _collision_regex,),
  contype=0,
  conaffinity=1,
  solref=(0.01, 1),
  condim={_wheel_regex: 6, _collision_regex: 1,},
  priority={_wheel_regex: 1, ".*": 0},
  friction={_wheel_regex: (1, 5e-3, 5e-4)},
)

##
# Final config.
##

HERO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    *HERO_ACTUATORS_HIP,
    *HERO_ACTUATORS_WHEEL,
    HERO_CONSTANT_FORCE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_wheeled_legged_hero_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=HERO_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_wheeled_legged_hero_robot_cfg())

  # viewer.launch(robot.spec.compile())

  # 调试工具：参数验证
  from dataclasses import asdict
  object = robot.spec.actuators
  for _ in range(len(object)):
    obj = object[_]
    for name in dir(obj):
      # 跳过内部属性
      if name.startswith('_'):
          continue
      try:
          val = getattr(obj, name)
          # 只取数据属性
          if not callable(val):
            # if name == "name":
            print(f"{name} = {val}")
      except Exception as e:
          print(f"{name}: 获取失败 - {e}")




