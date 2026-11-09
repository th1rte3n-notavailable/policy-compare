"""wheeled legged hero robot velocity environment configurations."""


from src.assets.robots import (
  get_wheeled_legged_hero_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def hero_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create wheeled legged hero robot flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.entities = {"robot": get_wheeled_legged_hero_robot_cfg()} 

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  return cfg
