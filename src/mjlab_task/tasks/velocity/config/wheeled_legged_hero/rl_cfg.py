"""RL configuration for wheeled legged hero robot velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def hero_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for wheeled legged hero robot velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 64, 32),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
      class_name="MLPModel"
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      class_name="MLPModel"
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.01,
      desired_kl=0.005,
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
    ),
    num_steps_per_env=48,
    max_iterations=8000,
    save_interval=100,
    experiment_name="hero_velocity",
    logger="tensorboard",
  )
