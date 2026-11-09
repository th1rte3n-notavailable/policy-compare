"""Shared constants for the dual-policy MuJoCo comparison demo.

Everything here mirrors the training configuration that produced the two
policies (see ``src/mjlab_task`` and the ``params/env.yaml`` files next to each
ONNX model), so that the comparison runs the policies under the same
observation, action, actuator and contact settings they were trained with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HERO_XML = (
    PROJECT_ROOT
    / "src"
    / "mjlab_task"
    / "assets"
    / "robots"
    / "wheeled_legged_hero"
    / "xmls"
    / "hero.xml"
)
POLICY_ROOT = PROJECT_ROOT / "src" / "policy"

DEFAULT_OLD_RUN = "2026-10-29_22-47-59"
DEFAULT_NEW_RUN = "2026-11-04_23-15-26"


def resolve_policy_file(value: str | Path) -> Path:
    """Resolve a policy argument to an ``.onnx`` file.

    Accepts an explicit ONNX path, a run directory, or a run name under
    ``src/policy``.  When given a directory the first ``*.onnx`` file in it is
    used (the exported actor).
    """
    path = Path(value).expanduser()
    candidates: list[Path] = []
    if path.suffix == ".onnx":
        candidates.append(path)
    elif path.is_dir():
        candidates.extend(sorted(path.glob("*.onnx")))
    else:
        run_dir = POLICY_ROOT / path.name
        if run_dir.is_dir():
            candidates.extend(sorted(run_dir.glob("*.onnx")))
        elif path.suffix == ".pt":
            raise ValueError(
                f"Policy {path} is a torch checkpoint; this demo loads the "
                "exported ONNX actor next to it. Pass the .onnx file or the "
                "run directory instead."
            )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find an ONNX policy for '{value}'. Pass a path to an "
            ".onnx file, a run directory, or a run name under "
            f"{POLICY_ROOT}."
        )
    return candidates[0].resolve()


# ---------------------------------------------------------------------------
# Robot model
# ---------------------------------------------------------------------------

# Joint order in hero.xml (and therefore in the ONNX metadata).
JOINT_NAMES: tuple[str, ...] = (
    "L_thigh_hipjoint",
    "L_leg5_joint",
    "L_wheel_joint",
    "L_leg1_hipjoint",
    "L_leg2_joint",
    "L_leg3_joint",
    "L_leg4_joint",
    "R_thigh_hipjoint",
    "R_leg5_joint",
    "R_wheel_joint",
    "R_leg1_hipjoint",
    "R_leg2_joint",
    "R_leg3_joint",
    "R_leg4_joint",
)

# Reset / default joint coordinates (the values stored in mjlab's
# ``INIT_STATE`` and exported as ``default_joint_pos`` in the ONNX metadata).
DEFAULT_JOINT_POS_BY_NAME: dict[str, float] = {
    "L_thigh_hipjoint": -0.463933,
    "L_leg5_joint": -1.965552,
    "L_wheel_joint": 0.0,
    "L_leg1_hipjoint": 0.463943,
    "L_leg2_joint": 0.0,
    "L_leg3_joint": 0.0,
    "L_leg4_joint": 0.0,
    "R_thigh_hipjoint": 0.463933,
    "R_leg5_joint": 1.965552,
    "R_wheel_joint": 0.0,
    "R_leg1_hipjoint": -0.463943,
    "R_leg2_joint": 0.0,
    "R_leg3_joint": 0.0,
    "R_leg4_joint": 0.0,
}
DEFAULT_JOINT_POS = np.array(
    [DEFAULT_JOINT_POS_BY_NAME[name] for name in JOINT_NAMES], dtype=np.float64
)

# Actuated joints, in the order used by the action terms:
#   joint_pos term -> HIP_JOINTS (4 position actuators)
#   joint_vel term -> WHEEL_JOINTS (2 velocity actuators)
HIP_JOINTS: tuple[str, ...] = (
    "L_thigh_hipjoint",
    "L_leg1_hipjoint",
    "R_thigh_hipjoint",
    "R_leg1_hipjoint",
)
WHEEL_JOINTS: tuple[str, ...] = ("L_wheel_joint", "R_wheel_joint")

# Joint order of the six ``joint_vel`` observations (natural XML order among
# the actuated joints): L_thigh, L_wheel, L_leg1, R_thigh, R_wheel, R_leg1.
LEG_JOINTS_OBS: tuple[str, ...] = (
    "L_thigh_hipjoint",
    "L_wheel_joint",
    "L_leg1_hipjoint",
    "R_thigh_hipjoint",
    "R_wheel_joint",
    "R_leg1_hipjoint",
)

# Identified passive joint parameters from hero_constants.py:
# (armature, viscous damping, friction loss).
HIP_ACTUATOR_PARAMS: dict[str, tuple[float, float, float]] = {
    "L_thigh_hipjoint": (0.032077, 0.102576, 0.524101),
    "L_leg1_hipjoint": (0.020803, 0.221320, 0.635953),
    "R_thigh_hipjoint": (0.039156, 0.019438, 0.747308),
    "R_leg1_hipjoint": (0.020687, 0.019569, 0.204171),
}
WHEEL_ACTUATOR_PARAMS: dict[str, tuple[float, float, float]] = {
    "L_wheel_joint": (0.000853, 0.000932, 0.024860),
    "R_wheel_joint": (0.000672, 0.002292, 0.029213),
}

HIP_KP = 30.0
HIP_KD = 1.0
HIP_EFFORT_LIMIT = 40.0
WHEEL_DAMPING = 0.2
WHEEL_EFFORT_LIMIT = 6.0
TENDON_EFFORT_LIMIT = 300.0

# Initial root state (world frame). Both robots start here.
INIT_ROOT_POS = np.array([0.0, 0.0, 0.32713], dtype=np.float64)
INIT_ROOT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z

# ---------------------------------------------------------------------------
# Observation / action definition (must match the actor used in training)
# ---------------------------------------------------------------------------

ACTOR_OBS_DIM = 25
ACTION_DIM = 6

ANG_VEL_SCALE = 0.5
JOINT_VEL_SCALE = np.array([0.5, 0.1, 0.5, 0.5, 0.1, 0.5], dtype=np.float32)
COMMAND_SCALE = np.array([1.0, 0.5, 5.0], dtype=np.float32)

# Joint-position action term: target = default + 1.0 * raw, then clipped.
HIP_ACTION_SCALE = 1.0
HIP_TARGET_CLIP: dict[str, tuple[float, float]] = {
    "L_thigh_hipjoint": (-1.892251, 1.249341),
    "L_leg1_hipjoint": (-1.249341, 1.892251),
    "R_thigh_hipjoint": (-1.249341, 1.892251),
    "R_leg1_hipjoint": (-1.892251, 1.249341),
}

# Joint-velocity action term: target = 0 + 10.0 * raw, then clipped.
WHEEL_ACTION_SCALE = 10.0
WHEEL_TARGET_CLIP = (-30.0, 30.0)

# ---------------------------------------------------------------------------
# Commands: [lin_vel_x (m/s), ang_vel_z (rad/s), base_height (m)]
# ---------------------------------------------------------------------------

COMMAND_LIMITS = np.array(
    [
        [-1.5, 1.5],
        [-6.0, 6.0],
        [0.215, 0.465],
    ],
    dtype=np.float64,
)
DEFAULT_COMMAND = np.array([0.0, 0.0, 0.33], dtype=np.float64)

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

PHYSICS_TIMESTEP = 0.005
DECIMATION = 4
POLICY_DT = PHYSICS_TIMESTEP * DECIMATION  # 50 Hz policy / 200 Hz physics

# Command latency from the actuator configs (1-4 physics steps during
# training).  The demo uses one fixed lag for both robots so the comparison is
# strictly identical apart from the policies.
DEFAULT_DELAY_STEPS = 2

# Tendon preload: training randomized this uniformly in [90, 150] N at reset.
TENDON_FORCE_RANGE = (90.0, 150.0)
DEFAULT_TENDON_FORCE = 120.0

FALL_TILT_DEG = 30.0

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# The old policy is rendered translucent blue; the new policy is opaque
# orange.  Alpha < 1 makes MuJoCo draw the geom with alpha blending.
OLD_VISUAL_RGBA = (0.15, 0.45, 1.0, 0.30)
NEW_VISUAL_RGBA = (0.95, 0.45, 0.12, 1.0)

# Checkerboard from scene_test.xml:
#   <texture type="2d" name="checker" builtin="checker"
#            rgb1="0.2 0.3 0.4" rgb2="0.9 0.9 0.9" width="500" height="500"/>
#   <material name="checker_mat" texture="checker" specular="0.1"
#             shininess="1" texrepeat="10 10"/>
# The demo floor is larger than scene_test's 10 m plane, so texrepeat is
# scaled proportionally to preserve the same 1 m checker period.
FLOOR_RGBA_1 = (0.2, 0.3, 0.4, 1.0)
FLOOR_RGBA_2 = (0.9, 0.9, 0.9, 1.0)
FLOOR_TEXREPEAT = 60.0