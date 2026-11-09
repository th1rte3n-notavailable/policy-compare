"""Build one MuJoCo scene containing two copies of the hero robot.

The two copies share the same physics world (and therefore the same floor,
gravity, contact parameters and solver settings) but have independent free
joints, actuators, tendons and closed-loop equality constraints.  Their
collision geoms are built with ``contype=0, conaffinity=1`` exactly like the
training model, which means they collide with the floor (``contype=1``) but
*not* with each other.  That is what lets both robots start at the exact same
pose without pushing each other apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import config


@dataclass(frozen=True)
class RobotRuntime:
    """Indices/addresses for one robot copy in the combined model."""

    key: str
    prefix: str
    root_body_id: int
    root_qpos_adr: int
    root_dof_adr: int
    all_qpos_adr: tuple[int, ...]
    hip_qpos_adr: tuple[int, ...]
    obs_dof_adr: tuple[int, ...]
    gyro_adr: int
    hip_ctrl_ids: tuple[int, ...]
    wheel_ctrl_ids: tuple[int, ...]
    tendon_ctrl_ids: tuple[int, ...]
    default_hip_qpos: np.ndarray


@dataclass(frozen=True)
class DualRobotModel:
    model: mujoco.MjModel
    robots: dict[str, RobotRuntime]


def _make_robot_template() -> mujoco.MjSpec:
    """Load hero.xml and apply the identified joint-level actuator params."""
    if not config.HERO_XML.exists():
        raise FileNotFoundError(f"Robot MJCF not found: {config.HERO_XML}")

    spec = mujoco.MjSpec.from_file(str(config.HERO_XML))
    joint_params = {
        **config.HIP_ACTUATOR_PARAMS,
        **config.WHEEL_ACTUATOR_PARAMS,
    }
    for joint_name, (armature, damping, frictionloss) in joint_params.items():
        joint = spec.joint(joint_name)
        joint.armature = float(armature)
        joint.damping[0] = float(damping)
        joint.frictionloss = float(frictionloss)
    return spec


def _add_floor_and_lights(spec: mujoco.MjSpec) -> None:
    texture = spec.add_texture()
    texture.name = "checker"
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    texture.width = 500
    texture.height = 500
    texture.rgb1 = np.asarray(config.FLOOR_RGBA_1[:3], dtype=np.float64)
    texture.rgb2 = np.asarray(config.FLOOR_RGBA_2[:3], dtype=np.float64)

    floor_material = spec.add_material()
    floor_material.name = "floor_material"
    floor_material.rgba = [1.0, 1.0, 1.0, 1.0]
    # MjSpec expects a 10-slot texture-role vector.  The RGB base-color role
    # is mjTEXROLE_RGB (=1), not slot 0 (which is the user role).
    texture_slots = [""] * 10
    texture_slots[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "checker"
    floor_material.textures = texture_slots
    floor_material.texrepeat = [config.FLOOR_TEXREPEAT, config.FLOOR_TEXREPEAT]
    floor_material.texuniform = False
    floor_material.specular = 0.1
    floor_material.shininess = 1.0

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [30.0, 30.0, 0.1]
    floor.material = "floor_material"
    floor.contype = 1
    floor.conaffinity = 1

    sun = spec.worldbody.add_light()
    sun.name = "sun"
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.pos = [-2.0, 2.0, 4.0]
    sun.dir = [0.35, -0.35, -1.0]
    sun.diffuse = [0.8, 0.8, 0.8]
    sun.specular = [0.3, 0.3, 0.3]
    # MuJoCo's classic renderer draws shadow penumbrae with stochastic
    # sampling, which looks like black noise on the checker floor in a static
    # frame.  Shadows are cosmetic here, so turn them off for a clean
    # side-by-side view (the checker floor still provides contact cues).
    sun.castshadow = False

    fill = spec.worldbody.add_light()
    fill.name = "fill"
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.pos = [3.0, -3.0, 3.0]
    fill.dir = [-0.5, 0.4, -1.0]
    fill.diffuse = [0.35, 0.35, 0.35]
    fill.specular = [0.0, 0.0, 0.0]
    fill.castshadow = False


def _add_visual_materials(spec: mujoco.MjSpec) -> None:
    for name, rgba in (
        ("old_visual", config.OLD_VISUAL_RGBA),
        ("new_visual", config.NEW_VISUAL_RGBA),
    ):
        material = spec.add_material()
        material.name = name
        material.rgba = list(rgba)
        material.specular = 0.35
        material.shininess = 0.45
        material.reflectance = 0.04

    # Collision geoms are hidden (group 3) in the viewer, but give them a
    # valid material so no geom inherits an unresolved class material.
    hidden = spec.add_material()
    hidden.name = "collision_hidden"
    hidden.rgba = [0.15, 0.15, 0.15, 0.2]


def _add_robot_actuators(spec: mujoco.MjSpec, prefix: str) -> None:
    # Hip joints: <position kp=30 kv=1> with a 40 Nm effort limit.
    for joint_name in config.HIP_JOINTS:
        actuator = spec.add_actuator(
            name=f"{prefix}/motor/{joint_name}",
            target=f"{prefix}/{joint_name}",
        )
        actuator.set_to_position(config.HIP_KP, config.HIP_KD)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.inheritrange = 0.0
        actuator.ctrllimited = False
        actuator.forcelimited = True
        actuator.forcerange = [-config.HIP_EFFORT_LIMIT, config.HIP_EFFORT_LIMIT]
        actuator.gear[0] = 1.0

    # Wheels: <velocity kv=0.2> with a 6 Nm effort limit.
    for joint_name in config.WHEEL_JOINTS:
        actuator = spec.add_actuator(
            name=f"{prefix}/motor/{joint_name}",
            target=f"{prefix}/{joint_name}",
        )
        actuator.set_to_velocity(config.WHEEL_DAMPING)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.inheritrange = 0.0
        actuator.ctrllimited = False
        actuator.forcelimited = True
        actuator.forcerange = [
            -config.WHEEL_EFFORT_LIMIT,
            config.WHEEL_EFFORT_LIMIT,
        ]
        actuator.gear[0] = 1.0

    # Constant-force tendons (preloaded closed-loop springs).
    for tendon_name in ("Left_force", "Right_force"):
        actuator = spec.add_actuator(
            name=f"{prefix}/motor/{tendon_name}",
            target=f"{prefix}/{tendon_name}",
        )
        actuator.set_to_motor()
        actuator.trntype = mujoco.mjtTrn.mjTRN_TENDON
        actuator.ctrllimited = True
        actuator.ctrlrange = [
            -config.TENDON_EFFORT_LIMIT,
            config.TENDON_EFFORT_LIMIT,
        ]
        actuator.forcelimited = True
        actuator.forcerange = [
            -config.TENDON_EFFORT_LIMIT,
            config.TENDON_EFFORT_LIMIT,
        ]
        actuator.gear[0] = 1.0


def _style_geoms(spec: mujoco.MjSpec) -> None:
    """Apply per-copy materials and the training collision parameters."""
    for geom in spec.geoms:
        name = geom.name
        if name == "floor":
            continue

        prefix = name.split("/", 1)[0]
        if name.endswith("_visual"):
            geom.material = f"{prefix}_visual"
            geom.group = 2
            geom.contype = 0
            geom.conaffinity = 0
            continue

        # Every remaining robot geom is a collision geom.  The training
        # collision config uses contype=0 / conaffinity=1: robot-vs-floor
        # contacts are kept, robot-vs-robot contacts are filtered out.
        wheel = "wheel_ground_collision" in name
        geom.material = "collision_hidden"
        geom.group = 3
        geom.contype = 0
        geom.conaffinity = 1
        geom.solref = [0.01, 1.0]
        geom.condim = 6 if wheel else 1
        geom.priority = 1 if wheel else 0
        if wheel:
            # (sliding, torsional, rolling)
            geom.friction = [1.0, 5e-3, 5e-4]


def _runtime_for(model: mujoco.MjModel, key: str, prefix: str) -> RobotRuntime:
    def joint_id(name: str) -> int:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint not found in compiled model: {name}")
        return jid

    def actuator_id(name: str) -> int:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"Actuator not found in compiled model: {name}")
        return aid

    root_joint_name = f"{prefix}/base_floating_joint"
    root_joint = joint_id(root_joint_name)
    root_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/base_link"
    )
    if root_body < 0:
        raise RuntimeError(f"Body not found: {prefix}/base_link")

    gyro_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}/imu_ang_vel"
    )
    if gyro_id < 0:
        raise RuntimeError(f"Gyro sensor not found: {prefix}/imu_ang_vel")

    all_qpos_adr = tuple(
        int(model.jnt_qposadr[joint_id(f"{prefix}/{name}")])
        for name in config.JOINT_NAMES
    )
    hip_qpos_adr = tuple(
        int(model.jnt_qposadr[joint_id(f"{prefix}/{name}")])
        for name in config.HIP_JOINTS
    )
    obs_dof_adr = tuple(
        int(model.jnt_dofadr[joint_id(f"{prefix}/{name}")])
        for name in config.LEG_JOINTS_OBS
    )

    default_hip_qpos = np.array(
        [config.DEFAULT_JOINT_POS_BY_NAME[name] for name in config.HIP_JOINTS],
        dtype=np.float64,
    )

    return RobotRuntime(
        key=key,
        prefix=prefix,
        root_body_id=int(root_body),
        root_qpos_adr=int(model.jnt_qposadr[root_joint]),
        root_dof_adr=int(model.jnt_dofadr[root_joint]),
        all_qpos_adr=all_qpos_adr,
        hip_qpos_adr=hip_qpos_adr,
        obs_dof_adr=obs_dof_adr,
        gyro_adr=int(model.sensor_adr[gyro_id]),
        hip_ctrl_ids=tuple(
            actuator_id(f"{prefix}/motor/{name}") for name in config.HIP_JOINTS
        ),
        wheel_ctrl_ids=tuple(
            actuator_id(f"{prefix}/motor/{name}") for name in config.WHEEL_JOINTS
        ),
        tendon_ctrl_ids=tuple(
            actuator_id(f"{prefix}/motor/{name}")
            for name in ("Left_force", "Right_force")
        ),
        default_hip_qpos=default_hip_qpos,
    )


def build_dual_robot_model() -> DualRobotModel:
    """Compile the two-robot scene used by the comparison demo."""
    spec = mujoco.MjSpec()
    spec.modelname = "policy_compare"
    spec.option.timestep = config.PHYSICS_TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.impratio = 10.0
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.iterations = 20
    spec.option.tolerance = 1e-8
    spec.option.ls_iterations = 20
    spec.option.ls_tolerance = 0.01
    spec.option.ccd_iterations = 50

    _add_floor_and_lights(spec)
    _add_visual_materials(spec)

    for key in ("old", "new"):
        frame = spec.worldbody.add_frame()
        frame.name = f"{key}_mount"
        spec.attach(_make_robot_template(), prefix=f"{key}/", frame=frame)

    _style_geoms(spec)
    for key in ("old", "new"):
        _add_robot_actuators(spec, key)

    model = spec.compile()
    robots = {
        key: _runtime_for(model, key, key) for key in ("old", "new")
    }
    return DualRobotModel(model=model, robots=robots)