"""Closed-loop simulation of two policies in one MuJoCo world."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import config
from .model import DualRobotModel, RobotRuntime
from .policy import OnnxActor


def _move_towards(current: float, target: float, max_delta: float) -> float:
    """Step ``current`` toward ``target`` by at most ``max_delta``."""
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + np.copysign(max_delta, delta)


class CommandDelay:
    """Fixed-size command delay, applied per physics step.

    With ``steps=2`` a command pushed at time t is written to the actuators at
    time t+2; until enough commands have accumulated the initial command is
    held.
    """

    def __init__(self, steps: int, initial: np.ndarray) -> None:
        self.steps = max(0, int(steps))
        self._queue: deque[np.ndarray] = deque(maxlen=self.steps + 1)
        self.reset(initial)

    def reset(self, initial: np.ndarray) -> None:
        self._queue.clear()
        for _ in range(self.steps + 1):
            self._queue.append(np.array(initial, dtype=np.float64, copy=True))

    def push(self, command: np.ndarray) -> np.ndarray:
        self._queue.append(np.array(command, dtype=np.float64, copy=True))
        return self._queue[0].copy()


@dataclass
class PolicyRobot:
    runtime: RobotRuntime
    policy: OnnxActor
    last_action: np.ndarray = field(
        default_factory=lambda: np.zeros(config.ACTION_DIM, dtype=np.float64)
    )
    target: np.ndarray = field(
        default_factory=lambda: np.zeros(config.ACTION_DIM, dtype=np.float64)
    )
    delay: CommandDelay | None = None
    tilt_deg: float = 0.0
    fell: bool = False


@dataclass(frozen=True)
class RobotMetrics:
    root_z: float
    lin_vel_x: float
    lin_vel_y: float
    ang_vel_z: float
    tilt_deg: float
    fell: bool


class ComparisonSim:
    """Both policies see the same model, command and physics timestep."""

    def __init__(
        self,
        dual_model: DualRobotModel,
        policies: dict[str, OnnxActor],
        *,
        delay_steps: int = config.DEFAULT_DELAY_STEPS,
        tendon_force: float | None = None,
        randomize_tendon_force: bool = True,
        seed: int = 0,
    ) -> None:
        missing = {"old", "new"} - set(policies)
        if missing:
            raise ValueError(f"Missing policies for: {sorted(missing)}")

        self.dual = dual_model
        self.model = dual_model.model
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.command = config.DEFAULT_COMMAND.copy()
        self._fixed_tendon_force = tendon_force
        self.randomize_tendon_force = randomize_tendon_force
        self.tendon_forces = np.full(
            2, config.DEFAULT_TENDON_FORCE, dtype=np.float64
        )
        self.time = 0.0
        self.steps = 0

        self.robots: dict[str, PolicyRobot] = {}
        for key in ("old", "new"):
            runtime = dual_model.robots[key]
            robot = PolicyRobot(runtime=runtime, policy=policies[key])
            robot.delay = CommandDelay(delay_steps, self._nominal_target(runtime))
            self.robots[key] = robot

        self.reset()

    # ------------------------------------------------------------------
    # State setup
    # ------------------------------------------------------------------

    def _nominal_target(self, runtime: RobotRuntime) -> np.ndarray:
        target = np.zeros(config.ACTION_DIM, dtype=np.float64)
        target[: len(config.HIP_JOINTS)] = runtime.default_hip_qpos
        target[len(config.HIP_JOINTS) :] = 0.0
        return target

    def _sample_tendon_forces(self) -> np.ndarray:
        if self._fixed_tendon_force is not None:
            return np.full(2, float(self._fixed_tendon_force), dtype=np.float64)
        if not self.randomize_tendon_force:
            return np.full(2, config.DEFAULT_TENDON_FORCE, dtype=np.float64)
        low, high = config.TENDON_FORCE_RANGE
        # Training sampled the left and right tendon preloads independently;
        # both robots receive the same pair for a fair comparison.
        return self.rng.uniform(low, high, size=2)

    def reset(self) -> None:
        """Reset both robots to the exact same initial state."""
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.data.qpos

        for robot in self.robots.values():
            rt = robot.runtime
            qpos[rt.root_qpos_adr : rt.root_qpos_adr + 3] = config.INIT_ROOT_POS
            qpos[rt.root_qpos_adr + 3 : rt.root_qpos_adr + 7] = config.INIT_ROOT_QUAT
            for adr, value in zip(rt.all_qpos_adr, config.DEFAULT_JOINT_POS):
                qpos[adr] = value

            robot.last_action[:] = 0.0
            robot.target = self._nominal_target(rt)
            assert robot.delay is not None
            robot.delay.reset(robot.target)
            robot.tilt_deg = 0.0
            robot.fell = False

        self.tendon_forces = self._sample_tendon_forces()
        self.data.ctrl[:] = 0.0
        for robot in self.robots.values():
            self.data.ctrl[list(robot.runtime.tendon_ctrl_ids)] = self.tendon_forces

        self.time = 0.0
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        self._update_fall_state()

    # ------------------------------------------------------------------
    # Observations / actions
    # ------------------------------------------------------------------

    def _observation(self, robot: PolicyRobot) -> np.ndarray:
        rt = robot.runtime
        data = self.data
        gyr = data.sensordata[rt.gyro_adr : rt.gyro_adr + 3]

        rot = data.xmat[rt.root_body_id].reshape(3, 3)
        # Gravity direction (0, 0, -1) expressed in the base frame.
        projected_gravity = -rot[2, :]

        joint_pos = data.qpos[list(rt.hip_qpos_adr)] - rt.default_hip_qpos
        joint_vel = data.qvel[list(rt.obs_dof_adr)] * config.JOINT_VEL_SCALE

        return np.concatenate(
            [
                gyr * config.ANG_VEL_SCALE,          # 3
                projected_gravity,                    # 3
                joint_pos,                            # 4
                joint_vel,                            # 6
                robot.last_action,                    # 6
                self.command * config.COMMAND_SCALE,  # 3
            ]
        ).astype(np.float32)

    def _target_from_action(
        self, runtime: RobotRuntime, action: np.ndarray
    ) -> np.ndarray:
        target = np.zeros(config.ACTION_DIM, dtype=np.float64)

        hip_actions = action[: len(config.HIP_JOINTS)]
        for i, joint_name in enumerate(config.HIP_JOINTS):
            target_qpos = (
                runtime.default_hip_qpos[i]
                + config.HIP_ACTION_SCALE * hip_actions[i]
            )
            low, high = config.HIP_TARGET_CLIP[joint_name]
            target[i] = np.clip(target_qpos, low, high)

        wheel_actions = action[len(config.HIP_JOINTS) :]
        target[len(config.HIP_JOINTS) :] = np.clip(
            config.WHEEL_ACTION_SCALE * wheel_actions,
            config.WHEEL_TARGET_CLIP[0],
            config.WHEEL_TARGET_CLIP[1],
        )
        return target

    def _write_ctrl(self) -> None:
        ctrl = self.data.ctrl
        for robot in self.robots.values():
            rt = robot.runtime
            assert robot.delay is not None
            delayed = robot.delay.push(robot.target)
            ctrl[list(rt.hip_ctrl_ids)] = delayed[: len(config.HIP_JOINTS)]
            ctrl[list(rt.wheel_ctrl_ids)] = delayed[len(config.HIP_JOINTS) :]
            ctrl[list(rt.tendon_ctrl_ids)] = self.tendon_forces

    def policy_step(self) -> dict[str, np.ndarray]:
        """Run one 50 Hz policy step (4 physics steps at 200 Hz)."""
        observations = {
            key: self._observation(robot) for key, robot in self.robots.items()
        }
        actions = {
            key: robot.policy(observations[key])
            for key, robot in self.robots.items()
        }

        for key, robot in self.robots.items():
            robot.last_action = actions[key]
            robot.target = self._target_from_action(robot.runtime, actions[key])

        for _ in range(config.DECIMATION):
            self._write_ctrl()
            mujoco.mj_step(self.model, self.data)

        self.time += config.POLICY_DT
        self.steps += 1
        self._update_fall_state()
        return actions

    # ------------------------------------------------------------------
    # Metrics / commands
    # ------------------------------------------------------------------

    def _update_fall_state(self) -> None:
        for robot in self.robots.values():
            rot = self.data.xmat[robot.runtime.root_body_id].reshape(3, 3)
            # Angle between the body's z axis and world z.
            cos_tilt = float(np.clip(rot[2, 2], -1.0, 1.0))
            robot.tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
            robot.fell = robot.tilt_deg > config.FALL_TILT_DEG

    def metrics(self, key: str) -> RobotMetrics:
        robot = self.robots[key]
        rt = robot.runtime
        rot = self.data.xmat[rt.root_body_id].reshape(3, 3)
        vel_world = self.data.qvel[rt.root_dof_adr : rt.root_dof_adr + 3]
        vel_body = rot.T @ vel_world
        ang_body = self.data.sensordata[rt.gyro_adr : rt.gyro_adr + 3]
        return RobotMetrics(
            root_z=float(self.data.qpos[rt.root_qpos_adr + 2]),
            lin_vel_x=float(vel_body[0]),
            lin_vel_y=float(vel_body[1]),
            ang_vel_z=float(ang_body[2]),
            tilt_deg=robot.tilt_deg,
            fell=robot.fell,
        )

    def set_command(self, command: np.ndarray | list[float]) -> None:
        cmd = np.asarray(command, dtype=np.float64).reshape(3)
        self.command = np.clip(
            cmd, config.COMMAND_LIMITS[:, 0], config.COMMAND_LIMITS[:, 1]
        )

    def adjust_command(
        self, dvx: float = 0.0, dwz: float = 0.0, dheight: float = 0.0
    ) -> None:
        self.set_command(
            self.command + np.array([dvx, dwz, dheight], dtype=np.float64)
        )

    def slew_planar_command(
        self,
        target_vx: float,
        target_wz: float,
        vx_rate: float,
        wz_rate: float,
        dt: float,
    ) -> None:
        """Move vx/wz toward targets at bounded rates (joystick-like).

        The height command is left untouched.
        """
        command = self.command.copy()
        command[0] = _move_towards(command[0], target_vx, vx_rate * dt)
        command[1] = _move_towards(command[1], target_wz, wz_rate * dt)
        self.set_command(command)

    def set_tendon_force(self, force: float | None) -> None:
        """Set the preload now; ``None`` restores reset-time randomization."""
        self._fixed_tendon_force = force
        if force is not None:
            self.tendon_forces = np.full(2, float(force), dtype=np.float64)
            for robot in self.robots.values():
                self.data.ctrl[list(robot.runtime.tendon_ctrl_ids)] = (
                    self.tendon_forces
                )

    @property
    def any_fell(self) -> bool:
        return any(robot.fell for robot in self.robots.values())