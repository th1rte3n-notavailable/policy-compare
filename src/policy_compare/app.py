"""Run the old (translucent) and new (solid) policies side by side."""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import config
from .keyboard import KeyboardInput
from .model import DualRobotModel, build_dual_robot_model
from .policy import OnnxActor
from .simulation import ComparisonSim, RobotMetrics
from .viewer import GlfwViewer

# Joystick-like command shaping: holding a key ramps toward the training
# limit, releasing it ramps back to zero.  Full scale is reached in 0.35 s.
VX_MAX = float(config.COMMAND_LIMITS[0, 1])
WZ_MAX = float(config.COMMAND_LIMITS[1, 1])
RAMP_TIME = 0.35
VX_RAMP_RATE = VX_MAX / RAMP_TIME
WZ_RAMP_RATE = WZ_MAX / RAMP_TIME
HEIGHT_RATE = 0.30  # m/s while Q/E is held


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="policy-compare",
        description=(
            "Simulate two trained policies on the wheeled-legged hero robot "
            "in the same MuJoCo scene, with identical commands and initial "
            "states. The old policy is translucent, the new one is solid."
        ),
    )
    parser.add_argument(
        "--old",
        default=config.DEFAULT_OLD_RUN,
        help="Old policy: .onnx path, run directory, or run name (default: %(default)s)",
    )
    parser.add_argument(
        "--new",
        default=config.DEFAULT_NEW_RUN,
        help="New policy: .onnx path, run directory, or run name (default: %(default)s)",
    )
    parser.add_argument(
        "--old-label",
        default=None,
        help="Display label for the old policy (default: run directory name)",
    )
    parser.add_argument(
        "--new-label",
        default=None,
        help="Display label for the new policy (default: run directory name)",
    )
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        metavar=("VX", "WZ", "HEIGHT"),
        default=tuple(float(v) for v in config.DEFAULT_COMMAND),
        help="Initial command: forward velocity (m/s), yaw rate (rad/s), base height (m)",
    )
    parser.add_argument(
        "--delay-steps",
        type=int,
        default=config.DEFAULT_DELAY_STEPS,
        help="Actuator command delay in physics steps (default: %(default)s)",
    )
    parser.add_argument(
        "--tendon-force",
        type=float,
        default=None,
        help=(
            "Fixed tendon preload in N. By default a value is sampled from "
            "[90, 150] N on each reset, as during training."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for preload")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Exit after this many seconds of simulation (0 = until closed)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a viewer (runs 10 s by default, or --duration)",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable keyboard command input",
    )
    parser.add_argument(
        "--auto-reset",
        action="store_true",
        help="Reset both robots whenever either one falls over",
    )
    parser.add_argument(
        "--follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Camera follows the robots (default: on; toggle with F in the "
            "viewer)"
        ),
    )
    return parser.parse_args(argv)


def _load_policies(args: argparse.Namespace) -> tuple[dict[str, OnnxActor], dict[str, str]]:
    old_path = config.resolve_policy_file(args.old)
    new_path = config.resolve_policy_file(args.new)
    if old_path == new_path:
        print("[warn] --old and --new resolve to the same ONNX file.")

    policies = {
        "old": OnnxActor(old_path),
        "new": OnnxActor(new_path),
    }
    labels = {
        "old": args.old_label or old_path.parent.name,
        "new": args.new_label or new_path.parent.name,
    }
    print(f"[policy] old: {old_path}")
    print(f"[policy] new: {new_path}")
    return policies, labels


def _metrics_text(metrics: RobotMetrics) -> str:
    state = "FELL" if metrics.fell else "ok"
    return (
        f"z={metrics.root_z:5.3f} m  vx={metrics.lin_vel_x:+5.2f} m/s  "
        f"wz={metrics.ang_vel_z:+5.2f} rad/s  tilt={metrics.tilt_deg:5.1f} deg  {state}"
    )


def _format_status(
    sim: ComparisonSim,
    labels: dict[str, str],
    paused: bool,
    follow: bool,
) -> str:
    cmd = sim.command
    old_metrics = sim.metrics("old")
    new_metrics = sim.metrics("new")
    running = "PAUSED" if paused else "running"
    return "\n".join(
        [
            f"cmd: vx={cmd[0]:+.2f} m/s  yaw={cmd[1]:+.2f} rad/s  height={cmd[2]:.3f} m",
            f"OLD  {labels['old']}  (translucent blue)",
            f"     {_metrics_text(old_metrics)}",
            f"NEW  {labels['new']}  (solid orange)",
            f"     {_metrics_text(new_metrics)}",
            f"t={sim.time:6.2f}s  tendon L/R="
            f"{sim.tendon_forces[0]:5.1f}/{sim.tendon_forces[1]:5.1f} N  "
            f"{running}  follow={'on' if follow else 'off'}",
        ]
    )


HELP_TEXT = "\n".join(
    [
        "W/S: vx (hold=ramp, release=center)   A/D: yaw (same)",
        "Q/E: height +/- (fast)   X/Space: stop vx/yaw   C: center command",
        "R: reset episode   P: pause   F: follow camera   Esc: quit",
    ]
)


def _handle_input(
    sim: ComparisonSim,
    keyboard: KeyboardInput | None,
    dt: float,
    state: dict[str, bool],
) -> None:
    if keyboard is None:
        return

    for event in keyboard.consume_events():
        if event in ("space", "x"):
            # Emergency stop: zero planar velocity immediately.  If a
            # directional key is still held, the normal ramp resumes.
            sim.set_command([0.0, 0.0, sim.command[2]])
        elif event == "c":
            sim.set_command(config.DEFAULT_COMMAND)
        elif event == "r":
            sim.reset()
        elif event == "p":
            state["paused"] = not state["paused"]
        elif event == "f":
            state["follow"] = not state["follow"]
        elif event == "escape":
            state["running"] = False

    # Virtual joystick: W/S and A/D set the target; with no key the target is
    # zero, so the command decelerates back to rest automatically.
    target_vx = 0.0
    if keyboard.is_pressed("w"):
        target_vx += VX_MAX
    if keyboard.is_pressed("s"):
        target_vx -= VX_MAX

    target_wz = 0.0
    if keyboard.is_pressed("a"):
        target_wz += WZ_MAX
    if keyboard.is_pressed("d"):
        target_wz -= WZ_MAX

    sim.slew_planar_command(
        target_vx,
        target_wz,
        VX_RAMP_RATE,
        WZ_RAMP_RATE,
        dt,
    )

    # Height command is persistent; Q/E adjusts it at a faster rate.
    if keyboard.is_pressed("q"):
        sim.adjust_command(dheight=HEIGHT_RATE * dt)
    if keyboard.is_pressed("e"):
        sim.adjust_command(dheight=-HEIGHT_RATE * dt)


def _robot_contact_pairs(sim: ComparisonSim) -> list[tuple[str, str]]:
    """Cross-robot geom contacts. Should always be empty by construction."""
    model = sim.model
    pairs: list[tuple[str, str]] = []
    for contact in sim.data.contact[: sim.data.ncon]:
        name1 = model.geom(contact.geom1).name
        name2 = model.geom(contact.geom2).name
        key1 = name1.split("/", 1)[0] if "/" in name1 else None
        key2 = name2.split("/", 1)[0] if "/" in name2 else None
        if key1 in ("old", "new") and key2 in ("old", "new") and key1 != key2:
            pairs.append((name1, name2))
    return pairs


def _camera_midpoint(sim: ComparisonSim) -> np.ndarray:
    old = sim.data.qpos[
        sim.robots["old"].runtime.root_qpos_adr :
        sim.robots["old"].runtime.root_qpos_adr + 3
    ]
    new = sim.data.qpos[
        sim.robots["new"].runtime.root_qpos_adr :
        sim.robots["new"].runtime.root_qpos_adr + 3
    ]
    midpoint = 0.5 * (old + new)
    midpoint[2] += 0.10
    return midpoint


def _run_headless(
    sim: ComparisonSim,
    labels: dict[str, str],
    duration: float,
    auto_reset: bool = False,
) -> int:
    if duration <= 0.0:
        duration = 10.0
    steps = max(1, int(round(duration / config.POLICY_DT)))
    print(
        f"[headless] running {duration:.1f} s ({steps} policy steps, "
        f"{1.0 / config.POLICY_DT:.0f} Hz)"
    )
    print(f"[headless] old: {labels['old']}")
    print(f"[headless] new: {labels['new']}")

    for i in range(steps):
        sim.policy_step()
        if not np.isfinite(sim.data.qpos).all():
            print(f"[error] non-finite state at t={sim.time:.3f}s")
            return 1
        if auto_reset and sim.any_fell:
            print(f"[headless] reset at t={sim.time:.2f}s (a robot fell)")
            sim.reset()
        if (i + 1) % int(round(1.0 / config.POLICY_DT)) == 0:
            print(
                f"t={sim.time:6.2f}s  "
                f"cmd=({sim.command[0]:+.2f},{sim.command[1]:+.2f},{sim.command[2]:.3f})  "
                f"old[{_metrics_text(sim.metrics('old'))}]  "
                f"new[{_metrics_text(sim.metrics('new'))}]"
            )

    cross = _robot_contact_pairs(sim)
    if cross:
        print(f"[warn] unexpected cross-robot contacts: {cross[:3]}")
    else:
        print("[check] no cross-robot contacts (robots are collision-filtered)")
    print(f"[done] simulated {sim.time:.2f}s of physics")
    return 0


def _run_viewer(
    sim: ComparisonSim,
    labels: dict[str, str],
    args: argparse.Namespace,
) -> int:
    keyboard = None if args.no_keyboard else KeyboardInput()

    try:
        viewer = GlfwViewer(sim.model, sim.data, keyboard)
    except Exception as exc:
        print(f"[error] could not open the viewer: {exc}")
        print("        Use --headless to run without a window.")
        return 1

    state = {
        "paused": False,
        "follow": bool(args.follow),
        "running": True,
    }
    status_text = _format_status(sim, labels, False, state["follow"])

    last_wall = time.perf_counter()
    last_overlay = 0.0

    try:
        while state["running"] and viewer.is_running():
            now = time.perf_counter()
            dt = min(0.1, now - last_wall)
            last_wall = now

            viewer.poll()
            _handle_input(sim, keyboard, dt, state)
            if not state["running"]:
                break

            if not state["paused"]:
                sim.policy_step()
                if args.auto_reset and sim.any_fell:
                    sim.reset()
                if args.duration > 0.0 and sim.time >= args.duration:
                    break

            if state["follow"]:
                viewer.set_lookat(_camera_midpoint(sim))

            if now - last_overlay >= 0.1:
                status_text = _format_status(
                    sim, labels, state["paused"], state["follow"]
                )
                last_overlay = now

            viewer.render(status_text, HELP_TEXT)

            elapsed = time.perf_counter() - now
            if elapsed < config.POLICY_DT:
                time.sleep(config.POLICY_DT - elapsed)
    finally:
        viewer.close()

    cross = _robot_contact_pairs(sim)
    if cross:
        print(f"[warn] unexpected cross-robot contacts: {cross[:3]}")
    print(f"[done] simulated {sim.time:.2f}s of physics")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policies, labels = _load_policies(args)

    dual: DualRobotModel = build_dual_robot_model()
    sim = ComparisonSim(
        dual,
        policies,
        delay_steps=max(0, args.delay_steps),
        tendon_force=args.tendon_force,
        seed=args.seed,
    )
    sim.set_command(args.command)
    start_pos = ", ".join(f"{value:.5f}" for value in config.INIT_ROOT_POS)
    print(
        f"[sim] both robots start at ({start_pos}); robot-robot collisions "
        "are disabled by the training collision mask"
    )
    start_contacts = _robot_contact_pairs(sim)
    if start_contacts:
        print(f"[warn] cross-robot contacts at reset: {start_contacts[:3]}")
    else:
        print("[check] robots overlap at t=0 with no robot-robot contacts")

    if args.headless:
        return _run_headless(sim, labels, args.duration, args.auto_reset)
    return _run_viewer(sim, labels, args)


if __name__ == "__main__":
    raise SystemExit(main())