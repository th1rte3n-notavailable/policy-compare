"""ONNX actor wrapper and metadata checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import config

EXPECTED_OBSERVATION_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "last_actions",
    "command",
)


@dataclass(frozen=True)
class PolicyMetadata:
    joint_names: tuple[str, ...] | None
    observation_names: tuple[str, ...] | None
    default_joint_pos: np.ndarray | None
    run_path: str | None


def _split_csv(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_metadata(raw: dict[str, str]) -> PolicyMetadata:
    default_pos = None
    if raw.get("default_joint_pos"):
        default_pos = np.array(
            [float(v) for v in raw["default_joint_pos"].split(",")],
            dtype=np.float64,
        )
    return PolicyMetadata(
        joint_names=_split_csv(raw.get("joint_names")),
        observation_names=_split_csv(raw.get("observation_names")),
        default_joint_pos=default_pos,
        run_path=raw.get("run_path"),
    )


class OnnxActor:
    """Single-output ONNX actor: ``obs [1, 25] -> actions [1, 6]``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"ONNX policy not found: {self.path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"{self.path.name}: expected one input and one output, got "
                f"{len(inputs)} and {len(outputs)}"
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        in_shape = inputs[0].shape
        out_shape = outputs[0].shape
        if isinstance(in_shape[-1], int) and in_shape[-1] != config.ACTOR_OBS_DIM:
            raise ValueError(
                f"{self.path.name}: actor input is {in_shape}, expected last "
                f"dim {config.ACTOR_OBS_DIM}"
            )
        if isinstance(out_shape[-1], int) and out_shape[-1] != config.ACTION_DIM:
            raise ValueError(
                f"{self.path.name}: actor output is {out_shape}, expected last "
                f"dim {config.ACTION_DIM}"
            )

        raw_metadata = self.session.get_modelmeta().custom_metadata_map
        self.metadata = _parse_metadata(raw_metadata)
        self._check_metadata()

    def _check_metadata(self) -> None:
        joint_names = self.metadata.joint_names
        if joint_names is not None and joint_names != config.JOINT_NAMES:
            print(
                f"[warn] {self.path.name}: ONNX joint order does not match the "
                "MJCF joint order. The demo assumes the training order."
            )
        obs_names = self.metadata.observation_names
        if obs_names is not None and obs_names != EXPECTED_OBSERVATION_NAMES:
            print(
                f"[warn] {self.path.name}: unexpected observation terms "
                f"{obs_names}; expected {EXPECTED_OBSERVATION_NAMES}."
            )
        default_pos = self.metadata.default_joint_pos
        if default_pos is not None:
            if default_pos.shape != config.DEFAULT_JOINT_POS.shape:
                print(
                    f"[warn] {self.path.name}: default_joint_pos has "
                    f"{default_pos.shape}, expected "
                    f"{config.DEFAULT_JOINT_POS.shape}."
                )
            elif not np.allclose(
                default_pos, config.DEFAULT_JOINT_POS, atol=2e-3
            ):
                print(
                    f"[warn] {self.path.name}: default_joint_pos does not "
                    "match the MJCF/default pose used by this demo."
                )

    @property
    def name(self) -> str:
        return self.path.parent.name

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(
            1, config.ACTOR_OBS_DIM
        )
        output = self.session.run([self.output_name], {self.input_name: obs})[0]
        return np.asarray(output, dtype=np.float64).reshape(config.ACTION_DIM)