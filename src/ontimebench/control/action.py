from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

ActionMode: TypeAlias = Literal["E7", "J8"]
ActionRepresentation: TypeAlias = Literal["delta", "absolute"]


@dataclass(frozen=True, slots=True)
class ActionSpec:
    mode: ActionMode
    representation: ActionRepresentation
    motion_limits: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.mode not in ("E7", "J8") or self.representation not in ("delta", "absolute"):
            raise ValueError("invalid ActionSpec mode or representation")
        expected = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5) if self.mode == "E7" else (0.05,) * 7
        if self.motion_limits != expected and not (
            self.mode == "E7"
            and self.representation == "delta"
            and self.motion_limits == (0.10, 0.10, 0.10, 0.5, 0.5, 0.5)
        ):
            raise ValueError(
                f"{self.mode} motion_limits must equal the default profile {expected} "
                "or the explicit E7 delta 0.10 m execution limits"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkAction:
    mode: ActionMode
    representation: ActionRepresentation
    value: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.mode not in ("E7", "J8"):
            raise ValueError("BenchmarkAction.mode must be E7 or J8")
        if self.representation not in ("delta", "absolute"):
            raise ValueError("BenchmarkAction.representation must be delta or absolute")
        expected = 7 if self.mode == "E7" else 8
        if not isinstance(self.value, np.ndarray):
            raise ValueError("BenchmarkAction.value must be a NumPy array")
        if self.value.dtype != np.float64 or self.value.shape != (expected,):
            raise ValueError(f"{self.mode} action must be float64[{expected}]")
        if not self.value.flags.c_contiguous or self.value.flags.writeable:
            raise ValueError("BenchmarkAction.value must be contiguous and read-only")
        if not np.isfinite(self.value).all():
            raise ValueError("BenchmarkAction.value must be finite")
        if float(self.value[-1]) not in {-1.0, 0.0, 1.0}:
            raise ValueError("gripper must be exactly -1, 0, or 1")
        if self.mode == "E7" and self.representation == "absolute":
            if float(np.linalg.norm(self.value[3:6])) > math.pi + 1e-6:
                raise ValueError("absolute E7 rotation must be a principal rotation vector")


def _validate_execution_translation_limit(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0.05, 0.10):
        raise ValueError("execution_translation_limit_m must be 0.05 or 0.10")


def validate_action_spec(action: BenchmarkAction, spec: ActionSpec) -> None:
    """Check a physical command against the episode's action specification."""
    if (action.mode, action.representation) != (spec.mode, spec.representation):
        raise ValueError("BenchmarkAction does not match the episode ActionSpec")
    if action.representation == "delta":
        limits = np.asarray(spec.motion_limits, dtype=np.float32).astype(np.float64)
        if np.any(np.abs(action.value[:-1]) > limits):
            raise ValueError("delta Action exceeds the configured execution-delta limits")


def _skew(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    x, y, z = value
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


def _so3_exp(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    vector = _state_vector(value, 3, "rotation vector")
    angle = float(np.linalg.norm(vector))
    generator = _skew(vector)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64) + generator + 0.5 * generator @ generator
    sine_scale = math.sin(angle) / angle
    cosine_scale = (1.0 - math.cos(angle)) / (angle * angle)
    squared = np.matmul(generator, generator)
    return np.asarray(
        np.eye(3, dtype=np.float64) + sine_scale * generator + cosine_scale * squared,
        dtype=np.float64,
    )


def _matrix_to_quaternion_xyzw(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            diagonal_term = 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            scale = 2.0 * math.sqrt(max(0.0, diagonal_term))
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            diagonal_term = 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            scale = 2.0 * math.sqrt(max(0.0, diagonal_term))
            quaternion = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            diagonal_term = 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            scale = 2.0 * math.sqrt(max(0.0, diagonal_term))
            quaternion = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_xyzw_to_principal_rotvec(
    value: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    quaternion = _state_vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError("quaternion must have non-zero norm")
    quaternion = quaternion / norm
    xyz = quaternion[:3]
    w = float(quaternion[3])
    first_nonzero = next((float(component) for component in xyz if component != 0.0), 0.0)
    if w < 0.0 or (w == 0.0 and first_nonzero < 0.0):
        quaternion = -quaternion
        xyz = quaternion[:3]
        w = float(quaternion[3])
    sine_half = float(np.linalg.norm(xyz))
    if sine_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(sine_half, min(1.0, max(0.0, w)))
    return xyz * (angle / sine_half)


def _so3_log(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    rotation = _state_rotation(value, "rotation")
    return _quaternion_xyzw_to_principal_rotvec(_matrix_to_quaternion_xyzw(rotation))


def _state_vector(value: object, dimension: int, name: str) -> npt.NDArray[np.float64]:
    if not isinstance(value, np.ndarray) or value.shape != (dimension,):
        raise ValueError(f"{name} must be a NumPy vector with shape ({dimension},)")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _state_rotation(value: object, name: str) -> npt.NDArray[np.float64]:
    if not isinstance(value, np.ndarray) or value.shape != (3, 3):
        raise ValueError(f"{name} must be a NumPy rotation matrix with shape (3, 3)")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(result.T @ result, np.eye(3), rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} must be orthogonal")
    if not math.isclose(float(np.linalg.det(result)), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{name} determinant must be +1")
    return result


def _controller_action(
    action: BenchmarkAction,
    spec: ActionSpec,
    *,
    achieved_eef_position: npt.NDArray[np.float64] | None,
    achieved_eef_rotation: npt.NDArray[np.float64] | None,
    achieved_joint_position: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64]:
    validate_action_spec(action, spec)
    configured = np.asarray(action.value, dtype=np.float64)
    if spec.mode == "E7":
        if spec.representation == "delta":
            execution_delta = configured[:6]
        else:
            if achieved_eef_position is None or achieved_eef_rotation is None:
                raise ValueError("absolute E7 requires achieved EEF position and rotation")
            achieved_position = _state_vector(achieved_eef_position, 3, "achieved EEF position")
            achieved_rotation = _state_rotation(achieved_eef_rotation, "achieved EEF rotation")
            goal_rotation = _so3_exp(configured[3:6])
            execution_delta = np.concatenate(
                (configured[:3] - achieved_position, _so3_log(goal_rotation @ achieved_rotation.T))
            )
    elif spec.representation == "delta":
        execution_delta = configured[:7]
    else:
        if achieved_joint_position is None:
            raise ValueError("absolute J8 requires achieved joint position")
        achieved_joints = _state_vector(achieved_joint_position, 7, "achieved joint position")
        execution_delta = configured[:7] - achieved_joints

    limits = np.asarray(spec.motion_limits, dtype=np.float32).astype(np.float64)
    if np.any(np.abs(execution_delta) > limits):
        raise ValueError("Action exceeds the configured execution-delta limits at applied tick")
    # Keep the existing float32 execution bounds; E7 transmission now retains
    # the nominal physical scale and the original normalized float32 boundary.
    scales = np.asarray(spec.motion_limits, dtype=np.float64) if spec.mode == "E7" else limits
    controller = np.concatenate((execution_delta / scales, configured[-1:]))
    checked_motion = controller[:-1].astype(np.float32) if spec.mode == "E7" else controller[:-1]
    if np.any(np.abs(checked_motion) > 1.0):
        raise ValueError("normalized controller motion must remain within [-1, 1]")
    return np.ascontiguousarray(controller, dtype=np.float64)


class _ActionDelayQueue:
    def __init__(self, action_spec: ActionSpec, delay_steps: int) -> None:
        if not isinstance(action_spec, ActionSpec):
            raise ValueError("action_spec must be an ActionSpec")
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, int) or delay_steps < 0:
            raise ValueError("delay_steps must be a non-negative integer")
        self._action_spec = action_spec
        self._delay_steps = delay_steps
        self._pending: deque[tuple[int, BenchmarkAction]] = deque()

    def reset(self) -> None:
        self._pending.clear()

    def issue(
        self,
        action: BenchmarkAction,
        *,
        issued_tick: int,
        achieved_eef_position: npt.NDArray[np.float64] | None = None,
        achieved_eef_rotation: npt.NDArray[np.float64] | None = None,
        achieved_joint_position: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], tuple[int, BenchmarkAction] | None]:
        if not isinstance(action, BenchmarkAction):
            raise ValueError("action must be a BenchmarkAction")
        if isinstance(issued_tick, bool) or not isinstance(issued_tick, int) or issued_tick < 0:
            raise ValueError("issued_tick must be a non-negative integer")
        validate_action_spec(action, self._action_spec)
        self._pending.append((issued_tick, action))
        if len(self._pending) <= self._delay_steps:
            dimension = 7 if self._action_spec.mode == "E7" else 8
            return np.zeros(dimension, dtype=np.float64), None
        applied_issued_tick, applied = self._pending.popleft()
        return (
            _controller_action(
                applied,
                self._action_spec,
                achieved_eef_position=achieved_eef_position,
                achieved_eef_rotation=achieved_eef_rotation,
                achieved_joint_position=achieved_joint_position,
            ),
            (applied_issued_tick, applied),
        )

    def advance(
        self,
        action: BenchmarkAction | None,
        *,
        tick: int,
        achieved_eef_position: npt.NDArray[np.float64] | None = None,
        achieved_eef_rotation: npt.NDArray[np.float64] | None = None,
        achieved_joint_position: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64] | None, tuple[int, BenchmarkAction] | None]:
        """Advance physical time even when no new command was issued."""
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a non-negative integer")
        if action is not None:
            validate_action_spec(action, self._action_spec)
            self._pending.append((tick, action))
        if not self._pending or self._pending[0][0] + self._delay_steps > tick:
            return None, None
        issued_tick, applied = self._pending.popleft()
        return _controller_action(
            applied,
            self._action_spec,
            achieved_eef_position=achieved_eef_position,
            achieved_eef_rotation=achieved_eef_rotation,
            achieved_joint_position=achieved_joint_position,
        ), (issued_tick, applied)


__all__ = ["ActionMode", "ActionSpec", "BenchmarkAction", "validate_action_spec"]
