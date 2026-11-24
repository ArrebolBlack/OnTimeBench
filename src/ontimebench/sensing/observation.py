from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt

from ontimebench.sensing.camera import _metric_depth, _validate_rigid_transform

_DType = TypeVar("_DType", bound=np.generic)


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    view_names: tuple[str, ...]
    image_height_px: int = 224
    image_width_px: int = 224

    def __post_init__(self) -> None:
        if self.view_names not in (("agentview",), ("agentview", "wrist")):
            raise ValueError("view_names must match a frozen observation profile")
        if (
            isinstance(self.image_height_px, bool)
            or not isinstance(self.image_height_px, int)
            or isinstance(self.image_width_px, bool)
            or not isinstance(self.image_width_px, int)
            or (self.image_height_px, self.image_width_px) != (224, 224)
        ):
            raise ValueError("v1 observations require a 224x224 raster")


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    tick: int
    timestamp_s: float
    view_names: tuple[str, ...]
    rgb: npt.NDArray[np.uint8]
    # Rendered geom-surface camera z in metres; zero denotes no surface hit.
    depth_m: npt.NDArray[np.float32]
    camera_intrinsics: npt.NDArray[np.float64]
    world_from_camera_cv: npt.NDArray[np.float64]
    T_robot_base_from_world: npt.NDArray[np.float64]
    joint_position: npt.NDArray[np.float32]
    joint_velocity: npt.NDArray[np.float32]
    eef_position: npt.NDArray[np.float32]
    eef_quaternion_xyzw: npt.NDArray[np.float32]
    gripper_position: npt.NDArray[np.float32]
    gripper_velocity: npt.NDArray[np.float32]
    # Target pose in the robot base frame, read from the simulator. Set only when
    # the evaluation runs with target_pose_source="simulator"; None otherwise.
    target_position: npt.NDArray[np.float64] | None = None
    target_quaternion_xyzw: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.tick, bool)
            or not isinstance(self.tick, int)
            or self.tick < 0
            or isinstance(self.timestamp_s, bool)
            or not isinstance(self.timestamp_s, (int, float))
            or not math.isfinite(self.timestamp_s)
            or self.timestamp_s < 0
        ):
            raise ValueError("observation tick and timestamp must be non-negative")
        ObservationSpec(self.view_names)
        views = len(self.view_names)
        arrays = {
            "rgb": (self.rgb, np.dtype(np.uint8), (views, 224, 224, 3)),
            "depth_m": (self.depth_m, np.dtype(np.float32), (views, 224, 224)),
            "camera_intrinsics": (self.camera_intrinsics, np.dtype(np.float64), (views, 3, 3)),
            "world_from_camera_cv": (
                self.world_from_camera_cv,
                np.dtype(np.float64),
                (views, 4, 4),
            ),
            "T_robot_base_from_world": (self.T_robot_base_from_world, np.dtype(np.float64), (4, 4)),
            "joint_position": (self.joint_position, np.dtype(np.float32), (7,)),
            "joint_velocity": (self.joint_velocity, np.dtype(np.float32), (7,)),
            "eef_position": (self.eef_position, np.dtype(np.float32), (3,)),
            "eef_quaternion_xyzw": (self.eef_quaternion_xyzw, np.dtype(np.float32), (4,)),
            "gripper_position": (self.gripper_position, np.dtype(np.float32), (2,)),
            "gripper_velocity": (self.gripper_velocity, np.dtype(np.float32), (2,)),
        }
        for name, (array, dtype, shape) in arrays.items():
            if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape:
                raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
            if array.flags.writeable or not array.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous and read-only")
            if array.dtype != np.uint8 and not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
        if np.any(self.depth_m < 0):
            raise ValueError("depth_m must be non-negative")
        _validate_rigid_transform(self.world_from_camera_cv, "world_from_camera_cv")
        _validate_rigid_transform(self.T_robot_base_from_world, "T_robot_base_from_world")
        if not math.isclose(float(np.linalg.norm(self.eef_quaternion_xyzw)), 1.0, abs_tol=1e-5):
            raise ValueError("eef_quaternion_xyzw must be unit length")
        if (self.target_position is None) != (self.target_quaternion_xyzw is None):
            raise ValueError("target_position and target_quaternion_xyzw are set together")
        if self.target_position is not None and self.target_quaternion_xyzw is not None:
            for name, array, shape in (
                ("target_position", self.target_position, (3,)),
                ("target_quaternion_xyzw", self.target_quaternion_xyzw, (4,)),
            ):
                if (
                    not isinstance(array, np.ndarray)
                    or array.dtype != np.float64
                    or array.shape != shape
                    or array.flags.writeable
                    or not np.isfinite(array).all()
                ):
                    raise ValueError(f"{name} must be a finite read-only float64 array {shape}")
            if not math.isclose(
                float(np.linalg.norm(self.target_quaternion_xyzw)), 1.0, abs_tol=1e-5
            ):
                raise ValueError("target_quaternion_xyzw must be unit length")


def _array(
    source: Mapping[str, object], key: str, shape: tuple[int, ...]
) -> npt.NDArray[np.generic]:
    try:
        value = source[key]
    except KeyError:
        raise ValueError(f"native observation is missing {key}") from None
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"native observation {key} must have shape {shape}")
    if value.dtype != np.uint8 and not np.isfinite(value).all():
        raise ValueError(f"native observation {key} must be finite")
    return value


def _readonly(value: npt.ArrayLike, dtype: np.dtype[_DType]) -> npt.NDArray[_DType]:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _quaternion_xyzw_to_matrix(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("native EEF quaternion must be finite and non-zero")
    x, y, z, w = quaternion / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _matrix_to_quaternion_xyzw(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        result = np.array(
            (
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
                0.25 * scale,
            ),
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(value)))
        cyclic = ((1, 2), (2, 0), (0, 1))
        j, k = cyclic[index]
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + value[index, index] - value[j, j] - value[k, k]))
        result = np.empty(4, dtype=np.float64)
        result[index] = 0.25 * scale
        result[j] = (value[index, j] + value[j, index]) / scale
        result[k] = (value[index, k] + value[k, index]) / scale
        result[3] = (value[k, j] - value[j, k]) / scale
    return result / np.linalg.norm(result)


def _continuous_quaternion_xyzw(
    value: npt.NDArray[np.float64],
    *,
    tick: int,
    previous: npt.NDArray[np.float32] | None,
) -> npt.NDArray[np.float64]:
    if tick == 0:
        if previous is not None:
            raise ValueError("previous EEF quaternion must be cleared at episode reset")
        return -value if value[3] < 0.0 else value
    if previous is None:
        raise ValueError("previous EEF quaternion is required after tick zero")
    if (
        not isinstance(previous, np.ndarray)
        or previous.dtype != np.float32
        or previous.shape != (4,)
        or not np.isfinite(previous).all()
        or not math.isclose(float(np.linalg.norm(previous)), 1.0, abs_tol=1e-5)
    ):
        raise ValueError("previous EEF quaternion must be a finite unit float32 vector")
    return -value if float(np.dot(previous, value)) < 0.0 else value


def _build_benchmark_observation(
    raw_observation: Mapping[str, object],
    *,
    observation_spec: ObservationSpec,
    camera_sensor_names: tuple[str, ...],
    depth_extent: float,
    depth_znear: float,
    depth_zfar: float,
    camera_intrinsics: npt.NDArray[np.float64],
    world_from_camera_cv: npt.NDArray[np.float64],
    T_robot_base_from_world: npt.NDArray[np.float64],
    tick: int,
    timestamp_s: float,
    previous_eef_quaternion_xyzw: npt.NDArray[np.float32] | None = None,
) -> BenchmarkObservation:
    if camera_sensor_names not in (("agentview",), ("agentview", "robot0_eye_in_hand")):
        raise ValueError("camera sensor names must match the frozen Robosuite view mapping")
    if len(camera_sensor_names) != len(observation_spec.view_names):
        raise ValueError("camera sensor names do not match ObservationSpec")
    view_count = len(camera_sensor_names)
    rgb_frames: list[npt.NDArray[np.uint8]] = []
    depth_frames: list[npt.NDArray[np.float32]] = []
    for sensor_name in camera_sensor_names:
        rgb = _array(raw_observation, f"{sensor_name}_image", (224, 224, 3))
        if rgb.dtype != np.uint8:
            raise ValueError(f"native observation {sensor_name}_image must be uint8")
        normalized = _array(raw_observation, f"{sensor_name}_depth", (224, 224, 1))
        if normalized.dtype != np.float32:
            raise ValueError(f"native observation {sensor_name}_depth must be float32")
        # Pinned Robosuite supplies bottom-up OpenGL rasters. Both policy images
        # use the frozen top-left CV raster, consistently with the calibration.
        rgb_frames.append(cast(npt.NDArray[np.uint8], rgb[::-1]))
        depth_frames.append(
            _metric_depth(
                cast(npt.NDArray[np.float32], normalized)[::-1, :, 0],
                extent=depth_extent,
                znear=depth_znear,
                zfar=depth_zfar,
            )
        )

    joint_position = _array(raw_observation, "robot0_joint_pos", (7,))
    joint_velocity = _array(raw_observation, "robot0_joint_vel", (7,))
    eef_position_world = np.asarray(
        _array(raw_observation, "robot0_eef_pos", (3,)), dtype=np.float64
    )
    eef_quaternion_world = np.asarray(
        _array(raw_observation, "robot0_eef_quat_site", (4,)), dtype=np.float64
    )
    gripper_position = _array(raw_observation, "robot0_gripper_qpos", (2,))
    gripper_velocity = _array(raw_observation, "robot0_gripper_qvel", (2,))

    if camera_intrinsics.shape != (view_count, 3, 3):
        raise ValueError("camera_intrinsics do not match the observation views")
    if world_from_camera_cv.shape != (view_count, 4, 4):
        raise ValueError("world_from_camera_cv do not match the observation views")
    if T_robot_base_from_world.shape != (4, 4):
        raise ValueError("T_robot_base_from_world must have shape (4, 4)")
    if not np.isfinite(camera_intrinsics).all():
        raise ValueError("camera_intrinsics must be finite")
    _validate_rigid_transform(world_from_camera_cv, "world_from_camera_cv")
    _validate_rigid_transform(T_robot_base_from_world, "T_robot_base_from_world")

    eef_homogeneous = np.concatenate((eef_position_world, np.ones(1, dtype=np.float64)))
    eef_position_base = (T_robot_base_from_world @ eef_homogeneous)[:3]
    eef_rotation_base = T_robot_base_from_world[:3, :3] @ _quaternion_xyzw_to_matrix(
        eef_quaternion_world
    )
    eef_quaternion_base = _continuous_quaternion_xyzw(
        _matrix_to_quaternion_xyzw(eef_rotation_base),
        tick=tick,
        previous=previous_eef_quaternion_xyzw,
    )

    return BenchmarkObservation(
        tick=tick,
        timestamp_s=timestamp_s,
        view_names=observation_spec.view_names,
        rgb=_readonly(np.stack(rgb_frames), np.dtype(np.uint8)),
        depth_m=_readonly(np.stack(depth_frames), np.dtype(np.float32)),
        camera_intrinsics=_readonly(camera_intrinsics, np.dtype(np.float64)),
        world_from_camera_cv=_readonly(world_from_camera_cv, np.dtype(np.float64)),
        T_robot_base_from_world=_readonly(T_robot_base_from_world, np.dtype(np.float64)),
        joint_position=_readonly(joint_position, np.dtype(np.float32)),
        joint_velocity=_readonly(joint_velocity, np.dtype(np.float32)),
        eef_position=_readonly(eef_position_base, np.dtype(np.float32)),
        eef_quaternion_xyzw=_readonly(eef_quaternion_base, np.dtype(np.float32)),
        gripper_position=_readonly(gripper_position, np.dtype(np.float32)),
        gripper_velocity=_readonly(gripper_velocity, np.dtype(np.float32)),
    )


__all__ = ["BenchmarkObservation", "ObservationSpec"]
