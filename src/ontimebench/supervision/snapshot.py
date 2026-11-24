from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ontimebench.sensing.camera import _validate_rigid_transform
from ontimebench.sensing.observation import BenchmarkObservation


@dataclass(frozen=True, slots=True)
class SupervisionSnapshot:
    tick: int
    timestamp_s: float
    view_names: tuple[str, ...]
    depth_m: npt.NDArray[np.float32]
    geom_id: npt.NDArray[np.int32]
    camera_intrinsics: npt.NDArray[np.float64]
    world_from_camera_cv: npt.NDArray[np.float64]
    world_from_geom: npt.NDArray[np.float64]
    target_object_pose_world: npt.NDArray[np.float64]
    geom_semantic_identity: tuple[str, ...]

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
            raise ValueError("snapshot tick and timestamp must be non-negative")
        if self.view_names not in (("agentview",), ("agentview", "wrist")):
            raise ValueError("invalid snapshot view_names")
        views = len(self.view_names)
        if not isinstance(self.geom_semantic_identity, tuple) or not all(
            isinstance(identity, str) and identity for identity in self.geom_semantic_identity
        ):
            raise ValueError("geom_semantic_identity must contain non-empty strings")
        if not self.geom_semantic_identity:
            raise ValueError("geom_semantic_identity must contain at least one compiled geom")
        geoms = len(self.geom_semantic_identity)
        contracts = {
            "depth_m": (self.depth_m, np.dtype(np.float32), (views, 224, 224)),
            "geom_id": (self.geom_id, np.dtype(np.int32), (views, 224, 224)),
            "camera_intrinsics": (self.camera_intrinsics, np.dtype(np.float64), (views, 3, 3)),
            "world_from_camera_cv": (
                self.world_from_camera_cv,
                np.dtype(np.float64),
                (views, 4, 4),
            ),
            "world_from_geom": (self.world_from_geom, np.dtype(np.float64), (geoms, 4, 4)),
            "target_object_pose_world": (self.target_object_pose_world, np.dtype(np.float64), (7,)),
        }
        for name, (array, dtype, shape) in contracts.items():
            if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape:
                raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
            if (
                array.flags.writeable
                or not array.flags.c_contiguous
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"{name} must be C-contiguous, read-only, and finite")
        if np.any(self.depth_m < 0):
            raise ValueError("depth_m must be non-negative")
        if np.any(self.geom_id < -1) or np.any(self.geom_id >= geoms):
            raise ValueError("geom_id must be background -1 or index geom_semantic_identity")
        _validate_rigid_transform(self.world_from_camera_cv, "world_from_camera_cv")
        _validate_rigid_transform(self.world_from_geom, "world_from_geom")
        if not math.isclose(
            float(np.linalg.norm(self.target_object_pose_world[3:])), 1.0, abs_tol=1e-8
        ):
            raise ValueError("target object quaternion must be unit length")


def _world_from_compiled_geoms(
    model: mujoco.MjModel, data: mujoco.MjData
) -> npt.NDArray[np.float64]:
    result = np.repeat(np.eye(4, dtype=np.float64)[None, ...], model.ngeom, axis=0)
    result[:, :3, :3] = np.asarray(data.geom_xmat, dtype=np.float64).reshape((-1, 3, 3))
    result[:, :3, 3] = np.asarray(data.geom_xpos, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("compiled geom transforms must be finite")
    _validate_rigid_transform(result, "world_from_geom")
    result.setflags(write=False)
    return result


def _cv_geom_segmentation(
    element_segmentation_opengl: npt.NDArray[np.int32],
    *,
    views: int,
    geoms: int,
) -> npt.NDArray[np.int32]:
    if (
        not isinstance(element_segmentation_opengl, np.ndarray)
        or element_segmentation_opengl.dtype != np.int32
        or element_segmentation_opengl.shape != (views, 224, 224, 2)
    ):
        raise ValueError(
            "element_segmentation_opengl must be int32 with shape (views, 224, 224, 2)"
        )
    cv_segmentation = element_segmentation_opengl[:, ::-1, :, :]
    is_geom = cv_segmentation[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
    raw_geom_id = cv_segmentation[..., 1]
    if np.any(is_geom & ((raw_geom_id < 0) | (raw_geom_id >= geoms))):
        raise ValueError("geom segmentation contains an invalid compiled geom id")
    result = np.full((views, 224, 224), -1, dtype=np.int32)
    result[is_geom] = raw_geom_id[is_geom]
    result.setflags(write=False)
    return result


def _assemble_supervision_snapshot(
    observation: BenchmarkObservation,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    element_segmentation_opengl: npt.NDArray[np.int32],
    target_pose_world: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    geom_semantic_identity: tuple[str, ...],
    previous_target_quaternion_xyzw: npt.NDArray[np.float64] | None,
) -> SupervisionSnapshot:
    if not isinstance(observation, BenchmarkObservation):
        raise ValueError("observation must be a BenchmarkObservation")
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise ValueError("model and data must be MuJoCo MjModel and MjData")
    if data.model is not model:
        raise ValueError("MjData must belong to the supplied MjModel")
    if not math.isclose(float(data.time), observation.timestamp_s, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("observation and MuJoCo data must describe the same timestamp")
    if not isinstance(geom_semantic_identity, tuple) or len(geom_semantic_identity) != model.ngeom:
        raise ValueError("geom_semantic_identity must contain one entry per compiled geom")
    geom_id = _cv_geom_segmentation(
        element_segmentation_opengl,
        views=len(observation.view_names),
        geoms=model.ngeom,
    )
    position, rotation = target_pose_world
    quaternion = Rotation.from_matrix(rotation).as_quat(canonical=True)
    if (
        previous_target_quaternion_xyzw is not None
        and previous_target_quaternion_xyzw @ quaternion < 0
    ):
        quaternion = -quaternion
    target_pose = np.concatenate((position, quaternion))
    target_pose.setflags(write=False)
    return SupervisionSnapshot(
        tick=observation.tick,
        timestamp_s=observation.timestamp_s,
        view_names=observation.view_names,
        depth_m=observation.depth_m,
        geom_id=geom_id,
        camera_intrinsics=observation.camera_intrinsics,
        world_from_camera_cv=observation.world_from_camera_cv,
        world_from_geom=_world_from_compiled_geoms(model, data),
        target_object_pose_world=target_pose,
        geom_semantic_identity=geom_semantic_identity,
    )


__all__ = ["SupervisionSnapshot"]
