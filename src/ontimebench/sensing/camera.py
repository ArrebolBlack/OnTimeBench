"""Metric depth and camera calibration."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt


def _validate_rigid_transform(value: npt.NDArray[np.float64], name: str) -> None:
    if value.ndim < 2 or value.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must contain 4x4 transforms")
    matrices = value.reshape((-1, 4, 4))
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(matrices[:, 3, :], expected_last_row, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} transform last row must be [0, 0, 0, 1]")
    rotations = matrices[:, :3, :3]
    gram = np.swapaxes(rotations, 1, 2) @ rotations
    if not np.allclose(gram, np.eye(3, dtype=np.float64), rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} rotation must be orthogonal")
    if not np.allclose(np.linalg.det(rotations), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} rotation determinant must be +1")


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> npt.NDArray[np.float64]:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must be a NumPy array with shape {shape}")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _rotation(value: object, name: str) -> npt.NDArray[np.float64]:
    result = _finite_array(value, (3, 3), name)
    if not np.allclose(result.T @ result, np.eye(3), rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} must be orthogonal")
    if not np.isclose(np.linalg.det(result), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} determinant must be +1")
    return result


def _metric_depth(
    normalized_depth: npt.NDArray[np.float32],
    *,
    extent: float,
    znear: float,
    zfar: float,
) -> npt.NDArray[np.float32]:
    if not isinstance(normalized_depth, np.ndarray) or normalized_depth.dtype != np.float32:
        raise ValueError("normalized depth must be a float32 NumPy array")
    if (
        not np.isfinite(normalized_depth).all()
        or np.any(normalized_depth < 0.0)
        or np.any(normalized_depth > 1.0)
    ):
        raise ValueError("normalized depth must be finite and in [0, 1]")
    parameters = (extent, znear, zfar)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value <= 0
        for value in parameters
    ):
        raise ValueError("depth extent, znear, and zfar must be finite positive numbers")
    near_m = float(extent) * float(znear)
    far_m = float(extent) * float(zfar)
    if near_m >= far_m:
        raise ValueError("compiled depth near plane must be closer than far plane")
    result = near_m / (1.0 - normalized_depth * (1.0 - near_m / far_m))
    return np.ascontiguousarray(result, dtype=np.float32)


def _camera_intrinsics(
    vertical_fov_deg: float, image_height_px: int, image_width_px: int
) -> npt.NDArray[np.float64]:
    if (
        isinstance(vertical_fov_deg, bool)
        or not isinstance(vertical_fov_deg, (int, float))
        or not np.isfinite(vertical_fov_deg)
        or not 0.0 < vertical_fov_deg < 180.0
    ):
        raise ValueError("vertical_fov_deg must be finite and in (0, 180)")
    if (
        isinstance(image_height_px, bool)
        or not isinstance(image_height_px, int)
        or isinstance(image_width_px, bool)
        or not isinstance(image_width_px, int)
        or (image_height_px, image_width_px) != (224, 224)
    ):
        raise ValueError("v1 camera intrinsics require a 224x224 raster")
    focal = 0.5 * image_height_px / np.tan(np.deg2rad(vertical_fov_deg) / 2.0)
    return np.array(
        (
            (focal, 0.0, (image_width_px - 1) / 2.0),
            (0.0, focal, (image_height_px - 1) / 2.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _world_from_camera_cv(
    camera_position_world_m: npt.NDArray[np.float64],
    camera_rotation_world: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    position = _finite_array(camera_position_world_m, (3,), "camera position")
    rotation = _rotation(camera_rotation_world, "camera rotation")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation @ np.diag((1.0, -1.0, -1.0))
    transform[:3, 3] = position
    _validate_rigid_transform(transform, "world_from_camera_cv")
    return transform


def _robot_base_from_world(
    base_position_world_m: npt.NDArray[np.float64],
    base_rotation_world: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    position = _finite_array(base_position_world_m, (3,), "robot base position")
    rotation = _rotation(base_rotation_world, "robot base rotation")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ position)
    _validate_rigid_transform(result, "T_robot_base_from_world")
    return result


@dataclass(frozen=True, slots=True)
class _CompiledCameraCalibration:
    camera_names: tuple[str, ...]
    depth_extent: float
    depth_znear: float
    depth_zfar: float
    camera_intrinsics: npt.NDArray[np.float64]
    world_from_camera_cv: npt.NDArray[np.float64]
    T_robot_base_from_world: npt.NDArray[np.float64]


def _ordered_names(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or not all(
            isinstance(item, str) and item.strip() and item == item.strip() for item in value
        )
    ):
        raise ValueError(f"{name} must be a non-empty tuple of trimmed names")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return value


def _named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"compiled model does not contain {name!r}")
    return object_id


def _compiled_camera_calibration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_names: tuple[str, ...],
    robot_base_body_name: str,
    image_height_px: int,
    image_width_px: int,
) -> _CompiledCameraCalibration:
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise ValueError("model and data must be MuJoCo MjModel and MjData")
    if data.model is not model:
        raise ValueError("MjData must belong to the supplied MjModel")
    names = _ordered_names(camera_names, "camera_names")
    if (
        not isinstance(robot_base_body_name, str)
        or not robot_base_body_name.strip()
        or robot_base_body_name != robot_base_body_name.strip()
    ):
        raise ValueError("robot_base_body_name must be a non-empty trimmed name")

    extent = float(model.stat.extent)
    znear = float(model.vis.map.znear)
    zfar = float(model.vis.map.zfar)
    if not all(np.isfinite(value) and value > 0.0 for value in (extent, znear, zfar)):
        raise ValueError("compiled depth extent, znear, and zfar must be finite and positive")
    if extent * znear >= extent * zfar:
        raise ValueError("compiled depth near plane must be closer than far plane")

    intrinsics: list[npt.NDArray[np.float64]] = []
    extrinsics: list[npt.NDArray[np.float64]] = []
    perspective = int(mujoco.mjtProjection.mjPROJ_PERSPECTIVE)
    for camera_name in names:
        camera_id = _named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if int(model.cam_projection[camera_id]) != perspective:
            raise ValueError(f"camera {camera_name!r} must use perspective projection")
        intrinsics.append(
            _camera_intrinsics(
                float(model.cam_fovy[camera_id]),
                image_height_px,
                image_width_px,
            )
        )
        extrinsics.append(
            _world_from_camera_cv(
                np.asarray(data.cam_xpos[camera_id], dtype=np.float64),
                np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape((3, 3)),
            )
        )

    base_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, robot_base_body_name)
    base_transform = _robot_base_from_world(
        np.asarray(data.xpos[base_id], dtype=np.float64),
        np.asarray(data.xmat[base_id], dtype=np.float64).reshape((3, 3)),
    )
    camera_intrinsics = np.stack(intrinsics)
    world_from_camera_cv = np.stack(extrinsics)
    camera_intrinsics.setflags(write=False)
    world_from_camera_cv.setflags(write=False)
    base_transform.setflags(write=False)
    return _CompiledCameraCalibration(
        camera_names=names,
        depth_extent=extent,
        depth_znear=znear,
        depth_zfar=zfar,
        camera_intrinsics=camera_intrinsics,
        world_from_camera_cv=world_from_camera_cv,
        T_robot_base_from_world=base_transform,
    )


__all__: list[str] = []
