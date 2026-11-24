from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import mujoco
import numpy as np
import robosuite.macros as robosuite_macros
from robosuite.controllers import controller_factory, load_composite_controller_config
from robosuite.controllers.composite.composite_controller import CompositeController
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.controllers.parts.generic.joint_pos import JointPositionController
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.robots.fixed_base_robot import FixedBaseRobot
from robosuite.utils.observables import sensor

from ontimebench._resources import resource_path
from ontimebench.control.action import (
    ActionSpec,
    BenchmarkAction,
    _ActionDelayQueue,
    _validate_execution_translation_limit,
    validate_action_spec,
)
from ontimebench.scenarios.spec import JointPhysicsSpec, ScenarioSpec
from ontimebench.sensing.camera import _compiled_camera_calibration
from ontimebench.sensing.observation import (
    BenchmarkObservation,
    ObservationSpec,
    _build_benchmark_observation,
)
from ontimebench.tasks.registry import load_yaml_strict
from ontimebench.tasks.spec import TaskSpec

_RawStageFact: TypeAlias = tuple[str, bool, float, int]
_RecordingFrameFacts: TypeAlias = tuple[
    bool,
    str | None,
    float | None,
    tuple[_RawStageFact, ...],
]
_RecordingGeomFact: TypeAlias = tuple[int, str, str, str, str]


def _reference_collision_masks(
    contype: np.ndarray[Any, Any],
    conaffinity: np.ndarray[Any, Any],
    robot_geoms: frozenset[int],
    target_geoms: frozenset[int],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Preserve native bit compatibility except for the robot--target cross product."""
    if robot_geoms & target_geoms:
        raise ValueError("reference robot and target subtrees must be disjoint")
    if np.any(contype < 0) or np.any(conaffinity < 0):
        raise ValueError("reference collision masks must be nonnegative")
    bits = max(1, int(max(contype.max(), conaffinity.max())).bit_length())
    if bits * 3 > 31:
        raise ValueError("reference collision masks exceed three signed-int bit banks")
    kinds = np.full(len(contype), 2, dtype=np.int32)  # robot, target, everything else
    kinds[list(robot_geoms)] = 0
    kinds[list(target_geoms)] = 1
    types = contype.copy()
    affinities = np.zeros_like(conaffinity)
    for kind in range(3):
        selected = kinds == kind
        types[selected] <<= kind * bits
        for other in range(3):
            if {kind, other} != {0, 1}:
                affinities[selected] |= conaffinity[selected] << (other * bits)
    return types, affinities


@dataclass(frozen=True, slots=True)
class _CompiledJointPhysics:
    spec: JointPhysicsSpec
    dof_address: int
    dof_count: int


def _resolve_joint_physics(
    model: mujoco.MjModel,
    specs: Sequence[JointPhysicsSpec],
    *,
    entity_body_ids: Mapping[str, int],
) -> tuple[_CompiledJointPhysics, ...]:
    dof_counts = {
        int(mujoco.mjtJoint.mjJNT_FREE): 6,
        int(mujoco.mjtJoint.mjJNT_BALL): 3,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    resolved: list[_CompiledJointPhysics] = []
    for spec in specs:
        body_id = entity_body_ids.get(spec.entity_id)
        if body_id is None:
            raise ValueError(f"joint physics entity does not resolve: {spec.entity_id!r}")
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name))
        if joint_id < 0:
            raise ValueError(f"joint physics element does not resolve: {spec.joint_name!r}")
        if int(model.jnt_bodyid[joint_id]) != body_id:
            raise ValueError("joint physics element does not belong to its declared entity")
        joint_type = int(model.jnt_type[joint_id])
        dof_count = dof_counts.get(joint_type)
        if dof_count is None:
            raise ValueError(f"joint physics uses an unsupported MuJoCo joint type: {joint_type}")
        if len(spec.damping_per_dof) != dof_count or len(spec.friction_loss_per_dof) != dof_count:
            raise ValueError(
                f"joint physics for {spec.joint_name!r} must contain exactly "
                f"{dof_count} per-DOF values"
            )
        resolved.append(
            _CompiledJointPhysics(
                spec=spec,
                dof_address=int(model.jnt_dofadr[joint_id]),
                dof_count=dof_count,
            )
        )
    return tuple(resolved)


def _apply_joint_physics(model: mujoco.MjModel, bindings: Sequence[_CompiledJointPhysics]) -> None:
    for binding in bindings:
        stop = binding.dof_address + binding.dof_count
        model.dof_damping[binding.dof_address : stop] = binding.spec.damping_per_dof
        model.dof_frictionloss[binding.dof_address : stop] = binding.spec.friction_loss_per_dof


def _validate_compiled_joint_physics(
    model: mujoco.MjModel,
    bindings: Sequence[_CompiledJointPhysics],
    *,
    owner: str,
) -> None:
    for binding in bindings:
        stop = binding.dof_address + binding.dof_count
        damping = tuple(float(value) for value in model.dof_damping[binding.dof_address : stop])
        friction = tuple(
            float(value) for value in model.dof_frictionloss[binding.dof_address : stop]
        )
        if damping != binding.spec.damping_per_dof or (
            friction != binding.spec.friction_loss_per_dof
        ):
            raise ValueError(f"compiled {owner} joint physics differs from ScenarioSpec")


def _incomplete_progress(value: float) -> float:
    """Clamp one active stage below the exact completed value."""
    progress = float(value)
    if not math.isfinite(progress):
        raise RuntimeError("stage progress must be finite")
    return min(max(progress, 0.0), math.nextafter(1.0, 0.0))


def _require_distance_stage_begin(
    start_distance: float | None, best_distance: float | None, *, owner: str
) -> None:
    if start_distance is None or best_distance is None:
        raise RuntimeError(f"{owner} requires begin() at scored t0")


def _distance_stage_progress(
    start_distance: float | None,
    best_distance: float | None,
    success_distance: float,
    *,
    completed: bool,
) -> float:
    if not math.isfinite(success_distance) or success_distance < 0.0:
        raise RuntimeError("stage success distance must be finite and non-negative")
    if (start_distance is None) != (best_distance is None):
        raise RuntimeError("stage distance progress requires both start and best distances")
    if start_distance is not None and best_distance is not None:
        if (
            not math.isfinite(start_distance)
            or not math.isfinite(best_distance)
            or start_distance < 0.0
            or best_distance < 0.0
            or best_distance > start_distance
        ):
            raise RuntimeError(
                "stage start and best distances must be finite, non-negative and monotonic"
            )
    if completed:
        return 1.0
    if start_distance is None or best_distance is None or start_distance <= success_distance:
        return 0.0
    return _incomplete_progress(
        (start_distance - best_distance) / (start_distance - success_distance)
    )


def _duration_stage_progress(current_steps: int, required_steps: int, *, completed: bool) -> float:
    if isinstance(current_steps, bool) or not isinstance(current_steps, int) or current_steps < 0:
        raise RuntimeError("stage duration count must be a non-negative integer")
    if (
        isinstance(required_steps, bool)
        or not isinstance(required_steps, int)
        or required_steps <= 0
    ):
        raise RuntimeError("stage duration must require a positive physics-step count")
    if completed:
        return 1.0
    return _incomplete_progress(current_steps / required_steps)


def _ordered_stage_facts(
    candidates: Sequence[_RawStageFact],
) -> tuple[_RawStageFact, ...]:
    """Expose a completed prefix and at most one active outcome-owned stage."""
    result: list[_RawStageFact] = []
    reached_active = False
    names: set[str] = set()
    for name, completed, progress, consecutive_steps in candidates:
        if not isinstance(name, str) or not name or name in names:
            raise RuntimeError("stage names must be non-empty and unique")
        names.add(name)
        if not isinstance(completed, bool):
            raise RuntimeError("stage completion must be boolean")
        if (
            isinstance(consecutive_steps, bool)
            or not isinstance(consecutive_steps, int)
            or consecutive_steps < 0
        ):
            raise RuntimeError("stage consecutive count must be a non-negative integer")
        if not math.isfinite(float(progress)):
            raise RuntimeError("stage progress must be finite")
        if reached_active:
            if completed:
                raise RuntimeError("a completed stage cannot follow an incomplete prerequisite")
            result.append((name, False, 0.0, 0))
            continue
        if completed:
            result.append((name, True, 1.0, consecutive_steps))
            continue
        result.append((name, False, _incomplete_progress(progress), consecutive_steps))
        reached_active = True
    return tuple(result)


def _collision_bottom_z(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids: tuple[int, ...]
) -> float:
    bottoms = []
    for geom_id in geom_ids:
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        size = model.geom_size[geom_id]
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            extent = float(np.dot(np.abs(rotation[2]), size))
        elif model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = float(
                size[0] * math.hypot(rotation[2, 0], rotation[2, 1]) + size[1] * abs(rotation[2, 2])
            )
        elif model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = float(size[0] + size[1] * abs(rotation[2, 2]))
        elif model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = float(size[0])
        else:
            raise ValueError(
                "capture target collision geoms must be frozen boxes, cylinders, "
                "capsules or spheres"
            )
        bottoms.append(float(data.geom_xpos[geom_id, 2]) - extent)
    if not bottoms:
        raise ValueError("capture target requires collision geometry")
    return min(bottoms)


_BAD_PHYSICS_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADCTRL,
)


def _physics_state_is_valid(data: mujoco.MjData) -> bool:
    return all(
        np.isfinite(value).all()
        for value in (data.qpos, data.qvel, data.qacc, data.ctrl, data.efc_force)
    ) and not any(data.warning[warning].number for warning in _BAD_PHYSICS_WARNINGS)


def _fingerpad_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_geom_ids: tuple[int, ...],
    left_pad_ids: frozenset[int],
    right_pad_ids: frozenset[int],
    *,
    contact_indices: tuple[int, ...] | None = None,
) -> tuple[bool, bool, bool]:
    left_normal = np.zeros(3, dtype=np.float64)
    right_normal = np.zeros(3, dtype=np.float64)
    left = right = False
    force = np.empty(6, dtype=np.float64)
    for index in range(data.ncon) if contact_indices is None else contact_indices:
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in target_geom_ids:
            other, direction = geom2, -1.0
        elif geom2 in target_geom_ids:
            other, direction = geom1, 1.0
        else:
            continue
        if other not in left_pad_ids and other not in right_pad_ids:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        if not np.isfinite(force).all():
            raise RuntimeError("invalid MuJoCo contact force")
        normal = direction * force[0] * contact.frame[:3]
        if other in left_pad_ids:
            left = True
            left_normal += normal
        else:
            right = True
            right_normal += normal
    return left, right, bool(np.dot(left_normal, right_normal) < 0.0)


@dataclass(frozen=True, slots=True)
class _IdentitySymmetry:
    pass


@dataclass(frozen=True, slots=True)
class _FiniteSymmetry:
    quaternions_xyzw: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True, slots=True)
class _AxialSymmetry:
    axis_local: tuple[float, float, float]
    bidirectional: bool


@dataclass(frozen=True, slots=True)
class _SphericalSymmetry:
    pass


_Symmetry: TypeAlias = _IdentitySymmetry | _FiniteSymmetry | _AxialSymmetry | _SphericalSymmetry


@dataclass(frozen=True, slots=True)
class _AssetRecord:
    asset_id: str
    kind: Literal["arena", "object", "fixture"]
    path: Path
    functional_family: str
    physical_identity: str
    visual_equivalence_group: str
    symmetry: _Symmetry
    grasp_class: Literal["none", "parallel_jaw", "handle", "enclosure"]


_CATALOG_FIELDS = {"schema_version", "assets"}
_ASSET_FIELDS = {
    "asset_id",
    "kind",
    "path",
    "functional_family",
    "physical_identity",
    "visual_equivalence_group",
    "symmetry",
    "grasp_class",
}


def _catalog_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _catalog_exact(value: Mapping[str, object], fields: set[str], name: str) -> None:
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} unknown fields: {sorted(unknown)}")


def _catalog_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _catalog_vector(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} finite numbers")
    result: list[float] = []
    for component in value:
        if (
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(component)
        ):
            raise ValueError(f"{name} must contain {length} finite numbers")
        result.append(float(component))
    return tuple(result)


def _parse_symmetry(value: object) -> _Symmetry:
    symmetry = _catalog_mapping(value, "asset symmetry")
    symmetry_type = symmetry.get("type")
    if symmetry_type == "identity":
        _catalog_exact(symmetry, {"type"}, "identity symmetry")
        return _IdentitySymmetry()
    if symmetry_type == "spherical":
        _catalog_exact(symmetry, {"type"}, "spherical symmetry")
        return _SphericalSymmetry()
    if symmetry_type == "axial":
        _catalog_exact(symmetry, {"type", "axis_local", "bidirectional"}, "axial symmetry")
        axis = _catalog_vector(symmetry["axis_local"], 3, "axis_local")
        if not math.isclose(math.sqrt(sum(component**2 for component in axis)), 1.0, abs_tol=1e-6):
            raise ValueError("axial symmetry axis_local must be unit length")
        bidirectional = symmetry["bidirectional"]
        if not isinstance(bidirectional, bool):
            raise ValueError("axial symmetry bidirectional must be bool")
        return _AxialSymmetry(cast(tuple[float, float, float], axis), bidirectional)
    if symmetry_type == "finite":
        _catalog_exact(symmetry, {"type", "quaternions_xyzw"}, "finite symmetry")
        raw_quaternions = symmetry["quaternions_xyzw"]
        if not isinstance(raw_quaternions, Sequence) or isinstance(raw_quaternions, (str, bytes)):
            raise ValueError("finite symmetry quaternions_xyzw must be a sequence")
        quaternions = tuple(
            _catalog_vector(item, 4, "finite symmetry quaternion") for item in raw_quaternions
        )
        if not quaternions:
            raise ValueError("finite symmetry must contain at least the identity quaternion")
        if any(
            not math.isclose(
                math.sqrt(sum(component**2 for component in quaternion)),
                1.0,
                abs_tol=1e-6,
            )
            for quaternion in quaternions
        ):
            raise ValueError("finite symmetry quaternions must be unit length")
        if not any(abs(quaternion[3]) >= 1.0 - 1e-6 for quaternion in quaternions):
            raise ValueError("finite symmetry must contain the identity quaternion")
        canonical = {
            tuple(-component for component in quaternion)
            if next((component for component in reversed(quaternion) if component != 0.0), 0.0)
            < 0.0
            else quaternion
            for quaternion in quaternions
        }
        if len(canonical) != len(quaternions):
            raise ValueError("finite symmetry contains duplicate equivalent quaternions")
        return _FiniteSymmetry(cast(tuple[tuple[float, float, float, float], ...], quaternions))
    raise ValueError(f"unknown asset symmetry type: {symmetry_type}")


def _parse_asset(value: object, *, repository_root: Path) -> _AssetRecord:
    asset = _catalog_mapping(value, "asset record")
    _catalog_exact(asset, _ASSET_FIELDS, "asset record")
    asset_id = _catalog_text(asset["asset_id"], "asset_id")
    kind = asset["kind"]
    if kind not in ("arena", "object", "fixture"):
        raise ValueError("asset kind must be arena, object, or fixture")
    expected_prefix = {"arena": "arena.", "object": "obj.", "fixture": "fixture."}[kind]
    if not asset_id.startswith(expected_prefix):
        raise ValueError(f"{kind} asset_id must start with {expected_prefix}")
    path_text = _catalog_text(asset["path"], "asset path")
    posix_path = PurePosixPath(path_text)
    if (
        posix_path.is_absolute()
        or not posix_path.parts
        or posix_path.parts[0] != "assets"
        or ".." in posix_path.parts
        or path_text != posix_path.as_posix()
    ):
        raise ValueError("asset path must be a normalized repository-relative path under assets/")
    resolved_path = (repository_root / Path(*posix_path.parts)).resolve()
    assets_root = (repository_root / "assets").resolve()
    if not resolved_path.is_relative_to(assets_root) or not resolved_path.is_file():
        raise ValueError(f"asset path does not resolve to a file under assets/: {path_text}")
    grasp_class = asset["grasp_class"]
    if grasp_class not in ("none", "parallel_jaw", "handle", "enclosure"):
        raise ValueError("invalid asset grasp_class")
    return _AssetRecord(
        asset_id=asset_id,
        kind=kind,
        path=resolved_path,
        functional_family=_catalog_text(asset["functional_family"], "functional_family"),
        physical_identity=_catalog_text(asset["physical_identity"], "physical_identity"),
        visual_equivalence_group=_catalog_text(
            asset["visual_equivalence_group"], "visual_equivalence_group"
        ),
        symmetry=_parse_symmetry(asset["symmetry"]),
        grasp_class=grasp_class,
    )


def _parse_asset_catalog(source: Path) -> dict[str, _AssetRecord]:
    if not source.is_file():
        raise FileNotFoundError(source)
    catalog = _catalog_mapping(load_yaml_strict(source), "asset catalog")
    _catalog_exact(catalog, _CATALOG_FIELDS, "asset catalog")
    version = catalog["schema_version"]
    if isinstance(version, bool) or version != 1:
        raise ValueError("asset catalog requires schema_version 1")
    values = catalog["assets"]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("asset catalog assets must be a sequence")
    records = tuple(_parse_asset(value, repository_root=source.parent.parent) for value in values)
    if not records:
        raise ValueError("asset catalog must contain at least one asset")
    result = {record.asset_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("asset catalog contains duplicate asset_id values")
    return result


@lru_cache(maxsize=1)
def _load_asset_catalog() -> Mapping[str, _AssetRecord]:
    return MappingProxyType(_parse_asset_catalog(resource_path("assets", "catalog.yaml")))


_ROBOSUITE_VERSION = "1.5.2"
_MUJOCO_VERSION = "3.11.0"


def _validate_runtime_versions(robosuite_version: object, mujoco_version: object) -> None:
    if robosuite_version != _ROBOSUITE_VERSION or mujoco_version != _MUJOCO_VERSION:
        raise RuntimeError("OnTimeBench requires exactly robosuite==1.5.2 and mujoco==3.11.0")


def _require_exact_runtime_versions() -> None:
    _validate_runtime_versions(version("robosuite"), mujoco.__version__)


def _panda_controller_config(action_mode: object) -> dict[str, Any]:
    _require_exact_runtime_versions()
    if action_mode not in ("E7", "J8"):
        raise ValueError(f"unsupported Benchmark action mode: {action_mode}")
    loaded = load_composite_controller_config(robot="Panda")
    if not isinstance(loaded, dict) or set(loaded) != {"type", "body_parts"}:
        raise RuntimeError("Robosuite Panda composite controller config is not flattened v1.5.2")
    if loaded["type"] != "BASIC":
        raise RuntimeError("Robosuite Panda requires the BASIC composite controller")
    body_parts = loaded["body_parts"]
    if not isinstance(body_parts, dict) or set(body_parts) != {"right"}:
        raise RuntimeError("Robosuite Panda config must contain one flattened right arm")
    right = body_parts["right"]
    if not isinstance(right, dict) or right.get("type") != "OSC_POSE":
        raise RuntimeError("Robosuite Panda default right arm must use OSC_POSE")
    if right.get("gripper") != {"type": "GRIP"}:
        raise RuntimeError("Robosuite Panda right arm must contain the standard GRIP config")

    config = cast(dict[str, Any], deepcopy(loaded))
    if action_mode == "E7":
        # The operational-space controller forms its goal directly in world coordinates. The public
        # Benchmark action remains in robot-base coordinates at the bridge.
        config["body_parts"]["right"]["input_ref_frame"] = "world"
        return config
    gripper = deepcopy(right["gripper"])
    config["body_parts"]["right"] = {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": 300,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "input_type": "delta",
        "gripper": gripper,
    }
    return config


class _MassMatrixUpdateMixin:
    def update(self: Any, force: bool = False) -> None:
        if self.new_update or force:
            # OnTimeBench refreshes kinematics and qM before every controller update.
            self.sim.forward()

            if self.ref_name is not None:
                self.update_reference_data()

            self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
            self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])

            mass_matrix: np.ndarray[Any, np.dtype[np.float64]] = np.ndarray(
                shape=(self.sim.model.nv, self.sim.model.nv),
                dtype=np.float64,
                order="C",
            )
            mujoco.mj_fullM(
                self.sim.model._model,
                self.sim.data._data,
                mass_matrix,
            )
            mass_matrix = np.reshape(
                mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel))
            )
            self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]
            self.new_update = False


class _E7OperationalSpaceController(
    _MassMatrixUpdateMixin,
    OperationalSpaceController,  # type: ignore[misc]
):
    def update_reference_data(self: Any) -> None:
        name = self.ref_name
        site = self.sim.model.site_name2id(name)
        self.ref_pos = np.array(self.sim.data.site_xpos[site])
        self.ref_ori_mat = np.array(self.sim.data.site_xmat[site].reshape((3, 3)))
        self.ref_pos_vel = np.array(self.sim.data.get_site_xvelp(name))
        self.ref_ori_vel = np.array(self.sim.data.get_site_xvelr(name))
        # Column indexing keeps the Jacobian in Fortran order. Changing its layout
        # changes the reduction order in J.T @ wrench.
        self.J_pos = np.array(
            self.sim.data.get_site_jacp(name).reshape((3, -1))[:, self.qvel_index]
        )
        self.J_ori = np.array(
            self.sim.data.get_site_jacr(name).reshape((3, -1))[:, self.qvel_index]
        )
        self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))

    def compute_goal_ori(
        self, delta: np.ndarray[Any, np.dtype[np.float64]], goal_update_mode: str | None = None
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        # A zero rotation command retains the previous orientation goal.
        # Robosuite 1.5.2 instead replaces it with the achieved orientation each tick,
        # allowing the wrist to drift during translation-only reset and grasp steps.
        if self.goal_ori is not None and all(math.isclose(float(value), 0.0) for value in delta):
            return np.array(self.goal_ori, dtype=np.float64, copy=True)
        return cast(
            np.ndarray[Any, np.dtype[np.float64]],
            super().compute_goal_ori(delta, goal_update_mode=goal_update_mode),
        )


class _J8JointPositionController(
    _MassMatrixUpdateMixin,
    JointPositionController,  # type: ignore[misc]
):
    pass


class _BenchmarkCompositeController(CompositeController):  # type: ignore[misc]
    record_arm_goal: (
        Callable[[np.ndarray[Any, np.dtype[np.float64]], _E7OperationalSpaceController], None]
        | None
    ) = None

    def set_goal(self, all_action: np.ndarray[Any, np.dtype[np.float64]]) -> None:
        self._gripper_inputs: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        for part_name, controller in self.part_controllers.items():
            start, end = self._action_split_indexes[part_name]
            action = all_action[start:end]
            if part_name in self.grippers:
                self._gripper_inputs[part_name] = action.copy()
            else:
                controller.set_goal(action)
                if self.record_arm_goal is not None and part_name == "right":
                    self.record_arm_goal(action, controller)

    def run_controller(self, enabled_parts: Mapping[str, bool]) -> dict[str, Any]:
        self.update_state()
        self._applied_action_dict.clear()
        for part_name, controller in self.part_controllers.items():
            if not enabled_parts.get(part_name, False):
                continue
            if part_name in self.grippers:
                # The Panda advances its gripper target at physics frequency.
                # Robosuite 1.4 increments by 0.01 here; 1.5 raised its default
                # to 0.2 when moving this update to policy frequency.
                gripper = self.grippers[part_name]
                gripper.current_action = np.clip(
                    gripper.current_action
                    + np.asarray((-1.0, 1.0)) * 0.01 * np.sign(self._gripper_inputs[part_name]),
                    -1.0,
                    1.0,
                )
                controller.set_goal(gripper.current_action)
            self._applied_action_dict[part_name] = controller.run_controller()
        return cast(dict[str, Any], self._applied_action_dict)

    def _init_controllers(self) -> None:
        # Before any policy/preparation command, retain the reset gripper target.
        # This is controller initialization, not an issued zero E7 action.
        self._gripper_inputs = {
            name: np.zeros(gripper.dof, dtype=np.float64) for name, gripper in self.grippers.items()
        }
        for part_name in self.part_controller_config:
            params = dict(self.part_controller_config[part_name])
            controller_type = params["type"]
            if part_name in self.arms:
                if params.get("interpolation") is not None:
                    raise ValueError("Benchmark Panda controllers require interpolation=None")
                if controller_type == "OSC_POSE":
                    params["control_ori"] = True
                    controller = _E7OperationalSpaceController(
                        interpolator_pos=None,
                        interpolator_ori=None,
                        **params,
                    )
                elif controller_type == "JOINT_POSITION":
                    controller = _J8JointPositionController(interpolator=None, **params)
                else:
                    raise ValueError(
                        f"unsupported Benchmark Panda arm controller: {controller_type}"
                    )
            else:
                controller = controller_factory(
                    part_name=part_name,
                    controller_type=controller_type,
                    controller_params=params,
                )
            self.part_controllers[part_name] = controller


def _native_joint_address(
    model: mujoco.MjModel,
    joint_name: str,
    *,
    state: Literal["qpos", "qvel"],
) -> int | tuple[int, int]:
    joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
    if joint_id < 0:
        raise RuntimeError(f"compiled Panda joint {joint_name!r} is missing")
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        qpos_width, qvel_width = 7, 6
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        qpos_width, qvel_width = 4, 3
    elif joint_type in (
        int(mujoco.mjtJoint.mjJNT_SLIDE),
        int(mujoco.mjtJoint.mjJNT_HINGE),
    ):
        qpos_width = qvel_width = 1
    else:
        raise RuntimeError(
            f"compiled Panda joint {joint_name!r} has unknown native type {joint_type}"
        )
    if state == "qpos":
        address = int(model.jnt_qposadr[joint_id])
        width = qpos_width
    elif state == "qvel":
        address = int(model.jnt_dofadr[joint_id])
        width = qvel_width
    else:
        raise ValueError(f"unsupported native joint state: {state!r}")
    return address if width == 1 else (address, address + width)


class _BenchmarkFixedBaseRobot(FixedBaseRobot):  # type: ignore[misc]
    def setup_references(self) -> None:
        _require_exact_runtime_versions()
        model = cast(mujoco.MjModel, self.sim.model._model)

        self.robot_joints = self.robot_model.joints
        self._ref_joint_pos_indexes = [
            _native_joint_address(model, joint, state="qpos") for joint in self.robot_joints
        ]
        self._ref_joint_vel_indexes = [
            _native_joint_address(model, joint, state="qvel") for joint in self.robot_joints
        ]

        self._ref_joint_indexes = [
            self.sim.model.joint_name2id(joint) for joint in self.robot_joints
        ]
        self._ref_arm_joint_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator) for actuator in self.robot_model.arm_actuators
        ]

        self.robot_arm_joints = self.robot_model.arm_joints
        self._ref_arm_joint_indexes = [
            self.sim.model.joint_name2id(joint) for joint in self.robot_arm_joints
        ]
        self._ref_arm_joint_pos_indexes = [
            _native_joint_address(model, joint, state="qpos") for joint in self.robot_arm_joints
        ]
        self._ref_arm_joint_vel_indexes = [
            _native_joint_address(model, joint, state="qvel") for joint in self.robot_arm_joints
        ]

        self._ref_base_joint_pos_indexes = [
            _native_joint_address(model, joint, state="qpos")
            for joint in self.robot_model._base_joints
        ]
        self._ref_torso_joint_pos_indexes = [
            _native_joint_address(model, joint, state="qpos")
            for joint in self.robot_model._torso_joints
        ]
        self._ref_all_joint_pos_indexes = [
            _native_joint_address(model, joint, state="qpos")
            for joint in self.robot_model.all_joints
        ]

        for arm in self.arms:
            start, end = (
                (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)
            )
            self._ref_joints_indexes_dict[arm] = [
                self.sim.model.joint_name2id(joint)
                for joint in self.robot_model.arm_joints[start:end]
            ]
            self._ref_actuators_indexes_dict[arm] = [
                self.sim.model.actuator_name2id(actuator)
                for actuator in self.robot_model.arm_actuators[start:end]
            ]
            if self.has_gripper[arm]:
                gripper = self.gripper[arm]
                self.gripper_joints[arm] = list(gripper.joints)
                self._ref_gripper_joint_pos_indexes[arm] = [
                    _native_joint_address(model, joint, state="qpos")
                    for joint in self.gripper_joints[arm]
                ]
                self._ref_gripper_joint_vel_indexes[arm] = [
                    _native_joint_address(model, joint, state="qvel")
                    for joint in self.gripper_joints[arm]
                ]
                self._ref_joint_gripper_actuator_indexes[arm] = [
                    self.sim.model.actuator_name2id(actuator) for actuator in gripper.actuators
                ]
                gripper_name = self.get_gripper_name(arm)
                self._ref_joints_indexes_dict[gripper_name] = [
                    self.sim.model.joint_name2id(joint) for joint in self.gripper_joints[arm]
                ]
                self._ref_actuators_indexes_dict[gripper_name] = (
                    self._ref_joint_gripper_actuator_indexes[arm]
                )

            self.eef_site_id[arm] = self.sim.model.site_name2id(
                self.gripper[arm].important_sites["grip_site"]
            )
            self.eef_cylinder_id[arm] = self.sim.model.site_name2id(
                self.gripper[arm].important_sites["grip_cylinder"]
            )

    def _load_controller(self) -> None:
        self.composite_controller = _BenchmarkCompositeController(
            sim=self.sim,
            robot_model=self.robot_model,
            grippers={self.get_gripper_name(arm): self.gripper[arm] for arm in self.arms},
        )
        self._load_arm_controllers()
        self._postprocess_part_controller_config()
        self.composite_controller.load_controller_config(
            self.part_controller_config,
            self.composite_controller_config.get("composite_controller_specific_configs", {}),
        )
        self.enable_parts()


def _xml_numbers(values: Sequence[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _rewrite_scenario_lighting_and_gravity_xml(
    xml_str: str,
    scenario: ScenarioSpec,
    *,
    light_id: str,
    invalid_message: str,
    duplicate_message: str,
    missing_message: str,
    headlight_before_gravity: bool,
) -> str:
    root = ET.fromstring(xml_str)
    lights = scenario.lighting.lights
    if len(lights) != 1 or lights[0].light_id != light_id:
        raise ValueError(invalid_message)
    selected: ET.Element | None = None
    for parent in root.iter():
        for light in tuple(parent.findall("light")):
            if light.get("name") == light_id:
                if selected is not None:
                    raise ValueError(duplicate_message)
                selected = light
            else:
                parent.remove(light)
    if selected is None:
        raise ValueError(missing_message)
    spec = lights[0]
    for field, values in (
        ("pos", spec.position_world_m),
        ("dir", spec.direction_world),
        ("ambient", spec.ambient_rgb),
        ("diffuse", spec.diffuse_rgb),
        ("specular", spec.specular_rgb),
        ("attenuation", spec.attenuation),
    ):
        selected.set(field, _xml_numbers(values))
    selected.set("cutoff", format(spec.cutoff_deg, ".17g"))
    selected.set("active", "true" if spec.active else "false")

    def restore_headlight() -> None:
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        headlight = visual.find("headlight")
        if headlight is None:
            headlight = ET.SubElement(visual, "headlight")
        headlight.set("active", "1")

    def set_gravity() -> None:
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("gravity", _xml_numbers(scenario.physics.gravity_world_m_s2))

    if headlight_before_gravity:
        restore_headlight()
        set_gravity()
    else:
        set_gravity()
        restore_headlight()
    return ET.tostring(root, encoding="unicode")


class _ScenarioResetRejected(ValueError):
    """A known physical preparation failure, retained across the factory boundary."""

    def __init__(self, reason: str, time_s: float) -> None:
        self.reason, self.time_s = reason, time_s
        super().__init__(f"scenario preparation rejected: {reason} at {time_s:.9g} s")


class DynamicManipulationEnv(ManipulationEnv):  # type: ignore[misc]
    """Robosuite environment for dynamic manipulation with a Franka Panda."""

    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        scenario_spec: ScenarioSpec,
        seed: int | None = None,
        execution_translation_limit_m: float = 0.05,
    ) -> None:
        if not isinstance(task_spec, TaskSpec):
            raise ValueError("task_spec must be a TaskSpec")
        if not isinstance(scenario_spec, ScenarioSpec):
            raise ValueError("scenario_spec must be a ScenarioSpec")
        _validate_execution_translation_limit(execution_translation_limit_m)
        if execution_translation_limit_m == 0.10 and scenario_spec.action_mode != "E7":
            raise ValueError("0.10 m execution translation limits require E7 delta")
        self._task_spec = task_spec
        self._scenario_spec = scenario_spec
        self._execution_translation_limit_m = execution_translation_limit_m
        self._episode_tick = 0
        self._action_spec: ActionSpec | None = None
        self._asynchronous_actions = False
        self._action_delay_queue: _ActionDelayQueue | None = None
        self._last_action_application: tuple[int, int, BenchmarkAction] | None = None
        self._previous_eef_quaternion_xyzw: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._dynamic_target_body_id: int | None = None
        self._recording_geom_table: tuple[_RecordingGeomFact, ...] | None = None
        self._single_sample_context: mujoco.MjrContext | None = None
        self._recording_observations = False
        self._recording_segmentation: tuple[int, np.ndarray[Any, np.dtype[np.int32]]] | None = None
        self._rgb_render_states: dict[str, tuple[np.ndarray[Any, Any], ...]] = {}
        self._camera_sensor_names = (
            ("agentview",)
            if scenario_spec.observation_profile == "agentview"
            else ("agentview", "robot0_eye_in_hand")
        )
        self._observation_spec = ObservationSpec(
            tuple(view.view_name for view in scenario_spec.cameras.views)
        )
        self._robot_base_body_name = ""
        self._fixed_reference_collision_masks: tuple[mujoco.MjModel, tuple[Any, ...]] | None = None
        super().__init__(
            robots="Panda",
            controller_configs=_panda_controller_config(
                "E7"
                if task_spec.family
                in (
                    "moving_placement",
                    "rotating_interaction",
                    "free_dynamics_interception",
                    "replanning",
                )
                else scenario_spec.action_mode
            ),
            base_types="default",
            gripper_types="default",
            initialization_noise=None,
            use_camera_obs=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="agentview",
            render_gpu_device_id=-1,
            renderer="mujoco",
            control_freq=scenario_spec.control_frequency_hz,
            # The training data were recorded with Robosuite 1.4's full forward/update sequence.
            lite_physics=False,
            horizon=task_spec.episode_horizon_steps,
            ignore_done=False,
            hard_reset=True,
            camera_names=list(self._camera_sensor_names),
            camera_heights=scenario_spec.cameras.image_height_px,
            camera_widths=scenario_spec.cameras.image_width_px,
            camera_depths=True,
            camera_segmentations=None,
            seed=seed,
        )
        self.set_ep_meta(scenario_spec)

    def set_ep_meta(self, meta: object) -> None:
        if not isinstance(meta, ScenarioSpec):
            raise ValueError("episode metadata must be the concrete ScenarioSpec")
        if meta != self._scenario_spec:
            raise ValueError("compiled environment cannot accept a different ScenarioSpec")
        super().set_ep_meta(meta)

    def edit_model_xml(self, xml_str: str) -> str:
        rewritten = super().edit_model_xml(xml_str)
        root = ET.fromstring(rewritten)
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        quality = visual.find("quality")
        if quality is None:
            quality = ET.SubElement(visual, "quality")
        # Keep the training camera's RGB sampling. Read depth and labels in a separate context.
        quality.set("offsamples", "4")
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        compiler.set("inertiagrouprange", "0 0")
        cameras: dict[str, list[ET.Element]] = {}
        for element in root.iter("camera"):
            name = element.get("name")
            if name is not None:
                cameras.setdefault(name, []).append(element)
        for camera_spec, sensor_name in zip(
            self._scenario_spec.cameras.views,
            self._camera_sensor_names,
            strict=True,
        ):
            matches = cameras.get(sensor_name, [])
            if len(matches) != 1:
                raise ValueError(
                    f"compiled XML must contain exactly one camera named {sensor_name!r}"
                )
            camera = matches[0]
            if camera.get("mode") != "fixed" or any(
                key in camera.attrib
                for key in ("focal", "focalpixel", "sensorsize", "resolution", "orthographic")
            ):
                raise ValueError(f"camera {sensor_name!r} must be a fixed perspective camera")
            x, y, z, w = camera_spec.quaternion_parent_xyzw
            camera.set("pos", _xml_numbers(camera_spec.position_parent_m))
            camera.set("quat", _xml_numbers((w, x, y, z)))
            camera.set("fovy", format(camera_spec.vertical_fov_deg, ".17g"))
        return ET.tostring(root, encoding="unicode")

    def _check_robot_configuration(self, robots: object) -> None:
        if isinstance(robots, str):
            robot_names = (robots,)
        elif isinstance(robots, Sequence) and not isinstance(robots, (str, bytes)):
            robot_names = tuple(robots)
        else:
            raise ValueError("OnTimeBench requires one explicit Panda robot")
        if robot_names != ("Panda",):
            raise ValueError("OnTimeBench requires one explicit Panda robot")

    def _load_robots(self) -> None:
        _require_exact_runtime_versions()
        if self.robot_names != ["Panda"] or len(self.robot_configs) != 1 or len(self.robots) != 1:
            raise ValueError("OnTimeBench requires one explicit Panda robot")
        config = self.robot_configs[0]
        if not isinstance(config, dict) or config.get("composite_controller_config") is None:
            raise ValueError("OnTimeBench requires an explicit Panda controller config")
        config = dict(config)
        config["initial_qpos"] = np.asarray(
            self._scenario_spec.robot_initial_state.joint_position_rad,
            dtype=np.float64,
        )
        robot = _BenchmarkFixedBaseRobot(robot_type="Panda", idn=0, **config)
        robot.load_model()
        # Robosuite 1.5.2 added a collision mesh that was absent from The Panda.
        # Keep the original collision surface while using the current robot API.
        arm_model = robot.robot_model
        link = arm_model.worldbody.find(f".//body[@name='{arm_model.naming_prefix}link7']")
        collision_name = "link7_collision"
        collision = (
            None
            if link is None
            else link.find(f"geom[@name='{arm_model.naming_prefix}{collision_name}']")
        )
        if link is None or collision is None:
            raise RuntimeError("Robosuite 1.5.2 Panda link7 collision binding is missing")
        link.remove(collision)
        arm_model._contact_geoms.remove(collision_name)
        arm_model._elements["contact_geoms"].remove(collision)
        gripper = robot.gripper["right"]
        if gripper is None or len(gripper.init_qpos) != 2:
            raise RuntimeError("Benchmark Panda must have the standard two-joint gripper")
        root_body = robot.robot_model.root_body
        if not isinstance(root_body, str) or not root_body:
            raise RuntimeError("Benchmark Panda must expose one explicit robot base body")
        self._robot_base_body_name = root_body
        self.robots[0] = robot

    def _reset_internal(self) -> None:
        self._restore_fixed_reference_collision_masks()
        self._recording_segmentation = None
        self._rgb_render_states.clear()
        super()._reset_internal()
        # The segmentation setup hides helper sites in both rasters.
        self.sim.model.site_size[:, :] = 1e-8
        robot = self.robots[0]
        if not isinstance(robot, _BenchmarkFixedBaseRobot):
            raise RuntimeError("Benchmark Panda private robot bridge is not active")
        gripper = robot.gripper["right"]
        if gripper is None:
            raise RuntimeError("Benchmark Panda gripper is missing")
        gripper.current_action = np.zeros(gripper.dof, dtype=np.float64)
        self.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]] = np.asarray(
            self._scenario_spec.robot_initial_state.gripper_position_m,
            dtype=np.float64,
        )
        if float(self.sim.model.opt.timestep) != self._scenario_spec.physics_timestep_s:
            raise RuntimeError("compiled MuJoCo timestep does not match ScenarioSpec")
        if self.control_freq != self._scenario_spec.control_frequency_hz:
            raise RuntimeError("Robosuite control frequency does not match ScenarioSpec")
        self._episode_tick = 0
        self._action_spec = None
        self._first_contact_snapshots: dict[str, dict[str, object]] = {}
        self._fixed_reference_state: tuple[Any, Any, Any] | None = None
        self._fixed_reference_contacts: dict[str, object] = {}
        self._asynchronous_actions = False
        self._action_delay_queue = None
        self._last_action_application = None
        self._previous_eef_quaternion_xyzw = None

    def _preparation_controller_action(
        self,
        position_error: np.ndarray[Any, np.dtype[np.float64]],
        rotation_error: np.ndarray[Any, np.dtype[np.float64]],
        gripper: float,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Build an unscored reset command for the operational-space controller."""
        part = self.robots[0].composite_controller.part_controllers["right"]
        if not isinstance(part, _E7OperationalSpaceController):
            raise RuntimeError("Episode preparation requires the E7 OSC controller")
        return np.concatenate(
            (
                np.clip(position_error / 0.05, -1.0, 1.0),
                np.clip(rotation_error / 0.5, -1.0, 1.0),
                [gripper],
            )
        )

    def _finish_preparation_controller(self) -> None:
        """Keep OnTimeBench preparation state while selecting the declared scoring controller."""
        if self._scenario_spec.action_mode == "E7":
            return
        robot = self.robots[0]
        if not isinstance(robot, _BenchmarkFixedBaseRobot) or not isinstance(
            robot.composite_controller, _BenchmarkCompositeController
        ):
            raise RuntimeError("Benchmark Panda private controller bridge is not active")
        composite = robot.composite_controller
        if not isinstance(composite.part_controllers["right"], _E7OperationalSpaceController):
            raise RuntimeError("Episode preparation must finish with the E7 OSC controller")
        bound = composite.part_controller_config["right"]
        params = {
            key: bound[key]
            for key in (
                "robot_name",
                "sim",
                "ref_name",
                "part_name",
                "naming_prefix",
                "eef_rot_offset",
                "ndim",
                "policy_freq",
                "lite_physics",
                "joint_indexes",
                "actuator_range",
                "load_urdf",
            )
        }
        joint_config = _panda_controller_config("J8")["body_parts"]["right"]
        params.update({key: value for key, value in joint_config.items() if key != "gripper"})
        controller = _J8JointPositionController(interpolator=None, **params)
        controller.reset_goal()
        # Replace only this live right-arm configuration; reset retains its
        # original OSC construction config and the existing gripper controller.
        robot.part_controller_config = {**robot.part_controller_config, "right": params}
        composite.part_controller_config = robot.part_controller_config
        composite.part_controllers["right"] = controller
        composite.setup_action_split_idx()
        self._action_dim = robot.action_dim

    def _achieved_controller_state(
        self,
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float64]] | None,
        np.ndarray[Any, np.dtype[np.float64]] | None,
        np.ndarray[Any, np.dtype[np.float64]] | None,
    ]:
        robot = self.robots[0]
        if not isinstance(robot, _BenchmarkFixedBaseRobot) or not isinstance(
            robot.composite_controller, _BenchmarkCompositeController
        ):
            raise RuntimeError("Benchmark Panda private controller bridge is not active")
        robot.composite_controller.update_state()
        part = robot.composite_controller.part_controllers.get("right")
        if self._scenario_spec.action_mode == "E7":
            if not isinstance(part, _E7OperationalSpaceController):
                raise RuntimeError("E7 scenario requires the private OSC_POSE controller")
            # Match the reference controller's refresh schedule and reset cache.
            part.update()
            eef_pose = np.asarray(part.goal_origin_to_eef_pose(), dtype=np.float64)
            if eef_pose.shape != (4, 4) or not np.isfinite(eef_pose).all():
                raise RuntimeError("private OSC controller returned an invalid achieved EEF pose")
            return (
                np.ascontiguousarray(eef_pose[:3, 3], dtype=np.float64),
                np.ascontiguousarray(eef_pose[:3, :3], dtype=np.float64),
                None,
            )
        if not isinstance(part, _J8JointPositionController):
            raise RuntimeError("J8 scenario requires the private JOINT_POSITION controller")
        part.update(force=True)
        joints = np.asarray(part.joint_pos, dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise RuntimeError("private joint controller returned invalid achieved joint state")
        return None, None, np.ascontiguousarray(joints, dtype=np.float64)

    def _delayed_controller_action(
        self, action: BenchmarkAction
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        tuple[int, int, BenchmarkAction] | None,
    ]:
        if not isinstance(action, BenchmarkAction):
            raise ValueError("OnTimeBench step requires a BenchmarkAction")
        if action.mode != self._scenario_spec.action_mode:
            raise ValueError("BenchmarkAction mode does not match ScenarioSpec")
        if self._action_spec is None:
            limits = (
                (self._execution_translation_limit_m,) * 3 + (0.5,) * 3
                if action.mode == "E7"
                else (0.05,) * 7
            )
            spec = ActionSpec(action.mode, action.representation, limits)
            validate_action_spec(action, spec)
            if self._execution_translation_limit_m == 0.10:
                part = self.robots[0].composite_controller.part_controllers["right"]
                if not isinstance(part, _E7OperationalSpaceController):
                    raise RuntimeError("E7 execution requires the private OSC_POSE controller")
                # Apply the 0.10 m limit to scored actions. Reset uses the 0.05 m limit.
                part.output_max = np.asarray(spec.motion_limits, dtype=np.float64)
                part.output_min = -part.output_max
                part.action_scale = None
            self._action_spec = spec
            self._action_delay_queue = _ActionDelayQueue(
                self._action_spec,
                self._scenario_spec.action_delay_steps,
            )
        elif (action.mode, action.representation) != (
            self._action_spec.mode,
            self._action_spec.representation,
        ):
            raise ValueError("BenchmarkAction does not match the episode ActionSpec")
        if self._action_delay_queue is None:
            raise RuntimeError("episode ActionSpec latch did not create its delay queue")
        eef_position, eef_rotation, joint_position = self._achieved_controller_state()
        controller_action, applied = self._action_delay_queue.issue(
            action,
            issued_tick=self._episode_tick,
            achieved_eef_position=eef_position,
            achieved_eef_rotation=eef_rotation,
            achieved_joint_position=joint_position,
        )
        if action.mode == "E7":
            base = int(self.sim.model._model.body(self._robot_base_body_name).id)
            world_from_base = self.sim.data._data.xmat[base].reshape(3, 3)
            controller_action[:3] = world_from_base @ controller_action[:3]
            controller_action[3:6] = world_from_base @ controller_action[3:6]
            controller_action = controller_action.astype(np.float32).astype(np.float64)
        application = None
        if applied is not None:
            issued_tick, applied_action = applied
            application = (
                issued_tick,
                self._episode_tick,
                applied_action,
            )
        return controller_action, application

    def _enable_asynchronous_actions(self, action_spec: ActionSpec | None = None) -> None:
        """Declare the controller contract before the first possibly-empty scoring tick."""
        if self._episode_tick != 0:
            raise ValueError("asynchronous actions require declaration immediately after reset")
        if action_spec is None:
            if self._scenario_spec.action_mode != "E7":
                raise ValueError("asynchronous actions require an explicit ActionSpec for J8")
            action_spec = ActionSpec(
                "E7", "delta", (self._execution_translation_limit_m,) * 3 + (0.5,) * 3
            )
        if action_spec.mode != self._scenario_spec.action_mode:
            raise ValueError("asynchronous ActionSpec must match ScenarioSpec")
        expected_limits = (
            (self._execution_translation_limit_m,) * 3 + (0.5,) * 3
            if action_spec.mode == "E7"
            else (0.05,) * 7
        )
        if action_spec.motion_limits != expected_limits:
            raise ValueError("asynchronous ActionSpec must match execution limits")
        self._action_spec = action_spec
        self._action_delay_queue = _ActionDelayQueue(
            self._action_spec, self._scenario_spec.action_delay_steps
        )
        part = self.robots[0].composite_controller.part_controllers["right"]
        if action_spec.mode == "J8":
            if not isinstance(part, _J8JointPositionController):
                raise RuntimeError("asynchronous J8 requires joint position controller")
            self._asynchronous_actions = True
            return
        if not isinstance(part, _E7OperationalSpaceController):
            raise RuntimeError("asynchronous E7 requires OSC controller")
        part.output_max = np.asarray(self._action_spec.motion_limits, dtype=np.float64)
        part.output_min = -part.output_max
        part.action_scale = None
        # Load the three lazy Numba kernels before the first paced tick. These
        # pure computations do not run the controller, change its goals, or step
        # physics. Use the real reset array layouts to match native signatures.
        from robosuite.utils.control_utils import (
            nullspace_torques,
            opspace_matrices,
            orientation_error,
        )

        orientation_error(np.dot(part.origin_ori, part.goal_ori), part.ref_ori_mat)
        _, _, _, nullspace = opspace_matrices(part.mass_matrix, part.J_full, part.J_pos, part.J_ori)
        nullspace_torques(
            part.mass_matrix, nullspace, part.initial_joint, part.joint_pos, part.joint_vel
        )
        # Reset/hold use zero rotation and bypass the nonzero orientation branch.
        # Warm its pure conversion with native float64 input before any issued
        # action; do not call set_goal or integrate this synthetic rotation.
        from robosuite.utils.transform_utils import axisangle2quat, quat2mat

        quat2mat(axisangle2quat(np.asarray([0.01, 0.02, 0.03], dtype=np.float64)))
        self._asynchronous_actions = True

    def _step_without_new_goal(self) -> tuple[Mapping[str, object], float, bool, dict[str, Any]]:
        """Run the native control tick with every substep marked as feedback-only.

        Preserve the native step hooks and timing. The previous controller input
        is only a carrier for hooks; policy_step=False never integrates its delta.
        """
        carrier = self.robots[0].recent_actions.current.copy()
        self.timestep += 1
        for _ in range(int(self.control_timestep / self.model_timestep)):
            if self.lite_physics:
                self.sim.step1()
            else:
                self.sim.forward()
            self._pre_action(carrier, False)
            if self.lite_physics:
                self.sim.step2()
            else:
                self.sim.step()
            self._update_observables()
        self.cur_time += self.control_timestep
        reward, done, info = self._post_action(carrier)
        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()
        elif self.has_renderer and self.renderer == "mjviewer" and self.viewer is None:
            self.initialize_renderer()
            assert self.viewer is not None
            self.viewer.update()
        observations = (
            self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        )
        return observations, reward, done, info

    def step(
        self, action: BenchmarkAction | None
    ) -> tuple[Mapping[str, object], float, bool, dict[str, Any]]:
        if self._fixed_reference_collision_masks is not None:
            raise ValueError(
                "fixed reference mode requires its private step path; reset before rollout"
            )
        if self.done:
            raise ValueError("executing action in terminated episode")
        self._recording_segmentation = None
        self._last_action_application = None
        if self._asynchronous_actions:
            assert self._action_delay_queue is not None
            eef_position, eef_rotation, joint_position = self._achieved_controller_state()
            controller_action, applied = self._action_delay_queue.advance(
                action,
                tick=self._episode_tick,
                achieved_eef_position=eef_position,
                achieved_eef_rotation=eef_rotation,
                achieved_joint_position=joint_position,
            )
            application = None if applied is None else (applied[0], self._episode_tick, applied[1])
            if controller_action is None:
                result = self._step_without_new_goal()
            else:
                if self._scenario_spec.action_mode == "E7":
                    base = int(self.sim.model._model.body(self._robot_base_body_name).id)
                    world_from_base = self.sim.data._data.xmat[base].reshape(3, 3)
                    controller_action[:3] = world_from_base @ controller_action[:3]
                    controller_action[3:6] = world_from_base @ controller_action[3:6]
                    controller_action = controller_action.astype(np.float32).astype(np.float64)
                result = super().step(controller_action)
        else:
            if action is None:
                raise ValueError("OnTimeBench step requires a BenchmarkAction")
            controller_action, application = self._delayed_controller_action(action)
            result = super().step(controller_action)
        self._last_action_application = application
        self._episode_tick += 1
        if result[2] and self._action_delay_queue is not None:
            self._action_delay_queue.reset()
        return result

    def _last_action_application_fact(self) -> tuple[int, int, BenchmarkAction] | None:
        return self._last_action_application

    def _enable_fixed_robot_reference(self, target_body_id: int) -> None:
        """Fix the initial robot and remove only its collisions with the reference target."""
        if self._episode_tick != 0:
            raise ValueError("fixed robot reference must start at scored t0")
        if self._fixed_reference_collision_masks is not None:
            raise ValueError("fixed robot reference is already enabled")
        if mujoco.get_mjcb_contactfilter() is not None:
            raise ValueError("fixed reference requires default MuJoCo collision filtering")
        model, data = self.sim.model._model, self.sim.data._data
        if not 0 < target_body_id < model.nbody:
            raise ValueError("reference target must be a model body other than world")
        robot = self.robots[0]
        joint_ids = [
            model.joint(name).id
            for name in (*robot.robot_model.joints, *robot.gripper["right"].joints)
        ]
        if not np.all(
            np.isin(
                model.jnt_type[joint_ids],
                [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE],
            )
        ):
            raise ValueError("fixed reference requires scalar arm and gripper joints")
        qadr, dadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
        robot_body = model.body(self._robot_base_body_name).id

        def below(body: int, root: int) -> bool:
            while body:
                if body == root:
                    return True
                body = int(model.body_parentid[body])
            return False

        self._fixed_reference_robot_geoms = frozenset(
            i for i in range(model.ngeom) if below(int(model.geom_bodyid[i]), robot_body)
        )
        self._fixed_reference_target_geoms = frozenset(
            i for i in range(model.ngeom) if below(int(model.geom_bodyid[i]), target_body_id)
        )
        robot_geoms, target_geoms = (
            self._fixed_reference_robot_geoms,
            self._fixed_reference_target_geoms,
        )
        for g1, g2 in zip(model.pair_geom1, model.pair_geom2, strict=True):
            if (g1 in robot_geoms and g2 in target_geoms) or (
                g2 in robot_geoms and g1 in target_geoms
            ):
                raise ValueError("explicit robot-target contact pairs bypass reference bit masks")
        types, affinities = _reference_collision_masks(
            model.geom_contype, model.geom_conaffinity, robot_geoms, target_geoms
        )
        self._fixed_reference_collision_masks = (
            model,
            tuple(
                array.copy()
                for array in (
                    model.geom_contype,
                    model.geom_conaffinity,
                    model.body_contype,
                    model.body_conaffinity,
                )
            ),
        )
        model.geom_contype[:] = types
        model.geom_conaffinity[:] = affinities
        # MuJoCo broadphase uses body aggregates before the individual geom masks.
        model.body_contype[:] = 0
        model.body_conaffinity[:] = 0
        np.bitwise_or.at(model.body_contype, model.geom_bodyid, types)
        np.bitwise_or.at(model.body_conaffinity, model.geom_bodyid, affinities)
        self._fixed_reference_state = qadr.copy(), dadr.copy(), data.qpos[qadr].copy()
        self._fixed_reference_contacts = {
            "robot_target_collisions_disabled": True,
            "robot_target_first_contact_s": None,
            "robot_target_contact_substeps": 0,
        }

    def _restore_fixed_reference_collision_masks(self) -> None:
        if self._fixed_reference_collision_masks is not None:
            model, saved = self._fixed_reference_collision_masks
            for array, original in zip(
                (
                    model.geom_contype,
                    model.geom_conaffinity,
                    model.body_contype,
                    model.body_conaffinity,
                ),
                saved,
                strict=True,
            ):
                array[:] = original
            self._fixed_reference_collision_masks = None
            self._fixed_reference_state = None

    def close(self) -> None:
        self._restore_fixed_reference_collision_masks()
        super().close()

    def _step_fixed_robot_reference(self) -> None:
        """Advance the world 50 ms, pinning arm and gripper before/after every substep.

        This is an offline reference path, not a controller or a policy rollout.
        Native drivers, environment contacts and task event bookkeeping still run.
        Only robot--reference-target contacts are disabled on this isolated model.
        """
        if self._fixed_reference_state is None:
            raise RuntimeError("fixed robot reference was not enabled")
        qadr, dadr, qinitial = self._fixed_reference_state
        data = self.sim.data._data
        carrier = self.robots[0].recent_actions.current.copy()
        self.timestep += 1
        for _ in range(int(self.control_timestep / self.model_timestep)):
            data.qpos[qadr] = qinitial
            data.qvel[dadr] = 0.0
            self.sim.forward()
            if self.lite_physics:
                self.sim.step1()
            self._pre_action(carrier, False)
            if self.lite_physics:
                self.sim.step2()
            else:
                self.sim.step()
            data.qpos[qadr] = qinitial
            data.qvel[dadr] = 0.0
            self.sim.forward()
            if any(
                (
                    int(contact.geom1) in self._fixed_reference_robot_geoms
                    and int(contact.geom2) in self._fixed_reference_target_geoms
                )
                or (
                    int(contact.geom2) in self._fixed_reference_robot_geoms
                    and int(contact.geom1) in self._fixed_reference_target_geoms
                )
                for contact in data.contact[: data.ncon]
            ):
                if self._fixed_reference_contacts["robot_target_first_contact_s"] is None:
                    self._fixed_reference_contacts["robot_target_first_contact_s"] = float(
                        data.time
                    )
                count = cast(int, self._fixed_reference_contacts["robot_target_contact_substeps"])
                self._fixed_reference_contacts["robot_target_contact_substeps"] = count + 1
            self._update_observables()
        self.cur_time += self.control_timestep
        self._post_action(carrier)
        self._episode_tick += 1

    def _fixed_robot_reference_facts(self) -> dict[str, object]:
        return dict(self._fixed_reference_contacts)

    def _capture_first_contact(
        self, kind: str, target_body_id: int, *, ramp_exited: bool | None = None
    ) -> None:
        """Copy poses at the first scored native contact, including failed contacts.

        Kinematics runs on a separate data object: live post-integration qpos and
        mocap poses are authoritative, and measuring must not refresh live caches.
        These are raw body/EEF poses, not an assumed task-specific grasp frame.
        """
        if kind in self._first_contact_snapshots:
            return
        model, data = self.sim.model._model, self.sim.data._data
        measured = mujoco.MjData(model)
        measured.qpos[:] = data.qpos
        measured.mocap_pos[:] = data.mocap_pos
        measured.mocap_quat[:] = data.mocap_quat
        mujoco.mj_kinematics(model, measured)
        eef_site = self.robots[0].eef_site_id["right"]
        self._first_contact_snapshots[kind] = {
            "contact_kind": kind,
            "contact_time_s": float(data.time),
            "target_body_name": model.body(target_body_id).name,
            "eef_position_world_m": measured.site_xpos[eef_site].tolist(),
            "eef_rotation_world": measured.site_xmat[eef_site].reshape(3, 3).tolist(),
            "target_body_position_world_m": measured.xpos[target_body_id].tolist(),
            "target_body_rotation_world": measured.xmat[target_body_id].reshape(3, 3).tolist(),
        }
        if ramp_exited is not None:
            self._first_contact_snapshots[kind]["ramp_exited"] = ramp_exited

    def _contact_snapshot_facts(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(value) for value in self._first_contact_snapshots.values())

    def _bind_scene_entities(
        self,
        *,
        target_body_id: int,
        entity_body_ids: Mapping[str, int],
    ) -> None:
        """Bind the moving target and scenario entities to their MuJoCo bodies."""
        model = self.sim.model._model
        asset_by_instance = {
            entity.instance_id: entity.asset_id for entity in self._scenario_spec.object_states
        }
        asset_by_instance.update(
            {entity.instance_id: entity.asset_id for entity in self._scenario_spec.fixture_states}
        )
        if set(entity_body_ids) != set(asset_by_instance):
            raise ValueError(
                "recording bindings must cover every concrete scenario entity exactly once"
            )
        if (
            isinstance(target_body_id, bool)
            or not isinstance(target_body_id, int)
            or target_body_id <= 0
            or target_body_id >= model.nbody
        ):
            raise ValueError("recording target must name one compiled non-world body")
        roots: dict[int, tuple[str, str]] = {}
        for instance_id, body_id in entity_body_ids.items():
            if (
                isinstance(body_id, bool)
                or not isinstance(body_id, int)
                or body_id <= 0
                or body_id >= model.nbody
            ):
                raise ValueError("recording entity binding names an invalid compiled body")
            if body_id in roots:
                raise ValueError("recording entity bindings must use distinct compiled roots")
            roots[body_id] = (instance_id, asset_by_instance[instance_id])
        if target_body_id not in roots:
            raise ValueError("the target body must belong to a scenario entity")

        robot_root = int(model.body(self._robot_base_body_name).id)

        def ancestors(body_id: int) -> tuple[int, ...]:
            result = [body_id]
            while result[-1] != 0:
                result.append(int(model.body_parentid[result[-1]]))
            return tuple(result)

        geom_table: list[_RecordingGeomFact] = []
        for geom_id in range(model.ngeom):
            geom_name = model.geom(geom_id).name
            body_id = int(model.geom_bodyid[geom_id])
            body_name = "world" if body_id == 0 else model.body(body_id).name
            if not isinstance(geom_name, str) or not geom_name:
                raise ValueError("every compiled geom must have a stable name for recording")
            if not isinstance(body_name, str) or not body_name:
                raise ValueError("every compiled geom body must have a stable identity")
            lineage = ancestors(body_id)
            matches = tuple(root for root in roots if root in lineage)
            if len(matches) > 1:
                raise ValueError("compiled geom belongs to overlapping scenario entity bindings")
            if matches:
                instance_identity, semantic_identity = roots[matches[0]]
            elif robot_root in lineage:
                instance_identity, semantic_identity = "robot0", "robot"
            else:
                scene_root = next(
                    (item for item in lineage if int(model.body_parentid[item]) == 0), 0
                )
                scene_name = "world" if scene_root == 0 else model.body(scene_root).name
                if not isinstance(scene_name, str) or not scene_name:
                    raise ValueError("compiled scene owner must have a stable identity")
                instance_identity, semantic_identity = scene_name, "scene"
            geom_table.append((geom_id, geom_name, body_name, instance_identity, semantic_identity))
        if not geom_table:
            raise ValueError("recording requires at least one compiled geom")
        self._dynamic_target_body_id = target_body_id
        self._recording_geom_table = tuple(geom_table)

    def _recording_geoms(self) -> tuple[_RecordingGeomFact, ...]:
        if self._recording_geom_table is None:
            raise RuntimeError("Family Env did not establish its recording bindings")
        return self._recording_geom_table

    def _recording_frame_facts(self) -> _RecordingFrameFacts:
        raise NotImplementedError("Family Env must expose its outcome-owned recording facts")

    def _target_pose_in_robot_base(
        self, observation: BenchmarkObservation
    ) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
        """Read the simulated target pose at this capture, without advancing physics."""
        from scipy.spatial.transform import Rotation

        if observation.tick != self._episode_tick:
            raise ValueError("target pose must be read at the observation's capture tick")
        position, rotation = self._dynamic_target_snapshot()
        transform = observation.T_robot_base_from_world
        return (
            transform[:3, :3] @ position + transform[:3, 3],
            Rotation.from_matrix(transform[:3, :3] @ rotation).as_quat(),
        )

    def _dynamic_target_snapshot(
        self,
    ) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
        """Copy the dynamic target's world position and rotation at the current tick.

        T2/T3 override this with the prescribed motion at the observation time.
        Other families retain native provider timing: integrated free-body qpos
        or cached xpos/xmat. Reading never calls forward or advances physics.
        """
        target = self._dynamic_target_body_id
        if target is None:
            raise RuntimeError("Family Env did not establish its dynamic target binding")
        model, data = self.sim.model._model, self.sim.data._data
        joint = int(model.body_jntadr[target])
        if model.body_jntnum[target] == 1 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            address = int(model.jnt_qposadr[joint])
            position = np.array(data.qpos[address : address + 3], dtype=np.float64, copy=True)
            quaternion = np.array(data.qpos[address + 3 : address + 7], dtype=np.float64, copy=True)
            quaternion /= np.linalg.norm(quaternion)
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, quaternion)
            rotation = rotation.reshape(3, 3)
        else:
            position = np.array(data.xpos[target], dtype=np.float64, copy=True)
            rotation = np.array(data.xmat[target], dtype=np.float64, copy=True).reshape(3, 3)
        return position, rotation

    def _render_geometry_state(self) -> tuple[np.ndarray[Any, Any], ...]:
        """Copy the rigid scene poses consumed by the native RGB renderer.

        Reading these cached quantities must not call forward: the original RGB
        sensor runs before outcome hooks that can activate an attachment.
        """
        data = self.sim.data._data
        return tuple(
            value.copy()
            for value in (
                data.geom_xpos,
                data.geom_xmat,
                data.site_xpos,
                data.site_xmat,
                data.cam_xpos,
                data.cam_xmat,
                data.light_xpos,
                data.light_xdir,
            )
        )

    def _create_camera_sensors(
        self,
        cam_name: str,
        cam_w: int,
        cam_h: int,
        cam_d: bool,
        cam_segs: object,
        modality: str = "image",
    ) -> tuple[list[Any], list[str]]:
        sensors, names = super()._create_camera_sensors(
            cam_name, cam_w, cam_h, cam_d, cam_segs, modality
        )
        rgb_index = names.index(f"{cam_name}_image")
        native_rgb = sensors[rgb_index]

        @sensor(modality=modality)  # type: ignore[untyped-decorator]
        def camera_rgb(obs_cache: dict[str, object]) -> np.ndarray[Any, Any]:
            render_state = self._render_geometry_state()
            image = native_rgb(obs_cache)
            self._rgb_render_states[cam_name] = render_state
            return cast(np.ndarray[Any, Any], image)

        sensors[rgb_index] = camera_rgb
        return sensors, names

    def _camera_rgb_at_observation(
        self, raw_observation: Mapping[str, object]
    ) -> dict[str, object]:
        """Pair RGB with the current depth, calibration and supervision scene.

        Normal frames retain the native RGB array. Attachment outcome hooks may
        refresh render poses after that sensor sampled; only those stale rasters
        are reacquired. No physics or observable/controller updates occur here.
        """
        current = self._render_geometry_state()
        result = dict(raw_observation)
        for name in self._camera_sensor_names:
            if name not in self._rgb_render_states:
                raise RuntimeError(f"RGB exposure state is missing for {name!r}")
            exposed = self._rgb_render_states[name]
            if not all(np.array_equal(a, b) for a, b in zip(exposed, current, strict=True)):
                result[f"{name}_image"] = self.sim.render(
                    camera_name=name,
                    width=self._scenario_spec.cameras.image_width_px,
                    height=self._scenario_spec.cameras.image_height_px,
                    depth=False,
                )
        return result

    def _render_single_sample(
        self, camera_name: str, *, segmentation: bool
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Render pixel-center depth/IDs without resolving multisampled category colors."""
        context = self.sim._render_context_offscreen
        context.gl_ctx.make_current()
        original = context.con
        original_buffer = original.currentBuffer
        original_flags = context.scn.flags.copy()
        try:
            if self._single_sample_context is None:
                samples = self.sim.model.vis.quality.offsamples
                self.sim.model.vis.quality.offsamples = 0
                try:
                    self._single_sample_context = mujoco.MjrContext(
                        self.sim.model._model, mujoco.mjtFontScale.mjFONTSCALE_150
                    )
                finally:
                    self.sim.model.vis.quality.offsamples = samples
            context.con = self._single_sample_context
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context.con)
            image, depth = self.sim.render(
                camera_name=camera_name,
                width=224,
                height=224,
                depth=True,
                segmentation=segmentation,
            )
            return np.asarray(image), np.asarray(depth)
        finally:
            context.con = original
            context.scn.flags[:] = original_flags
            mujoco.mjr_setBuffer(original_buffer, original)

    def _destroy_sim(self) -> None:
        self._recording_segmentation = None
        self._rgb_render_states.clear()
        if self._single_sample_context is not None:
            self.sim._render_context_offscreen.gl_ctx.make_current()
            self._single_sample_context.free()
            self._single_sample_context = None
        super()._destroy_sim()

    def _benchmark_observation(self, raw_observation: Mapping[str, object]) -> BenchmarkObservation:
        if robosuite_macros.IMAGE_CONVENTION != "opengl":
            raise ValueError(
                "Benchmark observation adapter requires native OpenGL image convention"
            )
        expected_timestamp = self._episode_tick / self._scenario_spec.sensor_frequency_hz
        if not math.isclose(
            float(self.sim.data.time),
            expected_timestamp,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise RuntimeError("simulator time is not aligned with the 20 Hz observation tick")
        calibration = _compiled_camera_calibration(
            self.sim.model._model,
            self.sim.data._data,
            camera_names=self._camera_sensor_names,
            robot_base_body_name=self._robot_base_body_name,
            image_height_px=self._scenario_spec.cameras.image_height_px,
            image_width_px=self._scenario_spec.cameras.image_width_px,
        )
        single_sample_observation = self._camera_rgb_at_observation(raw_observation)
        segmentations = []
        for camera_name in self._camera_sensor_names:
            # Keep the original metric-depth rasterization path. Segmentation
            # can change foreground depth by a depth-buffer quantization step.
            _, depth = self._render_single_sample(camera_name, segmentation=False)
            image, _ = self._render_single_sample(camera_name, segmentation=True)
            single_sample_observation[f"{camera_name}_depth"] = depth[:, :, None]
            segmentations.append(image)
        observation = _build_benchmark_observation(
            single_sample_observation,
            observation_spec=self._observation_spec,
            camera_sensor_names=self._camera_sensor_names,
            depth_extent=calibration.depth_extent,
            depth_znear=calibration.depth_znear,
            depth_zfar=calibration.depth_zfar,
            camera_intrinsics=calibration.camera_intrinsics,
            world_from_camera_cv=calibration.world_from_camera_cv,
            T_robot_base_from_world=calibration.T_robot_base_from_world,
            tick=self._episode_tick,
            timestamp_s=expected_timestamp,
            previous_eef_quaternion_xyzw=self._previous_eef_quaternion_xyzw,
        )
        # Match training carriers: depth measures rendered geom surfaces, not a
        # skybox or cleared far plane. Use IDs from the same pixel-center scene
        # and retain every valid surface depth without any distance threshold.
        segmentation = np.ascontiguousarray(np.stack(segmentations), dtype=np.int32)
        surface = segmentation[:, ::-1, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
        depth_m = np.ascontiguousarray(np.where(surface, observation.depth_m, 0), dtype=np.float32)
        depth_m.setflags(write=False)
        observation = replace(observation, depth_m=depth_m)
        self._previous_eef_quaternion_xyzw = observation.eef_quaternion_xyzw
        if self._recording_observations:
            self._recording_segmentation = (
                observation.tick,
                segmentation,
            )
        return observation


__all__ = ["DynamicManipulationEnv"]
