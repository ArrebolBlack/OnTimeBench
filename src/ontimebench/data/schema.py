"""HDF5 layout for recorded simulator rollouts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from types import EllipsisType
from typing import Any, Protocol, TypeVar, cast
from urllib.parse import quote

import numpy as np
import numpy.typing as npt

from ontimebench.control.action import (
    ActionMode,
    ActionRepresentation,
    ActionSpec,
    BenchmarkAction,
    validate_action_spec,
)
from ontimebench.evaluation.result import FAILURE_CODES
from ontimebench.scenarios.spec import scenario_from_dict, scenario_to_dict
from ontimebench.sensing.camera import _camera_intrinsics, _validate_rigid_transform
from ontimebench.tasks.registry import TASK_FAMILIES

SCHEMA_VERSION = 1

_POLICY_DATASETS = (
    "tick",
    "timestamp_s",
    "rgb",
    "depth_m",
    "camera_intrinsics",
    "world_from_camera_cv",
    "T_robot_base_from_world",
    "joint_position",
    "joint_velocity",
    "eef_position",
    "eef_quaternion_xyzw",
    "gripper_position",
    "gripper_velocity",
)
_ACTION_DATASETS = ("value", "issued_tick", "applied_tick")
_SUPERVISION_DATASETS = (
    "geom_id",
    "world_from_geom",
    "target_object_pose_world",
    "flow_2d_px",
    "flow_valid",
    "flow_future_visible",
    "reward",
    "done",
    "success",
    "event_flags",
)

_METADATA_FIELDS = {
    "episode_id",
    "protocol_version",
    "task_id",
    "scenario_id",
    "partition",
    "factor",
    "repeat_index",
    "scenario_spec",
    "observation_profile",
    "action_spec",
    "control_frequency_hz",
    "view_names",
    "geom_table",
    "event_names",
    "event_times_s",
    "outcome",
    "software_versions",
}
_ACTION_SPEC_FIELDS = {"mode", "representation", "motion_limits"}
_GEOM_FIELDS = {
    "geom_id",
    "geom_name",
    "body_identity",
    "instance_identity",
    "semantic_identity",
}
_OUTCOME_FIELDS = {"success", "termination_reason", "failure_code"}
_SOFTWARE_FIELDS = {"benchmark", "robosuite", "mujoco"}
_WINDOWS_SAFE_EPISODE_ID = re.compile(r"[A-Za-z0-9._%~-]+")

_DType = TypeVar("_DType", bound=np.generic)


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate episode metadata field: {name}")
        result[name] = value
    return result


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], name: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} unknown fields: {sorted(unknown)}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _string_sequence(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be a non-empty-string list")
    result = tuple(value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


def _episode_id(scenario_id: str, repeat_index: int) -> str:
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    repeat = _nonnegative_integer(repeat_index, "repeat_index")
    return f"{quote(scenario_id, safe='._~-')}~r{repeat:06d}"


def _validate_geom_table(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("geom_table must be a non-empty list")
    result: list[Mapping[str, object]] = []
    for expected_id, raw in enumerate(value):
        entry = _mapping(raw, "geom_table entry")
        _exact_fields(entry, _GEOM_FIELDS, "geom_table entry")
        if entry["geom_id"] != expected_id:
            raise ValueError("geom_table geom_id must be contiguous and zero-based")
        for field in _GEOM_FIELDS - {"geom_id"}:
            _text(entry[field], f"geom_table {field}")
        result.append(entry)
    geom_names = tuple(cast(str, entry["geom_name"]) for entry in result)
    if len(geom_names) != len(set(geom_names)):
        raise ValueError("geom_table geom_name values must be unique")
    return tuple(result)


def _validate_metadata_document(value: object) -> Mapping[str, object]:
    document = _mapping(value, "episode metadata")
    _exact_fields(document, _METADATA_FIELDS, "episode metadata")
    episode_id = _text(document["episode_id"], "episode_id")
    if _WINDOWS_SAFE_EPISODE_ID.fullmatch(episode_id) is None:
        raise ValueError("episode_id must use the deterministic Windows-safe encoding")
    if document["protocol_version"] != "v1":
        raise ValueError("episode metadata protocol_version must be v1")
    task_id = _text(document["task_id"], "task_id")
    if task_id not in TASK_FAMILIES:
        raise ValueError(f"unknown episode task_id: {task_id}")
    scenario_id = _text(document["scenario_id"], "scenario_id")
    partition = document["partition"]
    factor = document["factor"]
    if partition not in ("train", "test_id"):
        raise ValueError("episode metadata has an invalid partition")
    if factor is not None:
        raise ValueError("episode metadata factor must be null")
    expected = rf"v1:{partition}:{re.escape(task_id)}:\d{{6}}"
    if re.fullmatch(expected, scenario_id) is None:
        raise ValueError("episode scenario_id does not match task/partition/factor")
    repeat_index = _nonnegative_integer(document["repeat_index"], "repeat_index")
    if episode_id != _episode_id(scenario_id, repeat_index):
        raise ValueError("episode_id must encode the scenario identity and repeat_index")
    observation_profile = document["observation_profile"]
    if observation_profile not in ("agentview", "agentview_wrist"):
        raise ValueError("invalid episode observation_profile")
    view_names = _string_sequence(document["view_names"], "view_names")
    expected_views = (
        ("agentview",) if observation_profile == "agentview" else ("agentview", "wrist")
    )
    if view_names != expected_views:
        raise ValueError("view_names do not match observation_profile")
    event_names = _string_sequence(document["event_names"], "event_names")

    event_times = _mapping(document["event_times_s"], "event_times_s")
    for event_name, time_s in event_times.items():
        _text(event_name, "event name")
        if (
            isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not math.isfinite(time_s)
            or time_s < 0.0
        ):
            raise ValueError("event_times_s values must be finite and non-negative")

    action = _mapping(document["action_spec"], "action_spec")
    _exact_fields(action, _ACTION_SPEC_FIELDS, "action_spec")
    mode = action["mode"]
    representation = action["representation"]
    if mode not in ("E7", "J8"):
        raise ValueError("action_spec mode must be E7 or J8")
    if representation not in ("delta", "absolute"):
        raise ValueError("action_spec representation must be delta or absolute")
    limits = action["motion_limits"]
    if not isinstance(limits, list) or any(
        isinstance(limit, bool) or not isinstance(limit, (int, float)) for limit in limits
    ):
        raise ValueError("action_spec motion_limits must be a numeric list")
    action_spec = ActionSpec(
        mode,
        representation,
        cast(tuple[float, ...], tuple(limits)),
    )

    control_frequency = document["control_frequency_hz"]
    if isinstance(control_frequency, bool) or control_frequency != 20:
        raise ValueError("episode control_frequency_hz must be the frozen 20 Hz")
    scenario_value = _mapping(document["scenario_spec"], "scenario_spec")
    scenario = scenario_from_dict(scenario_value, family=TASK_FAMILIES[task_id])
    if scenario_to_dict(scenario) != scenario_value:
        raise ValueError("scenario_spec metadata is not in canonical value form")
    if (
        scenario.observation_profile != observation_profile
        or scenario.control_frequency_hz != control_frequency
        or scenario.action_mode != action_spec.mode
    ):
        raise ValueError("scenario_spec does not match episode observation/action metadata")

    geom_table = _validate_geom_table(document["geom_table"])
    expected_assets = {entity.instance_id: entity.asset_id for entity in scenario.object_states}
    expected_assets.update(
        {entity.instance_id: entity.asset_id for entity in scenario.fixture_states}
    )
    observed_instances = {cast(str, entry["instance_identity"]) for entry in geom_table}
    if not expected_assets.keys() <= observed_instances:
        raise ValueError("geom_table must cover every concrete scenario entity")
    for entry in geom_table:
        instance_identity = cast(str, entry["instance_identity"])
        if (
            instance_identity in expected_assets
            and entry["semantic_identity"] != expected_assets[instance_identity]
        ):
            raise ValueError("geom_table scenario semantics must match concrete asset identity")
    outcome = _mapping(document["outcome"], "outcome")
    _exact_fields(outcome, _OUTCOME_FIELDS, "outcome")
    success = outcome["success"]
    reason = outcome["termination_reason"]
    failure_code = outcome["failure_code"]
    if not isinstance(success, bool):
        raise ValueError("outcome success must be bool")
    if success:
        if reason != "success" or failure_code is not None:
            raise ValueError("successful outcome must use success and null failure_code")
    else:
        if not isinstance(failure_code, str) or failure_code not in FAILURE_CODES:
            raise ValueError("failed outcome requires a frozen failure_code")
        expected_reason = "timeout" if failure_code == "timeout" else "terminal_failure"
        if reason != expected_reason:
            raise ValueError("failed outcome termination_reason does not match failure_code")

    software = _mapping(document["software_versions"], "software_versions")
    _exact_fields(software, _SOFTWARE_FIELDS, "software_versions")
    for field in _SOFTWARE_FIELDS:
        _text(software[field], f"software_versions {field}")
    if not geom_table or not event_names:
        raise AssertionError("validated episode identity tables cannot be empty")
    return document


def _parse_metadata_json(value: object) -> Mapping[str, object]:
    if not isinstance(value, str):
        raise ValueError("metadata/episode_json must decode to a string")
    try:
        document = json.loads(value, object_pairs_hook=_strict_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata/episode_json must contain valid JSON") from exc
    return _validate_metadata_document(document)


def _require_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[_DType],
    shape: tuple[int, ...],
    finite: bool = True,
) -> npt.NDArray[_DType]:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
        raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _validate_quaternion_sequence(
    value: npt.NDArray[np.float32] | npt.NDArray[np.float64], name: str
) -> None:
    norms = np.linalg.norm(value, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} must contain unit quaternions")
    if value[0, 3] < 0.0 or np.any(np.sum(value[:-1] * value[1:], axis=1) < 0.0):
        raise ValueError(f"{name} must use the frozen continuous quaternion sign")


def _validate_applied_ticks(issued: npt.NDArray[np.int64], applied: npt.NDArray[np.int64]) -> None:
    pending = np.flatnonzero(applied == -1)
    if pending.size and np.any(applied[int(pending[0]) :] != -1):
        raise ValueError("unapplied Actions must form a terminal suffix")
    executed = applied[applied >= 0]
    if (
        np.any(applied < -1)
        or np.any(executed < issued[: executed.size])
        or np.any(executed >= issued.size)
        or (executed.size > 1 and np.any(np.diff(executed) <= 0))
    ):
        raise ValueError("applied_tick must preserve actual ordered Action execution")


_RASTER_FIELDS = frozenset(
    {"rgb", "depth_m", "geom_id", "flow_2d_px", "flow_valid", "flow_future_visible"}
)


class _ArraySource(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[np.generic]: ...

    def __getitem__(self, key: slice | EllipsisType) -> object: ...


def _validate_episode_sources(
    sources: Mapping[str, _ArraySource], metadata: Mapping[str, object]
) -> None:
    """Check dataset values with bounded HDF5 reads."""
    document = _validate_metadata_document(metadata)
    view_names = cast(list[str], document["view_names"])
    geom_table = cast(list[Mapping[str, object]], document["geom_table"])
    event_names = cast(list[str], document["event_names"])
    action_spec = cast(Mapping[str, object], document["action_spec"])
    action_mode = cast(ActionMode, action_spec["mode"])
    action_representation = cast(ActionRepresentation, action_spec["representation"])
    views, geoms, events = len(view_names), len(geom_table), len(event_names)
    frames = sources["tick"].shape[0] if sources["tick"].shape else -1
    transitions = frames - 1
    action_dimension = 7 if action_spec["mode"] == "E7" else 8
    if transitions <= 0:
        raise ValueError("episode must contain at least one transition")
    contracts: tuple[tuple[str, str, np.dtype[np.generic], tuple[int, ...]], ...] = (
        ("policy/tick", "tick", np.dtype(np.int64), (frames,)),
        ("policy/timestamp_s", "timestamp_s", np.dtype(np.float64), (frames,)),
        ("policy/rgb", "rgb", np.dtype(np.uint8), (frames, views, 224, 224, 3)),
        ("policy/depth_m", "depth_m", np.dtype(np.float32), (frames, views, 224, 224)),
        (
            "policy/camera_intrinsics",
            "camera_intrinsics",
            np.dtype(np.float64),
            (frames, views, 3, 3),
        ),
        (
            "policy/world_from_camera_cv",
            "world_from_camera_cv",
            np.dtype(np.float64),
            (frames, views, 4, 4),
        ),
        (
            "policy/T_robot_base_from_world",
            "T_robot_base_from_world",
            np.dtype(np.float64),
            (4, 4),
        ),
        ("policy/joint_position", "joint_position", np.dtype(np.float32), (frames, 7)),
        ("policy/joint_velocity", "joint_velocity", np.dtype(np.float32), (frames, 7)),
        ("policy/eef_position", "eef_position", np.dtype(np.float32), (frames, 3)),
        (
            "policy/eef_quaternion_xyzw",
            "eef_quaternion_xyzw",
            np.dtype(np.float32),
            (frames, 4),
        ),
        ("policy/gripper_position", "gripper_position", np.dtype(np.float32), (frames, 2)),
        ("policy/gripper_velocity", "gripper_velocity", np.dtype(np.float32), (frames, 2)),
        (
            "action/value",
            "action_value",
            np.dtype(np.float64),
            (transitions, action_dimension),
        ),
        ("action/issued_tick", "issued_tick", np.dtype(np.int64), (transitions,)),
        ("action/applied_tick", "applied_tick", np.dtype(np.int64), (transitions,)),
        ("supervision/geom_id", "geom_id", np.dtype(np.int32), (frames, views, 224, 224)),
        (
            "supervision/world_from_geom",
            "world_from_geom",
            np.dtype(np.float64),
            (frames, geoms, 4, 4),
        ),
        (
            "supervision/target_object_pose_world",
            "target_object_pose_world",
            np.dtype(np.float64),
            (frames, 7),
        ),
        (
            "supervision/flow_2d_px",
            "flow_2d_px",
            np.dtype(np.float32),
            (transitions, views, 224, 224, 2),
        ),
        (
            "supervision/flow_valid",
            "flow_valid",
            np.dtype(np.bool_),
            (transitions, views, 224, 224),
        ),
        (
            "supervision/flow_future_visible",
            "flow_future_visible",
            np.dtype(np.bool_),
            (transitions, views, 224, 224),
        ),
        ("supervision/reward", "reward", np.dtype(np.float32), (transitions,)),
        ("supervision/done", "done", np.dtype(np.bool_), (transitions,)),
        ("supervision/success", "success", np.dtype(np.bool_), (frames,)),
        ("supervision/event_flags", "event_flags", np.dtype(np.bool_), (frames, events)),
    )
    values: dict[str, npt.NDArray[Any]] = {}
    flow_fields = {"flow_2d_px", "flow_valid", "flow_future_visible"}
    for name, field, dtype, shape in contracts:
        source = sources[field]
        if source.dtype != dtype or source.shape != shape:
            raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
        if field in flow_fields:
            continue
        if field in _RASTER_FIELDS:
            for index in range(shape[0]):
                chunk: npt.NDArray[Any] = _require_array(
                    source[index : index + 1], name=name, dtype=dtype, shape=(1, *shape[1:])
                )
                if field == "depth_m" and np.any(chunk < 0.0):
                    raise ValueError("policy depth must be non-negative")
                if field == "geom_id" and (np.any(chunk < -1) or np.any(chunk >= geoms)):
                    raise ValueError("geom_id must use background -1 or a compiled geom index")
        else:
            values[field] = _require_array(source[...], name=name, dtype=dtype, shape=shape)
    for index in range(transitions):
        flow = _require_array(
            sources["flow_2d_px"][index : index + 1],
            name="supervision/flow_2d_px",
            dtype=np.dtype(np.float32),
            shape=(1, views, 224, 224, 2),
        )
        valid = _require_array(
            sources["flow_valid"][index : index + 1],
            name="supervision/flow_valid",
            dtype=np.dtype(np.bool_),
            shape=(1, views, 224, 224),
        )
        visible = _require_array(
            sources["flow_future_visible"][index : index + 1],
            name="supervision/flow_future_visible",
            dtype=np.dtype(np.bool_),
            shape=(1, views, 224, 224),
        )
        if np.any(visible & ~valid):
            raise ValueError("flow_future_visible must be a subset of flow_valid")
        if np.any(flow[~valid] != 0.0):
            raise ValueError("invalid optical flow values must be exactly zero")
    if not np.array_equal(values["tick"], np.arange(frames, dtype=np.int64)):
        raise ValueError("policy/tick must be contiguous and zero-based")
    expected_time = values["tick"].astype(np.float64) / cast(int, document["control_frequency_hz"])
    if not np.array_equal(values["timestamp_s"], expected_time):
        raise ValueError("policy timestamps must align exactly with control ticks")
    if not np.array_equal(values["issued_tick"], np.arange(transitions, dtype=np.int64)):
        raise ValueError("issued_tick must be contiguous and zero-based")
    spec = ActionSpec(
        action_mode,
        action_representation,
        tuple(cast(list[float], action_spec["motion_limits"])),
    )
    for raw_action in values["action_value"]:
        action = np.ascontiguousarray(raw_action, dtype=np.float64)
        action.setflags(write=False)
        validate_action_spec(BenchmarkAction(action_mode, action_representation, action), spec)
    _validate_applied_ticks(values["issued_tick"], values["applied_tick"])
    scenario = cast(Mapping[str, object], document["scenario_spec"])
    action_delay_steps = cast(int, scenario["action_delay_steps"])
    expected_applied = values["issued_tick"] + action_delay_steps
    expected_applied[expected_applied >= transitions] = -1
    if not np.array_equal(values["applied_tick"], expected_applied):
        raise ValueError("applied_tick does not match the frozen Action delay")
    scenario_spec = scenario_from_dict(
        scenario,
        family=TASK_FAMILIES[cast(str, document["task_id"])],
    )
    expected_intrinsics = np.stack(
        [
            _camera_intrinsics(
                view.vertical_fov_deg,
                scenario_spec.cameras.image_height_px,
                scenario_spec.cameras.image_width_px,
            )
            for view in scenario_spec.cameras.views
        ]
    )
    if not np.array_equal(
        values["camera_intrinsics"],
        np.broadcast_to(expected_intrinsics, values["camera_intrinsics"].shape),
    ):
        raise ValueError("camera_intrinsics must match the concrete ScenarioSpec calibration")
    _validate_rigid_transform(values["world_from_camera_cv"], "policy/world_from_camera_cv")
    _validate_rigid_transform(values["T_robot_base_from_world"], "policy/T_robot_base_from_world")
    _validate_rigid_transform(values["world_from_geom"], "supervision/world_from_geom")
    _validate_quaternion_sequence(values["eef_quaternion_xyzw"], "policy/eef_quaternion_xyzw")
    _validate_quaternion_sequence(
        values["target_object_pose_world"][:, 3:],
        "supervision/target_object_pose_world",
    )
    if transitions > 1 and np.any(values["done"][:-1]):
        raise ValueError("only the final transition may be done")
    if not bool(values["done"][-1]):
        raise ValueError("the final transition must be done")
    outcome = cast(Mapping[str, object], document["outcome"])
    if bool(values["success"][-1]) != bool(outcome["success"]):
        raise ValueError("terminal frame success does not match metadata outcome")
    if np.any(values["success"][:-1]):
        raise ValueError("success may become true only on the terminal frame")
    if np.any(values["event_flags"][:-1] & ~values["event_flags"][1:]):
        raise ValueError("event_flags must be cumulative across frames")
    if np.any(values["event_flags"][0]):
        raise ValueError("reset frame event_flags must all be false")
    if bool(outcome["success"]) and not np.all(values["event_flags"][-1]):
        raise ValueError("successful terminal frame must complete every formal event")


__all__ = ["SCHEMA_VERSION"]
