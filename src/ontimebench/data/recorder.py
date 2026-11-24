"""Record OnTimeBench episodes in HDF5 with a robosuite wrapper."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, TypeVar, cast

import h5py
import numpy as np
import numpy.typing as npt
from robosuite.wrappers.wrapper import Wrapper

from ontimebench.control.action import ActionSpec, BenchmarkAction
from ontimebench.data.schema import (
    _ACTION_DATASETS,
    _POLICY_DATASETS,
    _RASTER_FIELDS,
    _SUPERVISION_DATASETS,
    SCHEMA_VERSION,
    _ArraySource,
    _episode_id,
    _parse_metadata_json,
    _validate_episode_sources,
)
from ontimebench.data.training import TrainingEpisode
from ontimebench.envs.base import DynamicManipulationEnv
from ontimebench.scenarios.spec import ScenarioSpec, scenario_to_dict
from ontimebench.sensing.observation import BenchmarkObservation
from ontimebench.supervision.optical_flow import _exact_dense_optical_flow
from ontimebench.supervision.snapshot import (
    SupervisionSnapshot,
    _assemble_supervision_snapshot,
)
from ontimebench.tasks.spec import TaskSpec

_DType = TypeVar("_DType", bound=np.generic)


class _H5Attributes(Protocol):
    def __getitem__(self, name: str) -> object: ...

    def __setitem__(self, name: str, value: object) -> None: ...

    def keys(self) -> Iterable[str]: ...


class _H5Dataset(Protocol):
    @property
    def attrs(self) -> _H5Attributes: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[np.generic]: ...

    @property
    def chunks(self) -> tuple[int, ...] | None: ...

    @property
    def compression(self) -> str | None: ...

    @property
    def shuffle(self) -> bool: ...

    def __getitem__(self, key: object) -> object: ...

    def asstr(self) -> _H5Dataset: ...

    def __setitem__(self, key: object, value: object) -> None: ...

    def resize(self, size: int, axis: int) -> None: ...


class _H5Group(Protocol):
    @property
    def attrs(self) -> _H5Attributes: ...

    def keys(self) -> Iterable[str]: ...

    def __getitem__(self, name: str) -> object: ...

    def create_group(self, name: str) -> _H5Group: ...

    def create_dataset(self, name: str, *, data: object, **options: object) -> _H5Dataset: ...


class _H5File(_H5Group, Protocol):
    def close(self) -> None: ...


class _H5FileFactory(Protocol):
    def __call__(self, path: str | Path, mode: str) -> _H5File: ...


class _H5StringDtypeFactory(Protocol):
    def __call__(self, *, encoding: str) -> np.dtype[np.generic]: ...


class _H5StringInfo(Protocol):
    @property
    def encoding(self) -> str: ...

    @property
    def length(self) -> int | None: ...


class _H5CheckStringDtype(Protocol):
    def __call__(self, dtype: np.dtype[np.generic]) -> _H5StringInfo | None: ...


_H5_FILE = cast(_H5FileFactory, h5py.File)
_H5_STRING_DTYPE = cast(_H5StringDtypeFactory, h5py.string_dtype)
_H5_CHECK_STRING_DTYPE = cast(_H5CheckStringDtype, h5py.h5t.check_string_dtype)


def _open_hdf5(path: Path, mode: str) -> _H5File:
    return _H5_FILE(path, mode)


def _required_group(parent: _H5Group, name: str) -> _H5Group:
    value = parent[name]
    if not hasattr(value, "create_dataset"):
        raise ValueError(f"{name} must be an HDF5 group")
    return cast(_H5Group, value)


def _required_dataset(parent: _H5Group, name: str) -> _H5Dataset:
    value = parent[name]
    if not hasattr(value, "shape"):
        raise ValueError(f"{name} must be an HDF5 dataset")
    return cast(_H5Dataset, value)


def _exact_keys(
    value: _H5Group,
    expected: tuple[str, ...],
    name: str,
    *,
    allow_attributes: bool = False,
) -> None:
    actual = tuple(value.keys())
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError(f"{name} has an invalid frozen dataset set")
    if not allow_attributes and tuple(value.attrs.keys()):
        raise ValueError(f"{name} must not have HDF5 attributes")


def _dataset_options(
    value: _ArraySource, *, frame_view: bool = False, constant: bool = False
) -> dict[str, object]:
    if constant:
        return {}
    if frame_view:
        chunks = (1, 1, *value.shape[2:])
    else:
        chunks = (min(64, value.shape[0]), *value.shape[1:])
    return {
        "chunks": chunks,
        "compression": "lzf",
        "shuffle": value.dtype.itemsize > 1,
    }


def _write_dataset(
    group: _H5Group,
    name: str,
    value: npt.NDArray[np.generic],
    *,
    frame_view: bool = False,
    constant: bool = False,
) -> None:
    group.create_dataset(
        name,
        data=value,
        **_dataset_options(value, frame_view=frame_view, constant=constant),
    )


def _expected_storage(
    value: _ArraySource, *, frame_view: bool = False, constant: bool = False
) -> tuple[tuple[int, ...] | None, str | None, bool]:
    options = _dataset_options(value, frame_view=frame_view, constant=constant)
    return (
        cast(tuple[int, ...] | None, options.get("chunks")),
        cast(str | None, options.get("compression")),
        cast(bool, options.get("shuffle", False)),
    )


def _stack(values: list[npt.NDArray[_DType]], dtype: np.dtype[_DType]) -> npt.NDArray[_DType]:
    return np.ascontiguousarray(np.stack(values), dtype=dtype)


def _recording_stage_facts(
    value: object,
) -> tuple[tuple[str, bool, float, int], ...]:
    if not isinstance(value, tuple):
        raise RuntimeError("recording stage facts must be a tuple")
    result: list[tuple[str, bool, float, int]] = []
    for raw in value:
        if not isinstance(raw, tuple) or len(raw) != 4:
            raise RuntimeError("recording stage facts must be four-value tuples")
        name, completed, progress, consecutive_steps = raw
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(completed, bool)
            or isinstance(progress, bool)
            or not isinstance(progress, (int, float))
            or not np.isfinite(progress)
            or isinstance(consecutive_steps, bool)
            or not isinstance(consecutive_steps, int)
            or consecutive_steps < 0
        ):
            raise RuntimeError("recording stage facts contain an invalid value")
        result.append((name, completed, float(progress), consecutive_steps))
    names = tuple(item[0] for item in result)
    if not names or len(names) != len(set(names)):
        raise RuntimeError("recording stage names must be non-empty and unique")
    return tuple(result)


def _as_info(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RuntimeError("recorded Family Env info must be a string-keyed mapping")
    return value


def _append_raster(group: _H5Group, name: str, value: npt.NDArray[np.generic]) -> None:
    """Append one frame or flow transition using the unchanged schema-1 raster layout."""
    if name not in group.keys():
        group.create_dataset(
            name,
            data=value[None, ...],
            maxshape=(None, *value.shape),
            **_dataset_options(value[None, ...], frame_view=True),
        )
    else:
        dataset = _required_dataset(group, name)
        index = dataset.shape[0]
        dataset.resize(index + 1, axis=0)
        dataset[index] = value


class EpisodeRecorder(Wrapper):
    """Record one complete real episode without owning another interaction loop."""

    def __init__(
        self,
        environment: DynamicManipulationEnv,
        directory: str | Path,
        *,
        task: TaskSpec,
        scenario: ScenarioSpec,
        scenario_id: str,
        partition: str,
        factor: str | None,
        repeat_index: int,
        training: TrainingEpisode | None = None,
    ) -> None:
        if not isinstance(environment, DynamicManipulationEnv):
            raise ValueError("EpisodeRecorder requires a DynamicManipulationEnv")
        if environment._task_spec != task or environment._scenario_spec != scenario:
            raise ValueError("EpisodeRecorder task/scenario must match the compiled Family Env")
        destination_directory = Path(directory)
        if not destination_directory.is_dir():
            raise ValueError("EpisodeRecorder destination directory must already exist")
        self._environment = environment
        self._task = task
        self._scenario = scenario
        self._scenario_id = scenario_id
        self._partition = partition
        self._factor = factor
        self._repeat_index = repeat_index
        self._training = training
        self._episode_id = _episode_id(scenario_id, repeat_index)
        self._destination = destination_directory / (
            training.path if training is not None else f"{self._episode_id}.hdf5"
        )
        self._destination.parent.mkdir(parents=True, exist_ok=True)
        self._partial = self._destination.with_suffix(".partial.hdf5")
        if self._destination.exists() or self._partial.exists():
            raise FileExistsError(
                self._destination if self._destination.exists() else self._partial
            )
        partial = _open_hdf5(self._partial, "x")
        try:
            partial.attrs["schema_version"] = SCHEMA_VERSION
            partial.create_group("policy")
            partial.create_group("action")
            partial.create_group("supervision")
        finally:
            partial.close()
        self._frame_count = 0
        self._frame_values: dict[str, list[npt.NDArray[np.generic]]] = {}
        self._last_snapshot: SupervisionSnapshot | None = None
        self._base_transform: npt.NDArray[np.float64] | None = None
        self._view_names: tuple[str, ...] = ()
        self._actions: list[BenchmarkAction] = []
        self._applied_ticks: list[int | None] = []
        self._rewards: list[float] = []
        self._done: list[bool] = []
        self._frame_success: list[bool] = []
        self._frame_failure_code: list[str | None] = []
        self._frame_terminal_time_s: list[float | None] = []
        self._frame_stage_facts: list[tuple[tuple[str, bool, float, int], ...]] = []
        self._reset_returned = False
        self._pending_step: tuple[float, bool, Mapping[str, object]] | None = None
        self._event_times_s: dict[str, float] = {}
        self._finalized = False
        wrapper_init = cast(Callable[[Wrapper, object], None], Wrapper.__init__)
        wrapper_init(self, environment)
        environment._recording_observations = True

    @property
    def destination(self) -> Path:
        return self._destination

    def reset(self) -> Mapping[str, object]:
        if self._reset_returned or self._frame_count or self._actions:
            raise RuntimeError("EpisodeRecorder records exactly one reset and episode")
        raw = self._environment.reset()
        if not isinstance(raw, Mapping):
            raise RuntimeError("Family Env reset observation must be a mapping")
        self._reset_returned = True
        if self._training is not None:
            self._environment.robots[
                0
            ].composite_controller.record_arm_goal = self._training.capture_goal
        return raw

    def _capture_snapshot(self, observation: BenchmarkObservation) -> SupervisionSnapshot:
        geom_table = self._environment._recording_geoms()
        rendered = self._environment._recording_segmentation
        if rendered is None or rendered[0] != observation.tick:
            raise RuntimeError("recording requires the current observation's segmentation")
        segmentation = rendered[1]
        self._environment._recording_segmentation = None
        previous = (
            None
            if self._last_snapshot is None
            else self._last_snapshot.target_object_pose_world[3:]
        )
        return _assemble_supervision_snapshot(
            observation,
            self._environment.sim.model._model,
            self._environment.sim.data._data,
            element_segmentation_opengl=segmentation,
            target_pose_world=self._environment._dynamic_target_snapshot(),
            geom_semantic_identity=tuple(item[4] for item in geom_table),
            previous_target_quaternion_xyzw=previous,
        )

    def _current_frame_facts(
        self,
    ) -> tuple[bool, str | None, float | None, tuple[tuple[str, bool, float, int], ...]]:
        success, failure_code, terminal_time_s, raw_stages = (
            self._environment._recording_frame_facts()
        )
        if not isinstance(success, bool):
            raise RuntimeError("recording success fact must be bool")
        if failure_code is not None and not isinstance(failure_code, str):
            raise RuntimeError("recording failure_code fact must be a string or None")
        if terminal_time_s is not None and (
            isinstance(terminal_time_s, bool)
            or not isinstance(terminal_time_s, (int, float))
            or not np.isfinite(terminal_time_s)
            or terminal_time_s < 0.0
        ):
            raise RuntimeError("recording terminal time must be finite and non-negative")
        return (
            success,
            failure_code,
            None if terminal_time_s is None else float(terminal_time_s),
            _recording_stage_facts(raw_stages),
        )

    def _append_frame(self, observation: BenchmarkObservation) -> None:
        if observation.tick != self._frame_count:
            raise RuntimeError("recording frames must be contiguous and zero-based")
        snapshot = self._capture_snapshot(observation)
        success, failure_code, terminal_time_s, stages = self._current_frame_facts()
        if self._frame_stage_facts and tuple(item[0] for item in stages) != tuple(
            item[0] for item in self._frame_stage_facts[0]
        ):
            raise RuntimeError("recording stage vocabulary changed within an episode")
        output = _open_hdf5(self._partial, "r+")
        try:
            policy = _required_group(output, "policy")
            supervision = _required_group(output, "supervision")
            _append_raster(policy, "rgb", observation.rgb)
            depth = observation.depth_m
            if self._training is not None:
                depth = np.where(snapshot.geom_id >= 0, depth, np.float32(0))
            _append_raster(policy, "depth_m", depth)
            _append_raster(supervision, "geom_id", snapshot.geom_id)
            if self._last_snapshot is not None:
                flow, valid, visible = _exact_dense_optical_flow(self._last_snapshot, snapshot)
                _append_raster(supervision, "flow_2d_px", flow)
                _append_raster(supervision, "flow_valid", valid)
                _append_raster(supervision, "flow_future_visible", visible)
        finally:
            output.close()
        for name in _POLICY_DATASETS:
            if name not in _RASTER_FIELDS and name != "T_robot_base_from_world":
                value = np.asarray(
                    getattr(observation, name), dtype=np.int64 if name == "tick" else None
                )
                self._frame_values.setdefault(name, []).append(value)
        for name in ("world_from_geom", "target_object_pose_world"):
            self._frame_values.setdefault(name, []).append(getattr(snapshot, name))
        if self._frame_count == 0:
            self._base_transform = observation.T_robot_base_from_world
            self._view_names = observation.view_names
        self._last_snapshot = snapshot
        self._frame_count += 1
        self._frame_success.append(success)
        self._frame_failure_code.append(failure_code)
        self._frame_terminal_time_s.append(terminal_time_s)
        self._frame_stage_facts.append(stages)

    def capture_reset_frame(self, observation: BenchmarkObservation) -> None:
        if not self._reset_returned or self._frame_count or self._pending_step is not None:
            raise RuntimeError("reset frame must be captured exactly once immediately after reset")
        if observation.tick != 0:
            raise RuntimeError("reset frame must use scored observation tick zero")
        self._append_frame(observation)
        if (
            self._frame_success[0]
            or self._frame_failure_code[0] is not None
            or self._frame_terminal_time_s[0] is not None
        ):
            raise RuntimeError("reset frame cannot already carry a terminal outcome")

    def step(
        self, action: BenchmarkAction
    ) -> tuple[Mapping[str, object], float, bool, Mapping[str, object]]:
        if (
            not self._frame_count
            or self._pending_step is not None
            or (self._done and self._done[-1])
        ):
            raise RuntimeError("EpisodeRecorder step lifecycle is invalid")
        issued_tick = len(self._actions)
        raw_observation, reward, done, raw_info = self._environment.step(action)
        if not isinstance(raw_observation, Mapping):
            raise RuntimeError("Family Env step observation must be a mapping")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not np.isfinite(reward)
        ):
            raise RuntimeError("Family Env reward must be finite")
        if not isinstance(done, bool):
            raise RuntimeError("Family Env done fact must be bool")
        info = _as_info(raw_info)
        events = info.get("events")
        if not isinstance(events, Mapping):
            raise RuntimeError("recorded Family events must be a mapping")
        history: dict[str, float] = {}
        for name, time_s in events.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(time_s, bool)
                or not isinstance(time_s, (int, float))
                or not np.isfinite(time_s)
                or time_s < 0.0
            ):
                raise RuntimeError("recorded event names and times must be valid")
            history[name] = float(time_s)
        previous = tuple(self._event_times_s.items())
        if tuple(history.items())[: len(previous)] != previous:
            raise RuntimeError("recorded events must preserve the ordered history")
        self._event_times_s = history
        self._actions.append(action)
        self._applied_ticks.append(None)
        application = self._environment._last_action_application_fact()
        if application is not None:
            application_issued_tick, application_tick, application_action = application
            if application_tick != issued_tick:
                raise RuntimeError("recorded applied Action must name the current execution tick")
            if not 0 <= application_issued_tick < len(self._actions):
                raise RuntimeError("recorded applied Action names an unknown issued tick")
            if self._actions[application_issued_tick] is not application_action:
                raise RuntimeError("recorded applied Action does not match the issued command")
            if self._applied_ticks[application_issued_tick] is not None:
                raise RuntimeError("recorded issued Action cannot execute more than once")
            self._applied_ticks[application_issued_tick] = application_tick
        self._pending_step = (float(reward), bool(done), info)
        return raw_observation, float(reward), bool(done), info

    def capture_post_step_frame(self, observation: BenchmarkObservation) -> None:
        if self._pending_step is None:
            raise RuntimeError("post-step frame requires one preceding recorded step")
        reward, done, info = self._pending_step
        self._append_frame(observation)
        success = self._frame_success[-1]
        failure_code = self._frame_failure_code[-1]
        terminal_time_s = self._frame_terminal_time_s[-1]
        stages = self._frame_stage_facts[-1]
        expected_info = {
            "success": success,
            "failure_code": failure_code,
            "terminal_time_s": terminal_time_s,
            "stage_facts": stages,
        }
        for name, expected in expected_info.items():
            if name not in info or info[name] != expected:
                raise RuntimeError(f"recording frame fact disagrees with Family Env info {name!r}")
        if done != (terminal_time_s is not None):
            raise RuntimeError("recording done fact disagrees with Family terminal time")
        if done and success == (failure_code is not None):
            raise RuntimeError("recording terminal outcome facts are inconsistent")
        if not done and (success or failure_code is not None):
            raise RuntimeError("recording saw terminal facts before done")
        self._rewards.append(reward)
        self._done.append(done)
        self._pending_step = None

    def _small_arrays(self) -> dict[str, npt.NDArray[np.generic]]:
        if self._frame_count != len(self._actions) + 1 or self._base_transform is None:
            raise RuntimeError("recording requires exactly N+1 frames for N Actions")
        arrays = {name: np.stack(values) for name, values in self._frame_values.items()}
        arrays.update(
            T_robot_base_from_world=self._base_transform,
            action_value=_stack([item.value for item in self._actions], np.dtype(np.float64)),
            issued_tick=np.arange(len(self._actions), dtype=np.int64),
            applied_tick=np.asarray(
                [-1 if item is None else item for item in self._applied_ticks], dtype=np.int64
            ),
            reward=np.asarray(self._rewards, dtype=np.float32),
            done=np.asarray(self._done, dtype=np.bool_),
            success=np.asarray(self._frame_success, dtype=np.bool_),
            event_flags=np.asarray(
                [tuple(stage[1] for stage in frame) for frame in self._frame_stage_facts],
                dtype=np.bool_,
            ),
        )
        return arrays

    def _metadata(
        self,
        *,
        action_spec: ActionSpec,
        event_names: tuple[str, ...],
        success: bool,
        failure_code: str | None,
    ) -> Mapping[str, object]:
        geom_table = self._environment._recording_geoms()
        if success:
            termination_reason = "success"
        elif failure_code == "timeout":
            termination_reason = "timeout"
        else:
            termination_reason = "terminal_failure"
        return {
            "episode_id": self._episode_id,
            "protocol_version": "v1",
            "task_id": self._task.task_id,
            "scenario_id": self._scenario_id,
            "partition": self._partition,
            "factor": self._factor,
            "repeat_index": self._repeat_index,
            "scenario_spec": scenario_to_dict(self._scenario),
            "observation_profile": self._scenario.observation_profile,
            "action_spec": {
                "mode": action_spec.mode,
                "representation": action_spec.representation,
                "motion_limits": list(action_spec.motion_limits),
            },
            "control_frequency_hz": self._scenario.control_frequency_hz,
            "view_names": list(self._view_names),
            "geom_table": [
                {
                    "geom_id": geom_id,
                    "geom_name": geom_name,
                    "body_identity": body_identity,
                    "instance_identity": instance_identity,
                    "semantic_identity": semantic_identity,
                }
                for (
                    geom_id,
                    geom_name,
                    body_identity,
                    instance_identity,
                    semantic_identity,
                ) in geom_table
            ],
            "event_names": list(event_names),
            "event_times_s": dict(self._event_times_s),
            "outcome": {
                "success": success,
                "termination_reason": termination_reason,
                "failure_code": failure_code,
            },
            "software_versions": {
                "benchmark": version("ontimebench"),
                "robosuite": version("robosuite"),
                "mujoco": version("mujoco"),
            },
        }

    def finalize_episode(
        self,
        *,
        action_spec: ActionSpec,
        event_names: tuple[str, ...],
        success: bool,
        failure_code: str | None,
        terminal_time_s: float,
    ) -> Path:
        if self._finalized or self._pending_step is not None:
            raise RuntimeError("EpisodeRecorder cannot finalize its current lifecycle")
        if not self._done or not self._done[-1]:
            raise RuntimeError("EpisodeRecorder requires a complete terminal episode")
        if tuple(stage[0] for stage in self._frame_stage_facts[0]) != event_names:
            raise RuntimeError("recording event vocabulary does not match EpisodeTrace")
        if any(
            (action.mode, action.representation) != (action_spec.mode, action_spec.representation)
            for action in self._actions
        ):
            raise RuntimeError("recorded Actions do not match the terminal ActionSpec")
        if (
            self._frame_success[-1] != success
            or self._frame_failure_code[-1] != failure_code
            or self._frame_terminal_time_s[-1] != terminal_time_s
        ):
            raise RuntimeError("recorded terminal facts do not match EpisodeTrace")
        arrays = self._small_arrays()
        if self._training is None:
            metadata = self._metadata(
                action_spec=action_spec,
                event_names=event_names,
                success=success,
                failure_code=failure_code,
            )
        else:
            if not success:
                raise ValueError(f"expert failed in {self._scenario_id}: {failure_code}")
            env = self._environment
            target = env.sim.model._model.body(env._dynamic_target_body_id).name
            metadata = self._training.prepare(
                arrays,
                action_spec=action_spec,
                view_names=self._view_names,
                frequency=self._scenario.control_frequency_hz,
                event_times=self._event_times_s,
                geoms=env._recording_geoms(),
                target_body=target,
            )
        output = _open_hdf5(self._partial, "r+")
        try:
            if self._training is not None:
                output.attrs["format"] = "OnTimeBench"
            metadata_group = output.create_group("metadata")
            metadata_group.create_dataset(
                "episode_json",
                data=json.dumps(
                    metadata, allow_nan=False, ensure_ascii=False, separators=(",", ":")
                ),
                dtype=_H5_STRING_DTYPE(encoding="utf-8"),
            )
            for name, value in arrays.items():
                if name in _POLICY_DATASETS:
                    group_name = "policy"
                elif name in {
                    "action_value",
                    "issued_tick",
                    "applied_tick",
                    "normalized_value",
                    "valid",
                }:
                    group_name = "action"
                else:
                    group_name = "supervision"
                _write_dataset(
                    _required_group(output, group_name),
                    "value" if name == "action_value" else name,
                    value,
                    frame_view=name in {"camera_intrinsics", "world_from_camera_cv"},
                    constant=name == "T_robot_base_from_world",
                )
        finally:
            output.close()
        if self._training is None:
            _validate_episode_file(self._partial)
        os.replace(self._partial, self._destination)
        self._finalized = True
        self._environment._recording_observations = False
        return self._destination


def _episode_file_sources(
    episode: _H5File,
) -> tuple[Mapping[str, object], dict[str, _ArraySource]]:
    if tuple(episode.attrs.keys()) != ("schema_version",):
        raise ValueError("HDF5 root must contain only schema_version")
    schema_version = episode.attrs["schema_version"]
    if not isinstance(schema_version, (int, np.integer)) or int(schema_version) != SCHEMA_VERSION:
        raise ValueError("HDF5 schema_version must be 1")
    _exact_keys(
        episode,
        ("metadata", "policy", "action", "supervision"),
        "HDF5 root",
        allow_attributes=True,
    )
    metadata_group = _required_group(episode, "metadata")
    _exact_keys(metadata_group, ("episode_json",), "metadata")
    metadata_dataset = _required_dataset(metadata_group, "episode_json")
    if metadata_dataset.shape != () or metadata_dataset.chunks is not None:
        raise ValueError("metadata/episode_json must be an uncompressed scalar")
    if metadata_dataset.compression is not None or metadata_dataset.shuffle:
        raise ValueError("metadata/episode_json must be uncompressed")
    string_info = _H5_CHECK_STRING_DTYPE(metadata_dataset.dtype)
    if string_info is None or string_info.encoding != "utf-8" or string_info.length is not None:
        raise ValueError("metadata/episode_json must use variable-length UTF-8")
    raw_metadata = metadata_dataset.asstr()[()]
    metadata = _parse_metadata_json(raw_metadata)

    policy = _required_group(episode, "policy")
    action = _required_group(episode, "action")
    supervision = _required_group(episode, "supervision")
    _exact_keys(policy, _POLICY_DATASETS, "policy")
    _exact_keys(action, _ACTION_DATASETS, "action")
    _exact_keys(supervision, _SUPERVISION_DATASETS, "supervision")
    sources: dict[str, _ArraySource] = {}
    for group, names in (
        (policy, _POLICY_DATASETS),
        (action, _ACTION_DATASETS),
        (supervision, _SUPERVISION_DATASETS),
    ):
        for name in names:
            dataset = _required_dataset(group, name)
            field = "action_value" if group is action and name == "value" else name
            if tuple(dataset.attrs.keys()):
                raise ValueError(f"{name} must not have HDF5 attributes")
            expected = _expected_storage(
                dataset,
                frame_view=field in _RASTER_FIELDS | {"camera_intrinsics", "world_from_camera_cv"},
                constant=field == "T_robot_base_from_world",
            )
            if (dataset.chunks, dataset.compression, dataset.shuffle) != expected:
                raise ValueError(f"{name} has an invalid frozen storage layout")
            sources[field] = dataset
    return metadata, sources


def _validate_episode_file(path: Path) -> None:
    episode = _open_hdf5(path, "r")
    try:
        metadata, sources = _episode_file_sources(episode)
        _validate_episode_sources(sources, metadata)
    finally:
        episode.close()


__all__: list[str] = []
