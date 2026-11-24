"""Pass a held wand loop along a rotating wire (T3)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import mujoco
import numpy as np
import numpy.typing as npt
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.base import MujocoXML
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask

from ontimebench.envs.base import (
    DynamicManipulationEnv,
    _apply_joint_physics,
    _distance_stage_progress,
    _duration_stage_progress,
    _fingerpad_contacts,
    _load_asset_catalog,
    _ordered_stage_facts,
    _physics_state_is_valid,
    _RecordingFrameFacts,
    _require_distance_stage_begin,
    _resolve_joint_physics,
    _rewrite_scenario_lighting_and_gravity_xml,
    _ScenarioResetRejected,
    _validate_compiled_joint_physics,
    _xml_numbers,
)
from ontimebench.envs.components.motion import (
    _motion_state_at_time,
    _validate_motion_initial_pose,
)
from ontimebench.scenarios.spec import (
    BodyPhysicsSpec,
    ConstantAngularVelocityRotation,
    ContactPhysicsSpec,
    FixedTranslation,
    ObjectStateSpec,
    RotatingInteractionGeometrySpec,
    ScenarioSpec,
    Vector2,
    Vector3,
    Vector4,
)
from ontimebench.tasks.spec import (
    FullRotatingSuccessSpec,
    PhaseSuccessSpec,
    RotatingInteractionBindings,
    TaskSpec,
)

_CONTROL_FREQUENCY_HZ = 20
_PREPARATION_TICKS = 20
_BILATERAL_HOLD_S = 0.05
_MAXIMUM_WIRE_IMPULSE_N_S = 0.020
_MAXIMUM_PENETRATION_M = 0.004
_PENETRATION_GRACE_STEPS = 2
_MAXIMUM_PATH_DEVIATION_M = 0.030
_PATH_DEVIATION_DURATION_S = 0.050
_MAXIMUM_BACKTRACK = 0.05
_MAXIMUM_WORKSPACE_RADIUS_M = 0.50
_MINIMUM_WAND_BOTTOM_Z_M = 0.75


@dataclass(frozen=True, slots=True)
class _PathFacts:
    progress: float
    path_distance_m: float
    terminal_distance_m: float
    phase_alignment_distance_m: float


def _finite_path_facts(
    point_world_m: npt.NDArray[np.float64],
    *,
    frame_position_world_m: npt.NDArray[np.float64],
    world_from_frame: npt.NDArray[np.float64],
    path_points_frame_m: npt.NDArray[np.float64],
) -> _PathFacts:
    if (
        point_world_m.shape != (3,)
        or frame_position_world_m.shape != (3,)
        or world_from_frame.shape != (3, 3)
        or path_points_frame_m.ndim != 2
        or path_points_frame_m.shape[1] != 3
        or len(path_points_frame_m) < 2
        or not all(
            np.isfinite(value).all()
            for value in (
                point_world_m,
                frame_position_world_m,
                world_from_frame,
                path_points_frame_m,
            )
        )
    ):
        raise ValueError("finite wire-path inputs are invalid")
    point = (point_world_m - frame_position_world_m) @ world_from_frame
    lengths = np.linalg.norm(np.diff(path_points_frame_m, axis=0), axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("finite wire path contains a degenerate segment")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    best_distance, progress = math.inf, 0.0
    for index, (start, end) in enumerate(
        zip(path_points_frame_m[:-1], path_points_frame_m[1:], strict=True)
    ):
        edge = end - start
        length_squared = float(edge @ edge)
        fraction = float(np.clip(((point - start) @ edge) / length_squared, 0.0, 1.0))
        nearest = start + fraction * edge
        distance = float(np.linalg.norm(point - nearest))
        if distance < best_distance:
            best_distance = distance
            progress = (cumulative[index] + fraction * math.sqrt(length_squared)) / float(
                cumulative[-1]
            )
    path_world = frame_position_world_m + path_points_frame_m @ world_from_frame.T
    return _PathFacts(
        progress=float(np.clip(progress, 0.0, 1.0)),
        path_distance_m=best_distance,
        terminal_distance_m=float(np.linalg.norm(point_world_m - path_world[-1])),
        phase_alignment_distance_m=float(np.linalg.norm(point_world_m - path_world[0])),
    )


def _segment_distance(
    first_start: npt.NDArray[np.float64],
    first_end: npt.NDArray[np.float64],
    second_start: npt.NDArray[np.float64],
    second_end: npt.NDArray[np.float64],
) -> float:
    """Return the exact Euclidean distance between two finite 3-D segments."""
    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    aa, bb, cc = float(first @ first), float(first @ second), float(second @ second)
    dd, ee = float(first @ offset), float(second @ offset)
    denominator = aa * cc - bb * bb
    s = 0.0 if denominator <= 1e-18 else float(np.clip((bb * ee - cc * dd) / denominator, 0, 1))
    t = float(np.clip((bb * s + ee) / cc, 0, 1))
    s = float(np.clip((bb * t - dd) / aa, 0, 1))
    return float(np.linalg.norm(first_start + s * first - second_start - t * second))


class _RotatingOutcome:
    def __init__(
        self, task: TaskSpec, geometry: RotatingInteractionGeometrySpec, timestep_s: float
    ) -> None:
        if not isinstance(task.success, (PhaseSuccessSpec, FullRotatingSuccessSpec)):
            raise ValueError("T3 requires its rotating-interaction success record")
        self.spec = task.success
        self.geometry = geometry
        self.full = isinstance(task.success, FullRotatingSuccessSpec)
        self.timestep_s = timestep_s
        self.horizon_s = task.episode_horizon_steps / _CONTROL_FREQUENCY_HZ
        self.bilateral_steps_required = round(_BILATERAL_HOLD_S / timestep_s)
        self.deviation_steps_required = round(_PATH_DEVIATION_DURATION_S / timestep_s)
        self.reset()

    def reset(self) -> None:
        self.physics_step = 0
        self.bilateral_steps = 0
        self.deviation_steps = 0
        self.penetration_steps = 0
        self.maximum_progress = 0.0
        self.maximum_wire_impulse_n_s = 0.0
        self.maximum_penetration_m = 0.0
        self.events: dict[str, float] = {}
        self.failure_code: str | None = None
        self.terminal_time_s: float | None = None
        self.success = False
        self._phase_start: float | None = None
        self._phase_best: float | None = None
        self._entry_start: float | None = None
        self._entry_best: float | None = None
        self._terminal_start: float | None = None
        self._terminal_best: float | None = None
        self._wire_exit_start: float | None = None
        self._wire_exit_best: float | None = None
        self._frozen_stage_facts: tuple[tuple[str, bool, float, int], ...] | None = None

    def begin(self, path: _PathFacts) -> None:
        if not self.full:
            if (
                not math.isfinite(path.phase_alignment_distance_m)
                or path.phase_alignment_distance_m < 0.0
            ):
                raise ValueError(
                    "T3 phase initialization requires a finite non-negative alignment distance"
                )
            self._phase_start = self._phase_best = path.phase_alignment_distance_m
        self.events["scored_phase_sample"] = 0.0

    def _require_begun(self) -> None:
        if not self.full:
            _require_distance_stage_begin(
                self._phase_start,
                self._phase_best,
                owner="T3 phase outcome",
            )

    def _terminal_distance(self, path: _PathFacts) -> float:
        return max(
            max(path.terminal_distance_m / self.spec.terminal_distance_m - 1.0, 0.0),
            max(self.spec.terminal_progress_threshold - path.progress, 0.0)
            / self.spec.terminal_progress_threshold,
        )

    def _wire_exit_distance(self, path: _PathFacts) -> float:
        if not isinstance(self.spec, FullRotatingSuccessSpec):
            return 0.0
        clearance = self.spec.wire_exit_clearance_m
        return (
            max(
                max(clearance - path.path_distance_m, 0.0),
                max(clearance - path.terminal_distance_m, 0.0),
            )
            / clearance
        )

    def _stage_facts(self) -> tuple[tuple[str, bool, float, int], ...]:
        if self._frozen_stage_facts is not None:
            return self._frozen_stage_facts
        candidates: list[tuple[str, bool, float, int]] = []
        if self.full:
            candidates.append(
                (
                    "wand_acquisition",
                    "wand_acquisition" in self.events,
                    _duration_stage_progress(
                        self.bilateral_steps,
                        self.bilateral_steps_required,
                        completed="wand_acquisition" in self.events,
                    ),
                    min(self.bilateral_steps, self.bilateral_steps_required),
                )
            )
        candidates.extend(
            (
                (
                    "phase_alignment",
                    "phase_alignment" in self.events,
                    _distance_stage_progress(
                        self._phase_start,
                        self._phase_best,
                        self.spec.phase_alignment_distance_m,
                        completed="phase_alignment" in self.events,
                    ),
                    0,
                ),
                (
                    "loop_entry",
                    "loop_entry" in self.events,
                    _distance_stage_progress(
                        self._entry_start,
                        self._entry_best,
                        self.spec.loop_entry_distance_m,
                        completed="loop_entry" in self.events,
                    ),
                    0,
                ),
                (
                    "path_progress_0_60",
                    "path_progress_0_60" in self.events,
                    1.0
                    if "path_progress_0_60" in self.events
                    else self.maximum_progress / self.spec.path_progress_threshold,
                    0,
                ),
                (
                    "terminal_arrival",
                    "terminal_arrival" in self.events,
                    _distance_stage_progress(
                        self._terminal_start,
                        self._terminal_best,
                        0.0,
                        completed="terminal_arrival" in self.events,
                    ),
                    0,
                ),
            )
        )
        if self.full:
            candidates.extend(
                (
                    (
                        "wire_exit",
                        "wire_exit" in self.events,
                        _distance_stage_progress(
                            self._wire_exit_start,
                            self._wire_exit_best,
                            0.0,
                            completed="wire_exit" in self.events,
                        ),
                        0,
                    ),
                    ("wand_release", "wand_release" in self.events, 0.0, 0),
                )
            )
        return _ordered_stage_facts(candidates)

    def invalidate(self, time_s: float) -> None:
        if self.terminal_time_s is None:
            self._require_begun()
            self._frozen_stage_facts = self._stage_facts()
            self.failure_code = "invalid_state"
            self.terminal_time_s = time_s

    def update(
        self,
        *,
        time_s: float,
        finite: bool,
        path: _PathFacts,
        attachment_active: bool,
        bilateral_contact: bool,
        any_finger_contact: bool,
        loop_orientation_error_rad: float,
        unsafe_stage_contact: bool,
        release_requested: bool,
        wire_impulse_n_s: float,
        penetration_m: float,
        bottom_z_m: float,
        workspace_radius_m: float,
    ) -> None:
        if self.terminal_time_s is not None:
            return
        self._require_begun()
        stage_facts_before = self._stage_facts()
        if not math.isclose(
            time_s, (self.physics_step + 1) * self.timestep_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError("T3 outcome requires consecutive simulator physics states")
        self.physics_step += 1
        self.maximum_wire_impulse_n_s = max(self.maximum_wire_impulse_n_s, wire_impulse_n_s)
        self.maximum_penetration_m = max(self.maximum_penetration_m, penetration_m)
        self.penetration_steps = (
            self.penetration_steps + 1 if penetration_m > _MAXIMUM_PENETRATION_M else 0
        )
        numeric_values = (
            path.progress,
            path.path_distance_m,
            path.terminal_distance_m,
            path.phase_alignment_distance_m,
            wire_impulse_n_s,
            penetration_m,
            bottom_z_m,
            workspace_radius_m,
        )
        failure: str | None = None
        if not finite or not all(math.isfinite(value) for value in numeric_values):
            failure = "invalid_state"
        elif unsafe_stage_contact:
            failure = "stage_block"
        elif bottom_z_m < _MINIMUM_WAND_BOTTOM_Z_M:
            failure = "drop"
        elif workspace_radius_m > _MAXIMUM_WORKSPACE_RADIUS_M:
            failure = "workspace_exit"
        elif wire_impulse_n_s > _MAXIMUM_WIRE_IMPULSE_N_S:
            failure = "wire_contact"
        elif self.penetration_steps > _PENETRATION_GRACE_STEPS:
            failure = "wire_contact"
        if failure is not None:
            self._frozen_stage_facts = stage_facts_before
            self.failure_code, self.terminal_time_s = failure, time_s
            if failure != "invalid_state":
                self.events.setdefault(failure, time_s)
            return

        if self.full and "wand_acquisition" not in self.events:
            self.bilateral_steps = self.bilateral_steps + 1 if bilateral_contact else 0
            if self.bilateral_steps >= self.bilateral_steps_required:
                self.events["wand_acquisition"] = time_s
        if (
            self.full
            and release_requested
            and "wand_acquisition" in self.events
            and "wire_exit" not in self.events
        ):
            self._frozen_stage_facts = stage_facts_before
            self.failure_code, self.terminal_time_s = "premature_release", time_s
            self.events.setdefault("premature_release", time_s)
            return
        ready = not self.full or "wand_acquisition" in self.events
        if ready:
            if self._phase_start is None:
                if self._phase_best is not None:
                    raise RuntimeError("T3 phase distance state is inconsistent")
                self._phase_start = self._phase_best = path.phase_alignment_distance_m
            else:
                if self._phase_best is None:
                    raise RuntimeError("T3 phase distance state is inconsistent")
                self._phase_best = min(self._phase_best, path.phase_alignment_distance_m)
        if (
            ready
            and path.phase_alignment_distance_m <= self.spec.phase_alignment_distance_m
            and loop_orientation_error_rad <= 0.20
        ):
            self.events.setdefault("phase_alignment", time_s)
        if "phase_alignment" in self.events:
            if self._entry_start is None:
                self._entry_start = self._entry_best = path.phase_alignment_distance_m
            else:
                self._entry_best = min(
                    cast(float, self._entry_best), path.phase_alignment_distance_m
                )
        if (
            "phase_alignment" in self.events
            and path.phase_alignment_distance_m <= self.spec.loop_entry_distance_m
            and path.progress <= 0.15
        ):
            self.events.setdefault("loop_entry", time_s)
        if "loop_entry" in self.events and "wire_exit" not in self.events:
            if "terminal_arrival" not in self.events:
                self.deviation_steps = (
                    self.deviation_steps + 1
                    if path.path_distance_m > _MAXIMUM_PATH_DEVIATION_M
                    else 0
                )
            if self.deviation_steps >= self.deviation_steps_required:
                failure = "path_deviation"
            elif path.progress - self.maximum_progress < -_MAXIMUM_BACKTRACK:
                failure = "path_backtrack"
            if failure is not None:
                self._frozen_stage_facts = stage_facts_before
                self.failure_code, self.terminal_time_s = failure, time_s
                self.events.setdefault(failure, time_s)
                return
            self.maximum_progress = max(self.maximum_progress, path.progress)
        if (
            "loop_entry" in self.events
            and self.maximum_progress >= self.spec.path_progress_threshold
        ):
            self.events.setdefault("path_progress_0_60", time_s)
        if "path_progress_0_60" in self.events:
            terminal_distance = self._terminal_distance(path)
            if self._terminal_start is None:
                self._terminal_start = self._terminal_best = terminal_distance
            else:
                self._terminal_best = min(cast(float, self._terminal_best), terminal_distance)
        if (
            "path_progress_0_60" in self.events
            and self.maximum_progress >= self.spec.terminal_progress_threshold
            and path.terminal_distance_m <= self.spec.terminal_distance_m
        ):
            self.events.setdefault("terminal_arrival", time_s)
        if not self.full and "terminal_arrival" in self.events:
            self.success = True
            self.events.setdefault("success", time_s)
            self.terminal_time_s = time_s
            return
        if self.full:
            assert isinstance(self.spec, FullRotatingSuccessSpec)
            if "terminal_arrival" in self.events:
                wire_exit_distance = self._wire_exit_distance(path)
                if self._wire_exit_start is None:
                    self._wire_exit_start = self._wire_exit_best = wire_exit_distance
                else:
                    self._wire_exit_best = min(
                        cast(float, self._wire_exit_best), wire_exit_distance
                    )
            if (
                "terminal_arrival" in self.events
                and time_s > self.events["terminal_arrival"]
                and path.progress >= self.spec.terminal_progress_threshold
                and path.path_distance_m >= self.spec.wire_exit_clearance_m
                and path.terminal_distance_m >= self.spec.wire_exit_clearance_m
            ):
                self.events.setdefault("wire_exit", time_s)
            if "wire_exit" in self.events and release_requested and not any_finger_contact:
                self.events.setdefault("wand_release", time_s)
                self.success = True
                self.events.setdefault("success", time_s)
                self.terminal_time_s = time_s
        if self.terminal_time_s is None and time_s >= self.horizon_s - 1e-10:
            self._frozen_stage_facts = self._stage_facts()
            self.failure_code, self.terminal_time_s = "timeout", self.horizon_s
            self.events.setdefault("timeout", self.horizon_s)


class RotatingInteractionEnv(DynamicManipulationEnv):
    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        scenario_spec: ScenarioSpec,
        execution_translation_limit_m: float = 0.05,
    ) -> None:
        if task_spec.family != "rotating_interaction" or not isinstance(
            task_spec.bindings, RotatingInteractionBindings
        ):
            raise ValueError("RotatingInteractionEnv requires a registered T3 TaskSpec")
        if not isinstance(scenario_spec.geometry, RotatingInteractionGeometrySpec):
            raise ValueError("T3 requires RotatingInteractionGeometrySpec")
        bindings = task_spec.bindings
        objects = {item.instance_id: item for item in scenario_spec.object_states}
        fixtures = {item.instance_id: item for item in scenario_spec.fixture_states}
        bodies = {item.entity_id: item for item in scenario_spec.physics.bodies}
        contacts = {item.entity_id: item for item in scenario_spec.physics.contacts}
        required_fixtures = {bindings.wire_fixture_id, *bindings.cradle_fixture_ids}
        required_entities = {bindings.wand_object_id, *required_fixtures}
        if (
            set(objects) != {bindings.wand_object_id}
            or set(fixtures) != required_fixtures
            or set(bodies) != required_entities
            or set(contacts) != required_entities
            or len(scenario_spec.motion) != 1
            or scenario_spec.motion[0].entity_id != bindings.wire_fixture_id
            or scenario_spec.events
        ):
            raise ValueError(
                "T3 requires exactly its wand, wire, optional cradle and physical bindings"
            )
        self._full = task_spec.variant == "full"
        if self._full != (len(bindings.cradle_fixture_ids) == 1):
            raise ValueError("T3-Full alone requires exactly one cradle fixture")
        self._wand_spec = objects[bindings.wand_object_id]
        self._wire_spec = fixtures[bindings.wire_fixture_id]
        self._cradle_spec = fixtures[bindings.cradle_fixture_ids[0]] if self._full else None
        if (
            self._wand_spec.asset_id != "obj.wand_loop_v1"
            or self._wire_spec.asset_id != "fixture.rotating_wire_v1"
            or (
                self._cradle_spec is not None
                and self._cradle_spec.asset_id != "fixture.wand_cradle_v1"
            )
        ):
            raise ValueError("T3 requires its frozen functional wand, wire and cradle assets")
        self._motion = scenario_spec.motion[0]
        if not isinstance(self._motion.translation, FixedTranslation) or not isinstance(
            self._motion.rotation, ConstantAngularVelocityRotation
        ):
            raise ValueError("T3 wire requires fixed translation and constant angular velocity")
        if self._motion.rotation.axis_world != (1.0, 0.0, 0.0):
            raise ValueError("T3 wire must rotate around world X")
        if self._motion.rotation.angular_speed_rad_s == 0.0:
            raise ValueError("T3 wire angular speed must be non-zero")
        self._wire_rotation = self._motion.rotation
        _validate_motion_initial_pose(
            self._motion,
            initial_position_world_m=self._wire_spec.position_world_m,
            initial_quaternion_world_xyzw=self._wire_spec.quaternion_world_xyzw,
        )
        geometry = scenario_spec.geometry
        if self._wire_spec.position_world_m != geometry.frame_center_world_m:
            raise ValueError("T3 wire pose and GeometrySpec frame center must agree")
        if geometry.rotation_axis_world != (1.0, 0.0, 0.0):
            raise ValueError("T3 GeometrySpec requires world-X rotation")
        if len(geometry.wire_path_points_frame_m) < 2:
            raise ValueError("T3 requires a finite polyline with at least two points")
        catalog = _load_asset_catalog()
        self._arena_asset = catalog["arena.rotating_wire_v1"]
        self._wand_asset = catalog[self._wand_spec.asset_id]
        self._wire_asset = catalog[self._wire_spec.asset_id]
        self._cradle_asset = (
            catalog[self._cradle_spec.asset_id] if self._cradle_spec is not None else None
        )
        self._body_specs, self._contact_specs = bodies, contacts
        self._geometry = geometry
        self._path_points = np.asarray(geometry.wire_path_points_frame_m, dtype=np.float64)
        handle_points = np.asarray(
            [(x, y, z) for x in (-0.022, 0.022) for y in (-0.012, 0.012) for z in (-0.06, 0.06)]
        )
        ring_points = np.asarray(
            [
                (0.0, (0.035 + 0.002) * math.cos(angle), -0.1 + (0.035 + 0.002) * math.sin(angle))
                for angle in np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
            ]
        )
        self._wand_extent_points = np.concatenate((handle_points, ring_points))
        self._outcome = _RotatingOutcome(task_spec, geometry, scenario_spec.physics_timestep_s)
        self._preparing = False
        self._interaction_active = False
        self._prepared = False
        self._preparation_steps = 0
        self._pregrasp_steps = 0
        self._cradle_support_steps = 0
        self._release_requested = False
        self._attachment_active = False
        self._attachment_constraint_residual = math.nan
        self._preparation_diagnostics = (0, 0, 0, False)
        self._reset_scenario: ScenarioSpec | None = None
        self._wire_driver_active = False
        self._wire_driver_origin_s = 0.0
        self._wire_driver_phase_rad = 0.0
        super().__init__(
            task_spec=task_spec,
            scenario_spec=scenario_spec,
            seed=0,
            execution_translation_limit_m=execution_translation_limit_m,
        )

    def _load_model(self) -> None:
        self._interaction_active = self._preparing = False
        super()._load_model()
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(robot_model.base_xpos_offset["table"](0.8))
        arena = TableArena()
        arena.worldbody.remove(arena.table_body)
        for name in ("left_eef_target", "right_eef_target"):
            target = arena.worldbody.find(f"body[@name='{name}']")
            if target is None:
                raise RuntimeError("Robosuite 1.5.2 arena target binding is missing")
            arena.worldbody.remove(target)
        arena.merge(MujocoXML(str(self._arena_asset.path)))
        light = arena.worldbody.find("light")
        support = arena.worldbody.find("body[@name='wire_support']")
        if light is None or support is None:
            raise RuntimeError("T3 requires the table light and wire support")
        light.set("name", "rotating_wire_key")
        support.set("pos", _xml_numbers(self._geometry.frame_center_world_m))
        self._wand_model = MujocoXMLObject(
            str(self._wand_asset.path),
            name=self._wand_spec.instance_id,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self._wire_model = MujocoXMLObject(
            str(self._wire_asset.path),
            name=self._wire_spec.instance_id,
            joints=[
                {
                    "name": "hinge",
                    "type": "hinge",
                    "axis": "1 0 0",
                    "limited": "false",
                    "damping": "0",
                    "armature": "0",
                    "frictionloss": "0",
                }
            ],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        path = self._geometry.wire_path_points_frame_m
        for index, (start, end) in enumerate(zip(path[:-1], path[1:], strict=True)):
            for suffix, radius_scale in (("", 1.0), ("_visual", 1.22)):
                name = self._wire_model.naming_prefix + f"segment_{index:02d}{suffix}"
                geom = self._wire_model.get_obj().find(f"geom[@name='{name}']")
                if geom is None:
                    raise ValueError("T3 wire asset segment count differs from GeometrySpec")
                geom.set("fromto", _xml_numbers((*start, *end)))
                geom.set("size", format(self._geometry.wire_radius_m * radius_scale, ".17g"))
        self._wire_model.get_obj().set("pos", _xml_numbers(self._geometry.frame_center_world_m))
        objects = [self._wand_model, self._wire_model]
        if self._cradle_spec is not None:
            if self._cradle_asset is None:
                raise RuntimeError("T3-Full cradle asset did not resolve")
            self._cradle_model = MujocoXMLObject(
                str(self._cradle_asset.path),
                name=self._cradle_spec.instance_id,
                joints=None,
                obj_type="all",
                duplicate_collision_geoms=False,
            )
            x, y, z, w = self._cradle_spec.quaternion_world_xyzw
            cradle_body = self._cradle_model.get_obj()
            cradle_body.set("pos", _xml_numbers(self._cradle_spec.position_world_m))
            cradle_body.set("quat", _xml_numbers((w, x, y, z)))
            objects.append(self._cradle_model)
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot_model],
            mujoco_objects=objects,
        )
        hand_body = self.robots[0].gripper["right"].naming_prefix + "eef"
        if not isinstance(hand_body, str) or not hand_body:
            raise RuntimeError("T3 requires the explicit Panda right-hand body")
        ET.SubElement(
            self.model.equality,
            "weld",
            name="wand_attachment",
            body1=hand_body,
            body2=self._wand_model.root_body,
            active="false",
            solref="0.002 1",
            solimp="0.95 0.99 0.001",
        )

    def edit_model_xml(self, xml_str: str) -> str:
        return _rewrite_scenario_lighting_and_gravity_xml(
            super().edit_model_xml(xml_str),
            self._scenario_spec,
            light_id="rotating_wire_key",
            invalid_message="T3 requires the explicit canonical rotating-wire light set",
            duplicate_message="duplicate rotating-wire light",
            missing_message="rotating_wire_key must resolve in T3 arena",
            headlight_before_gravity=False,
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        model = self.sim.model._model
        self._wand_body_id = int(model.body(self._wand_model.root_body).id)
        self._wire_body_id = int(model.body(self._wire_model.root_body).id)
        self._wand_geom_ids = tuple(
            int(model.geom(name).id) for name in self._wand_model.contact_geoms
        )
        self._loop_geom_ids = tuple(
            int(model.geom(name).id)
            for name in self._wand_model.contact_geoms
            if name.removeprefix(self._wand_model.naming_prefix).startswith("loop_")
        )
        self._wire_geom_ids = tuple(
            int(model.geom(name).id) for name in self._wire_model.contact_geoms
        )
        self._loop_site_id = int(model.site(self._wand_model.naming_prefix + "loop_center").id)
        wand_joint = int(model.joint(self._wand_model.joints[0]).id)
        wire_joint = int(model.joint(self._wire_model.joints[0]).id)
        if (
            model.jnt_type[wand_joint] != mujoco.mjtJoint.mjJNT_FREE
            or model.jnt_type[wire_joint] != mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise ValueError("T3 requires a free wand and native wire hinge")
        self._wand_qpos_address = int(model.jnt_qposadr[wand_joint])
        self._wand_qvel_address = int(model.jnt_dofadr[wand_joint])
        self._wire_qpos_address = int(model.jnt_qposadr[wire_joint])
        self._wire_qvel_address = int(model.jnt_dofadr[wire_joint])
        self._attachment_eq_id = int(model.equality("wand_attachment").id)
        if model.eq_type[self._attachment_eq_id] != mujoco.mjtEq.mjEQ_WELD:
            raise ValueError("T3 attachment must be a native weld equality")
        self._cradle_geom_ids: tuple[int, ...] = ()
        self._cradle_body_id: int | None = None
        if self._cradle_spec is not None:
            self._cradle_body_id = int(model.body(self._cradle_model.root_body).id)
            self._cradle_geom_ids = tuple(
                int(model.geom(name).id) for name in self._cradle_model.contact_geoms
            )
        self._entity_bodies = {
            self._wand_spec.instance_id: self._wand_body_id,
            self._wire_spec.instance_id: self._wire_body_id,
        }
        self._entity_geoms = {
            self._wand_spec.instance_id: self._wand_geom_ids,
            self._wire_spec.instance_id: self._wire_geom_ids,
        }
        if self._cradle_spec is not None and self._cradle_body_id is not None:
            self._entity_bodies[self._cradle_spec.instance_id] = self._cradle_body_id
            self._entity_geoms[self._cradle_spec.instance_id] = self._cradle_geom_ids
        self._compiled_joint_physics = _resolve_joint_physics(
            model,
            self._scenario_spec.physics.joints,
            entity_body_ids=self._entity_bodies,
        )
        self._bind_scene_entities(
            target_body_id=self._wire_body_id,
            entity_body_ids=self._entity_bodies,
        )
        pads = self.robots[0].gripper["right"].important_geoms
        self._left_pad_ids = frozenset(int(model.geom(name).id) for name in pads["left_fingerpad"])
        self._right_pad_ids = frozenset(
            int(model.geom(name).id) for name in pads["right_fingerpad"]
        )
        self._gripper_geom_ids = tuple(
            int(model.geom(name).id) for name in self.robots[0].gripper["right"].contact_geoms
        )
        self._robot_geom_ids = frozenset(
            int(model.geom(name).id) for name in self.robots[0].robot_model.contact_geoms
        ) | frozenset(self._gripper_geom_ids)
        self._attachment_contact_masks = {
            geom: (int(model.geom_contype[geom]), int(model.geom_conaffinity[geom]))
            for geom in (*self._wand_geom_ids, *self._gripper_geom_ids)
        }
        self._validate_compiled_path()

    def _validate_compiled_path(self) -> None:
        model = self.sim.model._model
        if len(self._wire_geom_ids) != len(self._path_points) - 1:
            raise ValueError("compiled T3 wire segment count differs from GeometrySpec")
        for index, geom in enumerate(self._wire_geom_ids):
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CAPSULE or not math.isclose(
                float(model.geom_size[geom, 0]), self._geometry.wire_radius_m, abs_tol=1e-12
            ):
                raise ValueError("compiled T3 wire does not match finite GeometrySpec")
            center = model.geom_pos[geom]
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            half = float(model.geom_size[geom, 1])
            endpoints = np.asarray(
                [
                    center - rotation.reshape(3, 3)[:, 2] * half,
                    center + rotation.reshape(3, 3)[:, 2] * half,
                ]
            )
            expected = self._path_points[index : index + 2]
            if not (
                np.allclose(endpoints, expected, rtol=0.0, atol=1e-12)
                or np.allclose(endpoints[::-1], expected, rtol=0.0, atol=1e-12)
            ):
                raise ValueError("compiled T3 wire capsule endpoints differ from GeometrySpec")

    def _set_attachment(self, active: bool) -> None:
        if active == self._attachment_active:
            return
        model, data = self.sim.model._model, self.sim.data._data
        if active:
            body1 = int(model.eq_obj1id[self._attachment_eq_id])
            body2 = self._wand_body_id
            x1, x2 = data.xpos[body1].copy(), data.xpos[body2].copy()
            r1 = data.xmat[body1].reshape(3, 3).copy()
            r2 = data.xmat[body2].reshape(3, 3).copy()
            relative = np.empty(4)
            mujoco.mju_mat2Quat(relative, (r1.T @ r2).ravel())
            model.eq_data[self._attachment_eq_id, 3:6] = r1.T @ (x2 - x1)
            model.eq_data[self._attachment_eq_id, 6:10] = relative
        qpos, qvel = data.qpos.copy(), data.qvel.copy()
        data.eq_active[self._attachment_eq_id] = int(active)
        if active:
            for geom in self._wand_geom_ids:
                model.geom_contype[geom], model.geom_conaffinity[geom] = 2, 1
            for geom in self._gripper_geom_ids:
                model.geom_contype[geom], model.geom_conaffinity[geom] = 4, 1
        else:
            for geom, masks in self._attachment_contact_masks.items():
                model.geom_contype[geom], model.geom_conaffinity[geom] = masks
        self.sim.forward()
        if not (np.array_equal(data.qpos, qpos) and np.array_equal(data.qvel, qvel)):
            raise RuntimeError("T3 weld activation changed native pose or twist")
        if active:
            rows = np.flatnonzero(
                (data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                & (data.efc_id[: data.nefc] == self._attachment_eq_id)
            )
            if len(rows) != 6:
                raise RuntimeError("T3 weld must compile to one native six-dimensional constraint")
            self._attachment_constraint_residual = float(np.max(np.abs(data.efc_pos[rows])))
        self._attachment_active = active

    def _reset_internal(self) -> None:
        self._interaction_active = self._preparing = self._prepared = False
        self._wire_driver_active = False
        self._reset_scenario = None
        super()._reset_internal()
        model, data = self.sim.model._model, self.sim.data._data
        for entity_id, body_id in self._entity_bodies.items():
            body, contact = self._body_specs[entity_id], self._contact_specs[entity_id]
            model.body_mass[body_id] = body.mass_kg
            model.body_inertia[body_id] = body.diagonal_inertia_kg_m2
            for geom_id in self._entity_geoms[entity_id]:
                model.geom_friction[geom_id] = contact.friction_slide_spin_roll
                model.geom_solref[geom_id] = contact.solref
                model.geom_solimp[geom_id] = contact.solimp
        _apply_joint_physics(model, self._compiled_joint_physics)
        mujoco.mj_setConst(model, data)
        robot = self.robots[0]
        data.qpos[robot._ref_joint_pos_indexes] = (
            self._scenario_spec.robot_initial_state.joint_position_rad
        )
        data.qpos[robot._ref_gripper_joint_pos_indexes["right"]] = (
            self._scenario_spec.robot_initial_state.gripper_position_m
        )
        x, y, z, w = self._wand_spec.quaternion_world_xyzw
        address = self._wand_qpos_address
        data.qpos[address : address + 7] = (*self._wand_spec.position_world_m, w, x, y, z)
        quaternion = np.asarray((w, x, y, z), dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, quaternion)
        data.qvel[:] = 0.0
        data.qvel[self._wand_qvel_address : self._wand_qvel_address + 3] = (
            self._wand_spec.linear_velocity_world_m_s
        )
        data.qvel[self._wand_qvel_address + 3 : self._wand_qvel_address + 6] = (
            rotation.reshape(3, 3).T @ self._wand_spec.angular_velocity_world_rad_s
        )
        data.qpos[self._wire_qpos_address] = self._wire_initial_phase()
        data.qvel[self._wire_qvel_address] = 0.0
        data.eq_active[self._attachment_eq_id] = 0
        data.xfrc_applied[:] = 0.0
        data.qfrc_applied[:] = 0.0
        self._attachment_active = False
        self._attachment_constraint_residual = math.nan
        self.sim.forward()
        self._outcome.reset()
        self._preparation_steps = self._pregrasp_steps = self._cradle_support_steps = 0
        self._release_requested = False
        self._reset_scenario = self._read_preparation_input()
        data.qpos[self._wire_qpos_address] = (
            self._wire_initial_phase() - 0.4 * self._wire_rotation.angular_speed_rad_s
        )
        self.sim.forward()

    def _wire_initial_phase(self) -> float:
        x, y, z, w = self._wire_spec.quaternion_world_xyzw
        q = np.asarray((w, x, y, z), dtype=np.float64)
        q /= np.linalg.norm(q)
        if not np.allclose(q[2:], 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("T3 wire initial orientation must be a world-X rotation")
        return 2.0 * math.atan2(float(q[1]), float(q[0]))

    def _dynamic_target_snapshot(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        time_s = self._episode_tick / self._scenario_spec.sensor_frequency_hz
        reference = _motion_state_at_time(
            self._motion,
            initial_position_world_m=self._wire_spec.position_world_m,
            initial_quaternion_world_xyzw=self._wire_spec.quaternion_world_xyzw,
            simulator_time_s=time_s,
        )
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, reference.quaternion_world_xyzw[[3, 0, 1, 2]])
        return reference.position_world_m.copy(), rotation.reshape(3, 3)

    def reset(self) -> Mapping[str, object]:
        super().reset()
        self._preparing = True
        data = self.sim.data._data
        robot = self.robots[0]
        neutral = np.zeros(self.action_dim, dtype=np.float64)
        neutral[-1] = -1.0 if self._full else 1.0
        if not self._full:
            # Preserve the original Phase reset target's literal float64 values.
            target = np.asarray((-0.05, -0.02, 1.09), dtype=np.float64)
            for _ in range(140):
                error = target - data.site_xpos[robot.eef_site_id["right"]]
                if float(np.linalg.norm(error)) <= 0.008:
                    break
                action = self._preparation_controller_action(error, np.zeros(3), -1.0)
                ManipulationEnv.step(self, action)
                self.done, self.timestep = False, 0
            else:
                raise _ScenarioResetRejected("pregrasp_position_unreached", float(data.time))
            position = data.site_xpos[robot.eef_site_id["right"]] + (0.0, 0.0, -0.05)
            quaternion = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
            preparation_pose = np.concatenate((position, quaternion))
            for _ in range(20):
                data.qpos[self._wand_qpos_address : self._wand_qpos_address + 7] = preparation_pose
                data.qvel[self._wand_qvel_address : self._wand_qvel_address + 6] = 0.0
                self.sim.forward()
                ManipulationEnv.step(self, neutral)
        for _ in range(_PREPARATION_TICKS if self._full else 10):
            ManipulationEnv.step(self, neutral)
        self.cur_time, self.timestep, self.done = 0.0, 0, False
        self._wire_driver_active = True
        self._wire_driver_origin_s = float(data.time)
        self._wire_driver_phase_rad = (
            self._wire_initial_phase() - 0.4 * self._wire_rotation.angular_speed_rad_s
        )
        for _ in range(8):
            ManipulationEnv.step(self, neutral)
        self._preparation_diagnostics = (
            self._preparation_steps,
            self._cradle_support_steps,
            self._pregrasp_steps,
            self._attachment_active,
        )
        if self._full:
            if self._cradle_support_steps < 25 or self._pregrasp_steps or self._attachment_active:
                raise _ScenarioResetRejected("gravity_cradle_not_stable", float(self.sim.data.time))
        elif self._pregrasp_steps < 1:
            raise _ScenarioResetRejected("pregrasp_not_stable", float(self.sim.data.time))
        self._preparing = False
        data.qpos[self._wire_qpos_address] = self._wire_initial_phase()
        data.qvel[self._wire_qvel_address] = self._wire_rotation.angular_speed_rad_s
        data.time = 0.0
        self.cur_time = 0.0
        self.timestep = self._episode_tick = 0
        self.done = False
        self._action_spec = None
        self._action_delay_queue = None
        self._previous_eef_quaternion_xyzw = None
        self._obs_cache: dict[str, Any] = {}
        for observable in self._observables.values():
            observable.reset()
        self._outcome.reset()
        self._wire_driver_origin_s = 0.0
        self._wire_driver_phase_rad = self._wire_initial_phase()
        self._interaction_active = self._prepared = True
        self.sim.forward()
        if not self._full:
            self._set_attachment(True)
        self._verify_initial_threading()
        self._finish_preparation_controller()
        self._outcome.begin(self._path_facts())
        return cast(Mapping[str, object], self._get_observations(force_update=True))

    def _read_preparation_input(self) -> ScenarioSpec:
        model, data = self.sim.model._model, self.sim.data._data
        scenario, robot = self._scenario_spec, self.robots[0]
        if not _physics_state_is_valid(data):
            raise _ScenarioResetRejected("invalid_state", float(data.time))
        for name in (*robot.robot_model.joints, *robot.gripper["right"].joints):
            joint_id = int(model.joint(name).id)
            value = float(data.qpos[model.jnt_qposadr[joint_id]])
            if model.jnt_limited[joint_id] and not (
                model.jnt_range[joint_id, 0] <= value <= model.jnt_range[joint_id, 1]
            ):
                raise _ScenarioResetRejected("joint_out_of_range", float(data.time))
        robot_state = replace(
            scenario.robot_initial_state,
            joint_position_rad=cast(
                tuple[float, float, float, float, float, float, float],
                tuple(float(v) for v in data.qpos[robot._ref_joint_pos_indexes]),
            ),
            gripper_position_m=cast(
                Vector2,
                tuple(float(v) for v in data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]),
            ),
        )
        qpos = data.qpos[self._wand_qpos_address : self._wand_qpos_address + 7]
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_XBODY, self._wand_body_id, velocity, 0
        )
        expected_twist = (
            *self._wand_spec.angular_velocity_world_rad_s,
            *self._wand_spec.linear_velocity_world_m_s,
        )
        if not np.allclose(velocity, expected_twist, rtol=0.0, atol=1e-12):
            raise ValueError("compiled T3 preparation world twist differs from ScenarioSpec")
        wand = ObjectStateSpec(
            self._wand_spec.instance_id,
            self._wand_spec.asset_id,
            cast(Vector3, tuple(float(v) for v in qpos[:3])),
            cast(Vector4, tuple(float(v) for v in qpos[[4, 5, 6, 3]])),
            self._wand_spec.linear_velocity_world_m_s,
            self._wand_spec.angular_velocity_world_rad_s,
        )
        fixtures = [self._wire_spec]
        if self._cradle_spec is not None:
            fixtures.append(self._cradle_spec)
        for fixture in fixtures:
            body_id = self._entity_bodies[fixture.instance_id]
            expected_q = np.asarray(fixture.quaternion_world_xyzw, dtype=np.float64)[[3, 0, 1, 2]]
            expected_q /= np.linalg.norm(expected_q)
            if not (
                np.allclose(data.xpos[body_id], fixture.position_world_m, rtol=0.0, atol=1e-12)
                and math.isclose(
                    abs(float(np.dot(data.xquat[body_id], expected_q))),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("compiled T3 fixture pose differs from ScenarioSpec")
        bodies = tuple(
            BodyPhysicsSpec(
                item.entity_id,
                float(model.body_mass[self._entity_bodies[item.entity_id]]),
                cast(
                    Vector3,
                    tuple(
                        float(v) for v in model.body_inertia[self._entity_bodies[item.entity_id]]
                    ),
                ),
            )
            for item in scenario.physics.bodies
        )
        for item in scenario.physics.contacts:
            for geom_id in self._entity_geoms[item.entity_id]:
                actual = ContactPhysicsSpec(
                    item.entity_id,
                    cast(Vector3, tuple(float(v) for v in model.geom_friction[geom_id])),
                    cast(Vector2, tuple(float(v) for v in model.geom_solref[geom_id])),
                    cast(
                        tuple[float, float, float, float, float],
                        tuple(float(v) for v in model.geom_solimp[geom_id]),
                    ),
                )
                if actual != item:
                    raise ValueError("compiled T3 contact parameters differ from ScenarioSpec")
        _validate_compiled_joint_physics(
            model,
            self._compiled_joint_physics,
            owner="T3",
        )
        for name, camera in zip(self._camera_sensor_names, scenario.cameras.views, strict=True):
            index = model.camera(name).id
            q = np.asarray(camera.quaternion_parent_xyzw, dtype=np.float64)[[3, 0, 1, 2]]
            q /= np.linalg.norm(q)
            if not (
                np.allclose(model.cam_pos[index], camera.position_parent_m, rtol=0.0, atol=1e-12)
                and math.isclose(abs(float(np.dot(model.cam_quat[index], q))), 1.0, abs_tol=1e-12)
                and float(model.cam_fovy[index]) == camera.vertical_fov_deg
            ):
                raise ValueError("compiled T3 camera differs from ScenarioSpec")
        if (
            model.nlight != 1
            or model.light(0).name != "rotating_wire_key"
            or not model.vis.headlight.active
            or model.light_type[0] != mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            or model.light_castshadow[0]
        ):
            raise ValueError("compiled T3 canonical light set is invalid")
        light = scenario.lighting.lights[0]
        for actual_values, expected_values in (
            (model.light_pos[0], light.position_world_m),
            (model.light_dir[0], light.direction_world),
            (model.light_ambient[0], light.ambient_rgb),
            (model.light_diffuse[0], light.diffuse_rgb),
            (model.light_specular[0], light.specular_rgb),
            (model.light_attenuation[0], light.attenuation),
        ):
            if not np.allclose(actual_values, expected_values, rtol=0.0, atol=5e-8):
                raise ValueError("compiled T3 light differs from ScenarioSpec")
        if (
            float(model.light_cutoff[0]) != light.cutoff_deg
            or bool(model.light_active[0]) != light.active
        ):
            raise ValueError("compiled T3 light scalar differs from ScenarioSpec")
        if not math.isclose(
            float(data.qpos[self._wire_qpos_address]),
            self._wire_initial_phase(),
            abs_tol=1e-12,
        ):
            raise ValueError("compiled T3 wire hinge differs from ScenarioSpec")
        exported = replace(
            scenario,
            robot_initial_state=robot_state,
            object_states=(wand,),
            fixture_states=tuple(fixtures),
            physics=replace(
                scenario.physics,
                gravity_world_m_s2=cast(Vector3, tuple(float(v) for v in model.opt.gravity)),
                bodies=bodies,
            ),
        )
        if exported != scenario:
            raise ValueError("compiled T3 preparation input differs from ScenarioSpec")
        return exported

    def export_scenario(self) -> ScenarioSpec:
        if not self._prepared or self._reset_scenario is None:
            raise RuntimeError("T3 requires verified physical preparation before export")
        return self._reset_scenario

    def _wand_rotation(self) -> npt.NDArray[np.float64]:
        w, x, y, z = (float(value) for value in self.sim.data._data.xquat[self._wand_body_id])
        return np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )

    def _path_facts(self) -> _PathFacts:
        data = self.sim.data._data
        phase = (
            self._wire_initial_phase() + float(data.time) * self._wire_rotation.angular_speed_rad_s
        )
        cosine, sine = math.cos(phase), math.sin(phase)
        axis = np.asarray(self._geometry.rotation_axis_world)
        cross = np.asarray(
            ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
        )
        rotation = cosine * np.eye(3) + (1.0 - cosine) * np.outer(axis, axis) + sine * cross
        loop_center = data.xpos[self._wand_body_id] + self._wand_rotation() @ np.asarray(
            (0.0, 0.0, -0.1)
        )
        return _finite_path_facts(
            loop_center,
            frame_position_world_m=np.asarray(self._geometry.frame_center_world_m),
            world_from_frame=rotation,
            path_points_frame_m=self._path_points,
        )

    def _wire_contacts(self) -> tuple[float, float]:
        model, data = self.sim.model._model, self.sim.data._data
        impulse, penetration = 0.0, 0.0
        force = np.empty(6, dtype=np.float64)
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            if not (
                (a in self._wire_geom_ids and b in self._wand_geom_ids)
                or (b in self._wire_geom_ids and a in self._wand_geom_ids)
            ):
                continue
            mujoco.mj_contactForce(model, data, index, force)
            if not np.isfinite(force).all() or force[0] < 0.0:
                raise RuntimeError("invalid native T3 wire contact force")
            impulse = max(impulse, float(np.linalg.norm(force[:3])) * self.model_timestep)
            penetration = max(penetration, max(0.0, -float(contact.dist)))
        return impulse, penetration

    def _capsule_segment(self, geom_id: int) -> tuple[npt.NDArray[np.float64], ...]:
        model, data = self.sim.model._model, self.sim.data._data
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise ValueError("T3 threading requires capsule collision geometry")
        axis = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
        half = float(model.geom_size[geom_id, 1])
        return data.geom_xpos[geom_id] - axis * half, data.geom_xpos[geom_id] + axis * half

    def _verify_initial_threading(self) -> None:
        model, data = self.sim.model._model, self.sim.data._data
        self.sim.forward()
        loop_center = data.site_xpos[self._loop_site_id]
        loop_normal = data.xmat[self._wand_body_id].reshape(3, 3)[:, 0]
        entry = data.xpos[self._wire_body_id] + (
            data.xmat[self._wire_body_id].reshape(3, 3) @ self._path_points[0]
        )
        if float(np.linalg.norm(loop_center - entry)) > 0.030 or abs(
            float(loop_normal[0])
        ) < math.cos(0.20):
            raise _ScenarioResetRejected("wand_not_threaded", float(data.time))
        minimum_clearance = math.inf
        for wire_id in self._wire_geom_ids:
            wire_start, wire_end = self._capsule_segment(wire_id)
            for geom_id in self._loop_geom_ids:
                loop_start, loop_end = self._capsule_segment(geom_id)
                minimum_clearance = min(
                    minimum_clearance,
                    _segment_distance(wire_start, wire_end, loop_start, loop_end)
                    - float(model.geom_size[geom_id, 0])
                    - self._geometry.wire_radius_m,
                )
        impulse, penetration = self._wire_contacts()
        if minimum_clearance <= 0.0 or impulse or penetration:
            raise _ScenarioResetRejected("wand_not_threaded", float(data.time))

    def _pre_action(self, action: npt.NDArray[np.float64], policy_step: bool = False) -> None:
        data = self.sim.data._data
        if not _physics_state_is_valid(data):
            if self._preparing:
                raise _ScenarioResetRejected("invalid_state", float(data.time))
            self._outcome.invalidate(self._outcome.physics_step * self.model_timestep)
        if policy_step and not self._preparing:
            self._release_requested = bool(action[-1] < -0.5)
            if self._full and self._release_requested and self._attachment_active:
                self._set_attachment(False)
        if self._wire_driver_active:
            data.qpos[self._wire_qpos_address] = (
                self._wire_driver_phase_rad
                + (float(data.time) - self._wire_driver_origin_s)
                * self._wire_rotation.angular_speed_rad_s
            )
            data.qvel[self._wire_qvel_address] = self._wire_rotation.angular_speed_rad_s
            self.sim.forward()
        super()._pre_action(action, policy_step)

    def _update_observables(self, force: bool = False) -> None:
        super()._update_observables(force)
        if not force and (self._preparing or self._interaction_active):
            model, data = self.sim.model._model, self.sim.data._data
            finite = _physics_state_is_valid(data)
            if not finite:
                if self._preparing:
                    raise _ScenarioResetRejected("invalid_state", float(data.time))
                self._outcome.invalidate((self._outcome.physics_step + 1) * self.model_timestep)
            left, right, _ = _fingerpad_contacts(
                model, data, self._wand_geom_ids, self._left_pad_ids, self._right_pad_ids
            )
            bilateral = left and right
            if self._preparing:
                self._preparation_steps += 1
                self._pregrasp_steps = self._pregrasp_steps + 1 if bilateral else 0
                cradle_contact = False
                wand_ids, cradle_ids = set(self._wand_geom_ids), set(self._cradle_geom_ids)
                for index in range(data.ncon):
                    pair = {int(data.contact[index].geom1), int(data.contact[index].geom2)}
                    if pair & wand_ids and pair & cradle_ids:
                        cradle_contact = True
                        break
                self._cradle_support_steps = self._cradle_support_steps + 1 if cradle_contact else 0
            else:
                impulse, penetration = self._wire_contacts()
                if self._full and (left or right) and self._outcome.terminal_time_s is None:
                    self._capture_first_contact("fingerpad_wand", self._wand_body_id)
                path = self._path_facts()
                rotation = self._wand_rotation()
                wand_points = data.xpos[self._wand_body_id] + self._wand_extent_points @ rotation.T
                unsafe = any(
                    (
                        int(contact.geom1) in self._wire_geom_ids
                        and int(contact.geom2) in self._robot_geom_ids
                    )
                    or (
                        int(contact.geom2) in self._wire_geom_ids
                        and int(contact.geom1) in self._robot_geom_ids
                    )
                    for contact in data.contact[: data.ncon]
                )
                self._outcome.update(
                    time_s=float(data.time),
                    finite=finite,
                    path=path,
                    attachment_active=self._attachment_active,
                    bilateral_contact=bilateral,
                    any_finger_contact=left or right,
                    loop_orientation_error_rad=math.acos(
                        float(np.clip(abs(rotation[0, 0]), 0.0, 1.0))
                    ),
                    unsafe_stage_contact=unsafe,
                    release_requested=self._release_requested,
                    wire_impulse_n_s=impulse,
                    penetration_m=penetration,
                    bottom_z_m=float(wand_points[:, 2].min()),
                    workspace_radius_m=float(
                        np.linalg.norm(wand_points - data.xpos[self._wire_body_id], axis=1).max()
                    ),
                )

    def _post_action(self, action: npt.NDArray[np.float64]) -> tuple[float, bool, dict[str, Any]]:
        reward, _, info = super()._post_action(action)
        if (
            self._interaction_active
            and self._full
            and "wand_acquisition" in self._outcome.events
            and not self._attachment_active
            and not self._release_requested
        ):
            self._set_attachment(True)
        self.done = self._interaction_active and self._outcome.terminal_time_s is not None
        source_event_names = {
            "path_progress_0_60": "wire_path_progress",
            "stage_block": "unsafe_stage_block",
            "drop": "wand_drop",
            "wire_contact": "forbidden_wire_contact",
        }
        info.update(
            success=self._outcome.success,
            failure_code=self._outcome.failure_code,
            terminal_time_s=self._outcome.terminal_time_s,
            events={
                source_event_names.get(name, name): time_s
                for name, time_s in self._outcome.events.items()
            },
            stage_facts=self._outcome._stage_facts(),
            maximum_wire_impulse_n_s=self._outcome.maximum_wire_impulse_n_s,
            maximum_penetration_m=self._outcome.maximum_penetration_m,
        )
        return float(reward), bool(self.done), info

    def reward(self, action: npt.NDArray[np.float64] | None = None) -> float:
        del action
        return float(self._outcome.success)

    def _check_success(self) -> bool:
        return self._outcome.success

    def _recording_frame_facts(self) -> _RecordingFrameFacts:
        return (
            self._outcome.success,
            self._outcome.failure_code,
            self._outcome.terminal_time_s,
            self._outcome._stage_facts(),
        )


__all__: list[str] = []
