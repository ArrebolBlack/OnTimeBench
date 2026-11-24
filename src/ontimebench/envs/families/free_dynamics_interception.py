"""Capture an object released from a gate and moving under its own dynamics (T4)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import fields, replace
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
from ontimebench.envs.components.events import (
    _EventRuntime,
    _ReleaseGateCommand,
)
from ontimebench.scenarios.spec import (
    ContactPhysicsSpec,
    FixtureStateSpec,
    FreeDynamicsInterceptionGeometrySpec,
    ReleaseGateRetractionEffectSpec,
    RobotInitialStateSpec,
    ScenarioSpec,
    Vector2,
    Vector3,
)
from ontimebench.tasks.spec import (
    FreeDynamicsInterceptionBindings,
    SupportedCaptureSuccessSpec,
    TaskSpec,
    UnsupportedCaptureSuccessSpec,
)

_PREPARATION_CONTROL_TICKS = 12
_RELEASE_LINEAR_LIMIT_M_S = 0.005
_RELEASE_ANGULAR_LIMIT_RAD_S = 0.05
_RELEASE_IMPULSE_LIMIT_N_S = 0.002
_STOP_SPEED_M_S = 0.03
_STOP_DURATION_S = 0.25
_CAN_VERTICAL_ALIGNMENT_LIMIT = 0.45
_RELEASE_DYNAMICS_INTERVAL_STEPS = 25


def _normal_force_between(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_ids: frozenset[int],
    right_ids: frozenset[int],
) -> float:
    total = 0.0
    force = np.empty(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if not (
            (geom1 in left_ids and geom2 in right_ids) or (geom2 in left_ids and geom1 in right_ids)
        ):
            continue
        mujoco.mj_contactForce(model, data, index, force)
        normal = float(force[0])
        if not np.isfinite(force).all() or normal < 0.0:
            raise RuntimeError("invalid MuJoCo contact force")
        total += normal
    if not math.isfinite(total):
        raise RuntimeError("aggregated contact force must be finite")
    return total


def _contact_impulse_between(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_ids: frozenset[int],
    right_ids: frozenset[int],
    *,
    timestep_s: float,
) -> float:
    maximum_force = 0.0
    force = np.empty(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if not (
            (geom1 in left_ids and geom2 in right_ids) or (geom2 in left_ids and geom1 in right_ids)
        ):
            continue
        mujoco.mj_contactForce(model, data, index, force)
        if not np.isfinite(force).all() or force[0] < 0.0:
            raise RuntimeError("invalid MuJoCo contact force")
        maximum_force = max(maximum_force, float(np.linalg.norm(force[:3])))
    impulse = maximum_force * timestep_s
    if not math.isfinite(impulse):
        raise RuntimeError("aggregated contact impulse must be finite")
    return impulse


def _has_contact_between(
    data: mujoco.MjData,
    left_ids: frozenset[int],
    right_ids: frozenset[int],
) -> bool:
    return any(
        (int(data.contact[index].geom1) in left_ids and int(data.contact[index].geom2) in right_ids)
        or (
            int(data.contact[index].geom2) in left_ids
            and int(data.contact[index].geom1) in right_ids
        )
        for index in range(data.ncon)
    )


class _FreeDynamicsOutcome:
    """Track capture success and failure at the physics rate."""

    def __init__(self, task: TaskSpec, geometry: FreeDynamicsInterceptionGeometrySpec) -> None:
        if not isinstance(
            task.success, (UnsupportedCaptureSuccessSpec, SupportedCaptureSuccessSpec)
        ):
            raise ValueError("T4 requires a free-dynamics capture SuccessSpec")
        self.task = task
        self.success_spec = task.success
        self.geometry = geometry
        self.timestep_s = 0.002
        self.horizon_s = task.episode_horizon_steps / 20.0
        self.hold_steps = round(task.success.bilateral_contact_duration_s / self.timestep_s)
        self.dwell_steps = round(task.success.stable_hold_duration_s / self.timestep_s)
        self.stop_steps = round(_STOP_DURATION_S / self.timestep_s)
        self.supported = isinstance(task.success, SupportedCaptureSuccessSpec)
        self.can = task.variant == "supported_can"
        self.reset()

    def reset(self) -> None:
        self.physics_step = 0
        self.bilateral_steps = 0
        self.stable_steps = 0
        self.stopped_steps = 0
        self.capture_qualified = False
        self.capture_attempted = False
        self.capture_reference_com_z_m: float | None = None
        self.post_hold_contact_valid = False
        self.release_impulse_n_s = 0.0
        self.events: dict[str, float] = {}
        self.failure_code: str | None = None
        self.terminal_time_s: float | None = None
        self.success = False
        self._approach_start_m: float | None = None
        self._approach_best_m: float | None = None
        self._clearance_start_m: float | None = None
        self._clearance_best_m: float | None = None
        self._frozen_stage_facts: tuple[tuple[str, bool, float, int], ...] | None = None

    def begin(self, *, eef_distance_m: float) -> None:
        if not math.isfinite(eef_distance_m) or eef_distance_m < 0.0:
            raise ValueError(
                "T4 stage initialization requires a finite non-negative approach distance"
            )
        self._approach_start_m = self._approach_best_m = eef_distance_m

    def _require_begun(self) -> None:
        _require_distance_stage_begin(
            self._approach_start_m,
            self._approach_best_m,
            owner="T4 outcome",
        )

    def _stage_facts(self) -> tuple[tuple[str, bool, float, int], ...]:
        if self._frozen_stage_facts is not None:
            return self._frozen_stage_facts
        return _ordered_stage_facts(
            (
                (
                    "approach",
                    "approach" in self.events,
                    _distance_stage_progress(
                        self._approach_start_m,
                        self._approach_best_m,
                        0.10,
                        completed="approach" in self.events,
                    ),
                    0,
                ),
                ("fingerpad_contact", "fingerpad_contact" in self.events, 0.0, 0),
                (
                    "bilateral_hold",
                    "bilateral_hold" in self.events,
                    _duration_stage_progress(
                        self.bilateral_steps,
                        self.hold_steps,
                        completed="bilateral_hold" in self.events,
                    ),
                    min(self.bilateral_steps, self.hold_steps),
                ),
                (
                    "clearance",
                    "clearance" in self.events,
                    _distance_stage_progress(
                        self._clearance_start_m,
                        self._clearance_best_m,
                        0.0,
                        completed="clearance" in self.events,
                    ),
                    0,
                ),
                (
                    "stable_dwell",
                    "stable_dwell" in self.events,
                    _duration_stage_progress(
                        self.stable_steps,
                        self.dwell_steps,
                        completed="stable_dwell" in self.events,
                    ),
                    min(self.stable_steps, self.dwell_steps),
                ),
            )
        )

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
        position: npt.NDArray[np.float64],
        planar_speed_m_s: float,
        eef_distance_m: float,
        ramp_coordinate_m: npt.NDArray[np.float64],
        ramp_exited: bool,
        target_gate_impulse_n_s: float,
        target_downstream_contact: bool,
        robot_target_contact: bool,
        robot_ramp_contact: bool,
        left: bool,
        right: bool,
        can_axis_vertical_alignment: float,
    ) -> None:
        if self.terminal_time_s is not None:
            return
        self._require_begun()
        stage_facts_before = self._stage_facts()
        next_step = self.physics_step + 1
        if not math.isclose(time_s, next_step * self.timestep_s, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("T4 outcome requires consecutive simulator physics states")
        self.physics_step = next_step
        if robot_target_contact:
            self.events.setdefault("robot_contact", time_s)
        if target_downstream_contact:
            self.events.setdefault("downstream_contact", time_s)
        self.release_impulse_n_s = max(self.release_impulse_n_s, target_gate_impulse_n_s)

        failure: str | None = None
        if (
            not finite
            or not np.isfinite(position).all()
            or not all(
                math.isfinite(value)
                for value in (
                    planar_speed_m_s,
                    eef_distance_m,
                    can_axis_vertical_alignment,
                )
            )
        ):
            failure = "invalid_state"
        elif self.release_impulse_n_s > _RELEASE_IMPULSE_LIMIT_N_S:
            failure = "release_impulse"
        elif robot_ramp_contact:
            failure = "unsafe_contact"
        elif not self.supported and target_downstream_contact:
            failure = "tray_contact"
        elif (
            self.can
            and can_axis_vertical_alignment > _CAN_VERTICAL_ALIGNMENT_LIMIT
            and "fingerpad_contact" not in self.events
            and not (left or right)
        ):
            failure = "unstable_tipping"

        if failure is None and ramp_exited:
            self.events.setdefault("ramp_exit", time_s)
        exited = "ramp_exit" in self.events
        any_pad = left or right
        lower, upper = self.geometry.capture_region_bounds_world_m
        inside_capture = all(lower[index] <= position[index] <= upper[index] for index in (0, 1))
        eligible = exited
        if self.supported:
            assert isinstance(self.success_spec, SupportedCaptureSuccessSpec)
            eligible = (
                exited
                and target_downstream_contact
                and inside_capture
                and planar_speed_m_s >= self.success_spec.minimum_contact_speed_m_s
            )
        if failure is None:
            if not self.supported and any_pad and not exited:
                failure = "early_robot_contact"
            elif position[2] < (0.785 if self.supported else 0.805):
                failure = "drop"
            if self.supported:
                exit_lower, exit_upper = self.geometry.exit_region_bounds_xy_m
                if failure is None and abs(float(position[0])) > exit_upper[0]:
                    failure = "workspace_exit"
                if (
                    failure is None
                    and all(exit_lower[i] <= position[i] <= exit_upper[i] for i in (0, 1))
                    and "bilateral_hold" not in self.events
                ):
                    failure = "downstream_exit"
            elif failure is None and ramp_coordinate_m[1] > 0.48:
                failure = "downstream_exit"
            elif failure is None and abs(float(ramp_coordinate_m[0])) > 0.25:
                failure = "workspace_exit"
        if self.supported:
            self.stopped_steps = (
                self.stopped_steps + 1
                if exited
                and target_downstream_contact
                and not any_pad
                and planar_speed_m_s <= _STOP_SPEED_M_S
                else 0
            )
            if failure is None and self.stopped_steps >= self.stop_steps:
                failure = "stop_or_deflect_only"
            if (
                failure is None
                and any_pad
                and not eligible
                and "fingerpad_contact" not in self.events
            ):
                assert isinstance(self.success_spec, SupportedCaptureSuccessSpec)
                failure = (
                    "stop_or_deflect_only"
                    if exited and planar_speed_m_s < self.success_spec.minimum_contact_speed_m_s
                    else "early_robot_contact"
                )
        if failure is not None:
            self._frozen_stage_facts = stage_facts_before
            self.failure_code = failure
            self.terminal_time_s = time_s
            if failure != "invalid_state":
                self.events[failure] = time_s
            return

        bilateral = left and right
        self.bilateral_steps = self.bilateral_steps + 1 if bilateral else 0
        self._approach_best_m = min(cast(float, self._approach_best_m), eef_distance_m)
        if eef_distance_m <= 0.10:
            self.events.setdefault("approach", time_s)
        if "approach" in self.events and eligible and any_pad:
            self.events.setdefault("fingerpad_contact", time_s)
            self.capture_attempted = self.capture_qualified = True
        if (
            "fingerpad_contact" in self.events
            and "bilateral_hold" not in self.events
            and self.bilateral_steps >= self.hold_steps
        ):
            self.events["bilateral_hold"] = time_s
            self.capture_reference_com_z_m = float(position[2])
            self.post_hold_contact_valid = True
        if "bilateral_hold" in self.events and "clearance" not in self.events:
            self.post_hold_contact_valid = bilateral and self.bilateral_steps >= self.hold_steps
        com_rise = (
            0.0
            if self.capture_reference_com_z_m is None
            else float(position[2]) - self.capture_reference_com_z_m
        )
        clearance_distance = max(self.success_spec.minimum_lift_or_clearance_m - com_rise, 0.0)
        if "bilateral_hold" in self.events:
            if self._clearance_start_m is None:
                self._clearance_start_m = self._clearance_best_m = clearance_distance
            else:
                self._clearance_best_m = min(
                    cast(float, self._clearance_best_m), clearance_distance
                )
        clearance = clearance_distance == 0.0
        if (
            "bilateral_hold" in self.events
            and bilateral
            and self.post_hold_contact_valid
            and clearance
        ):
            self.events.setdefault("clearance", time_s)
        stable = (
            "clearance" in self.events and bilateral and self.post_hold_contact_valid and clearance
        )
        self.stable_steps = self.stable_steps + 1 if stable else 0
        if self.stable_steps >= self.dwell_steps:
            self.events["stable_dwell"] = time_s
            self.success = True
            self.terminal_time_s = time_s
            self.events["success"] = time_s
        elif time_s >= self.horizon_s - 1e-10:
            self._frozen_stage_facts = self._stage_facts()
            self.failure_code = "timeout"
            self.terminal_time_s = self.horizon_s
            self.events["timeout"] = self.horizon_s


class FreeDynamicsInterceptionEnv(DynamicManipulationEnv):
    """Release gate, ramp and capture environment."""

    _ROBOT_STAGING = {
        "ball_capture": ((0.06, 0.04, 0.961), math.pi / 2.0, math.pi / 2.0, 0.002, 100),
        "ball_capture_on_table": ((0.0, 0.07, 0.89), math.pi / 2.0, 0.0, 0.008, 100),
        "slider_capture": ((0.0, 0.07, 0.88), math.pi / 2.0, 0.0, 0.008, 100),
        "can_capture": ((0.005, 0.071, 0.89), 0.8, 0.0, 0.008, 160),
    }

    _PREPARATION_SURFACE_FRICTION = {
        "ball_capture": 0.8,
        "ball_capture_on_table": 0.8,
        "slider_capture": 0.035,
        "can_capture": 0.35,
    }

    _ASSETS = {
        "ball_capture": ("arena.unsupported_ramp_v1", "obj.sphere_r030_v1"),
        "ball_capture_on_table": ("arena.supported_ramp_v1", "obj.sphere_r030_v1"),
        "slider_capture": ("arena.supported_ramp_v1", "obj.slider_v1"),
        "can_capture": ("arena.supported_ramp_v1", "obj.can_v1"),
    }

    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        scenario_spec: ScenarioSpec,
        execution_translation_limit_m: float = 0.05,
    ) -> None:
        if task_spec.family != "free_dynamics_interception" or not isinstance(
            task_spec.bindings, FreeDynamicsInterceptionBindings
        ):
            raise ValueError("FreeDynamicsInterceptionEnv requires a registered T4 TaskSpec")
        if task_spec.task_id not in self._ASSETS:
            raise NotImplementedError(f"T4 task is not implemented: {task_spec.task_id}")
        if not isinstance(scenario_spec.geometry, FreeDynamicsInterceptionGeometrySpec):
            raise ValueError("T4 requires FreeDynamicsInterceptionGeometrySpec")
        bindings = task_spec.bindings
        fixture_ids = (
            bindings.ramp_fixture_id,
            bindings.release_gate_fixture_id,
            bindings.downstream_surface_fixture_id,
        )
        fixtures = {item.instance_id: item for item in scenario_spec.fixture_states}
        bodies = {item.entity_id: item for item in scenario_spec.physics.bodies}
        contacts = {item.entity_id: item for item in scenario_spec.physics.contacts}
        if (
            len(scenario_spec.object_states) != 1
            or scenario_spec.object_states[0].instance_id != bindings.target_object_id
            or len(fixtures) != 3
            or set(fixtures) != set(fixture_ids)
            or len(bodies) != 1
            or set(bodies) != {bindings.target_object_id}
            or len(contacts) != 4
            or set(contacts) != {bindings.target_object_id, *fixture_ids}
            or any(
                item.entity_id == bindings.release_gate_fixture_id
                for item in scenario_spec.physics.joints
            )
            or scenario_spec.motion
            or len(scenario_spec.events) != 1
        ):
            raise ValueError(
                "T4 requires exactly its target, three static fixtures, and tick-0 release event"
            )
        event = scenario_spec.events[0]
        if not isinstance(event.effect, ReleaseGateRetractionEffectSpec) or (
            event.effect.gate_entity_id != bindings.release_gate_fixture_id
        ):
            raise ValueError("T4 requires its bound ReleaseGateRetractionEffect")
        if scenario_spec.object_states[0].linear_velocity_world_m_s != (
            0.0,
            0.0,
            0.0,
        ) or scenario_spec.object_states[0].angular_velocity_world_rad_s != (0.0, 0.0, 0.0):
            raise ValueError("T4 preparation input requires zero target twist")
        arena_id, target_id = self._ASSETS[task_spec.task_id]
        if scenario_spec.object_states[0].asset_id != target_id or any(
            fixture.asset_id != arena_id for fixture in fixtures.values()
        ):
            raise ValueError("T4 requires its frozen target and composite arena assets")

        catalog = _load_asset_catalog()
        self._arena_asset = catalog[arena_id]
        self._target_asset = catalog[target_id]
        self._target_spec = scenario_spec.object_states[0]
        self._preparation_target_spec = scenario_spec.preparation_object_states[0]
        self._fixture_specs = fixtures
        self._body_spec = bodies[bindings.target_object_id]
        self._contact_specs = contacts
        self._geometry = scenario_spec.geometry
        self._bindings = bindings
        self._event_runtime = _EventRuntime(scenario_spec.events)
        self._outcome = _FreeDynamicsOutcome(task_spec, self._geometry)
        self._settling = False
        self._outcome_active = False
        self._prepared = False
        self._preparation_steps = 0
        self._ramp_support_steps = 0
        self._release_command: _ReleaseGateCommand | None = None
        self._release_event_time_s: float | None = None
        self._release_linear_speed_m_s: float | None = None
        self._release_angular_speed_rad_s: float | None = None
        self._release_event_fired = False
        self._release_interval_first_time_s: float | None = None
        self._release_interval_last_time_s: float | None = None
        self._release_interval_first_velocity_m_s: float | None = None
        self._release_interval_last_velocity_m_s: float | None = None
        self._release_along_slope_velocity_gain_m_s: float | None = None
        self._release_mean_along_slope_acceleration_m_s2: float | None = None
        self._release_pre_robot_physics_steps = 0
        self._release_robot_contact_seen = False
        self._reset_scenario: ScenarioSpec | None = None
        super().__init__(
            task_spec=task_spec,
            scenario_spec=scenario_spec,
            seed=0,
            execution_translation_limit_m=execution_translation_limit_m,
        )

    def _load_model(self) -> None:
        self._settling = self._outcome_active = self._prepared = False
        super()._load_model()
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(np.asarray(robot_model.base_xpos_offset["table"](0.8)))
        arena = TableArena(table_offset=(0.0, 0.0, self._geometry.support_surface_z_m))
        arena.set_origin((0.0, 0.0, 0.0))
        for name in ("left_eef_target", "right_eef_target"):
            body = arena.worldbody.find(f"./body[@name='{name}']")
            if body is None:
                raise ValueError(f"native TableArena is missing {name}")
            arena.worldbody.remove(body)
        lights = arena.worldbody.findall("light")
        if len(lights) != 1:
            raise ValueError("native TableArena requires one canonical light")
        lights[0].set("name", "free_dynamics_key")
        downstream_id = self._bindings.downstream_surface_fixture_id
        arena.table_body.set("name", downstream_id)
        table_collision = arena.table_body.find("geom[@name='table_collision']")
        table_visual = arena.table_body.find("geom[@name='table_visual']")
        if table_collision is None or table_visual is None:
            raise ValueError("native TableArena requires its collision and visual table")
        table_collision.set("name", "table_base_collision")
        table_collision.set("pos", "0 0 -0.005")
        table_collision.set("size", "0.4 0.4 0.02")
        table_visual.attrib.pop("material", None)
        table_visual.set("rgba", "0.58 0.60 0.62 1")
        ET.SubElement(
            arena.table_body,
            "geom",
            {
                "name": downstream_id + "_collision",
                "type": "box",
                "size": "0.28 0.4 0.005",
                "pos": "0 0 0.02",
                "group": "0",
                "friction": "1 0.005 0.0001",
                "rgba": "0 0 0 0",
                "condim": "3",
            },
        )
        arena.merge(MujocoXML(str(self._arena_asset.path)))
        for entity_id in (
            self._bindings.ramp_fixture_id,
            self._bindings.release_gate_fixture_id,
            self._bindings.downstream_surface_fixture_id,
        ):
            element = arena.worldbody.find(f"./body[@name='{entity_id}']")
            if element is None:
                raise ValueError(f"T4 composite arena is missing fixture body {entity_id!r}")
            fixture = self._fixture_specs[entity_id]
            x, y, z, w = fixture.quaternion_world_xyzw
            element.set("pos", _xml_numbers(fixture.position_world_m))
            element.set("quat", _xml_numbers((w, x, y, z)))
        ramp = arena.worldbody.find(
            f"./body[@name='{self._bindings.ramp_fixture_id}']/geom[@name='ramp_collision']"
        )
        ramp_visual = arena.worldbody.find(
            f"./body[@name='{self._bindings.ramp_fixture_id}']/geom[@name='ramp_visual']"
        )
        if ramp is None or ramp_visual is None:
            raise ValueError(
                "T4 composite arena must expose its ramp collision and visual geometry"
            )
        angle = self._geometry.ramp_angle_rad
        ramp_rotation = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, math.cos(angle), math.sin(angle)),
                (0.0, -math.sin(angle), math.cos(angle)),
            ),
            dtype=np.float64,
        )
        ramp_center = ramp_rotation @ np.asarray(
            (0.0, -self._geometry.ramp_half_size_m[1], -self._geometry.ramp_half_size_m[2])
        )
        ramp_quaternion_array = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ramp_quaternion_array, ramp_rotation.ravel())
        if self._task_spec.task_id == "ball_capture_on_table":
            ramp_quaternion_array /= np.linalg.norm(ramp_quaternion_array)
        ramp_quaternion = tuple(float(value) for value in ramp_quaternion_array)
        for geom in (ramp, ramp_visual):
            geom.set("pos", _xml_numbers(ramp_center))
            geom.set("quat", _xml_numbers(ramp_quaternion))
            geom.set("size", _xml_numbers(self._geometry.ramp_half_size_m))
        ramp_visual.set(
            "rgba",
            _xml_numbers(
                {
                    "can_capture": (0.20, 0.24, 0.29, 1.0),
                    "slider_capture": (0.22, 0.26, 0.30, 1.0),
                    "ball_capture": (0.18, 0.24, 0.30, 1.0),
                    "ball_capture_on_table": (0.18, 0.24, 0.30, 1.0),
                }[self._task_spec.task_id]
            ),
        )
        ramp_body = arena.worldbody.find(f"./body[@name='{self._bindings.ramp_fixture_id}']")
        assert ramp_body is not None
        posts = ramp_body.findall("body")
        supported = self._task_spec.task_id != "ball_capture"
        if len(posts) != (2 if supported else 4):
            raise ValueError("T4 arena has the wrong number of vertical ramp supports")
        half_width, half_length, half_thickness = self._geometry.ramp_half_size_m
        post_coordinates = (
            ((-0.08, -0.12), (0.08, -0.12))
            if supported
            else tuple(
                (x, y)
                for y in (-half_length + 0.035, half_length - 0.035)
                for x in (-half_width + 0.02, half_width - 0.02)
            )
        )
        ramp_position = np.asarray(
            self._fixture_specs[self._bindings.ramp_fixture_id].position_world_m
        )
        for post, (local_x, local_y) in zip(posts, post_coordinates, strict=True):
            underside = (
                ramp_position
                + ramp_center
                + ramp_rotation @ np.asarray((local_x, local_y, -half_thickness))
            )
            half_height = (float(underside[2]) - self._geometry.support_surface_z_m) / 2.0
            if half_height <= 0:
                raise ValueError("T4 support post must reach from the table to the ramp underside")
            post_position = underside.copy()
            post_position[2] = self._geometry.support_surface_z_m + half_height
            post.set(
                "pos", _xml_numbers(tuple(float(value) for value in post_position - ramp_position))
            )
            for geom in post.findall("geom"):
                geom.set(
                    "size",
                    _xml_numbers(
                        (0.008 if supported else 0.01, 0.008 if supported else 0.01, half_height)
                    ),
                )
        gate_collision = arena.worldbody.find(
            f"./body[@name='{self._bindings.release_gate_fixture_id}']/geom[@name='gate_collision']"
        )
        gate_visual = arena.worldbody.find(
            f"./body[@name='{self._bindings.release_gate_fixture_id}']/geom[@name='gate_visual']"
        )
        if gate_collision is None or gate_visual is None:
            raise ValueError(
                "T4 composite arena must expose its gate collision and visual geometry"
            )
        for geom in (gate_collision, gate_visual):
            geom.set("quat", _xml_numbers(ramp_quaternion))
        self._target_model = MujocoXMLObject(
            str(self._target_asset.path),
            name=self._target_spec.instance_id,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot_model],
            mujoco_objects=[self._target_model],
        )
        # Preserve the original contact ordering: its ramp and gate follow the target.
        for entity_id in (self._bindings.ramp_fixture_id, self._bindings.release_gate_fixture_id):
            fixture_body = self.model.worldbody.find(f"./body[@name='{entity_id}']")
            assert fixture_body is not None
            self.model.worldbody.remove(fixture_body)
            self.model.worldbody.append(fixture_body)

    def edit_model_xml(self, xml_str: str) -> str:
        return _rewrite_scenario_lighting_and_gravity_xml(
            super().edit_model_xml(xml_str),
            self._scenario_spec,
            light_id="free_dynamics_key",
            invalid_message="T4 requires the explicit free_dynamics_key light",
            duplicate_message="duplicate free_dynamics_key light",
            missing_message="free_dynamics_key must resolve in the composite arena",
            headlight_before_gravity=False,
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        model = self.sim.model._model
        self._target_body_id = int(model.body(self._target_model.root_body).id)
        self._target_geom_ids = tuple(
            int(model.geom(name).id) for name in self._target_model.contact_geoms
        )
        target_joint = int(model.joint(self._target_model.joints[0]).id)
        if model.jnt_type[target_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("T4 target must remain a native free body")
        if np.any(model.body_ipos[self._target_body_id] != 0.0):
            raise ValueError("T4 target asset frame must be its compiled center of mass")
        self._target_qpos_address = int(model.jnt_qposadr[target_joint])
        self._target_qvel_address = int(model.jnt_dofadr[target_joint])
        self._ramp_body_id = int(model.body(self._bindings.ramp_fixture_id).id)
        self._gate_body_id = int(model.body(self._bindings.release_gate_fixture_id).id)
        self._downstream_body_id = int(model.body(self._bindings.downstream_surface_fixture_id).id)
        self._ramp_geom_ids = (int(model.geom("ramp_collision").id),) + tuple(
            int(model.geom(f"ramp_support_{index}_collision").id)
            for index in range(2 if self._task_spec.task_id != "ball_capture" else 4)
        )
        self._gate_geom_ids = (int(model.geom("gate_collision").id),)
        downstream_name = self._bindings.downstream_surface_fixture_id + "_collision"
        self._downstream_geom_ids = (int(model.geom(downstream_name).id),)
        self._fixture_body_ids = {
            self._bindings.ramp_fixture_id: self._ramp_body_id,
            self._bindings.release_gate_fixture_id: self._gate_body_id,
            self._bindings.downstream_surface_fixture_id: self._downstream_body_id,
        }
        self._fixture_geom_ids = {
            self._bindings.ramp_fixture_id: self._ramp_geom_ids,
            self._bindings.release_gate_fixture_id: self._gate_geom_ids,
            self._bindings.downstream_surface_fixture_id: self._downstream_geom_ids,
        }
        if model.body_jntnum[self._gate_body_id] != 0:
            raise ValueError("T4 release gate must be a static body without a joint")
        pads = self.robots[0].gripper["right"].important_geoms
        self._left_pad_ids = frozenset(int(model.geom(name).id) for name in pads["left_fingerpad"])
        self._right_pad_ids = frozenset(
            int(model.geom(name).id) for name in pads["right_fingerpad"]
        )
        self._gripper_geom_ids = frozenset(
            int(model.geom(name).id) for name in self.robots[0].gripper["right"].contact_geoms
        )
        robot_root = int(model.body(self.robots[0].robot_model.root_body).id)
        robot_bodies = {robot_root}
        for body_id in range(robot_root + 1, model.nbody):
            if int(model.body_parentid[body_id]) in robot_bodies:
                robot_bodies.add(body_id)
        self._robot_geom_ids = frozenset(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in robot_bodies
        )
        self._barrel_geom_ids: tuple[int, ...]
        if self._task_spec.task_id == "can_capture":
            self._barrel_geom_ids = (
                int(model.geom(self._target_model.naming_prefix + "barrel_collision").id),
            )
            if self._barrel_geom_ids != self._target_geom_ids:
                raise ValueError("T4 can requires the complete target collision to be its barrel")
        else:
            self._barrel_geom_ids = ()
        self._bind_scene_entities(
            target_body_id=self._target_body_id,
            entity_body_ids={
                self._bindings.target_object_id: self._target_body_id,
                **self._fixture_body_ids,
            },
        )
        self._compiled_joint_physics = _resolve_joint_physics(
            model,
            self._scenario_spec.physics.joints,
            entity_body_ids={
                self._bindings.target_object_id: self._target_body_id,
                **self._fixture_body_ids,
            },
        )

    def _reset_internal(self) -> None:
        self._settling = self._outcome_active = self._prepared = False
        self._reset_scenario = None
        self._release_command = None
        self._release_event_time_s = None
        self._release_linear_speed_m_s = None
        self._release_angular_speed_rad_s = None
        self._release_event_fired = False
        self._release_interval_first_time_s = None
        self._release_interval_last_time_s = None
        self._release_interval_first_velocity_m_s = None
        self._release_interval_last_velocity_m_s = None
        self._release_along_slope_velocity_gain_m_s = None
        self._release_mean_along_slope_acceleration_m_s2 = None
        self._release_pre_robot_physics_steps = 0
        self._release_robot_contact_seen = False
        super()._reset_internal()
        model, data = self.sim.model._model, self.sim.data._data
        for entity_id, body_id in self._fixture_body_ids.items():
            fixture = self._fixture_specs[entity_id]
            x, y, z, w = fixture.quaternion_world_xyzw
            model.body_pos[body_id] = fixture.position_world_m
            model.body_quat[body_id] = (w, x, y, z)
        model.body_mass[self._target_body_id] = self._body_spec.mass_kg
        model.body_inertia[self._target_body_id] = self._body_spec.diagonal_inertia_kg_m2
        entity_geoms = {
            self._bindings.target_object_id: self._target_geom_ids,
            **self._fixture_geom_ids,
        }
        for entity_id, geom_ids in entity_geoms.items():
            contact = self._contact_specs[entity_id]
            for geom_id in geom_ids:
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
        x, y, z, w = self._target_spec.quaternion_world_xyzw
        target = self._target_qpos_address
        data.qpos[target : target + 7] = (*self._target_spec.position_world_m, w, x, y, z)
        data.qvel[:] = 0.0
        data.xfrc_applied[:] = 0.0
        data.qfrc_applied[:] = 0.0
        robot.gripper["right"].current_action = np.zeros(2, dtype=np.float64)
        self.sim.forward()
        robot.composite_controller.part_controllers["right"].reset_goal()
        self._outcome.reset()
        self._preparation_steps = self._ramp_support_steps = 0
        self._reset_scenario = self._read_preparation_input()

    def reset(self) -> Mapping[str, object]:
        super().reset()
        model, data = self.sim.model._model, self.sim.data._data
        support_body_ids = tuple(int(model.geom_bodyid[geom]) for geom in self._ramp_geom_ids[1:])
        fixture_positions = {
            body_id: model.body_pos[body_id].copy()
            for body_id in (self._ramp_body_id, self._gate_body_id, *support_body_ids)
        }
        for body_id in (self._ramp_body_id, self._gate_body_id):
            model.body_pos[body_id] = (0.0, 0.0, -1.0)
        # Support bodies are children of the ramp in the current arena asset.
        # Zero local offsets park every support at the same old world position.
        for body_id in support_body_ids:
            model.body_pos[body_id] = 0.0
        preparation_friction = (
            self._PREPARATION_SURFACE_FRICTION[self._task_spec.task_id],
            0.005,
            0.0001,
        )
        for geom_id in (*self._target_geom_ids, *self._ramp_geom_ids, *self._gate_geom_ids):
            model.geom_friction[geom_id] = preparation_friction
        for geom_id in self._downstream_geom_ids:
            model.geom_friction[geom_id] = (1.0, 0.005, 0.0001)
        preparation = self._preparation_target_spec
        x, y, z, w = preparation.quaternion_world_xyzw
        address = self._target_qpos_address
        data.qpos[address : address + 7] = (*preparation.position_world_m, w, x, y, z)
        data.qvel[self._target_qvel_address : self._target_qvel_address + 6] = 0.0
        self.sim.forward()
        self._preposition_robot_for_release()

        # OnTimeBench installs ramp, supports, downstream friction, gate, and target in
        # this order. Forward calls retain the complete staging warm start.
        ramp_position = fixture_positions[self._ramp_body_id]
        ramp_rotation = data.xmat[self._ramp_body_id].reshape(3, 3)
        for body_id in support_body_ids:
            model.body_pos[body_id] = ramp_rotation.T @ (
                np.asarray((0.0, 0.0, -1.0)) - ramp_position
            )
        model.body_pos[self._ramp_body_id] = ramp_position
        for geom_id in self._ramp_geom_ids:
            model.geom_friction[geom_id] = self._contact_specs[
                self._bindings.ramp_fixture_id
            ].friction_slide_spin_roll
        self.sim.forward()
        for body_id in support_body_ids:
            model.body_pos[body_id] = fixture_positions[body_id]
        self.sim.forward()
        for geom_id in self._downstream_geom_ids:
            model.geom_friction[geom_id] = self._contact_specs[
                self._bindings.downstream_surface_fixture_id
            ].friction_slide_spin_roll
        model.body_pos[self._gate_body_id] = fixture_positions[self._gate_body_id]
        for geom_id in self._gate_geom_ids:
            model.geom_friction[geom_id] = self._contact_specs[
                self._bindings.release_gate_fixture_id
            ].friction_slide_spin_roll
        self.sim.forward()
        for geom_id in self._target_geom_ids:
            model.geom_friction[geom_id] = self._contact_specs[
                self._bindings.target_object_id
            ].friction_slide_spin_roll
        x, y, z, w = self._target_spec.quaternion_world_xyzw
        data.qpos[address : address + 7] = (*self._target_spec.position_world_m, w, x, y, z)
        data.qvel[self._target_qvel_address : self._target_qvel_address + 6] = 0.0
        self.sim.forward()
        self._settling = True
        neutral = np.zeros(self.action_dim, dtype=np.float64)
        neutral[-1] = -1.0
        for _ in range(_PREPARATION_CONTROL_TICKS):
            ManipulationEnv.step(self, neutral)
        self._settling = False
        if self._preparation_steps != _PREPARATION_CONTROL_TICKS * 25:
            raise _ScenarioResetRejected("hidden_settle_incomplete", float(self.sim.data.time))
        if _has_contact_between(
            data, self._robot_geom_ids, frozenset((*self._ramp_geom_ids, *self._gate_geom_ids))
        ):
            raise _ScenarioResetRejected("robot_ramp_contact_during_preparation", float(data.time))
        settled_twist = data.qvel[self._target_qvel_address : self._target_qvel_address + 6]
        if (
            np.linalg.norm(settled_twist[:3]) > _RELEASE_LINEAR_LIMIT_M_S
            or np.linalg.norm(settled_twist[3:]) > _RELEASE_ANGULAR_LIMIT_RAD_S
        ):
            raise _ScenarioResetRejected("target_release_twist_not_settled", float(data.time))
        # OnTimeBench clears only the target twist once after settling. Static gate parking
        # then changes no generalized position, velocity, or controller goal.
        data.qvel[self._target_qvel_address : self._target_qvel_address + 6] = 0.0
        self.sim.forward()
        if np.any(data.qvel[self._target_qvel_address : self._target_qvel_address + 6]):
            raise _ScenarioResetRejected("target_twist_not_zero_at_scored_t0", float(data.time))
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
        self._event_runtime.reset(simulator_time_s=0.0, joint_states={})
        self._outcome.reset()
        self._park_release_gate()
        self._outcome_active = self._prepared = True
        self.sim.forward()
        self._outcome.begin(
            eef_distance_m=float(
                np.linalg.norm(
                    data.site_xpos[self.robots[0].eef_site_id["right"]]
                    - data.xipos[self._target_body_id]
                )
            )
        )
        self._finish_preparation_controller()
        return cast(Mapping[str, object], self._get_observations(force_update=True))

    @staticmethod
    def _rotation_vector(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
        if angle <= 1e-9:
            return np.zeros(3, dtype=np.float64)
        if math.pi - angle <= 1e-6:
            eigenvalues, eigenvectors = np.linalg.eig(rotation)
            axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
            return angle * axis / np.linalg.norm(axis)
        axis = np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            )
        )
        return angle * axis / (2.0 * math.sin(angle))

    def _preposition_robot_for_release(self) -> None:
        position, yaw, pitch, tolerance, maximum_steps = self._ROBOT_STAGING[
            self._task_spec.task_id
        ]
        target = np.asarray(position, dtype=np.float64)
        data = self.sim.data._data
        site_id = self.robots[0].eef_site_id["right"]
        rotation_z = np.asarray(
            (
                (math.cos(yaw), -math.sin(yaw), 0.0),
                (math.sin(yaw), math.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        rotation_y = np.asarray(
            (
                (math.cos(pitch), 0.0, math.sin(pitch)),
                (0.0, 1.0, 0.0),
                (-math.sin(pitch), 0.0, math.cos(pitch)),
            )
        )
        desired_rotation = rotation_z @ rotation_y @ data.site_xmat[site_id].reshape(3, 3).copy()
        waypoints = (
            (target,)
            if self._task_spec.task_id == "ball_capture"
            else (
                np.asarray((target[0], target[1], max(target[2] + 0.15, 1.02))),
                target,
            )
        )
        for waypoint in waypoints:
            for _ in range(maximum_steps):
                error = waypoint - data.site_xpos[site_id]
                rotation_error = self._rotation_vector(
                    desired_rotation @ data.site_xmat[site_id].reshape(3, 3).T
                )
                if np.linalg.norm(error) <= tolerance and np.linalg.norm(rotation_error) <= 0.15:
                    break
                action = self._preparation_controller_action(error, rotation_error, -1.0)
                ManipulationEnv.step(self, action)
                self.done = False
                self.timestep = 0
                if _has_contact_between(
                    data,
                    self._robot_geom_ids,
                    frozenset((*self._ramp_geom_ids, *self._gate_geom_ids)),
                ):
                    raise _ScenarioResetRejected(
                        "robot_ramp_contact_during_staging", float(data.time)
                    )
            else:
                raise _ScenarioResetRejected("robot_staging_not_reached", float(data.time))

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
        robot_state = RobotInitialStateSpec(
            cast(
                tuple[float, float, float, float, float, float, float],
                tuple(float(v) for v in data.qpos[robot._ref_joint_pos_indexes]),
            ),
            cast(
                Vector2,
                tuple(float(v) for v in data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]),
            ),
        )
        address = self._target_qpos_address
        qpos = data.qpos[address : address + 7]
        expected_target = np.asarray(self._target_spec.quaternion_world_xyzw)[[3, 0, 1, 2]]
        expected_target /= np.linalg.norm(expected_target)
        if not math.isclose(
            abs(float(np.dot(qpos[3:], expected_target))), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("compiled T4 target orientation differs from ScenarioSpec")
        target = replace(
            self._target_spec,
            position_world_m=cast(Vector3, tuple(float(v) for v in qpos[:3])),
        )
        if np.any(data.qvel[self._target_qvel_address : self._target_qvel_address + 6]):
            raise ValueError("T4 preparation input target twist must be zero")
        if not (
            float(model.body_mass[self._target_body_id]) == self._body_spec.mass_kg
            and np.array_equal(
                model.body_inertia[self._target_body_id], self._body_spec.diagonal_inertia_kg_m2
            )
        ):
            raise ValueError("compiled T4 target body physics differs from ScenarioSpec")
        for entity_id, geom_ids in {
            self._bindings.target_object_id: self._target_geom_ids,
            **self._fixture_geom_ids,
        }.items():
            expected_contact = self._contact_specs[entity_id]
            for geom_id in geom_ids:
                actual = ContactPhysicsSpec(
                    entity_id,
                    cast(Vector3, tuple(float(v) for v in model.geom_friction[geom_id])),
                    cast(Vector2, tuple(float(v) for v in model.geom_solref[geom_id])),
                    cast(
                        tuple[float, float, float, float, float],
                        tuple(float(v) for v in model.geom_solimp[geom_id]),
                    ),
                )
                if actual != expected_contact:
                    raise ValueError("compiled T4 contact properties differ from ScenarioSpec")
        _validate_compiled_joint_physics(
            model,
            self._compiled_joint_physics,
            owner="T4",
        )
        fixtures: list[FixtureStateSpec] = []
        for original in scenario.fixture_states:
            body_id = self._fixture_body_ids[original.instance_id]
            expected_q = np.asarray(original.quaternion_world_xyzw)[[3, 0, 1, 2]]
            expected_q /= np.linalg.norm(expected_q)
            if not math.isclose(
                abs(float(np.dot(data.xquat[body_id], expected_q))),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("compiled T4 fixture orientation differs from ScenarioSpec")
            position = data.xpos[body_id].copy()
            fixtures.append(
                replace(original, position_world_m=cast(Vector3, tuple(float(v) for v in position)))
            )
        ramp_geom = self._ramp_geom_ids[0]
        ramp_rotation = data.geom_xmat[ramp_geom].reshape(3, 3)
        compiled_lip = data.geom_xpos[ramp_geom] + ramp_rotation @ np.asarray(
            (0.0, self._geometry.ramp_half_size_m[1], self._geometry.ramp_half_size_m[2])
        )
        if not (
            np.allclose(
                model.geom_size[self._ramp_geom_ids[0]],
                self._geometry.ramp_half_size_m,
                rtol=0.0,
                atol=1e-12,
            )
            and np.allclose(
                compiled_lip,
                self._geometry.ramp_downstream_lip_world_m,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("compiled T4 ramp geometry differs from GeometrySpec")
        compiled_angle = math.atan2(-ramp_rotation[2, 1], ramp_rotation[1, 1])
        if not math.isclose(
            compiled_angle,
            self._geometry.ramp_angle_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("compiled T4 ramp angle differs from GeometrySpec")
        support_id = self._downstream_geom_ids[0]
        support_top = float(data.geom_xpos[support_id, 2] + model.geom_size[support_id, 2])
        if not math.isclose(
            support_top, self._geometry.support_surface_z_m, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("compiled T4 downstream support differs from GeometrySpec")
        for name, camera in zip(self._camera_sensor_names, scenario.cameras.views, strict=True):
            camera_id = int(model.camera(name).id)
            expected_q = np.asarray(camera.quaternion_parent_xyzw)[[3, 0, 1, 2]]
            expected_q /= np.linalg.norm(expected_q)
            if not (
                np.allclose(
                    model.cam_pos[camera_id],
                    camera.position_parent_m,
                    rtol=0.0,
                    atol=1e-12,
                )
                and math.isclose(
                    abs(float(np.dot(model.cam_quat[camera_id], expected_q))),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and float(model.cam_fovy[camera_id]) == camera.vertical_fov_deg
            ):
                raise ValueError("compiled T4 camera differs from ScenarioSpec")
        if (
            model.nlight != 1
            or model.light(0).name != "free_dynamics_key"
            or not model.vis.headlight.active
            or model.light_type[0] != mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            or model.light_castshadow[0]
        ):
            raise ValueError("compiled T4 canonical light set is invalid")
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
                raise ValueError("compiled T4 light differs from ScenarioSpec")
        if (
            float(model.light_cutoff[0]) != light.cutoff_deg
            or bool(model.light_active[0]) != light.active
        ):
            raise ValueError("compiled T4 light scalar differs from ScenarioSpec")
        exported = replace(
            scenario,
            robot_initial_state=robot_state,
            object_states=(target,),
            fixture_states=tuple(fixtures),
            physics=replace(
                scenario.physics,
                gravity_world_m_s2=cast(Vector3, tuple(float(v) for v in model.opt.gravity)),
            ),
        )
        if exported != scenario:
            changed = tuple(
                field.name
                for field in fields(scenario)
                if getattr(exported, field.name) != getattr(scenario, field.name)
            )
            raise ValueError(f"compiled T4 preparation input differs from ScenarioSpec: {changed}")
        return exported

    def export_scenario(self) -> ScenarioSpec:
        if not self._prepared or self._reset_scenario is None:
            raise RuntimeError("T4 requires verified native preparation before scenario export")
        return self._reset_scenario

    def _park_release_gate(self) -> None:
        model, data = self.sim.model._model, self.sim.data._data
        if float(data.time) != 0.0 or self._release_event_fired:
            raise RuntimeError("T4 static gate release must occur once at scored t0")
        result = self._event_runtime.step(
            simulator_time_s=0.0,
            control_tick=0,
            contacts=(),
            joint_states={},
        )
        if (
            result.fired_event_ids != ("release_gate",)
            or len(result.actuator_commands) != 1
            or not isinstance(result.actuator_commands[0], _ReleaseGateCommand)
        ):
            raise RuntimeError("T4 reset must fire its single static gate release")
        command = result.actuator_commands[0]
        if command.gate_entity_id != self._bindings.release_gate_fixture_id:
            raise RuntimeError("release event command is bound to the wrong native gate")
        model.body_pos[self._gate_body_id] = command.parked_position_world_m
        self.sim.forward()
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, self._target_body_id, velocity, 0
        )
        if not np.isfinite(velocity).all():
            raise RuntimeError("T4 release event twist must be finite")
        self._release_command = command
        self._release_event_fired = True
        self._release_event_time_s = 0.0
        self._release_angular_speed_rad_s = float(np.linalg.norm(velocity[:3]))
        self._release_linear_speed_m_s = float(np.linalg.norm(velocity[3:]))
        self._outcome.events["release_gate_opens"] = 0.0

    def _pre_action(self, action: npt.NDArray[np.float64], policy_step: bool = False) -> None:
        data = self.sim.data._data
        if not _physics_state_is_valid(data):
            if not self._outcome_active:
                raise _ScenarioResetRejected("invalid_state", float(data.time))
            self._outcome.invalidate(self._outcome.physics_step * self.model_timestep)
        super()._pre_action(action, policy_step)

    def _ramp_exited(self, *, target_ramp_contact: bool) -> bool:
        return (
            not target_ramp_contact
            and self._ramp_coordinate()[1] >= self._geometry.ramp_half_size_m[1]
        )

    def _ramp_coordinate(self) -> npt.NDArray[np.float64]:
        data = self.sim.data._data
        geom_id = self._ramp_geom_ids[0]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        origin = data.geom_xpos[geom_id].copy()
        if self._task_spec.task_id == "ball_capture_on_table":
            angle = self._geometry.ramp_angle_rad
            rotation = np.asarray(
                (
                    (1.0, 0.0, 0.0),
                    (0.0, math.cos(angle), math.sin(angle)),
                    (0.0, -math.sin(angle), math.cos(angle)),
                ),
                dtype=np.float64,
            )
        if self._task_spec.task_id == "ball_capture":
            origin += rotation[:, 2] * self._geometry.ramp_half_size_m[2]
        return np.asarray(rotation.T @ (data.xpos[self._target_body_id] - origin), dtype=np.float64)

    def _scored_contact_facts(
        self,
    ) -> tuple[float, float, bool, bool, bool, bool, bool, bool, bool]:
        """Read one physics sample, retaining the distinct force-sum and impulse-max rules."""
        model, data = self.sim.model._model, self.sim.data._data
        target_ids = frozenset(self._target_geom_ids)
        ramp_ids = frozenset(self._ramp_geom_ids)
        gate_ids = frozenset(self._gate_geom_ids)
        downstream_ids = frozenset(self._downstream_geom_ids)
        stage_ids = ramp_ids | gate_ids
        left = right = ramp = gate = downstream = robot_target = robot_ramp = False
        ramp_force = gate_force = 0.0
        force = np.empty(6, dtype=np.float64)
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            robot_ramp |= (a in self._robot_geom_ids and b in stage_ids) or (
                b in self._robot_geom_ids and a in stage_ids
            )
            if a not in target_ids and b not in target_ids:
                continue
            other = b if a in target_ids else a
            left |= other in self._left_pad_ids
            right |= other in self._right_pad_ids
            downstream |= other in downstream_ids
            robot_target |= other in self._robot_geom_ids
            if other in ramp_ids or other in gate_ids:
                mujoco.mj_contactForce(model, data, index, force)
                if not np.isfinite(force).all() or force[0] < 0.0:
                    raise RuntimeError("invalid MuJoCo contact force")
                if other in ramp_ids:
                    ramp = True
                    ramp_force += float(force[0])
                if other in gate_ids:
                    gate = True
                    gate_force = max(gate_force, float(np.linalg.norm(force[:3])))
        if not math.isfinite(ramp_force):
            raise RuntimeError("aggregated contact force must be finite")
        gate_impulse = gate_force * self.model_timestep
        if not math.isfinite(gate_impulse):
            raise RuntimeError("aggregated contact impulse must be finite")
        return (
            ramp_force,
            gate_impulse,
            left,
            right,
            ramp,
            gate,
            downstream,
            robot_target,
            robot_ramp,
        )

    def _update_observables(self, force: bool = False) -> None:
        if not force and (self._settling or self._outcome_active):
            data = self.sim.data._data
            finite = _physics_state_is_valid(data)
            model = self.sim.model._model
            if not finite:
                if self._settling:
                    raise _ScenarioResetRejected("invalid_state", float(data.time))
                self._outcome.invalidate((self._outcome.physics_step + 1) * self.model_timestep)
                super()._update_observables(force)
                return
            if self._settling:
                ramp_force = _normal_force_between(
                    model, data, frozenset(self._target_geom_ids), frozenset(self._ramp_geom_ids)
                )
                self._preparation_steps += 1
                self._ramp_support_steps = self._ramp_support_steps + 1 if ramp_force > 0.0 else 0
            else:
                (
                    ramp_force,
                    gate_impulse,
                    left,
                    right,
                    target_ramp_contact,
                    target_gate_contact,
                    downstream_contact,
                    robot_target,
                    robot_ramp,
                ) = self._scored_contact_facts()
                contact_ramp_exited = "ramp_exit" in self._outcome.events or self._ramp_exited(
                    target_ramp_contact=target_ramp_contact
                )
                if robot_target and self._outcome.terminal_time_s is None:
                    self._capture_first_contact(
                        "robot_target", self._target_body_id, ramp_exited=contact_ramp_exited
                    )
                if (left or right) and self._outcome.terminal_time_s is None:
                    self._capture_first_contact(
                        "fingerpad_target", self._target_body_id, ramp_exited=contact_ramp_exited
                    )
                velocity = np.empty(6, dtype=np.float64)
                mujoco.mj_objectVelocity(
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self._target_body_id,
                    velocity,
                    0,
                )
                if (
                    self._release_event_time_s is not None
                    and self._task_spec.task_id in ("slider_capture", "can_capture")
                    and self._release_along_slope_velocity_gain_m_s is None
                    and not self._release_robot_contact_seen
                ):
                    if robot_target:
                        self._release_robot_contact_seen = True
                    elif not target_gate_contact and ramp_force > 0.0:
                        ramp_axis = data.geom_xmat[self._ramp_geom_ids[0]].reshape(3, 3)[:, 1]
                        along_slope = float(np.dot(velocity[3:], ramp_axis))
                        if not math.isfinite(along_slope):
                            raise RuntimeError("T4 along-slope velocity must be finite")
                        if self._release_pre_robot_physics_steps == 0:
                            self._release_interval_first_time_s = float(data.time)
                            self._release_interval_first_velocity_m_s = along_slope
                        else:
                            previous_time = self._release_interval_last_time_s
                            if previous_time is None or not math.isclose(
                                float(data.time),
                                previous_time + self.model_timestep,
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            ):
                                raise RuntimeError(
                                    "T4 release dynamics require consecutive physics states"
                                )
                        self._release_pre_robot_physics_steps += 1
                        self._release_interval_last_time_s = float(data.time)
                        self._release_interval_last_velocity_m_s = along_slope
                        if (
                            self._release_pre_robot_physics_steps
                            == _RELEASE_DYNAMICS_INTERVAL_STEPS
                        ):
                            first_time = self._release_interval_first_time_s
                            first_velocity = self._release_interval_first_velocity_m_s
                            if first_time is None or first_velocity is None:
                                raise RuntimeError("T4 release dynamics interval is incomplete")
                            duration = float(data.time) - first_time
                            expected_duration = (
                                _RELEASE_DYNAMICS_INTERVAL_STEPS - 1
                            ) * self.model_timestep
                            if not math.isclose(
                                duration, expected_duration, rel_tol=0.0, abs_tol=1e-12
                            ):
                                raise RuntimeError("T4 release dynamics duration is invalid")
                            gain = along_slope - first_velocity
                            mean_acceleration = gain / duration
                            if not math.isfinite(gain) or not math.isfinite(mean_acceleration):
                                raise RuntimeError("T4 release dynamics must be finite")
                            self._release_along_slope_velocity_gain_m_s = gain
                            self._release_mean_along_slope_acceleration_m_s2 = mean_acceleration
                    else:
                        self._release_interval_first_time_s = None
                        self._release_interval_last_time_s = None
                        self._release_interval_first_velocity_m_s = None
                        self._release_interval_last_velocity_m_s = None
                        self._release_pre_robot_physics_steps = 0
                can_alignment = (
                    abs(float(data.xmat[self._target_body_id].reshape(3, 3)[2, 2]))
                    if self._barrel_geom_ids
                    else 0.0
                )
                self._outcome.update(
                    time_s=float(data.time),
                    finite=finite,
                    position=data.xpos[self._target_body_id],
                    planar_speed_m_s=float(np.linalg.norm(velocity[3:5])),
                    eef_distance_m=float(
                        np.linalg.norm(
                            data.site_xpos[self.robots[0].eef_site_id["right"]]
                            - data.xpos[self._target_body_id]
                        )
                    ),
                    ramp_coordinate_m=self._ramp_coordinate(),
                    ramp_exited=self._ramp_exited(target_ramp_contact=target_ramp_contact),
                    target_gate_impulse_n_s=gate_impulse,
                    target_downstream_contact=downstream_contact,
                    robot_target_contact=robot_target,
                    robot_ramp_contact=robot_ramp,
                    left=left,
                    right=right,
                    can_axis_vertical_alignment=can_alignment,
                )
        super()._update_observables(force)

    def _release_dynamics_rejection(self) -> str | None:
        if self._release_event_time_s is None:
            return "release_event_twist_missing"
        if self._release_linear_speed_m_s is None or self._release_angular_speed_rad_s is None:
            return "release_twist_missing"
        if (
            self._release_linear_speed_m_s > _RELEASE_LINEAR_LIMIT_M_S
            or self._release_angular_speed_rad_s > _RELEASE_ANGULAR_LIMIT_RAD_S
        ):
            return "release_twist_exceeded"
        if self._task_spec.task_id in ("slider_capture", "can_capture") and (
            self._release_pre_robot_physics_steps < _RELEASE_DYNAMICS_INTERVAL_STEPS
            or self._release_along_slope_velocity_gain_m_s is None
            or self._release_mean_along_slope_acceleration_m_s2 is None
            or self._release_along_slope_velocity_gain_m_s <= 0.0
            or self._release_mean_along_slope_acceleration_m_s2 <= 0.0
        ):
            return "nonpositive_along_slope_dynamics"
        return None

    def _post_action(self, action: npt.NDArray[np.float64]) -> tuple[float, bool, dict[str, Any]]:
        reward, _, info = super()._post_action(action)
        self.done = self._outcome_active and self._outcome.terminal_time_s is not None
        info.update(
            success=self._outcome.success,
            failure_code=self._outcome.failure_code,
            terminal_time_s=self._outcome.terminal_time_s,
            events={
                name: time_s
                for name, time_s in self._outcome.events.items()
                if name not in ("robot_contact", "downstream_contact")
            },
            stage_facts=self._outcome._stage_facts(),
            release_impulse_n_s=self._outcome.release_impulse_n_s,
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
