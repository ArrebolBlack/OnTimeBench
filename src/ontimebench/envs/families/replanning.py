"""Grasp a puck on a conveyor after a striker hits it (T5)."""

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
    _collision_bottom_z,
    _distance_stage_progress,
    _duration_stage_progress,
    _fingerpad_contacts,
    _load_asset_catalog,
    _ordered_stage_facts,
    _physics_state_is_valid,
    _RecordingFrameFacts,
    _resolve_joint_physics,
    _rewrite_scenario_lighting_and_gravity_xml,
    _ScenarioResetRejected,
    _validate_compiled_joint_physics,
    _xml_numbers,
)
from ontimebench.envs.components.events import (
    _EventRuntime,
    _JointState,
    _mujoco_contact_samples,
    _StrikerCommand,
    _StrikerPulseSnapshot,
)
from ontimebench.envs.components.motion import _conveyor_surface_velocity
from ontimebench.scenarios.spec import (
    BodyPhysicsSpec,
    ConstantVelocityTranslation,
    ContactPhysicsSpec,
    FixedRotation,
    FixtureStateSpec,
    ReplanningGeometrySpec,
    RobotInitialStateSpec,
    ScenarioSpec,
    StateTriggerSpec,
    StrikerPulseEffectSpec,
    Vector2,
    Vector3,
    Vector4,
)
from ontimebench.tasks.spec import ReplanningBindings, ReplanningSuccessSpec, TaskSpec

_OUTGOING_VALIDATION_DELAY_S = 0.030
_MINIMUM_LATERAL_SPEED_M_S = 0.035
_MINIMUM_YAW_RATE_RAD_S = 0.35


class _ReplanningOutcome:
    """Track the terminal state and capture at the physics rate."""

    def __init__(
        self,
        task: TaskSpec,
        geometry: ReplanningGeometrySpec,
        effect: StrikerPulseEffectSpec,
        *,
        selected_branch: str,
    ) -> None:
        if not isinstance(task.success, ReplanningSuccessSpec):
            raise ValueError("T5 requires ReplanningSuccessSpec")
        if selected_branch not in ("left", "right"):
            raise ValueError("T5 selected branch must be left or right")
        self.success_spec = task.success
        self.geometry = geometry
        self.effect = effect
        self.selected_branch = selected_branch
        self.timestep_s = 0.002
        self.horizon_s = task.episode_horizon_steps / 20.0
        self.hold_steps = round(task.success.bilateral_contact_duration_s / self.timestep_s)
        self.dwell_steps = round(task.success.stable_hold_duration_s / self.timestep_s)
        self.reset()

    def reset(self) -> None:
        self.physics_step = 0
        self.event_trigger_time_s: float | None = None
        self.impact_onset_time_s: float | None = None
        self.impact_end_time_s: float | None = None
        self._impact_end_step: int | None = None
        self.outgoing_validation_time_s: float | None = None
        self.incoming_twist_world: tuple[float, ...] | None = None
        self.outgoing_twist_world: tuple[float, ...] | None = None
        self._previous_impact_active = False
        self._maximum_impact_group_count = 0
        self._impact_contact_steps = self._impact_separation_steps = 0
        self._completed_impacts = 0
        self.bilateral_steps = 0
        self.stable_steps = 0
        self._capture_reference_z_m: float | None = None
        self._post_hold_contact_valid = False
        self.events: dict[str, float] = {}
        self.failure_code: str | None = None
        self.terminal_time_s: float | None = None
        self.success = False
        self._approach_start_m: float | None = None
        self._approach_best_m: float | None = None
        self._clearance_start_m: float | None = None
        self._clearance_best_m: float | None = None
        self._frozen_stage_facts: tuple[tuple[str, bool, float, int], ...] | None = None

    def _stage_facts(self) -> tuple[tuple[str, bool, float, int], ...]:
        if self._frozen_stage_facts is not None:
            return self._frozen_stage_facts
        return _ordered_stage_facts(
            (
                (
                    "valid_selected_impact",
                    "valid_selected_impact" in self.events,
                    0.0,
                    0,
                ),
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
            self._frozen_stage_facts = self._stage_facts()
            self.failure_code = "invalid_state"
            self.terminal_time_s = time_s
            self.events["invalid_state"] = time_s

    def update(
        self,
        *,
        time_s: float,
        finite: bool,
        position: npt.NDArray[np.float64],
        bottom_z_m: float,
        twist_world: npt.NDArray[np.float64],
        eef_distance_m: float,
        event_fired: bool,
        striker_state: _StrikerPulseSnapshot | None,
        selected_contact: bool,
        nonselected_contact: bool,
        robot_target_contact: bool,
        unsafe_contact: bool,
        left: bool,
        right: bool,
        opposing: bool,
        grasp_axis_error_rad: float,
    ) -> None:
        del opposing
        if self.terminal_time_s is not None:
            return
        stage_facts_before = self._stage_facts()
        next_step = self.physics_step + 1
        if not math.isclose(time_s, next_step * self.timestep_s, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("T5 outcome requires consecutive simulator physics states")
        self.physics_step = next_step
        if (
            not finite
            or not np.isfinite(position).all()
            or not np.isfinite(twist_world).all()
            or not all(
                math.isfinite(value) for value in (bottom_z_m, eef_distance_m, grasp_axis_error_rad)
            )
        ):
            self._frozen_stage_facts = stage_facts_before
            self.failure_code = "invalid_state"
            self.terminal_time_s = time_s
            self.events["invalid_state"] = time_s
            return

        if event_fired and self.event_trigger_time_s is None:
            if striker_state is None:
                raise RuntimeError("T5 fired event is missing its striker state")
            self.event_trigger_time_s = striker_state.trigger_time_s
            self.events["selected_strike_triggered"] = striker_state.trigger_time_s

        failure: str | None = None
        if unsafe_contact:
            failure = "unsafe_contact"
        elif robot_target_contact and self._completed_impacts == 0:
            failure = "early_robot_contact"
        elif nonselected_contact:
            failure = "wrong_striker_contact"
        if failure is None:
            if selected_contact:
                if not self._previous_impact_active:
                    self._maximum_impact_group_count += 1
                    if self._completed_impacts:
                        failure = "extra_impact"
                    else:
                        self._previous_impact_active = True
                        self._impact_contact_steps = 0
                        self.impact_onset_time_s = time_s
                        self.incoming_twist_world = tuple(float(value) for value in twist_world)
                        self.events["selected_impact"] = time_s
                self._impact_contact_steps += 1
                self._impact_separation_steps = 0
                if self._impact_contact_steps > round(
                    self.effect.maximum_contact_duration_s / self.timestep_s
                ):
                    failure = "striker_trap"
            elif self._previous_impact_active:
                self._impact_separation_steps += 1
                separation_steps = round(
                    self.effect.contact_grouping_hysteresis_s / self.timestep_s
                )
                if self._impact_separation_steps >= separation_steps:
                    self._previous_impact_active = False
                    self._completed_impacts += 1
                    self._impact_end_step = self.physics_step - separation_steps
                    self.impact_end_time_s = time_s - separation_steps * self.timestep_s
                    self.events.setdefault("selected_impact_end", self.impact_end_time_s)
        if (
            failure is None
            and self._impact_end_step is not None
            and self.outgoing_validation_time_s is None
            and self.physics_step
            >= self._impact_end_step + round(_OUTGOING_VALIDATION_DELAY_S / self.timestep_s)
        ):
            branch_sign = 1.0 if self.selected_branch == "left" else -1.0
            lateral = float(twist_world[3])
            yaw_rate = float(twist_world[2])
            if (
                branch_sign * lateral < _MINIMUM_LATERAL_SPEED_M_S
                or branch_sign * yaw_rate < _MINIMUM_YAW_RATE_RAD_S
            ):
                failure = "insufficient_outgoing_change"
            else:
                self.outgoing_validation_time_s = time_s
                self.outgoing_twist_world = tuple(float(value) for value in twist_world)
                self.events["valid_selected_impact"] = time_s
                self.events["outgoing_change_validated"] = time_s
        if failure is None:
            lower, upper = self.geometry.workspace_bounds_xy_m
            if abs(float(position[0])) > 0.270 or position[1] < -0.350:
                failure = "workspace_exit"
            elif position[1] > 0.340:
                failure = "downstream_exit"
            elif position[2] < 0.760:
                failure = "drop"
            elif not (lower[0] <= position[0] <= upper[0] and lower[1] <= position[1] <= upper[1]):
                failure = "workspace_exit"
        if failure is not None:
            self._frozen_stage_facts = stage_facts_before
            self.failure_code = failure
            self.terminal_time_s = time_s
            self.events[failure] = time_s
            return

        eligible = (
            self._completed_impacts == 1
            and not self._previous_impact_active
            and self.outgoing_validation_time_s is not None
        )
        bilateral = left and right
        oriented = bilateral and grasp_axis_error_rad <= math.radians(
            self.success_spec.maximum_grasp_axis_error_deg
        )
        if eligible:
            if self._approach_start_m is None:
                self._approach_start_m = self._approach_best_m = eef_distance_m
            else:
                self._approach_best_m = min(cast(float, self._approach_best_m), eef_distance_m)
        if eligible and eef_distance_m <= 0.10:
            self.events.setdefault("approach", time_s)
        if eligible and "approach" in self.events and (left or right):
            self.events.setdefault("fingerpad_contact", time_s)
        self.bilateral_steps = self.bilateral_steps + 1 if oriented else 0
        if (
            "fingerpad_contact" in self.events
            and "bilateral_hold" not in self.events
            and self.bilateral_steps >= self.hold_steps
        ):
            self.events["bilateral_hold"] = time_s
            self._capture_reference_z_m = float(position[2])
            self._post_hold_contact_valid = True
        if "bilateral_hold" in self.events and "clearance" not in self.events:
            if not oriented:
                self._post_hold_contact_valid = False
            elif self.bilateral_steps >= self.hold_steps:
                self._post_hold_contact_valid = True
        clearance = (
            0.0
            if self._capture_reference_z_m is None
            else float(position[2]) - self._capture_reference_z_m
        )
        clearance_distance = max(
            self.success_spec.minimum_lift_or_clearance_m - clearance,
            0.0,
        )
        if "bilateral_hold" in self.events:
            if self._clearance_start_m is None:
                self._clearance_start_m = self._clearance_best_m = clearance_distance
            else:
                self._clearance_best_m = min(
                    cast(float, self._clearance_best_m), clearance_distance
                )
        if (
            "bilateral_hold" in self.events
            and oriented
            and self._post_hold_contact_valid
            and clearance_distance == 0.0
        ):
            self.events.setdefault("clearance", time_s)
        stable = (
            "clearance" in self.events
            and oriented
            and self._post_hold_contact_valid
            and clearance_distance == 0.0
        )
        self.stable_steps = self.stable_steps + 1 if stable else 0
        if self.stable_steps >= self.dwell_steps:
            self.events["stable_dwell"] = time_s
            self.success = True
            self.terminal_time_s = time_s
            self.events["success"] = time_s
        elif time_s >= self.horizon_s - 1e-10:
            self._frozen_stage_facts = self._stage_facts()
            self.failure_code = "no_selected_impact" if self._completed_impacts == 0 else "timeout"
            self.terminal_time_s = self.horizon_s
            self.events[self.failure_code] = self.horizon_s


class ReplanningEnv(DynamicManipulationEnv):
    """Conveyor with two strikers, one of which is selected per scene."""

    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        scenario_spec: ScenarioSpec,
        execution_translation_limit_m: float = 0.05,
    ) -> None:
        if (
            task_spec.task_id != "post_impact_grasp"
            or task_spec.family != "replanning"
            or not isinstance(task_spec.bindings, ReplanningBindings)
        ):
            raise ValueError("ReplanningEnv requires the registered post_impact_grasp TaskSpec")
        if not isinstance(scenario_spec.geometry, ReplanningGeometrySpec):
            raise ValueError("T5 requires ReplanningGeometrySpec")
        bindings = task_spec.bindings
        fixtures = {item.instance_id: item for item in scenario_spec.fixture_states}
        contacts = {item.entity_id: item for item in scenario_spec.physics.contacts}
        joints = {(item.entity_id, item.joint_name): item for item in scenario_spec.physics.joints}
        if (
            len(scenario_spec.object_states) != 1
            or scenario_spec.object_states[0].instance_id != bindings.target_object_id
            or len(fixtures) != 3
            or set(fixtures)
            != {
                bindings.belt_fixture_id,
                bindings.left_striker_fixture_id,
                bindings.right_striker_fixture_id,
            }
            or len(scenario_spec.motion) != 1
            or scenario_spec.motion[0].entity_id != bindings.belt_fixture_id
            or len(scenario_spec.physics.bodies) != 1
            or scenario_spec.physics.bodies[0].entity_id != bindings.target_object_id
            or set(contacts)
            != {
                bindings.target_object_id,
                bindings.belt_fixture_id,
                bindings.left_striker_fixture_id,
                bindings.right_striker_fixture_id,
            }
            or len(scenario_spec.events) != 1
        ):
            raise ValueError(
                "T5 requires exactly its target, belt, two strikers, and selected event"
            )
        event = scenario_spec.events[0]
        if not isinstance(event.trigger, StateTriggerSpec) or not isinstance(
            event.effect, StrikerPulseEffectSpec
        ):
            raise ValueError("T5 requires one StateTrigger and StrikerPulseEffect")
        effect = event.effect
        selected_branch = (
            "left"
            if effect.striker_entity_id == bindings.left_striker_fixture_id
            else "right"
            if effect.striker_entity_id == bindings.right_striker_fixture_id
            else ""
        )
        if (
            event.event_id != "selected_strike"
            or not selected_branch
            or effect.joint_name != f"{selected_branch}_slide"
            or effect.strike_position_m != scenario_spec.geometry.striker_travel_m
        ):
            raise ValueError("T5 event must select one bound striker and its native travel")
        expected_joints = {
            (bindings.left_striker_fixture_id, "left_slide"),
            (bindings.right_striker_fixture_id, "right_slide"),
        }
        if not expected_joints.issubset(joints):
            raise ValueError("T5 requires both native striker joint-physics records")
        motion = scenario_spec.motion[0]
        if not isinstance(motion.translation, ConstantVelocityTranslation) or not isinstance(
            motion.rotation, FixedRotation
        ):
            raise ValueError("T5 belt requires constant translation and fixed rotation")
        velocity = motion.translation.velocity_world_m_s
        if velocity[0] != 0.0 or velocity[1] <= 0.0 or velocity[2] != 0.0:
            raise ValueError("T5 belt must move in world +Y")
        target = scenario_spec.object_states[0]
        if target.asset_id != "obj.keyed_puck_v1" or any(
            item.asset_id != "arena.replanning_conveyor_v1" for item in fixtures.values()
        ):
            raise ValueError("T5 requires its frozen puck and composite conveyor assets")

        catalog = _load_asset_catalog()
        self._arena_asset = catalog["arena.replanning_conveyor_v1"]
        self._target_asset = catalog["obj.keyed_puck_v1"]
        self._bindings = bindings
        self._target_spec = target
        self._fixture_specs = fixtures
        self._contact_specs = contacts
        self._body_spec = scenario_spec.physics.bodies[0]
        self._motion = motion
        self._geometry = scenario_spec.geometry
        self._effect = effect
        self._selected_branch = selected_branch
        self._event_runtime = _EventRuntime(
            scenario_spec.events,
            state_trigger_target_entity_id=bindings.target_object_id,
            observation_boundary_y_m=self._geometry.observation_boundary_y_m,
        )
        self._outcome = _ReplanningOutcome(
            task_spec, self._geometry, effect, selected_branch=selected_branch
        )
        self._outcome_active = False
        self._event_fired = False
        self._last_striker_state: _StrikerPulseSnapshot | None = None
        self._striker_forces = {"left": 0.0, "right": 0.0}
        self._reset_scenario: ScenarioSpec | None = None
        super().__init__(
            task_spec=task_spec,
            scenario_spec=scenario_spec,
            seed=0,
            execution_translation_limit_m=execution_translation_limit_m,
        )

    def _load_model(self) -> None:
        self._outcome_active = False
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
        if light is None:
            raise RuntimeError("T5 requires the table light")
        light.set("name", "replanning_key")
        for entity_id in (
            self._bindings.belt_fixture_id,
            self._bindings.left_striker_fixture_id,
            self._bindings.right_striker_fixture_id,
        ):
            element = arena.worldbody.find(f"./body[@name='{entity_id}']")
            if element is None:
                raise ValueError(f"T5 composite arena is missing body {entity_id!r}")
            fixture = self._fixture_specs[entity_id]
            x, y, z, w = fixture.quaternion_world_xyzw
            element.set("pos", _xml_numbers(fixture.position_world_m))
            element.set("quat", _xml_numbers((w, x, y, z)))
        belt_geom = arena.worldbody.find(
            f"./body[@name='{self._bindings.belt_fixture_id}']/geom[@name='belt_collision']"
        )
        if belt_geom is None:
            raise ValueError("T5 composite arena is missing belt_collision")
        belt = self._fixture_specs[self._bindings.belt_fixture_id]
        x, y, z, w = belt.quaternion_world_xyzw
        quaternion = np.asarray((w, x, y, z), dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, quaternion)
        surface_velocity = _conveyor_surface_velocity(
            self._motion, world_from_geom_rotation=rotation.reshape(3, 3)
        )
        belt_geom.set("surfacevel", _xml_numbers(tuple(float(value) for value in surface_velocity)))
        self._target_model = MujocoXMLObject(
            str(self._target_asset.path),
            name=self._target_spec.instance_id,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        strikers = []
        for branch in ("left", "right"):
            body = arena.worldbody.find(f"body[@name='{branch}_striker']")
            if body is None:
                raise RuntimeError("T5 native striker body is missing")
            arena.worldbody.remove(body)
            strikers.append(body)
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot_model],
            mujoco_objects=[self._target_model],
        )
        self.model.worldbody.extend(strikers)
        ET.SubElement(
            self.model.equality,
            "weld",
            name="verified_grasp_attachment",
            body1=self.robots[0].gripper["right"].naming_prefix + "eef",
            body2=self._target_model.root_body,
            active="false",
            solref="0.002 1",
            solimp="0.95 0.99 0.001",
        )

    def edit_model_xml(self, xml_str: str) -> str:
        return _rewrite_scenario_lighting_and_gravity_xml(
            super().edit_model_xml(xml_str),
            self._scenario_spec,
            light_id="replanning_key",
            invalid_message="T5 requires the explicit replanning_key light",
            duplicate_message="duplicate replanning_key light",
            missing_message="replanning_key must resolve in the composite arena",
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
            raise ValueError("T5 target must remain a native free body")
        self._attachment_eq_id = int(model.equality("verified_grasp_attachment").id)
        self._target_qpos_address = int(model.jnt_qposadr[target_joint])
        self._target_qvel_address = int(model.jnt_dofadr[target_joint])
        self._fixture_body_ids = {
            entity_id: int(model.body(entity_id).id)
            for entity_id in (
                self._bindings.belt_fixture_id,
                self._bindings.left_striker_fixture_id,
                self._bindings.right_striker_fixture_id,
            )
        }
        self._fixture_geom_ids = {
            self._bindings.belt_fixture_id: (int(model.geom("belt_collision").id),),
            self._bindings.left_striker_fixture_id: (int(model.geom("left_striker_collision").id),),
            self._bindings.right_striker_fixture_id: (
                int(model.geom("right_striker_collision").id),
            ),
        }
        self._belt_geom_id = self._fixture_geom_ids[self._bindings.belt_fixture_id][0]
        self._joint_ids = {
            "left": int(model.joint("left_slide").id),
            "right": int(model.joint("right_slide").id),
        }
        self._joint_qpos_addresses = {
            branch: int(model.jnt_qposadr[joint_id]) for branch, joint_id in self._joint_ids.items()
        }
        self._joint_qvel_addresses = {
            branch: int(model.jnt_dofadr[joint_id]) for branch, joint_id in self._joint_ids.items()
        }
        for branch, joint_id in self._joint_ids.items():
            if (
                model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE
                or not model.jnt_limited[joint_id]
                or not np.array_equal(model.jnt_range[joint_id], (0.0, 0.110))
            ):
                raise ValueError(f"T5 {branch} striker requires one finite native slide")
        pads = self.robots[0].gripper["right"].important_geoms
        self._left_pad_ids = frozenset(int(model.geom(name).id) for name in pads["left_fingerpad"])
        self._right_pad_ids = frozenset(
            int(model.geom(name).id) for name in pads["right_fingerpad"]
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
        self._gripper_geom_ids = tuple(
            int(model.geom(name).id) for name in self.robots[0].gripper["right"].contact_geoms
        )
        self._attachment_contact_masks = {
            geom: (int(model.geom_contype[geom]), int(model.geom_conaffinity[geom]))
            for geom in (*self._target_geom_ids, *self._gripper_geom_ids)
        }
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
        self._outcome_active = False
        self._event_fired = False
        self._last_striker_state = None
        self._striker_forces = {"left": 0.0, "right": 0.0}
        self._reset_scenario = None
        super()._reset_internal()
        model, data = self.sim.model._model, self.sim.data._data
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
        address = self._target_qpos_address
        data.qpos[address : address + 7] = (*self._target_spec.position_world_m, w, x, y, z)
        data.qvel[:] = 0.0
        quaternion = np.asarray((w, x, y, z), dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, quaternion)
        velocity = self._target_qvel_address
        data.qvel[velocity : velocity + 3] = self._target_spec.linear_velocity_world_m_s
        data.qvel[velocity + 3 : velocity + 6] = (
            rotation.reshape(3, 3).T @ self._target_spec.angular_velocity_world_rad_s
        )
        for branch in ("left", "right"):
            data.qpos[self._joint_qpos_addresses[branch]] = 0.0
            data.qvel[self._joint_qvel_addresses[branch]] = 0.0
        data.xfrc_applied[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.eq_active[self._attachment_eq_id] = 0
        self._attachment_active = False
        for geom_id in self._left_pad_ids | self._right_pad_ids:
            model.geom_friction[geom_id] = (2.0, 0.05, 0.0001)
        self.sim.forward()
        selected_key = (self._effect.striker_entity_id, self._effect.joint_name)
        self._event_runtime.reset(
            simulator_time_s=0.0,
            joint_states={selected_key: _JointState(0.0, 0.0)},
            target_body_y_m=float(data.xpos[self._target_body_id, 1]),
        )
        self._outcome.reset()
        self._reset_scenario = self._read_reset_scenario()

    def reset(self) -> Mapping[str, object]:
        super().reset()
        model, data = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        belt_velocity = model.geom_surfacevel[self._belt_geom_id].copy()
        model.geom_surfacevel[self._belt_geom_id] = 0.0
        address, velocity = self._target_qpos_address, self._target_qvel_address
        start_pose = data.qpos[address : address + 7].copy()
        data.qpos[address : address + 7] = (0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0)
        data.qvel[velocity : velocity + 6] = 0.0
        self.sim.forward()
        site = robot.eef_site_id["right"]
        # Hidden preparation uses the original OSC inputs for either
        # scoring interface: two waypoints, zero rotation, and open gripper.
        for waypoint in ((0.0, 0.09, 1.05), (0.0, 0.09, 0.93)):
            for _ in range(180):
                position_error = np.asarray(waypoint) - data.site_xpos[site]
                if np.linalg.norm(position_error) <= 0.008:
                    break
                ManipulationEnv.step(
                    self,
                    self._preparation_controller_action(
                        position_error, np.zeros(3, dtype=np.float64), -1.0
                    ),
                )
                self.done, self.timestep = False, 0
            else:
                raise _ScenarioResetRejected("robot_staging_unreached", float(data.time))
        data.qpos[address : address + 7] = start_pose
        data.qvel[velocity : velocity + 6] = 0.0
        data.time = self.cur_time = 0.0
        self.timestep = self._episode_tick = 0
        self.done = False
        self.sim.forward()
        model.geom_surfacevel[self._belt_geom_id] = belt_velocity
        self._action_spec = self._action_delay_queue = None
        self._last_action_application = None
        self._previous_eef_quaternion_xyzw = None
        self._obs_cache: dict[str, Any] = {}
        for observable in self._observables.values():
            observable.reset()
        selected_key = (self._effect.striker_entity_id, self._effect.joint_name)
        self._event_runtime.reset(
            simulator_time_s=0.0,
            joint_states={selected_key: _JointState(0.0, 0.0)},
            target_body_y_m=float(data.xpos[self._target_body_id, 1]),
        )
        self._outcome.reset()
        self._outcome_active = True
        self._outcome.events["keyed_puck_enters_observation_lane"] = 0.0
        self._finish_preparation_controller()
        return cast(Mapping[str, object], self._get_observations(force_update=True))

    def _read_reset_scenario(self) -> ScenarioSpec:
        model, data = self.sim.model._model, self.sim.data._data
        scenario, robot = self._scenario_spec, self.robots[0]
        if not _physics_state_is_valid(data):
            raise ValueError("T5 reset has invalid native state")
        robot_state = RobotInitialStateSpec(
            cast(
                tuple[float, float, float, float, float, float, float],
                tuple(float(value) for value in data.qpos[robot._ref_joint_pos_indexes]),
            ),
            cast(
                Vector2,
                tuple(
                    float(value)
                    for value in data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]
                ),
            ),
        )
        address = self._target_qpos_address
        qpos = data.qpos[address : address + 7]
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_XBODY, self._target_body_id, velocity, 0
        )
        target = replace(
            self._target_spec,
            position_world_m=cast(Vector3, tuple(float(value) for value in qpos[:3])),
            quaternion_world_xyzw=cast(
                Vector4, tuple(float(value) for value in qpos[[4, 5, 6, 3]])
            ),
            linear_velocity_world_m_s=cast(Vector3, tuple(float(value) for value in velocity[3:])),
            angular_velocity_world_rad_s=cast(
                Vector3, tuple(float(value) for value in velocity[:3])
            ),
        )
        if float(
            model.body_mass[self._target_body_id]
        ) != self._body_spec.mass_kg or not np.array_equal(
            model.body_inertia[self._target_body_id], self._body_spec.diagonal_inertia_kg_m2
        ):
            raise ValueError("compiled T5 target body physics differs from ScenarioSpec")
        for entity_id, geom_ids in {
            self._bindings.target_object_id: self._target_geom_ids,
            **self._fixture_geom_ids,
        }.items():
            expected = self._contact_specs[entity_id]
            for geom_id in geom_ids:
                actual = ContactPhysicsSpec(
                    entity_id,
                    cast(Vector3, tuple(float(value) for value in model.geom_friction[geom_id])),
                    cast(Vector2, tuple(float(value) for value in model.geom_solref[geom_id])),
                    cast(
                        tuple[float, float, float, float, float],
                        tuple(float(value) for value in model.geom_solimp[geom_id]),
                    ),
                )
                if actual != expected:
                    raise ValueError("compiled T5 contact properties differ from ScenarioSpec")
        fixtures: list[FixtureStateSpec] = []
        for original in scenario.fixture_states:
            body_id = self._fixture_body_ids[original.instance_id]
            actual_q = data.xquat[body_id][[1, 2, 3, 0]]
            expected_q = np.asarray(original.quaternion_world_xyzw, dtype=np.float64)
            expected_q /= np.linalg.norm(expected_q)
            if not math.isclose(
                abs(float(np.dot(actual_q, expected_q))), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("compiled T5 fixture orientation differs from ScenarioSpec")
            fixtures.append(
                replace(
                    original,
                    position_world_m=cast(
                        Vector3, tuple(float(value) for value in data.xpos[body_id])
                    ),
                )
            )
        belt_rotation = data.geom_xmat[self._belt_geom_id].reshape(3, 3)
        world_surface = belt_rotation @ model.geom_surfacevel[self._belt_geom_id, :3]
        assert isinstance(self._motion.translation, ConstantVelocityTranslation)
        if (
            not model.flg_surfacevel
            or np.any(model.geom_surfacevel[self._belt_geom_id, 3:] != 0.0)
            or not np.allclose(
                world_surface,
                self._motion.translation.velocity_world_m_s,
                rtol=0.0,
                atol=1e-12,
            )
            or np.any(np.delete(model.geom_surfacevel, self._belt_geom_id, axis=0) != 0.0)
        ):
            raise ValueError("compiled T5 surface velocity differs from MotionSpec")
        support_top = float(
            data.geom_xpos[self._belt_geom_id, 2] + model.geom_size[self._belt_geom_id, 2]
        )
        if not math.isclose(
            support_top, self._geometry.support_surface_z_m, rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(
            _collision_bottom_z(model, data, self._target_geom_ids),
            support_top,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("compiled T5 support geometry differs from ScenarioSpec")
        _validate_compiled_joint_physics(
            model,
            self._compiled_joint_physics,
            owner="T5",
        )
        if any(
            data.qpos[self._joint_qpos_addresses[branch]] != 0.0 for branch in ("left", "right")
        ):
            raise ValueError("compiled T5 striker joint differs from ScenarioSpec")
        for name, spec in zip(self._camera_sensor_names, scenario.cameras.views, strict=True):
            camera_id = int(model.camera(name).id)
            expected_q = np.asarray(spec.quaternion_parent_xyzw)[[3, 0, 1, 2]]
            expected_q /= np.linalg.norm(expected_q)
            if not (
                np.allclose(model.cam_pos[camera_id], spec.position_parent_m, rtol=0.0, atol=1e-12)
                and math.isclose(
                    abs(float(np.dot(model.cam_quat[camera_id], expected_q))),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                and float(model.cam_fovy[camera_id]) == spec.vertical_fov_deg
            ):
                raise ValueError("compiled T5 camera differs from ScenarioSpec")
        if (
            model.nlight != 1
            or model.light(0).name != "replanning_key"
            or not model.vis.headlight.active
            or model.light_type[0] != mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            or model.light_castshadow[0]
        ):
            raise ValueError("compiled T5 canonical light set is invalid")
        exported = replace(
            scenario,
            robot_initial_state=robot_state,
            object_states=(target,),
            fixture_states=tuple(fixtures),
            physics=replace(
                scenario.physics,
                gravity_world_m_s2=cast(
                    Vector3, tuple(float(value) for value in model.opt.gravity)
                ),
                bodies=(
                    BodyPhysicsSpec(
                        self._bindings.target_object_id,
                        float(model.body_mass[self._target_body_id]),
                        cast(
                            Vector3,
                            tuple(
                                float(value) for value in model.body_inertia[self._target_body_id]
                            ),
                        ),
                    ),
                ),
            ),
        )
        if exported != scenario:
            changed = tuple(
                field.name
                for field in fields(scenario)
                if getattr(exported, field.name) != getattr(scenario, field.name)
            )
            raise ValueError(f"compiled T5 reset differs from ScenarioSpec: {changed}")
        return exported

    def export_scenario(self) -> ScenarioSpec:
        if self._reset_scenario is None:
            raise RuntimeError("T5 requires a verified native reset before export")
        return self._reset_scenario

    def _joint_states(self) -> dict[tuple[str, str], _JointState]:
        data = self.sim.data._data
        branch = self._selected_branch
        return {
            (self._effect.striker_entity_id, self._effect.joint_name): _JointState(
                float(data.qpos[self._joint_qpos_addresses[branch]]),
                float(data.qvel[self._joint_qvel_addresses[branch]]),
            )
        }

    def _set_attachment(self, active: bool) -> None:
        model, data = self.sim.model._model, self.sim.data._data
        equality = self._attachment_eq_id
        if active:
            hand = int(model.eq_obj1id[equality])
            target = self._target_body_id
            hand_rotation = data.xmat[hand].reshape(3, 3)
            relative_rotation = hand_rotation.T @ data.xmat[target].reshape(3, 3)
            relative_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.ravel())
            model.eq_data[equality, 3:6] = hand_rotation.T @ (data.xpos[target] - data.xpos[hand])
            model.eq_data[equality, 6:10] = relative_quaternion
            for geoms, category in ((self._target_geom_ids, 2), (self._gripper_geom_ids, 4)):
                for geom in geoms:
                    model.geom_contype[geom], model.geom_conaffinity[geom] = category, 1
        else:
            for geom, masks in self._attachment_contact_masks.items():
                model.geom_contype[geom], model.geom_conaffinity[geom] = masks
        data.eq_active[equality] = int(active)
        self._attachment_active = active
        self.sim.forward()

    def _advance_striker_event(self) -> None:
        model, data = self.sim.model._model, self.sim.data._data
        selected_entity = self._effect.striker_entity_id
        selected_ids = self._fixture_geom_ids[selected_entity]
        contacts = _mujoco_contact_samples(
            model,
            data,
            requested_entity_pairs=((selected_entity, self._bindings.target_object_id),),
            entity_geom_ids={
                selected_entity: selected_ids,
                self._bindings.target_object_id: self._target_geom_ids,
            },
        )
        result = self._event_runtime.step(
            simulator_time_s=float(data.time),
            control_tick=self._episode_tick,
            contacts=contacts,
            joint_states=self._joint_states(),
            target_body_y_m=float(data.xpos[self._target_body_id, 1]),
        )
        if result.fired_event_ids:
            if result.fired_event_ids != ("selected_strike",) or self._event_fired:
                raise RuntimeError("T5 selected strike must fire exactly once")
            self._event_fired = True
        command = next(
            (item for item in result.actuator_commands if isinstance(item, _StrikerCommand)),
            None,
        )
        self._striker_forces = self._next_striker_forces(command)
        self._last_striker_state = (
            result.striker_states[0] if result.striker_states else self._last_striker_state
        )

    def _pre_action(self, action: npt.NDArray[np.float64], policy_step: bool = False) -> None:
        if not _physics_state_is_valid(self.sim.data._data):
            self._outcome.invalidate(self._outcome.physics_step * self.model_timestep)
        if self._outcome_active:
            self._advance_striker_event()
        super()._pre_action(action, policy_step)
        if self._outcome_active:
            self._apply_gripper_endpoint_control()
            data = self.sim.data._data
            for branch, force in self._striker_forces.items():
                data.qfrc_applied[self._joint_qvel_addresses[branch]] = force

    def _apply_gripper_endpoint_control(self) -> None:
        """Retain The post-impact controller's physics-rate endpoint braking after Panda control."""
        model, data = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        for position_address, velocity_address, actuator_id in zip(
            robot._ref_gripper_joint_pos_indexes["right"],
            robot._ref_gripper_joint_vel_indexes["right"],
            robot._ref_joint_gripper_actuator_indexes["right"],
            strict=True,
        ):
            position = float(data.qpos[position_address])
            velocity = float(data.qvel[velocity_address])
            target = float(data.ctrl[actuator_id])
            joint = int(model.actuator_trnid[actuator_id, 0])
            lower, upper = model.jnt_range[joint]
            bounded = target
            if position >= upper - 0.00005 and target > upper - 0.00005:
                bounded = upper - 0.00005
            if position <= lower + 0.00005 and target < lower + 0.00005:
                bounded = lower + 0.00005
            if position <= lower + 0.0025 and velocity < 0.0 and target > position:
                bounded = upper
            if (
                position + self.model_timestep * velocity < lower
                and velocity < 0.0
                and target < position
            ):
                bounded = upper
            data.ctrl[actuator_id] = bounded

    def _next_striker_forces(self, command: _StrikerCommand | None) -> dict[str, float]:
        data = self.sim.data._data
        maximum = self._effect.maximum_force_n
        result: dict[str, float] = {}
        for branch in ("left", "right"):
            if command is not None and branch == self._selected_branch:
                result[branch] = command.force_n
                continue
            qpos = float(data.qpos[self._joint_qpos_addresses[branch]])
            qvel = float(data.qvel[self._joint_qvel_addresses[branch]])
            force = -self._effect.kp_n_m * qpos - self._effect.kd_n_s_m * qvel
            if branch == self._selected_branch:
                force *= self._effect.actuator_scale
            result[branch] = float(
                np.clip(
                    force,
                    -maximum,
                    maximum,
                )
            )
        return result

    def _update_observables(self, force: bool = False) -> None:
        if not force and self._outcome_active:
            data = self.sim.data._data
            finite = _physics_state_is_valid(data)
            model = self.sim.model._model
            if not finite:
                self._outcome.invalidate((self._outcome.physics_step + 1) * self.model_timestep)
                super()._update_observables(force)
                return
            target_ids = frozenset(self._target_geom_ids)
            left_striker_ids = frozenset(
                self._fixture_geom_ids[self._bindings.left_striker_fixture_id]
            )
            right_striker_ids = frozenset(
                self._fixture_geom_ids[self._bindings.right_striker_fixture_id]
            )
            striker_ids = left_striker_ids | right_striker_ids
            selected_ids = (
                left_striker_ids if self._selected_branch == "left" else right_striker_ids
            )
            nonselected_ids = striker_ids - selected_ids
            pad_ids = self._left_pad_ids | self._right_pad_ids
            selected_contact = nonselected_contact = robot_target = unsafe = False
            fingerpad_indices = []
            for index in range(data.ncon):
                contact = data.contact[index]
                a, b = int(contact.geom1), int(contact.geom2)
                if a in target_ids or b in target_ids:
                    other = b if a in target_ids else a
                    selected_contact |= other in selected_ids
                    nonselected_contact |= other in nonselected_ids
                    robot_target |= other in self._robot_geom_ids
                    if other in pad_ids:
                        fingerpad_indices.append(index)
                unsafe |= (a in self._robot_geom_ids and b in striker_ids) or (
                    b in self._robot_geom_ids and a in striker_ids
                )
            left, right, opposing = _fingerpad_contacts(
                model,
                data,
                self._target_geom_ids,
                self._left_pad_ids,
                self._right_pad_ids,
                contact_indices=tuple(fingerpad_indices),
            )
            left_center = np.mean(data.geom_xpos[sorted(self._left_pad_ids)], axis=0)
            if robot_target and self._outcome.terminal_time_s is None:
                self._capture_first_contact("robot_target", self._target_body_id)
            if (left or right) and self._outcome.terminal_time_s is None:
                self._capture_first_contact("fingerpad_target", self._target_body_id)
            right_center = np.mean(data.geom_xpos[sorted(self._right_pad_ids)], axis=0)
            closing_axis = right_center - left_center
            closing_axis /= max(float(np.linalg.norm(closing_axis)), np.finfo(np.float64).tiny)
            target_axis = data.xmat[self._target_body_id].reshape(3, 3)[:, 0]
            axis_error = math.acos(
                float(np.clip(abs(np.dot(closing_axis, target_axis)), -1.0, 1.0))
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
            self._outcome.update(
                time_s=float(data.time),
                finite=finite,
                position=data.xpos[self._target_body_id],
                bottom_z_m=_collision_bottom_z(model, data, self._target_geom_ids),
                twist_world=velocity,
                eef_distance_m=float(
                    np.linalg.norm(
                        data.site_xpos[self.robots[0].eef_site_id["right"]]
                        - data.xpos[self._target_body_id]
                    )
                ),
                event_fired=self._event_fired,
                striker_state=self._last_striker_state,
                selected_contact=selected_contact,
                nonselected_contact=nonselected_contact,
                robot_target_contact=robot_target,
                unsafe_contact=unsafe,
                left=left or self._attachment_active,
                right=right or self._attachment_active,
                opposing=opposing,
                grasp_axis_error_rad=axis_error,
            )
        super()._update_observables(force)

    def _post_action(self, action: npt.NDArray[np.float64]) -> tuple[float, bool, dict[str, Any]]:
        if (
            self._outcome_active
            and not self._attachment_active
            and "bilateral_hold" in self._outcome.events
        ):
            self._set_attachment(True)
            self._outcome.events["verified_rigid_grasp_engaged"] = self.timestep / self.control_freq
        reward, _, info = super()._post_action(action)
        self.done = self._outcome_active and self._outcome.terminal_time_s is not None
        raw_event_names = {
            "selected_strike_triggered": "side_strike_triggered",
            "selected_impact": "striker_impact_onset",
            "selected_impact_end": "striker_impact_end",
            "outgoing_change_validated": "postevent_state_validated",
            "extra_impact": "extra_striker_impact",
            "insufficient_outgoing_change": "wrong_postevent_branch",
            "no_selected_impact": "no_striker_impact",
        }
        info.update(
            success=self._outcome.success,
            failure_code=self._outcome.failure_code,
            terminal_time_s=self._outcome.terminal_time_s,
            events={
                raw_event_names.get(name, name): time_s
                for name, time_s in self._outcome.events.items()
                if name != "valid_selected_impact"
            },
            stage_facts=self._outcome._stage_facts(),
            impact_group_count=self._outcome._maximum_impact_group_count,
            incoming_twist_world=self._outcome.incoming_twist_world,
            outgoing_twist_world=self._outcome.outgoing_twist_world,
            striker_forces_n=dict(self._striker_forces),
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
