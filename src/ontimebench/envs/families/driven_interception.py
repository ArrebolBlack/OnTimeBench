"""Grasp a target moved by a conveyor surface or an applied wrench (T1)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import fields, replace
from typing import Any, cast

import mujoco
import numpy as np
import numpy.typing as npt
from robosuite.models.arenas import TableArena
from robosuite.models.base import MujocoXML
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask

from ontimebench.envs.base import (
    DynamicManipulationEnv,
    _apply_joint_physics,
    _collision_bottom_z,
    _duration_stage_progress,
    _fingerpad_contacts,
    _load_asset_catalog,
    _ordered_stage_facts,
    _physics_state_is_valid,
    _RecordingFrameFacts,
    _resolve_joint_physics,
    _rewrite_scenario_lighting_and_gravity_xml,
    _validate_compiled_joint_physics,
    _xml_numbers,
)
from ontimebench.envs.components.motion import (
    _bounded_planar_wrench,
    _conveyor_surface_velocity,
    _motion_state_at_time,
    _validate_motion_initial_pose,
)
from ontimebench.envs.components.occlusion import (
    _canopy_contact_facts,
    _canopy_model,
    _compiled_lift_envelope_clearance,
    _CompiledOcclusionFacts,
    _read_canopy_fixture,
)
from ontimebench.scenarios.spec import (
    BodyPhysicsSpec,
    ConstantAngularVelocityRotation,
    ConstantVelocityTranslation,
    ContactPhysicsSpec,
    DrivenInterceptionGeometrySpec,
    FixedRotation,
    FixtureStateSpec,
    HarmonicTranslation,
    ObjectStateSpec,
    RobotInitialStateSpec,
    ScenarioSpec,
    Vector2,
    Vector3,
    Vector4,
)
from ontimebench.tasks.spec import (
    CaptureSuccessSpec,
    DrivenInterceptionBindings,
    GraspMotion,
    TaskSpec,
)


class _DrivenCapture:
    """Track capture success and failure at the physics rate."""

    def __init__(
        self, task: TaskSpec, geometry: DrivenInterceptionGeometrySpec, timestep_s: float
    ) -> None:
        if not isinstance(task.success, CaptureSuccessSpec):
            raise ValueError("T1 requires CaptureSuccessSpec")
        self.success_spec = task.success
        self.unsafe_contact_enabled = task.has_occluder
        self.geometry = geometry
        self.timestep_s = timestep_s
        self.horizon_s = task.episode_horizon_steps / 20.0
        self.hold_steps = round(task.success.bilateral_contact_duration_s / timestep_s)
        self.dwell_steps = round(task.success.stable_hold_duration_s / timestep_s)
        self.reset()

    def reset(self) -> None:
        self.physics_step = 0
        self.bilateral_steps = 0
        self.post_hold_contact_valid = False
        self.stable_steps = 0
        self.events: dict[str, float] = {}
        self.failure_code: str | None = None
        self.terminal_time_s: float | None = None
        self.success = False
        self._eef_distance_m: float | None = None
        self._clearance_m = 0.0
        self._fingerpad_contact = False
        self._frozen_stage_facts: tuple[tuple[str, bool, float, int], ...] | None = None

    def begin(self, *, eef_distance_m: float) -> None:
        if not math.isfinite(eef_distance_m) or eef_distance_m < 0.0:
            raise ValueError(
                "T1 stage initialization requires a finite non-negative approach distance"
            )
        self._eef_distance_m = eef_distance_m
        # All four fixed T1 registrations share this scored-start event.
        self.events.setdefault("target_enters_observation_region", 0.0)

    def _require_begun(self) -> None:
        if self._eef_distance_m is None:
            raise RuntimeError("T1 capture requires begin() at scored t0")

    def _stage_facts(self) -> tuple[tuple[str, bool, float, int], ...]:
        if self._frozen_stage_facts is not None:
            return self._frozen_stage_facts
        return _ordered_stage_facts(
            (
                (
                    "approach",
                    "approach" in self.events,
                    (
                        float(
                            np.clip(
                                (0.30 - cast(float, self._eef_distance_m)) / (0.30 - 0.10), 0, 1
                            )
                        )
                        if self.physics_step
                        else 0.0
                    ),
                    0,
                ),
                (
                    "fingerpad_contact",
                    "fingerpad_contact" in self.events,
                    float(self._fingerpad_contact),
                    0,
                ),
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
                    (
                        float(
                            np.clip(
                                self._clearance_m / self.success_spec.minimum_lift_or_clearance_m,
                                0,
                                1,
                            )
                        )
                        if self.post_hold_contact_valid
                        else 0.0
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
            self.events["invalid_state"] = time_s

    def update(
        self,
        *,
        time_s: float,
        finite: bool,
        position: npt.NDArray[np.float64],
        bottom_z_m: float,
        eef_distance_m: float,
        left: bool,
        right: bool,
        opposing: bool,
        unsafe_canopy_contact: bool = False,
    ) -> None:
        if self.terminal_time_s is not None:
            return
        self._require_begun()
        stage_facts_before = self._stage_facts()
        next_step = self.physics_step + 1
        if not math.isclose(time_s, next_step * self.timestep_s, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("T1 capture requires consecutive simulator physics states")
        self.physics_step = next_step
        lower, upper = self.geometry.workspace_bounds_xy_m
        failure: str | None = None
        if (
            not finite
            or not np.isfinite(position).all()
            or not math.isfinite(bottom_z_m)
            or not math.isfinite(eef_distance_m)
        ):
            failure = "invalid_state"
        else:
            # A finite failure sample still counts toward progress.
            self._eef_distance_m = eef_distance_m
            self._clearance_m = bottom_z_m - self.geometry.support_surface_z_m
            self._fingerpad_contact = left or right
            if self.unsafe_contact_enabled and unsafe_canopy_contact:
                failure = "unsafe_contact"
            elif position[0] < lower[0] or position[0] > upper[0]:
                failure = "workspace_exit"
            elif position[1] > self.geometry.downstream_exit_y_m:
                failure = "downstream_exit"
            elif bottom_z_m < self.geometry.support_surface_z_m - 0.08:
                failure = "drop"
        if failure is not None:
            self._frozen_stage_facts = (
                stage_facts_before if failure == "invalid_state" else self._stage_facts()
            )
            self.failure_code = failure
            self.terminal_time_s = time_s
            self.events[failure] = time_s
            return

        bilateral = left and right
        if eef_distance_m <= 0.10:
            self.events.setdefault("approach", time_s)
        if "approach" in self.events and (left or right):
            self.events.setdefault("fingerpad_contact", time_s)
        self.bilateral_steps = self.bilateral_steps + 1 if bilateral else 0
        if "fingerpad_contact" in self.events and self.bilateral_steps >= self.hold_steps:
            self.events.setdefault("bilateral_hold", time_s)
        if "bilateral_hold" in self.events and "clearance" not in self.events:
            if not bilateral:
                self.post_hold_contact_valid = False
            elif self.bilateral_steps >= self.hold_steps:
                self.post_hold_contact_valid = True
        clearance_distance = max(
            self.geometry.support_surface_z_m
            + self.success_spec.minimum_lift_or_clearance_m
            - bottom_z_m,
            0.0,
        )
        clear = clearance_distance == 0.0
        if "bilateral_hold" in self.events and self.bilateral_steps >= self.hold_steps and clear:
            self.events.setdefault("clearance", time_s)
        stable = "clearance" in self.events and bilateral and clear
        self.stable_steps = self.stable_steps + 1 if stable else 0
        if self.stable_steps >= self.dwell_steps:
            self.events["stable_dwell"] = time_s
            self.success = True
            self.terminal_time_s = time_s
            self.events["success"] = time_s
        elif time_s >= self.horizon_s - 1e-10:
            # OnTimeBench first scores every physics sample in the last control tick.
            self._frozen_stage_facts = self._stage_facts()
            self.failure_code = "timeout"
            # Its timeout event uses the discrete episode clock, not accumulated sim time.
            self.terminal_time_s = self.horizon_s
            self.events["timeout"] = self.horizon_s


class DrivenInterceptionEnv(DynamicManipulationEnv):
    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        scenario_spec: ScenarioSpec,
        execution_translation_limit_m: float = 0.05,
    ) -> None:
        if task_spec.family != "driven_interception" or not isinstance(
            task_spec.bindings, DrivenInterceptionBindings
        ):
            raise ValueError("DrivenInterceptionEnv requires a registered T1 TaskSpec")
        if not isinstance(scenario_spec.geometry, DrivenInterceptionGeometrySpec):
            raise ValueError("T1 requires DrivenInterceptionGeometrySpec")
        self._is_occluded = task_spec.has_occluder
        self._is_surface_driven = task_spec.grasp_motion is GraspMotion.CONVEYOR
        target_id = task_spec.bindings.target_object_id
        driver_id = task_spec.bindings.driver_entity_id
        occluder_ids = task_spec.bindings.occluder_fixture_ids
        contacts = {item.entity_id: item for item in scenario_spec.physics.contacts}
        if (
            len(scenario_spec.object_states) != 1
            or scenario_spec.object_states[0].instance_id != target_id
            or scenario_spec.events
            or len(scenario_spec.motion) != 1
            or scenario_spec.motion[0].entity_id != driver_id
            or len(scenario_spec.physics.bodies) != 1
            or scenario_spec.physics.bodies[0].entity_id != target_id
            or len(scenario_spec.physics.contacts)
            != (2 if self._is_surface_driven else 1) + int(self._is_occluded)
            or set(contacts) != {target_id, driver_id, *occluder_ids}
        ):
            raise ValueError("T1 requires exactly its target, driver, motion and physical bindings")
        self._target_spec = scenario_spec.object_states[0]
        self._motion = scenario_spec.motion[0]
        self._target_contact = contacts[target_id]
        catalog = _load_asset_catalog()
        initial_driver: ObjectStateSpec | FixtureStateSpec
        if self._is_surface_driven:
            fixtures = {item.instance_id: item for item in scenario_spec.fixture_states}
            if len(scenario_spec.fixture_states) != 1 + int(self._is_occluded) or set(fixtures) != {
                driver_id,
                *occluder_ids,
            }:
                raise ValueError("surface-driven T1 requires its bound static fixtures")
            self._belt_spec = fixtures[driver_id]
            self._belt_contact = contacts[driver_id]
            if (
                self._target_spec.asset_id != "obj.cube_44_v1"
                or self._belt_spec.asset_id != "arena.conveyor_v1"
            ):
                raise ValueError("T1-Belt requires the frozen cube and conveyor assets")
            self._belt_asset = catalog[self._belt_spec.asset_id]
            if not isinstance(
                self._motion.translation, ConstantVelocityTranslation
            ) or not isinstance(self._motion.rotation, FixedRotation):
                raise ValueError(
                    "T1-Belt requires constant surface translation and fixed fixture rotation"
                )
            velocity = self._motion.translation.velocity_world_m_s
            initial_driver = self._belt_spec
            if self._is_occluded:
                self._canopy_spec = fixtures[occluder_ids[0]]
                if self._canopy_spec.asset_id != "fixture.occlusion_canopy_v1":
                    raise ValueError("T1-OCC requires its fixed canopy asset")
                self._canopy_asset = catalog[self._canopy_spec.asset_id]
                self._canopy_contact = contacts[self._canopy_spec.instance_id]
        else:
            if scenario_spec.fixture_states:
                raise ValueError("matched T1 requires its target as driver and no fixtures")
            if self._target_spec.asset_id != "obj.asym_t_t1_v1":
                raise ValueError("matched T1 requires the frozen asymmetric T asset")
            if isinstance(self._motion.translation, HarmonicTranslation):
                velocity = self._motion.translation.drift_velocity_world_m_s
                if self._motion.translation.amplitude_m[1:] != (0.0, 0.0):
                    raise ValueError("matched T1 harmonic amplitude is X-only")
            elif isinstance(self._motion.translation, ConstantVelocityTranslation):
                velocity = self._motion.translation.velocity_world_m_s
            else:
                raise ValueError("matched T1 requires harmonic drift or constant world translation")
            rotation = self._motion.rotation
            if not isinstance(rotation, ConstantAngularVelocityRotation) or rotation.axis_world != (
                0.0,
                0.0,
                1.0,
            ):
                raise ValueError("matched T1 requires explicit world-Z yaw rotation")
            if (rotation.angular_speed_rad_s == 0.0) != (
                task_spec.grasp_motion is GraspMotion.MATCHED_XYZ
            ):
                raise ValueError("XYZ yaw rate must be zero and SO3 yaw rate must be nonzero")
            initial_driver = self._target_spec
        if velocity[0] != 0.0 or velocity[1] <= 0.0 or velocity[2] != 0.0:
            raise ValueError("T1 forward velocity must be positive world +Y")
        _validate_motion_initial_pose(
            self._motion,
            initial_position_world_m=initial_driver.position_world_m,
            initial_quaternion_world_xyzw=initial_driver.quaternion_world_xyzw,
        )
        self._asset = catalog[self._target_spec.asset_id]
        self._light_id = "conveyor_key" if self._is_surface_driven else "table_key"
        self._geometry = scenario_spec.geometry
        self._capture = _DrivenCapture(task_spec, self._geometry, scenario_spec.physics_timestep_s)
        self._capture_active = False
        self._reset_scenario: ScenarioSpec | None = None
        self._last_wrench = np.zeros(6, dtype=np.float64)
        super().__init__(
            task_spec=task_spec,
            scenario_spec=scenario_spec,
            seed=0,
            execution_translation_limit_m=execution_translation_limit_m,
        )

    def _load_model(self) -> None:
        self._capture_active = False
        super()._load_model()
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(robot_model.base_xpos_offset["table"](0.8))
        arena = TableArena(table_offset=(0.0, 0.0, self._geometry.support_surface_z_m))
        arena.set_origin((0.0, 0.0, 0.0))
        for name in ("left_eef_target", "right_eef_target"):
            target = arena.worldbody.find(f"body[@name='{name}']")
            if target is None:
                raise RuntimeError("Robosuite 1.5.2 arena target binding is missing")
            arena.worldbody.remove(target)
        if self._is_surface_driven:
            arena.worldbody.remove(arena.table_body)
            arena.merge(MujocoXML(str(self._belt_asset.path)))
            if self._is_occluded:
                parked = arena.worldbody.find("body[@name='occluder']")
                if parked is None:
                    raise RuntimeError("T1 conveyor requires its parked canopy binding")
                arena.worldbody.remove(parked)
            body = arena.worldbody.find("./body[@name='belt']")
            geom = arena.worldbody.find(".//geom[@name='belt_collision']")
            if body is None or geom is None:
                raise ValueError("canonical conveyor must resolve its static belt and contact geom")
            body.set("pos", _xml_numbers(self._belt_spec.position_world_m))
            x, y, z, w = self._belt_spec.quaternion_world_xyzw
            body.set("quat", _xml_numbers((w, x, y, z)))
            rotation = np.empty(9, dtype=np.float64)
            unit_quaternion = np.asarray([w, x, y, z], dtype=np.float64)
            unit_quaternion /= np.linalg.norm(unit_quaternion)
            mujoco.mju_quat2Mat(rotation, unit_quaternion)
            local_velocity = _conveyor_surface_velocity(
                self._motion, world_from_geom_rotation=rotation.reshape(3, 3)
            )
            # Compile the non-zero field so MuJoCo sets flg_surfacevel itself.
            geom.set("surfacevel", _xml_numbers(tuple(float(value) for value in local_velocity)))
        else:
            collision = arena.table_body.find("geom[@name='table_collision']")
            visual = arena.table_body.find("geom[@name='table_visual']")
            if collision is None or visual is None:
                raise RuntimeError("T1 requires the native table collision and visual")
            collision.set("size", "0.4 0.4 0.02")
            collision.set("pos", "0 0 -0.005")
            visual.attrib.pop("material", None)
            visual.set("rgba", "0.58 0.6 0.62 1")
            ET.SubElement(
                arena.table_body,
                "geom",
                name="work_surface_collision",
                type="box",
                size="0.28 0.4 0.005",
                pos="0 0 0.02",
                group="0",
                friction="1 0.005 0.0001",
                rgba="0 0 0 0",
            )
        lights = arena.worldbody.findall("light")
        if len(lights) != 1:
            raise RuntimeError("frozen TableArena must contain one canonical light")
        lights[0].set("name", self._light_id)
        self._target_model = MujocoXMLObject(
            str(self._asset.path),
            name=self._target_spec.instance_id,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        objects = [self._target_model]
        if self._is_occluded:
            self._canopy_model = _canopy_model(self._canopy_asset.path, self._canopy_spec)
            objects.append(self._canopy_model)
        self.model = ManipulationTask(
            mujoco_arena=arena, mujoco_robots=[robot_model], mujoco_objects=objects
        )

    def edit_model_xml(self, xml_str: str) -> str:
        return _rewrite_scenario_lighting_and_gravity_xml(
            super().edit_model_xml(xml_str),
            self._scenario_spec,
            light_id=self._light_id,
            invalid_message=f"T1 requires the explicit {self._light_id} light set",
            duplicate_message=f"duplicate canonical {self._light_id} light",
            missing_message=f"{self._light_id} must resolve in the assembled arena",
            headlight_before_gravity=True,
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        model = self.sim.model._model
        self._target_body_id = int(model.body(self._target_model.root_body).id)
        self._target_geom_ids = tuple(
            int(model.geom(name).id) for name in self._target_model.contact_geoms
        )
        joint_id = int(model.joint(self._target_model.joints[0]).id)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("T1 target must be a free body")
        self._target_qpos_address = int(model.jnt_qposadr[joint_id])
        self._target_qvel_address = int(model.jnt_dofadr[joint_id])
        pads = self.robots[0].gripper["right"].important_geoms
        self._left_pad_ids = frozenset(int(model.geom(name).id) for name in pads["left_fingerpad"])
        self._right_pad_ids = frozenset(
            int(model.geom(name).id) for name in pads["right_fingerpad"]
        )
        self._support_geom_ids: tuple[int, ...]
        if self._is_surface_driven:
            self._belt_geom_id = int(model.geom("belt_collision").id)
            self._belt_body_id = int(model.body("belt").id)
            self._support_geom_ids = (self._belt_geom_id,)
            if model.body_dofnum[self._belt_body_id] != 0:
                raise ValueError("T1-Belt surface velocity must not introduce a fixture DOF")
        else:
            self._support_geom_ids = (
                int(model.geom("work_surface_collision").id),
                int(model.geom("table_collision").id),
            )
        if self._is_occluded:
            self._canopy_body_id = int(model.body(self._canopy_model.root_body).id)
            self._canopy_geom_ids = tuple(
                int(model.geom(name).id) for name in self._canopy_model.contact_geoms
            )
            robot_root_id = int(model.body(self.robots[0].robot_model.root_body).id)
            robot_body_ids = {robot_root_id}
            for body_id in range(robot_root_id + 1, model.nbody):
                if int(model.body_parentid[body_id]) in robot_body_ids:
                    robot_body_ids.add(body_id)
            self._robot_geom_ids = frozenset(
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in robot_body_ids
            )
        entity_body_ids = {self._target_spec.instance_id: self._target_body_id}
        if self._is_surface_driven:
            entity_body_ids[self._belt_spec.instance_id] = self._belt_body_id
        if self._is_occluded:
            entity_body_ids[self._canopy_spec.instance_id] = self._canopy_body_id
        self._bind_scene_entities(
            target_body_id=self._target_body_id,
            entity_body_ids=entity_body_ids,
        )
        self._compiled_joint_physics = _resolve_joint_physics(
            model,
            self._scenario_spec.physics.joints,
            entity_body_ids=entity_body_ids,
        )

    def _reset_internal(self) -> None:
        self._capture_active = False
        self._reset_scenario = None
        super()._reset_internal()
        model, data = self.sim.model._model, self.sim.data._data
        body = self._scenario_spec.physics.bodies[0]
        contact = self._target_contact
        model.body_mass[self._target_body_id] = body.mass_kg
        model.body_inertia[self._target_body_id] = body.diagonal_inertia_kg_m2
        for geom_id in self._target_geom_ids:
            model.geom_friction[geom_id] = contact.friction_slide_spin_roll
            model.geom_solref[geom_id] = contact.solref
            model.geom_solimp[geom_id] = contact.solimp
        if self._is_surface_driven:
            belt_contact = self._belt_contact
            model.geom_friction[self._belt_geom_id] = belt_contact.friction_slide_spin_roll
            model.geom_solref[self._belt_geom_id] = belt_contact.solref
            model.geom_solimp[self._belt_geom_id] = belt_contact.solimp
        if self._is_occluded:
            for geom_id in self._canopy_geom_ids:
                model.geom_friction[geom_id] = self._canopy_contact.friction_slide_spin_roll
                model.geom_solref[geom_id] = self._canopy_contact.solref
                model.geom_solimp[geom_id] = self._canopy_contact.solimp
        _apply_joint_physics(model, self._compiled_joint_physics)
        mujoco.mj_setConst(model, data)
        # mj_setConst leaves data at model qpos0; restore the explicit reset state afterwards.
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
        velocity = self._target_qvel_address
        data.qvel[velocity : velocity + 3] = self._target_spec.linear_velocity_world_m_s
        data.qvel[velocity + 3 : velocity + 6] = self._native_target_angular_velocity(
            data.qpos[address + 3 : address + 7]
        )
        data.xfrc_applied[:] = 0.0
        data.qfrc_applied[:] = 0.0
        self._last_wrench[:] = 0.0
        self.sim.forward()
        self._capture.reset()
        self._capture.begin(
            eef_distance_m=float(
                np.linalg.norm(
                    data.site_xpos[self.robots[0].eef_site_id["right"]]
                    - data.xpos[self._target_body_id]
                )
            )
        )
        self._reset_scenario = self._read_reset_scenario()
        self._capture_active = True

    def _native_target_angular_velocity(
        self, quaternion_wxyz: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        angular = np.asarray(self._target_spec.angular_velocity_world_rad_s, dtype=np.float64)
        if np.all(quaternion_wxyz[1:3] == 0.0) and np.all(angular[:2] == 0.0):
            # Rotation about world Z leaves its own angular-velocity component unchanged.
            # Preserve the original direct yaw-rate reset without a matrix-rounding step.
            return angular.copy()
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, quaternion_wxyz)
        return rotation.reshape(3, 3).T @ angular

    def _read_reset_scenario(self) -> ScenarioSpec:
        model, data = self.sim.model._model, self.sim.data._data
        scenario = self._scenario_spec
        if not _physics_state_is_valid(self.sim.data._data):
            raise ValueError("T1 reset has invalid state")
        robot = self.robots[0]
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
        address_velocity = self._target_qvel_address
        linear_velocity = data.qvel[address_velocity : address_velocity + 3]
        expected_local_angular_velocity = self._native_target_angular_velocity(qpos[3:])
        if not np.array_equal(
            data.qvel[address_velocity + 3 : address_velocity + 6],
            expected_local_angular_velocity,
        ):
            raise ValueError("compiled target angular velocity differs from ScenarioSpec")
        target = ObjectStateSpec(
            self._target_spec.instance_id,
            self._target_spec.asset_id,
            cast(Vector3, tuple(float(value) for value in qpos[:3])),
            cast(Vector4, tuple(float(value) for value in qpos[[4, 5, 6, 3]])),
            cast(Vector3, tuple(float(value) for value in linear_velocity)),
            # Validate in MuJoCo's native frame; a second rotation introduces rounding.
            self._target_spec.angular_velocity_world_rad_s,
        )
        body = BodyPhysicsSpec(
            target.instance_id,
            float(model.body_mass[self._target_body_id]),
            cast(
                Vector3, tuple(float(value) for value in model.body_inertia[self._target_body_id])
            ),
        )
        contact = self._target_contact
        for geom_id in self._target_geom_ids:
            actual = ContactPhysicsSpec(
                target.instance_id,
                cast(Vector3, tuple(float(value) for value in model.geom_friction[geom_id])),
                cast(Vector2, tuple(float(value) for value in model.geom_solref[geom_id])),
                cast(
                    tuple[float, float, float, float, float],
                    tuple(float(value) for value in model.geom_solimp[geom_id]),
                ),
            )
            if actual != contact:
                raise ValueError("compiled target contact properties differ from ScenarioSpec")
        _validate_compiled_joint_physics(
            model,
            self._compiled_joint_physics,
            owner="T1",
        )
        fixture_values = {}
        if self._is_surface_driven:
            fixture_values[self._belt_spec.instance_id] = self._read_surface_fixture()
        if self._is_occluded:
            fixture_values[self._canopy_spec.instance_id] = _read_canopy_fixture(
                model,
                data,
                fixture=self._canopy_spec,
                contact=self._canopy_contact,
                body_id=self._canopy_body_id,
                geom_ids=self._canopy_geom_ids,
            )
            self._canopy_clearance_m = _compiled_lift_envelope_clearance(
                model,
                data,
                target_body_id=self._target_body_id,
                target_geom_ids=self._target_geom_ids,
                canopy_geom_ids=self._canopy_geom_ids,
                geometry=self._geometry,
                minimum_lift_m=self._capture.success_spec.minimum_lift_or_clearance_m,
            )
        fixtures = tuple(fixture_values[item.instance_id] for item in scenario.fixture_states)
        if not self._is_surface_driven and (
            model.flg_surfacevel or np.any(model.geom_surfacevel != 0.0)
        ):
            raise ValueError("matched T1 uses bounded wrench, not surface velocity")
        support_geom = self._support_geom_ids[0]
        support_top = float(data.geom_xpos[support_geom, 2] + model.geom_size[support_geom, 2])
        if not math.isclose(
            support_top, self._geometry.support_surface_z_m, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("compiled support top differs from GeometrySpec")
        if not math.isclose(
            _collision_bottom_z(model, data, self._target_geom_ids),
            support_top,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("T1 reset target bottom must lie on compiled support")
        for name, spec in zip(self._camera_sensor_names, scenario.cameras.views, strict=True):
            camera_id = int(model.camera(name).id)
            expected_quaternion = np.asarray(spec.quaternion_parent_xyzw)[[3, 0, 1, 2]]
            if not (
                np.allclose(model.cam_pos[camera_id], spec.position_parent_m, rtol=0.0, atol=1e-12)
                and math.isclose(
                    abs(float(np.dot(model.cam_quat[camera_id], expected_quaternion))),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                and float(model.cam_fovy[camera_id]) == spec.vertical_fov_deg
            ):
                raise ValueError("compiled camera differs from ScenarioSpec")
        if (
            model.nlight != 1
            or model.light(0).name != self._light_id
            or not model.vis.headlight.active
        ):
            raise ValueError("compiled light set differs from canonical T1 lighting")
        if (
            model.light_type[0] != mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            or model.light_castshadow[0]
        ):
            raise ValueError("canonical T1 arena light must be directional with shadows disabled")
        light = scenario.lighting.lights[0]
        for actual_values, expected in (
            (model.light_pos[0], light.position_world_m),
            (model.light_dir[0], light.direction_world),
            (model.light_ambient[0], light.ambient_rgb),
            (model.light_diffuse[0], light.diffuse_rgb),
            (model.light_specular[0], light.specular_rgb),
            (model.light_attenuation[0], light.attenuation),
        ):
            if not np.allclose(actual_values, expected, rtol=0.0, atol=5e-8):
                raise ValueError("compiled light values differ from ScenarioSpec")
        if (
            float(model.light_cutoff[0]) != light.cutoff_deg
            or bool(model.light_active[0]) != light.active
        ):
            raise ValueError("compiled light scalar values differ from ScenarioSpec")
        exported = replace(
            scenario,
            robot_initial_state=robot_state,
            object_states=(target,),
            fixture_states=fixtures,
            physics=replace(
                scenario.physics,
                gravity_world_m_s2=cast(Vector3, tuple(float(x) for x in model.opt.gravity)),
                bodies=(body,),
            ),
        )
        if exported != scenario:
            changed = tuple(
                field.name
                for field in fields(scenario)
                if getattr(exported, field.name) != getattr(scenario, field.name)
            )
            raise ValueError(f"compiled reset differs from ScenarioSpec: {changed}")
        return exported

    def _read_surface_fixture(self) -> FixtureStateSpec:
        model, data = self.sim.model._model, self.sim.data._data
        belt_contact = ContactPhysicsSpec(
            self._belt_spec.instance_id,
            cast(Vector3, tuple(float(v) for v in model.geom_friction[self._belt_geom_id])),
            cast(Vector2, tuple(float(v) for v in model.geom_solref[self._belt_geom_id])),
            cast(
                tuple[float, float, float, float, float],
                tuple(float(v) for v in model.geom_solimp[self._belt_geom_id]),
            ),
        )
        if belt_contact != self._belt_contact:
            raise ValueError("compiled belt contact properties differ from ScenarioSpec")
        local_velocity = np.asarray(model.geom_surfacevel[self._belt_geom_id])
        world_velocity = data.geom_xmat[self._belt_geom_id].reshape(3, 3) @ local_velocity[:3]
        assert isinstance(self._motion.translation, ConstantVelocityTranslation)
        if (
            not model.flg_surfacevel
            or not np.isfinite(local_velocity).all()
            or np.any(local_velocity[3:] != 0.0)
            or not np.allclose(
                world_velocity,
                self._motion.translation.velocity_world_m_s,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("compiled conveyor surface velocity differs from MotionSpec")
        if np.any(np.delete(model.geom_surfacevel, self._belt_geom_id, axis=0) != 0.0):
            raise ValueError("T1-Belt permits surface velocity only on its belt contact geom")
        actual_quaternion = data.xquat[self._belt_body_id][[1, 2, 3, 0]]
        expected = np.asarray(self._belt_spec.quaternion_world_xyzw, dtype=np.float64)
        expected /= np.linalg.norm(expected)
        if not math.isclose(
            abs(float(np.dot(actual_quaternion, expected))), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("compiled belt orientation differs from FixtureStateSpec")
        return FixtureStateSpec(
            self._belt_spec.instance_id,
            self._belt_spec.asset_id,
            cast(Vector3, tuple(float(v) for v in data.xpos[self._belt_body_id])),
            self._belt_spec.quaternion_world_xyzw,
        )

    def export_scenario(self) -> ScenarioSpec:
        if self._reset_scenario is None:
            raise RuntimeError("T1 requires a verified real reset before scenario export")
        return self._reset_scenario

    def _compiled_occlusion_facts(self) -> _CompiledOcclusionFacts:
        if not self._is_occluded:
            raise ValueError("compiled occlusion facts require T1-OCC")
        model, data = self.sim.model._model, self.sim.data._data
        robot_contact, target_contact = _canopy_contact_facts(
            data,
            canopy_geom_ids=frozenset(self._canopy_geom_ids),
            robot_geom_ids=self._robot_geom_ids,
            target_geom_ids=frozenset(self._target_geom_ids),
        )
        return _CompiledOcclusionFacts(
            all(
                bool(model.geom_contype[gid] and model.geom_conaffinity[gid])
                for gid in self._canopy_geom_ids
            ),
            self._canopy_clearance_m,
            robot_contact,
            target_contact,
        )

    def _pre_action(self, action: npt.NDArray[np.float64], policy_step: bool = False) -> None:
        if not _physics_state_is_valid(self.sim.data._data):
            self._capture.invalidate(self._capture.physics_step * self.model_timestep)
        if not self._is_surface_driven:
            self._apply_target_wrench()
        super()._pre_action(action, policy_step)

    def _apply_target_wrench(self) -> None:
        model, data = self.sim.model._model, self.sim.data._data
        reference = _motion_state_at_time(
            self._motion,
            initial_position_world_m=self._target_spec.position_world_m,
            initial_quaternion_world_xyzw=self._target_spec.quaternion_world_xyzw,
            simulator_time_s=float(data.time),
        )
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, self._target_body_id, velocity, 0
        )
        self._last_wrench[:] = _bounded_planar_wrench(
            reference,
            position_world_m=data.xpos[self._target_body_id],
            quaternion_world_wxyz=data.xquat[self._target_body_id],
            linear_velocity_world_m_s=velocity[3:],
            angular_velocity_world_rad_s=velocity[:3],
            mass_kg=float(model.body_mass[self._target_body_id]),
            translation_kp_n_m=50.0,
            translation_kd_n_s_m=5.0,
            friction_compensation_acceleration_m_s2=9.0,
            yaw_kp_nm_rad=0.50,
            yaw_kd_nm_s_rad=0.10,
            tilt_kp_nm_rad=0.60,
            tilt_kd_nm_s_rad=0.05,
            force_moment_arm_m=0.0,
            maximum_force_n=3.0,
            maximum_yaw_torque_nm=0.25,
            maximum_tilt_torque_nm=0.15,
        )
        data.xfrc_applied[self._target_body_id] = self._last_wrench

    def _update_observables(self, force: bool = False) -> None:
        if not force and self._capture_active:
            model, data = self.sim.model._model, self.sim.data._data
            finite = _physics_state_is_valid(data)
            if not finite:
                self._capture.invalidate((self._capture.physics_step + 1) * self.model_timestep)
                super()._update_observables(force)
                return
            left, right, opposing = _fingerpad_contacts(
                model, data, self._target_geom_ids, self._left_pad_ids, self._right_pad_ids
            )
            if (left or right) and self._capture.terminal_time_s is None:
                self._capture_first_contact("fingerpad_target", self._target_body_id)
            self._capture.update(
                time_s=float(data.time),
                finite=finite,
                position=data.xpos[self._target_body_id],
                bottom_z_m=float(
                    data.xpos[self._target_body_id, 2] + self._target_model.bottom_offset[2]
                ),
                eef_distance_m=float(
                    np.linalg.norm(
                        data.site_xpos[self.robots[0].eef_site_id["right"]]
                        - data.xpos[self._target_body_id]
                    )
                ),
                left=left,
                right=right,
                opposing=opposing,
                unsafe_canopy_contact=(
                    any(
                        _canopy_contact_facts(
                            data,
                            canopy_geom_ids=frozenset(self._canopy_geom_ids),
                            robot_geom_ids=self._robot_geom_ids,
                            target_geom_ids=frozenset(self._target_geom_ids),
                        )
                    )
                    if self._is_occluded
                    else False
                ),
            )
        super()._update_observables(force)

    def _post_action(self, action: npt.NDArray[np.float64]) -> tuple[float, bool, dict[str, Any]]:
        reward, _, info = super()._post_action(action)
        self.done = self._capture.terminal_time_s is not None
        info.update(
            success=self._capture.success,
            failure_code=self._capture.failure_code,
            terminal_time_s=self._capture.terminal_time_s,
            events=dict(self._capture.events),
            stage_facts=self._capture._stage_facts(),
        )
        return float(reward), bool(self.done), info

    def reward(self, action: npt.NDArray[np.float64] | None = None) -> float:
        del action
        return float(self._check_success())

    def _check_success(self) -> bool:
        return self._capture.success

    def _recording_frame_facts(self) -> _RecordingFrameFacts:
        return (
            self._capture.success,
            self._capture.failure_code,
            self._capture.terminal_time_s,
            self._capture._stage_facts(),
        )


__all__: list[str] = []
