"""Grasp a free-body target driven by a continuous wrench (harmonic grasp)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import fields, replace
from typing import Any, cast

import mujoco
import numpy as np
import numpy.typing as npt
from robosuite.models.arenas import TableArena
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
)
from ontimebench.envs.components.motion import (
    _bounded_planar_wrench,
    _cross3,
    _motion_state_at_time,
)
from ontimebench.scenarios.spec import (
    BodyPhysicsSpec,
    ContactPhysicsSpec,
    GraspGeometrySpec,
    ObjectStateSpec,
    RobotInitialStateSpec,
    ScenarioSpec,
    Vector2,
    Vector3,
    Vector4,
)
from ontimebench.tasks.spec import CaptureSuccessSpec, GraspBindings, TaskSpec


class _GraspCapture:
    """Track capture success and failure at the physics rate."""

    def __init__(self, task: TaskSpec, geometry: GraspGeometrySpec, timestep_s: float) -> None:
        if not isinstance(task.success, CaptureSuccessSpec):
            raise ValueError("Grasp requires CaptureSuccessSpec")
        self.success_spec = task.success
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
        self._initial_eef_distance_m: float | None = None
        self._current_eef_distance_m: float | None = None
        self._current_clearance_m: float | None = None
        self._current_fingerpad_contact = False
        self._frozen_stage_facts: tuple[tuple[str, bool, float, int], ...] | None = None

    def begin(self, *, eef_distance_m: float) -> None:
        if not math.isfinite(eef_distance_m) or eef_distance_m < 0.0:
            raise ValueError(
                "Grasp stage initialization requires a finite non-negative approach distance"
            )
        self._initial_eef_distance_m = eef_distance_m
        self.events.setdefault("target_enters_observation_region", 0.0)

    def _require_begun(self) -> None:
        if self._initial_eef_distance_m is None:
            raise RuntimeError("Grasp capture requires begin() at scored t0 before physics updates")

    def _stage_facts(self) -> tuple[tuple[str, bool, float, int], ...]:
        if self._frozen_stage_facts is not None:
            return self._frozen_stage_facts
        approach_progress = (
            0.0
            if self._current_eef_distance_m is None
            else float(np.clip((0.30 - self._current_eef_distance_m) / (0.30 - 0.10), 0.0, 1.0))
        )
        clearance_progress = (
            0.0
            if self._current_clearance_m is None or not self.post_hold_contact_valid
            else float(
                np.clip(
                    self._current_clearance_m / self.success_spec.minimum_lift_or_clearance_m,
                    0.0,
                    1.0,
                )
            )
        )
        return _ordered_stage_facts(
            (
                (
                    "approach",
                    "approach" in self.events,
                    approach_progress,
                    0,
                ),
                (
                    "fingerpad_contact",
                    "fingerpad_contact" in self.events,
                    float(self._current_fingerpad_contact),
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
                    clearance_progress,
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
    ) -> None:
        if self.terminal_time_s is not None:
            return
        self._require_begun()
        stage_facts_before = self._stage_facts()
        next_step = self.physics_step + 1
        if not math.isclose(time_s, next_step * self.timestep_s, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("Grasp capture requires consecutive simulator physics states")
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
        elif position[0] < lower[0] or position[0] > upper[0]:
            failure = "workspace_exit"
        elif position[1] > self.geometry.downstream_exit_y_m:
            failure = "downstream_exit"
        elif bottom_z_m < self.geometry.support_surface_z_m - 0.08:
            failure = "drop"
        if failure != "invalid_state":
            self._current_eef_distance_m = eef_distance_m
            self._current_clearance_m = bottom_z_m - self.geometry.support_surface_z_m
            self._current_fingerpad_contact = left or right
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
        clear = (
            bottom_z_m - self.geometry.support_surface_z_m
            >= self.success_spec.minimum_lift_or_clearance_m
        )
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
            self._frozen_stage_facts = self._stage_facts()
            self.failure_code = "timeout"
            self.terminal_time_s = self.horizon_s
            self.events["timeout"] = self.horizon_s


class GraspEnv(DynamicManipulationEnv):
    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        scenario_spec: ScenarioSpec,
        execution_translation_limit_m: float = 0.05,
    ) -> None:
        if task_spec.task_id != "harmonic_grasp" or not isinstance(
            task_spec.bindings, GraspBindings
        ):
            raise ValueError("GraspEnv requires the registered harmonic_grasp TaskSpec")
        if not isinstance(scenario_spec.geometry, GraspGeometrySpec):
            raise ValueError("GraspEnv requires GraspGeometrySpec")
        target_id = task_spec.bindings.target_object_id
        if (
            len(scenario_spec.object_states) != 1
            or scenario_spec.object_states[0].instance_id != target_id
            or scenario_spec.fixture_states
            or scenario_spec.events
            or len(scenario_spec.motion) != 1
            or scenario_spec.motion[0].entity_id != target_id
            or len(scenario_spec.physics.bodies) != 1
            or scenario_spec.physics.bodies[0].entity_id != target_id
            or len(scenario_spec.physics.contacts) != 1
            or scenario_spec.physics.contacts[0].entity_id != target_id
        ):
            raise ValueError("Grasp requires one bound target, motion, body and contact record")
        self._target_spec = scenario_spec.object_states[0]
        catalog = _load_asset_catalog()
        if self._target_spec.asset_id not in catalog:
            raise ValueError("Grasp target asset must resolve through the frozen catalog")
        self._asset = catalog[self._target_spec.asset_id]
        self._geometry = scenario_spec.geometry
        self._capture = _GraspCapture(task_spec, self._geometry, scenario_spec.physics_timestep_s)
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
        collision = arena.table_body.find("geom[@name='table_collision']")
        visual = arena.table_body.find("geom[@name='table_visual']")
        if collision is None or visual is None:
            raise RuntimeError("Grasp requires the native table collision and visual")
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
        lights[0].set("name", "table_key")
        self._target_model = MujocoXMLObject(
            str(self._asset.path),
            name=self._target_spec.instance_id,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.model = ManipulationTask(
            mujoco_arena=arena, mujoco_robots=[robot_model], mujoco_objects=[self._target_model]
        )

    def edit_model_xml(self, xml_str: str) -> str:
        return _rewrite_scenario_lighting_and_gravity_xml(
            super().edit_model_xml(xml_str),
            self._scenario_spec,
            light_id="table_key",
            invalid_message="Grasp requires the explicit table_key light set",
            duplicate_message="duplicate canonical table_key light",
            missing_message="table_key must resolve in the assembled TableArena",
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
            raise ValueError("Grasp target must be a free body")
        self._target_qpos_address = int(model.jnt_qposadr[joint_id])
        self._target_qvel_address = int(model.jnt_dofadr[joint_id])
        pads = self.robots[0].gripper["right"].important_geoms
        self._left_pad_ids = frozenset(int(model.geom(name).id) for name in pads["left_fingerpad"])
        self._right_pad_ids = frozenset(
            int(model.geom(name).id) for name in pads["right_fingerpad"]
        )
        self._table_geom_id = int(model.geom("work_surface_collision").id)
        self._support_geom_ids = (self._table_geom_id, int(model.geom("table_collision").id))
        self._bind_scene_entities(
            target_body_id=self._target_body_id,
            entity_body_ids={self._target_spec.instance_id: self._target_body_id},
        )
        self._compiled_joint_physics = _resolve_joint_physics(
            model,
            self._scenario_spec.physics.joints,
            entity_body_ids={self._target_spec.instance_id: self._target_body_id},
        )

    def _reset_internal(self) -> None:
        self._capture_active = False
        self._reset_scenario = None
        super()._reset_internal()
        model, data = self.sim.model._model, self.sim.data._data
        body = self._scenario_spec.physics.bodies[0]
        contact = self._scenario_spec.physics.contacts[0]
        model.body_mass[self._target_body_id] = body.mass_kg
        model.body_inertia[self._target_body_id] = body.diagonal_inertia_kg_m2
        for geom_id in self._target_geom_ids:
            model.geom_friction[geom_id] = contact.friction_slide_spin_roll
            model.geom_solref[geom_id] = contact.solref
            model.geom_solimp[geom_id] = contact.solimp
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
        if (x, y) != (0.0, 0.0) or self._target_spec.angular_velocity_world_rad_s[:2] != (
            0.0,
            0.0,
        ):
            raise ValueError("Grasp reset requires yaw-only orientation and angular velocity")
        address = self._target_qpos_address
        data.qpos[address : address + 7] = (*self._target_spec.position_world_m, w, x, y, z)
        data.qvel[:] = 0.0
        velocity = self._target_qvel_address
        data.qvel[velocity : velocity + 3] = self._target_spec.linear_velocity_world_m_s
        # The free joint starts with yaw only, whose local and world axes coincide.
        data.qvel[velocity + 3 : velocity + 6] = self._target_spec.angular_velocity_world_rad_s
        data.xfrc_applied[:] = 0.0
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

    def _read_reset_scenario(self) -> ScenarioSpec:
        model, data = self.sim.model._model, self.sim.data._data
        scenario = self._scenario_spec
        if not _physics_state_is_valid(self.sim.data._data):
            raise ValueError("Grasp reset has invalid state")
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
        velocity = data.qvel[self._target_qvel_address : self._target_qvel_address + 6]
        target = ObjectStateSpec(
            self._target_spec.instance_id,
            self._target_spec.asset_id,
            cast(Vector3, tuple(float(value) for value in qpos[:3])),
            cast(Vector4, tuple(float(value) for value in qpos[[4, 5, 6, 3]])),
            cast(Vector3, tuple(float(value) for value in velocity[:3])),
            cast(Vector3, tuple(float(value) for value in velocity[3:])),
        )
        body = BodyPhysicsSpec(
            target.instance_id,
            float(model.body_mass[self._target_body_id]),
            cast(
                Vector3, tuple(float(value) for value in model.body_inertia[self._target_body_id])
            ),
        )
        contact = scenario.physics.contacts[0]
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
            owner="Grasp",
        )
        table_top = float(
            data.geom_xpos[self._table_geom_id, 2] + model.geom_size[self._table_geom_id, 2]
        )
        if not math.isclose(
            table_top, self._geometry.support_surface_z_m, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("compiled table top differs from GeometrySpec")
        if not math.isclose(
            _collision_bottom_z(model, data, self._target_geom_ids),
            table_top,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Grasp reset target bottom must lie on compiled table top")
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
            or model.light(0).name != "table_key"
            or not model.vis.headlight.active
        ):
            raise ValueError("compiled light set differs from canonical Grasp lighting")
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

    def export_scenario(self) -> ScenarioSpec:
        if self._reset_scenario is None:
            raise RuntimeError("Grasp requires a verified real reset before scenario export")
        return self._reset_scenario

    def _pre_action(self, action: npt.NDArray[np.float64], policy_step: bool = False) -> None:
        if not _physics_state_is_valid(self.sim.data._data):
            # step1 may have reset corrupt qpos and time; the warning is still an episode fact.
            self._capture.invalidate(self._capture.physics_step * self.model_timestep)
        model, data = self.sim.model._model, self.sim.data._data
        reference = _motion_state_at_time(
            self._scenario_spec.motion[0],
            initial_position_world_m=self._target_spec.position_world_m,
            initial_quaternion_world_xyzw=self._target_spec.quaternion_world_xyzw,
            simulator_time_s=float(data.time),
        )
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, self._target_body_id, velocity, 0
        )
        origin_velocity = velocity[3:] + _cross3(
            velocity[:3], data.xpos[self._target_body_id] - data.xipos[self._target_body_id]
        )
        self._last_wrench[:] = _bounded_planar_wrench(
            reference,
            position_world_m=data.xpos[self._target_body_id],
            quaternion_world_wxyz=data.xquat[self._target_body_id],
            linear_velocity_world_m_s=origin_velocity,
            angular_velocity_world_rad_s=velocity[:3],
            mass_kg=float(model.body_mass[self._target_body_id]),
            translation_kp_n_m=60.0,
            translation_kd_n_s_m=6.0,
            friction_compensation_acceleration_m_s2=5.0,
            yaw_kp_nm_rad=0.36,
            yaw_kd_nm_s_rad=0.02,
            tilt_kp_nm_rad=0.60,
            tilt_kd_nm_s_rad=0.01,
            force_moment_arm_m=0.020,
            maximum_force_n=3.0,
            maximum_yaw_torque_nm=0.12,
            maximum_tilt_torque_nm=0.15,
        )
        data.xfrc_applied[self._target_body_id] = self._last_wrench
        super()._pre_action(action, policy_step)

    def _update_observables(self, force: bool = False) -> None:
        if not force and self._capture_active:
            model, data = self.sim.model._model, self.sim.data._data
            if not _physics_state_is_valid(self.sim.data._data):
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
                finite=all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc)),
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
