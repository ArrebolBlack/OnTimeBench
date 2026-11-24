"""Place an object into a moving or rotating receptacle (T2)."""

from __future__ import annotations

import math
from collections import deque
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
    _collision_bottom_z,
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
from ontimebench.envs.components.motion import (
    _motion_state_at_time,
    _MotionState,
    _validate_motion_initial_pose,
)
from ontimebench.scenarios.spec import (
    BodyPhysicsSpec,
    ConstantAngularVelocityRotation,
    ConstantVelocityTranslation,
    ContactPhysicsSpec,
    FixedRotation,
    MovingPlacementGeometrySpec,
    ObjectStateSpec,
    RobotInitialStateSpec,
    ScenarioSpec,
    Vector2,
    Vector3,
    Vector4,
)
from ontimebench.tasks.spec import (
    MovingPlacementBindings,
    SE3PlacementSuccessSpec,
    TaskSpec,
    TranslatingPlacementSuccessSpec,
)

_BOX_SIGNS = np.asarray(
    [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64
)


def _rectangle_corners(
    rectangle: tuple[float, float, float, float],
) -> npt.NDArray[np.float64]:
    x0, x1, y0, y1 = rectangle
    return np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=np.float64)


def _polygon_area(polygon: npt.NDArray[np.float64]) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(
        float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )


def _clipped_area(
    polygon: npt.NDArray[np.float64], rectangle: tuple[float, float, float, float]
) -> float:
    clipped = polygon
    for axis, boundary, sign in (
        (0, rectangle[0], 1),
        (0, rectangle[1], -1),
        (1, rectangle[2], 1),
        (1, rectangle[3], -1),
    ):
        result: list[npt.NDArray[np.float64]] = []
        if not len(clipped):
            return 0.0
        start = clipped[-1]
        start_inside = sign * (start[axis] - boundary) >= -1e-10
        for end in clipped:
            end_inside = sign * (end[axis] - boundary) >= -1e-10
            if start_inside != end_inside:
                denominator = float(end[axis] - start[axis])
                if abs(denominator) <= 1e-10:
                    intersection = (start + end) / 2.0
                else:
                    fraction = float(np.clip((boundary - start[axis]) / denominator, 0.0, 1.0))
                    intersection = start + fraction * (end - start)
                intersection[axis] = boundary
                result.append(intersection)
            if end_inside:
                result.append(end)
            start, start_inside = end, end_inside
        clipped = np.asarray(result, dtype=np.float64)
    return _polygon_area(clipped)


def _point_segment_distances(
    points: npt.NDArray[np.float64], starts: npt.NDArray[np.float64], ends: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    edge = ends - starts
    relative = points[:, None, :] - starts[None, :, :]
    fraction = np.sum(relative * edge[None, :, :], axis=2) / np.sum(edge * edge, axis=1)
    closest = starts[None, :, :] + np.clip(fraction, 0.0, 1.0)[:, :, None] * edge[None, :, :]
    return cast(
        npt.NDArray[np.float64],
        np.linalg.norm(points[:, None, :] - closest, axis=2),
    )


def _polygon_boundary_distance(
    first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]
) -> float:
    first_ends, second_ends = np.roll(first, -1, axis=0), np.roll(second, -1, axis=0)
    a, b = (first_ends - first)[:, None, :], (second_ends - second)[None, :, :]

    def cross(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0])

    first_a, first_b = (
        cross(a, second[None, :, :] - first[:, None, :]),
        cross(a, second_ends[None, :, :] - first[:, None, :]),
    )
    second_a, second_b = (
        cross(b, first[:, None, :] - second[None, :, :]),
        cross(b, first_ends[:, None, :] - second[None, :, :]),
    )
    if np.any((first_a * first_b < -1e-10) & (second_a * second_b < -1e-10)):
        return 0.0
    return min(
        float(_point_segment_distances(first, second, second_ends).min()),
        float(_point_segment_distances(second, first, first_ends).min()),
    )


@dataclass(frozen=True, slots=True)
class _PlacementGeometryFacts:
    contained: bool
    planar_clearance_m: float
    floor_gap_m: float
    entrance_gap_m: float
    insertion_depth_m: float
    orientation_error_rad: float
    tilt_rad: float


class _PlacementOutcome:
    """Placed means sunk into the opening; success is placed and ungripped for a short hold.

    Pressing a still-gripped payload onto the goal walls/rim (forced insertion) is the only
    contact failure; free contact after release and light grazing are allowed.
    """

    def __init__(
        self, task: TaskSpec, geometry: MovingPlacementGeometrySpec, timestep_s: float
    ) -> None:
        if not isinstance(task.success, (TranslatingPlacementSuccessSpec, SE3PlacementSuccessSpec)):
            raise ValueError("T2 requires its placement success record")
        self.spec = task.success
        self.geometry = geometry
        self.timestep_s = timestep_s
        self.horizon_s = task.episode_horizon_steps / 20.0
        self.hold_steps_required = round(self.spec.placement_hold_duration_s / timestep_s)
        self.force_window_steps = round(self.spec.held_rim_force_window_s / timestep_s)
        self.reset()

    def reset(self) -> None:
        self.physics_step = 0
        self.hold_steps = 0
        self.events: dict[str, float] = {}
        self.failure_code: str | None = None
        self.terminal_time_s: float | None = None
        self.success = False
        self.maximum_held_rim_force_n = 0.0
        self._held_rim_forces: deque[float] = deque(maxlen=self.force_window_steps)
        self._approach_distance_m: float | None = None
        self._sample_geometry: _PlacementGeometryFacts | None = None
        self._gripper_contact = True
        self._frozen_stage_facts: tuple[tuple[str, bool, float, int], ...] | None = None

    def begin(self, *, approach_distance_m: float) -> None:
        if not math.isfinite(approach_distance_m) or approach_distance_m < 0.0:
            raise ValueError(
                "T2 stage initialization requires a finite non-negative approach distance"
            )
        self._approach_distance_m = approach_distance_m
        self.events.setdefault("goal_enters_observation_region", 0.0)

    def _require_begun(self) -> None:
        if self._approach_distance_m is None:
            raise RuntimeError("T2 outcome requires begin() at scored t0")

    def _stage_facts(self) -> tuple[tuple[str, bool, float, int], ...]:
        if self._frozen_stage_facts is not None:
            return self._frozen_stage_facts
        geometry = self._sample_geometry
        approach_progress = 0.0
        if geometry is not None:
            threshold = 0.05 if isinstance(self.spec, SE3PlacementSuccessSpec) else 0.10
            approach_progress = float(
                np.clip(1.0 - cast(float, self._approach_distance_m) / (3.0 * threshold), 0.0, 1.0)
            )
            if isinstance(self.spec, SE3PlacementSuccessSpec):
                orientation_progress = float(
                    np.clip(
                        1.0
                        - geometry.orientation_error_rad
                        / (3.0 * self.spec.maximum_orientation_error_rad),
                        0.0,
                        1.0,
                    )
                )
                approach_progress = min(approach_progress, orientation_progress)
        return _ordered_stage_facts(
            (
                (
                    "goal_approach",
                    "goal_approach" in self.events,
                    approach_progress,
                    0,
                ),
                (
                    "full_containment",
                    "full_containment" in self.events,
                    0.0
                    if geometry is None
                    else float(
                        np.clip(
                            geometry.insertion_depth_m / self.spec.minimum_insertion_depth_m,
                            0.0,
                            1.0,
                        )
                    ),
                    0,
                ),
                ("release", "release" in self.events, float(not self._gripper_contact), 0),
                (
                    "stable_dwell",
                    "stable_dwell" in self.events,
                    _duration_stage_progress(
                        self.hold_steps,
                        self.hold_steps_required,
                        completed="stable_dwell" in self.events,
                    ),
                    min(self.hold_steps, self.hold_steps_required),
                ),
            )
        )

    def invalidate(self, time_s: float) -> None:
        if self.terminal_time_s is None:
            self._require_begun()
            self._frozen_stage_facts = self._stage_facts()
            self.events["invalid_state"] = time_s
            self.failure_code = "invalid_state"
            self.terminal_time_s = time_s

    def update(
        self,
        *,
        time_s: float,
        finite: bool,
        geometry: _PlacementGeometryFacts,
        approach_distance_m: float,
        gripper_contact: bool,
        bottom_z_m: float,
        goal_y_m: float,
        rim_force_n: float,
    ) -> None:
        if self.terminal_time_s is not None:
            return
        self._require_begun()
        stage_facts_before = self._stage_facts()
        if not math.isclose(
            time_s, (self.physics_step + 1) * self.timestep_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError("T2 outcome requires consecutive simulator physics states")
        self.physics_step += 1
        failure: str | None = None
        if not finite or not all(
            math.isfinite(value)
            for value in (
                approach_distance_m,
                bottom_z_m,
                goal_y_m,
                rim_force_n,
                geometry.planar_clearance_m,
                geometry.floor_gap_m,
                geometry.entrance_gap_m,
                geometry.insertion_depth_m,
                geometry.orientation_error_rad,
                geometry.tilt_rad,
            )
        ):
            failure = "invalid_state"
        else:
            self._approach_distance_m = approach_distance_m
            self._sample_geometry = geometry
            self._gripper_contact = gripper_contact
            self._held_rim_forces.append(rim_force_n if gripper_contact else 0.0)
        held_force = (
            sum(self._held_rim_forces) / self.force_window_steps if failure is None else 0.0
        )
        self.maximum_held_rim_force_n = max(self.maximum_held_rim_force_n, held_force)
        if failure is None and held_force > self.spec.maximum_held_rim_force_n:
            failure = "forced_insertion"
        elif failure is None and bottom_z_m < 0.72:
            failure = "drop"
        elif failure is None and goal_y_m > self.geometry.goal_window_exit_y_m:
            failure = "goal_window_expired"
        if failure is not None:
            self._frozen_stage_facts = (
                stage_facts_before if failure == "invalid_state" else self._stage_facts()
            )
            self.events[failure] = time_s
            self.failure_code, self.terminal_time_s = failure, time_s
            return
        approach_threshold = 0.05 if isinstance(self.spec, SE3PlacementSuccessSpec) else 0.10
        if approach_distance_m <= approach_threshold and (
            not isinstance(self.spec, SE3PlacementSuccessSpec)
            or geometry.orientation_error_rad <= self.spec.maximum_orientation_error_rad
        ):
            self.events.setdefault("goal_approach", time_s)
        if geometry.contained:
            self.events.setdefault("goal_approach", time_s)
            self.events.setdefault("full_containment", time_s)
        # Placement is a state: the payload sits in the goal and no fingerpad touches it.
        # Where the hand goes afterwards, and how the payload settled, are not scored.
        placed_free = geometry.contained and not gripper_contact
        if placed_free:
            self.events.setdefault("release", time_s)
        self.hold_steps = self.hold_steps + 1 if placed_free else 0
        if self.hold_steps >= self.hold_steps_required:
            self.events["stable_dwell"] = time_s
            self.events["success"] = time_s
            self.success = True
            self.terminal_time_s = time_s
        elif time_s >= self.horizon_s - 1e-10:
            self._frozen_stage_facts = self._stage_facts()
            # OnTimeBench emits timeout on its integer control clock, unlike physical events.
            self.events["timeout"] = self.horizon_s
            self.failure_code, self.terminal_time_s = "timeout", self.horizon_s


class MovingPlacementEnv(DynamicManipulationEnv):
    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        scenario_spec: ScenarioSpec,
        execution_translation_limit_m: float = 0.05,
    ) -> None:
        if task_spec.family != "moving_placement" or not isinstance(
            task_spec.bindings, MovingPlacementBindings
        ):
            raise ValueError("MovingPlacementEnv requires a registered T2 TaskSpec")
        if not isinstance(scenario_spec.geometry, MovingPlacementGeometrySpec):
            raise ValueError("T2 requires MovingPlacementGeometrySpec")
        payload_id, goal_id = (
            task_spec.bindings.payload_object_id,
            task_spec.bindings.goal_fixture_id,
        )
        bodies = {item.entity_id: item for item in scenario_spec.physics.bodies}
        contacts = {item.entity_id: item for item in scenario_spec.physics.contacts}
        if (
            len(scenario_spec.object_states) != 1
            or scenario_spec.object_states[0].instance_id != payload_id
            or len(scenario_spec.fixture_states) != 1
            or scenario_spec.fixture_states[0].instance_id != goal_id
            or len(scenario_spec.motion) != 1
            or scenario_spec.motion[0].entity_id != goal_id
            or set(bodies) != {payload_id, goal_id}
            or set(contacts) != {payload_id, goal_id}
            or scenario_spec.events
        ):
            raise ValueError("T2 requires exactly its payload, driven goal and physical bindings")
        self._payload_spec, self._goal_spec = (
            scenario_spec.object_states[0],
            scenario_spec.fixture_states[0],
        )
        self._motion = scenario_spec.motion[0]
        self._geometry = scenario_spec.geometry
        self._is_se3 = task_spec.variant == "se3"
        expected_assets = (
            ("obj.asym_t_t2_handle_v1", "fixture.t_slot_v1")
            if self._is_se3
            else ("obj.cuboid_t2_handle_v1", "fixture.open_tray_v1")
        )
        if (self._payload_spec.asset_id, self._goal_spec.asset_id) != expected_assets:
            raise ValueError("T2 requires its frozen functional payload and goal assets")
        if not isinstance(self._motion.translation, ConstantVelocityTranslation):
            raise ValueError("T2 requires constant world +Y goal translation")
        vx, vy, vz = self._motion.translation.velocity_world_m_s
        if vx != 0.0 or vz != 0.0 or vy <= 0.0:
            raise ValueError("T2 goal must translate along world +Y")
        rotation = self._motion.rotation
        if self._is_se3:
            if not isinstance(rotation, ConstantAngularVelocityRotation) or rotation.axis_world != (
                0.0,
                0.0,
                1.0,
            ):
                raise ValueError("T2-SE3 requires explicit world-Z goal angular velocity")
        elif not isinstance(rotation, FixedRotation):
            raise ValueError("T2-TRANS requires fixed goal rotation")
        if self._goal_spec.quaternion_world_xyzw[:2] != (0.0, 0.0):
            raise ValueError("T2 goal initial orientation is planar yaw")
        if self._goal_spec.position_world_m[2] != self._geometry.goal_bottom_center_z_m:
            raise ValueError("goal fixture pose and GeometrySpec bottom center must agree")
        _validate_motion_initial_pose(
            self._motion,
            initial_position_world_m=self._goal_spec.position_world_m,
            initial_quaternion_world_xyzw=self._goal_spec.quaternion_world_xyzw,
        )
        catalog = _load_asset_catalog()
        self._payload_asset, self._goal_asset = (catalog[name] for name in expected_assets)
        self._arena_asset = catalog["arena.conveyor_v1"]
        self._body_specs, self._contact_specs = bodies, contacts
        self._placement = _PlacementOutcome(
            task_spec, self._geometry, scenario_spec.physics_timestep_s
        )
        self._settling = False
        self._placement_active = False
        self._prepared = False
        self._preparation_steps = 0
        self._pregrasp_steps = 0
        self._reset_scenario: ScenarioSpec | None = None
        super().__init__(
            task_spec=task_spec,
            scenario_spec=scenario_spec,
            seed=0,
            execution_translation_limit_m=execution_translation_limit_m,
        )

    def _load_model(self) -> None:
        self._placement_active = self._settling = False
        super()._load_model()
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(robot_model.base_xpos_offset["table"](0.8))
        arena = TableArena(table_offset=(0.0, 0.0, self._geometry.support_surface_z_m))
        arena.set_origin((0.0, 0.0, 0.0))
        arena.worldbody.remove(arena.table_body)
        for name in ("left_eef_target", "right_eef_target"):
            target = arena.worldbody.find(f"body[@name='{name}']")
            if target is None:
                raise RuntimeError("Robosuite 1.5.2 arena target binding is missing")
            arena.worldbody.remove(target)
        arena.merge(MujocoXML(str(self._arena_asset.path)))
        lights = arena.worldbody.findall("light")
        if len(lights) != 1:
            raise RuntimeError("T2 TableArena requires its one canonical light")
        lights[0].set("name", "conveyor_key")
        belt = arena.worldbody.find("./body[@name='belt']")
        if belt is None:
            raise ValueError("T2 conveyor support body must resolve")
        belt.set("pos", _xml_numbers((0.0, 0.0, self._geometry.support_surface_z_m - 0.020)))
        parked_canopy = arena.worldbody.find("./body[@name='occluder']")
        if parked_canopy is not None:
            arena.worldbody.remove(parked_canopy)
        self._payload_model = MujocoXMLObject(
            str(self._payload_asset.path),
            name=self._payload_spec.instance_id,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self._goal_model = MujocoXMLObject(
            str(self._goal_asset.path),
            name=self._goal_spec.instance_id,
            joints=[
                {
                    "name": "x",
                    "type": "slide",
                    "axis": "1 0 0",
                    "limited": "false",
                    "damping": "0",
                },
                {
                    "name": "travel",
                    "type": "slide",
                    "axis": "0 1 0",
                    "limited": "false",
                    "damping": "0",
                    "armature": "0",
                    "frictionloss": "0",
                },
                {
                    "name": "z",
                    "type": "slide",
                    "axis": "0 0 1",
                    "limited": "false",
                    "damping": "0",
                },
                {
                    "name": "yaw",
                    "type": "hinge",
                    "axis": "0 0 1",
                    "limited": "false",
                    "damping": "0",
                    "armature": "0",
                    "frictionloss": "0",
                },
            ],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self._resize_goal_opening()
        body = self._goal_model.get_obj()
        body.set("pos", "0 0 -1")
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot_model],
            mujoco_objects=[self._payload_model, self._goal_model],
        )

    def _resize_goal_opening(self) -> None:
        """Only the existing clearance field changes the asset's planar opening."""
        clearance = self._geometry.goal_planar_clearance_m
        nominal = 0.004 if self._is_se3 else 0.008
        delta = clearance - nominal
        for geom in self._goal_model.get_obj().findall("geom"):
            name = geom.get("name", "").removeprefix(self._goal_model.naming_prefix)
            part = name.removesuffix("_collision").removesuffix("_visual")
            position = np.fromstring(geom.get("pos", ""), sep=" ")
            size = np.fromstring(geom.get("size", ""), sep=" ")
            if position.shape != (3,) or size.shape != (3,):
                raise ValueError("T2 fixture must contain explicit box position and size")
            if part == "floor":
                size[:2] += delta
            elif part in {"left", "right", "bar_left", "bar_right"}:
                position[0] += -delta if "left" in part else delta
                size[1] += delta
            elif part in {"near", "far"}:
                position[1] += -delta if part == "near" else delta
                size[0] += delta
            elif part in {"stem_left", "stem_right"}:
                position[0] += -delta if "left" in part else delta
                position[1] -= delta
            else:
                raise ValueError(f"unknown T2 fixture box: {part}")
            if np.any(size <= 0.0):
                raise ValueError("T2 fixture clearance produces non-positive geometry")
            geom.set("pos", _xml_numbers(position.tolist()))
            geom.set("size", _xml_numbers(size.tolist()))

    def edit_model_xml(self, xml_str: str) -> str:
        return _rewrite_scenario_lighting_and_gravity_xml(
            super().edit_model_xml(xml_str),
            self._scenario_spec,
            light_id="conveyor_key",
            invalid_message="T2 requires the explicit canonical conveyor light set",
            duplicate_message="duplicate conveyor light",
            missing_message="conveyor_key must resolve in T2 arena",
            headlight_before_gravity=False,
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        model = self.sim.model._model
        self._payload_body_id = int(model.body(self._payload_model.root_body).id)
        self._goal_body_id = int(model.body(self._goal_model.root_body).id)
        self._payload_geom_ids = tuple(
            int(model.geom(name).id) for name in self._payload_model.contact_geoms
        )
        self._scored_payload_geom_ids = tuple(
            index
            for index in self._payload_geom_ids
            if model.geom(index).name != self._payload_model.naming_prefix + "handle_collision"
        )
        self._goal_geom_ids = tuple(
            int(model.geom(name).id) for name in self._goal_model.contact_geoms
        )
        self._floor_geom_id = int(model.geom(self._goal_model.naming_prefix + "floor_collision").id)
        self._rim_geom_ids = tuple(
            index for index in self._goal_geom_ids if index != self._floor_geom_id
        )
        payload_joint = model.joint(self._payload_model.joints[0]).id
        if model.jnt_type[payload_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("T2 payload must be a free body")
        self._payload_qpos_address = int(model.jnt_qposadr[payload_joint])
        self._payload_qvel_address = int(model.jnt_dofadr[payload_joint])
        goal_joints = [model.joint(name).id for name in self._goal_model.joints]
        if len(goal_joints) != 4 or tuple(model.jnt_type[goal_joints]) != (
            mujoco.mjtJoint.mjJNT_SLIDE,
            mujoco.mjtJoint.mjJNT_SLIDE,
            mujoco.mjtJoint.mjJNT_SLIDE,
            mujoco.mjtJoint.mjJNT_HINGE,
        ):
            raise ValueError("T2 goal requires its three native slides and yaw joint")
        self._goal_qpos_indexes = model.jnt_qposadr[goal_joints].copy()
        self._goal_qvel_indexes = model.jnt_dofadr[goal_joints].copy()
        self._entity_bodies = {
            self._payload_spec.instance_id: self._payload_body_id,
            self._goal_spec.instance_id: self._goal_body_id,
        }
        self._entity_geoms = {
            self._payload_spec.instance_id: self._payload_geom_ids,
            self._goal_spec.instance_id: self._goal_geom_ids,
        }
        self._compiled_joint_physics = _resolve_joint_physics(
            model,
            self._scenario_spec.physics.joints,
            entity_body_ids=self._entity_bodies,
        )
        self._bind_scene_entities(
            target_body_id=self._goal_body_id,
            entity_body_ids=self._entity_bodies,
        )
        pads = self.robots[0].gripper["right"].important_geoms
        self._left_pad_ids = frozenset(int(model.geom(name).id) for name in pads["left_fingerpad"])
        self._right_pad_ids = frozenset(
            int(model.geom(name).id) for name in pads["right_fingerpad"]
        )
        self._gripper_geom_ids = frozenset(
            int(model.geom(name).id) for name in self.robots[0].gripper["right"].contact_geoms
        )
        self._opening_rectangles, self._opening_boundary = self._read_opening()

    def _read_opening(
        self,
    ) -> tuple[tuple[tuple[float, float, float, float], ...], npt.NDArray[np.float64]]:
        model = self.sim.model._model

        def bounds(name: str) -> tuple[float, float, float, float]:
            index = model.geom(self._goal_model.naming_prefix + name + "_collision").id
            p, size = model.geom_pos[index], model.geom_size[index]
            if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_BOX or not np.allclose(
                model.geom_quat[index], (1, 0, 0, 0), rtol=0.0, atol=1e-12
            ):
                raise ValueError("T2 opening requires the actual asset's axis-aligned wall boxes")
            return (
                float(p[0] - size[0]),
                float(p[0] + size[0]),
                float(p[1] - size[1]),
                float(p[1] + size[1]),
            )

        if not self._is_se3:
            x0, x1, y0, y1 = (
                bounds("left")[1],
                bounds("right")[0],
                bounds("near")[3],
                bounds("far")[2],
            )
            return ((x0, x1, y0, y1),), np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
        x0, x1 = bounds("bar_left")[1], bounds("bar_right")[0]
        sx0, sx1 = bounds("stem_left")[1], bounds("stem_right")[0]
        y0, join, y1 = bounds("near")[3], bounds("bar_left")[2], bounds("far")[2]
        return ((x0, x1, join, y1), (sx0, sx1, y0, join)), np.asarray(
            (
                (x0, y1),
                (x0, join),
                (sx0, join),
                (sx0, y0),
                (sx1, y0),
                (sx1, join),
                (x1, join),
                (x1, y1),
            )
        )

    def _reset_internal(self) -> None:
        self._placement_active = self._settling = self._prepared = False
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
        x, y, z, w = self._payload_spec.quaternion_world_xyzw
        address = self._payload_qpos_address
        data.qpos[address : address + 7] = (*self._payload_spec.position_world_m, w, x, y, z)
        data.qpos[self._goal_qpos_indexes] = 0.0
        data.qvel[:] = 0.0
        # Free-joint translation is world-frame; its angular coordinates are body-local.
        # ObjectStateSpec velocity is measured at the object body origin (XBODY), not COM.
        quaternion = np.asarray((w, x, y, z), dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, quaternion)
        velocity_address = self._payload_qvel_address
        data.qvel[velocity_address : velocity_address + 3] = (
            self._payload_spec.linear_velocity_world_m_s
        )
        data.qvel[velocity_address + 3 : velocity_address + 6] = (
            rotation.reshape(3, 3).T @ self._payload_spec.angular_velocity_world_rad_s
        )
        data.xfrc_applied[:] = 0.0
        data.qfrc_applied[:] = 0.0
        gripper_controller = robot.composite_controller.part_controllers["right_gripper"]
        lower = np.asarray(gripper_controller.actuator_min, dtype=np.float64)
        upper = np.asarray(gripper_controller.actuator_max, dtype=np.float64)
        initial_fingers = np.asarray(self._scenario_spec.robot_initial_state.gripper_position_m)
        if np.any(initial_fingers < lower) or np.any(initial_fingers > upper):
            raise _ScenarioResetRejected("joint_out_of_range", float(data.time))
        robot.gripper["right"].current_action = np.zeros(2)
        self._set_goal_reference(0.0)
        self.sim.forward()
        self._placement.reset()
        self._preparation_steps = self._pregrasp_steps = 0
        self._reset_scenario = self._read_preparation_input()

    def reset(self) -> Mapping[str, object]:
        super().reset()
        self._settling = True
        neutral = np.zeros(self.action_dim, dtype=np.float64)
        neutral[-1] = 1.0
        for _ in range(20):
            ManipulationEnv.step(self, neutral)
            address = self._payload_qpos_address
            x, y, z, w = self._payload_spec.quaternion_world_xyzw
            self.sim.data.qpos[address : address + 7] = (
                *self._payload_spec.position_world_m,
                w,
                x,
                y,
                z,
            )
            address = self._payload_qvel_address
            self.sim.data.qvel[address : address + 6] = 0.0
            self.sim.forward()
        self._settling = False
        left, right, _ = _fingerpad_contacts(
            self.sim.model._model,
            self.sim.data._data,
            self._payload_geom_ids,
            self._left_pad_ids,
            self._right_pad_ids,
        )
        if self._preparation_steps != 500 or not (left and right):
            raise _ScenarioResetRejected("pregrasp_not_stable", float(self.sim.data.time))
        # Export is the verified preparation input, not a second, settled snapshot.
        # Re-running preparation always starts from that same input; velocities survive.
        self.sim.data.time = 0.0
        self.cur_time = 0.0
        self.timestep = self._episode_tick = 0
        self.done = False
        self._action_spec = None
        self._action_delay_queue = None
        self._previous_eef_quaternion_xyzw = None
        self._obs_cache: dict[str, Any] = {}
        for observable in self._observables.values():
            observable.reset()
        self._set_goal_reference(0.0)
        self._placement.reset()
        self._placement_active = self._prepared = True
        self.sim.forward()
        data = self.sim.data._data
        self._placement.begin(
            approach_distance_m=float(
                np.linalg.norm(
                    data.xpos[self._payload_body_id]
                    - self._goal_spec.position_world_m
                    - (0.0, 0.0, 0.022 if self._is_se3 else 0.027)
                )
            )
        )
        self._finish_preparation_controller()
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
        qpos = data.qpos[self._payload_qpos_address : self._payload_qpos_address + 7]
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_XBODY, self._payload_body_id, velocity, 0
        )
        if not np.allclose(
            velocity,
            (
                *self._payload_spec.angular_velocity_world_rad_s,
                *self._payload_spec.linear_velocity_world_m_s,
            ),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("compiled T2 preparation world twist differs from ScenarioSpec")
        payload = ObjectStateSpec(
            self._payload_spec.instance_id,
            self._payload_spec.asset_id,
            cast(Vector3, tuple(float(v) for v in qpos[:3])),
            cast(Vector4, tuple(float(v) for v in qpos[[4, 5, 6, 3]])),
            self._payload_spec.linear_velocity_world_m_s,
            self._payload_spec.angular_velocity_world_rad_s,
        )
        expected_q = np.asarray(self._goal_spec.quaternion_world_xyzw)[[3, 0, 1, 2]]
        expected_q /= np.linalg.norm(expected_q)
        if not math.isclose(
            abs(float(np.dot(data.xquat[self._goal_body_id], expected_q))),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("compiled T2 goal orientation differs from ScenarioSpec")
        if not np.allclose(
            data.xpos[self._goal_body_id], self._goal_spec.position_world_m, rtol=0.0, atol=1e-12
        ):
            raise ValueError("compiled T2 goal position differs from ScenarioSpec")
        goal_reference = self._goal_reference(0.0)
        if not np.array_equal(
            data.qvel[self._goal_qvel_indexes],
            (
                *goal_reference.linear_velocity_world_m_s,
                goal_reference.angular_velocity_world_rad_s[2],
            ),
        ):
            raise ValueError("compiled T2 goal joint velocities differ from declared motion")
        goal = self._goal_spec
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
            for index in self._entity_geoms[item.entity_id]:
                actual = ContactPhysicsSpec(
                    item.entity_id,
                    cast(Vector3, tuple(float(v) for v in model.geom_friction[index])),
                    cast(Vector2, tuple(float(v) for v in model.geom_solref[index])),
                    cast(
                        tuple[float, float, float, float, float],
                        tuple(float(v) for v in model.geom_solimp[index]),
                    ),
                )
                if actual != item:
                    raise ValueError("compiled T2 contact parameters differ from ScenarioSpec")
        _validate_compiled_joint_physics(
            model,
            self._compiled_joint_physics,
            owner="T2",
        )
        support = model.geom("belt_collision").id
        if not math.isclose(
            float(data.geom_xpos[support, 2] + model.geom_size[support, 2]),
            self._geometry.support_surface_z_m,
            abs_tol=1e-12,
        ):
            raise ValueError("compiled T2 support differs from GeometrySpec")
        if model.flg_surfacevel or np.any(model.geom_surfacevel):
            raise ValueError("T2 moving goal uses native joints, not a conveyor surface field")
        clearance = self._geometry.goal_planar_clearance_m

        def footprint(name: str) -> tuple[float, float, float, float]:
            index = model.geom(self._payload_model.naming_prefix + name + "_collision").id
            position, size = model.geom_pos[index], model.geom_size[index]
            return (
                float(position[0] - size[0]),
                float(position[0] + size[0]),
                float(position[1] - size[1]),
                float(position[1] + size[1]),
            )

        expected_opening: tuple[tuple[float, float, float, float], ...]
        if self._is_se3:
            bar, stem = footprint("bar"), footprint("stem")
            expected_opening = (
                (bar[0] - clearance, bar[1] + clearance, bar[2] - clearance, bar[3] + clearance),
                (stem[0] - clearance, stem[1] + clearance, stem[2] - clearance, bar[2] - clearance),
            )
        else:
            body = footprint("body")
            expected_opening = (
                (
                    body[0] - clearance,
                    body[1] + clearance,
                    body[2] - clearance,
                    body[3] + clearance,
                ),
            )
        actual_opening, _ = self._read_opening()
        if not np.allclose(actual_opening, expected_opening, rtol=0.0, atol=1e-12):
            raise ValueError("compiled T2 opening differs from payload footprint and clearance")
        for name, camera in zip(self._camera_sensor_names, scenario.cameras.views, strict=True):
            index = model.camera(name).id
            unit = np.asarray(camera.quaternion_parent_xyzw)[[3, 0, 1, 2]]
            unit /= np.linalg.norm(unit)
            if not (
                np.allclose(model.cam_pos[index], camera.position_parent_m, rtol=0.0, atol=1e-12)
                and math.isclose(
                    abs(float(np.dot(model.cam_quat[index], unit))), 1.0, rel_tol=0.0, abs_tol=1e-12
                )
                and float(model.cam_fovy[index]) == camera.vertical_fov_deg
            ):
                raise ValueError("compiled T2 camera differs from ScenarioSpec")
        if (
            model.nlight != 1
            or model.light(0).name != "conveyor_key"
            or not model.vis.headlight.active
            or model.light_type[0] != mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            or model.light_castshadow[0]
        ):
            raise ValueError("compiled T2 canonical light set is invalid")
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
                raise ValueError("compiled T2 light values differ from ScenarioSpec")
        if (
            float(model.light_cutoff[0]) != light.cutoff_deg
            or bool(model.light_active[0]) != light.active
        ):
            raise ValueError("compiled T2 light scalars differ from ScenarioSpec")
        exported = replace(
            scenario,
            robot_initial_state=robot_state,
            object_states=(payload,),
            fixture_states=(goal,),
            physics=replace(
                scenario.physics,
                gravity_world_m_s2=cast(Vector3, tuple(float(v) for v in model.opt.gravity)),
                bodies=bodies,
            ),
        )
        other_velocity = np.concatenate((data.qvel[: self._payload_qvel_address],))
        if np.any(other_velocity) or exported != scenario:
            raise ValueError("compiled T2 preparation input differs from ScenarioSpec")
        return exported

    def export_scenario(self) -> ScenarioSpec:
        if not self._prepared or self._reset_scenario is None:
            raise RuntimeError("T2 requires verified preparation and real pregrasp before export")
        return self._reset_scenario

    def _se3_planar_geometry(
        self,
        relative_translation: npt.NDArray[np.float64],
        relative_rotation: npt.NDArray[np.float64],
        overlap_tolerance_m: float = 0.0,
    ) -> tuple[bool, float]:
        """Return (footprint inside the opening within the overlap tolerance, clearance).

        The clearance is the untolerated boundary distance, negative when overlapping."""
        model = self.sim.model._model
        bar = model.geom(self._payload_model.naming_prefix + "bar_collision").id
        stem = model.geom(self._payload_model.naming_prefix + "stem_collision").id
        bar_position, bar_size = model.geom_pos[bar], model.geom_size[bar]
        stem_position, stem_size = model.geom_pos[stem], model.geom_size[stem]
        x0, x1 = float(bar_position[0] - bar_size[0]), float(bar_position[0] + bar_size[0])
        sx0, sx1 = float(stem_position[0] - stem_size[0]), float(stem_position[0] + stem_size[0])
        y0 = float(stem_position[1] - stem_size[1])
        join, y1 = float(bar_position[1] - bar_size[1]), float(bar_position[1] + bar_size[1])
        rectangles = ((x0, x1, join, y1), (sx0, sx1, y0, join))
        boundary = np.asarray(
            (
                (x0, y1),
                (x0, join),
                (sx0, join),
                (sx0, y0),
                (sx1, y0),
                (sx1, join),
                (x1, join),
                (x1, y1),
            ),
            dtype=np.float64,
        )
        yaw = math.atan2(float(relative_rotation[1, 0]), float(relative_rotation[0, 0]))
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
        object_boundary = boundary @ rotation.T + relative_translation[:2]
        inside_area = 0.0
        for rectangle in rectangles:
            transformed = _rectangle_corners(rectangle) @ rotation.T + relative_translation[:2]
            inside_area += sum(
                _clipped_area(transformed, opening) for opening in self._opening_rectangles
            )
        object_area = sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in rectangles)
        outside_area = max(0.0, object_area - min(object_area, inside_area))
        exact = outside_area <= 1e-12
        clearance = (
            _polygon_boundary_distance(object_boundary, self._opening_boundary)
            if exact
            else -math.sqrt(max(outside_area, 1e-12))
        )
        if exact or overlap_tolerance_m <= 0.0:
            return exact, clearance
        # Contact penetration only: the footprint shrunk by the tolerance must lie inside.
        t = overlap_tolerance_m
        shrunk = tuple((a + t, b - t, c + t, d - t) for a, b, c, d in rectangles)
        inside_area = sum(
            _clipped_area(_rectangle_corners(rectangle) @ rotation.T + relative_translation[:2], o)
            for rectangle in shrunk
            for o in self._opening_rectangles
        )
        shrunk_area = sum((b - a) * (d - c) for a, b, c, d in shrunk)
        return shrunk_area - inside_area <= 1e-12, clearance

    def _geometry_facts(self) -> _PlacementGeometryFacts:
        model, data = self.sim.model._model, self.sim.data._data
        reference = self._goal_reference(float(data.time))
        q = reference.quaternion_world_xyzw
        goal_rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(goal_rotation, q[[3, 0, 1, 2]])
        goal_rotation = goal_rotation.reshape(3, 3)
        origin = reference.position_world_m
        planar = True
        clearance = math.inf
        bottom, top = math.inf, -math.inf
        for index in self._scored_payload_geom_ids:
            if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("T2 payload geometry must use its fixed collision boxes")
            corners = (
                data.geom_xpos[index]
                + (_BOX_SIGNS * model.geom_size[index]) @ data.geom_xmat[index].reshape(3, 3).T
            )
            local = (corners - origin) @ goal_rotation
            bottom, top = min(bottom, float(local[:, 2].min())), max(top, float(local[:, 2].max()))
            if not self._is_se3:
                x0, x1, y0, y1 = self._opening_rectangles[0]
                clearance = min(
                    clearance,
                    float(local[:, 0].min() - x0),
                    float(x1 - local[:, 0].max()),
                    float(local[:, 1].min() - y0),
                    float(y1 - local[:, 1].max()),
                )
                continue
        floor = self._floor_geom_id
        floor_top = float(model.geom_pos[floor, 2] + model.geom_size[floor, 2])
        entrance = min(
            float(model.geom_pos[index, 2] + model.geom_size[index, 2])
            for index in self._rim_geom_ids
        )
        relative_rotation = goal_rotation.T @ data.xmat[self._payload_body_id].reshape(3, 3)
        spec = self._placement.spec
        tolerance = spec.planar_overlap_tolerance_m
        if self._is_se3:
            planar, clearance = self._se3_planar_geometry(
                goal_rotation.T @ (data.xpos[self._payload_body_id] - origin),
                relative_rotation,
                tolerance,
            )
        else:
            planar = clearance >= -tolerance
        orientation = math.acos(
            float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
        )
        tilt = math.acos(float(np.clip(relative_rotation[2, 2], -1.0, 1.0)))
        floor_gap, entrance_gap, depth = bottom - floor_top, entrance - top, entrance - bottom
        # Placed: footprint inside the opening and sunk well below the rim, i.e. in the box
        # rather than resting on its edge. Seating height, speed and settling are not scored.
        contained = bool(planar and depth >= spec.minimum_insertion_depth_m)
        if isinstance(spec, SE3PlacementSuccessSpec):
            contained = contained and tilt <= spec.maximum_tilt_rad
        return _PlacementGeometryFacts(
            contained, clearance, floor_gap, entrance_gap, depth, orientation, tilt
        )

    def _pre_action(self, action: npt.NDArray[np.float64], policy_step: bool = False) -> None:
        data = self.sim.data._data
        if not _physics_state_is_valid(data):
            if self._settling:
                raise _ScenarioResetRejected("invalid_state", float(data.time))
            self._placement.invalidate(self._placement.physics_step * self.model_timestep)
        if self._placement_active:
            self._set_goal_reference(float(data.time))
        super()._pre_action(action, policy_step)

    def _goal_reference(self, time_s: float) -> _MotionState:
        reference = _motion_state_at_time(
            self._motion,
            initial_position_world_m=self._goal_spec.position_world_m,
            initial_quaternion_world_xyzw=self._goal_spec.quaternion_world_xyzw,
            simulator_time_s=time_s,
        )
        if isinstance(self._motion.rotation, ConstantAngularVelocityRotation):
            initial = self._goal_spec.quaternion_world_xyzw
            initial_yaw = 2.0 * math.atan2(initial[2], initial[3])
            yaw = initial_yaw + time_s * self._motion.rotation.angular_speed_rad_s
            reference = replace(
                reference,
                quaternion_world_xyzw=np.array(
                    (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)),
                    dtype=np.float64,
                ),
            )
        return reference

    def _dynamic_target_snapshot(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        time_s = self._episode_tick / self._scenario_spec.sensor_frequency_hz
        reference = self._goal_reference(time_s)
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, reference.quaternion_world_xyzw[[3, 0, 1, 2]])
        return reference.position_world_m.copy(), rotation.reshape(3, 3)

    def _set_goal_reference(self, time_s: float) -> None:
        data = self.sim.data._data
        reference = self._goal_reference(time_s)
        x, y, z, w = reference.quaternion_world_xyzw
        position = reference.position_world_m
        data.qpos[self._goal_qpos_indexes] = (
            position[0],
            position[1],
            position[2] + 1.0,
            2.0 * np.arctan2(z, w),
        )
        data.qvel[self._goal_qvel_indexes] = (
            *reference.linear_velocity_world_m_s,
            reference.angular_velocity_world_rad_s[2],
        )
        self.sim.forward()

    def _update_observables(self, force: bool = False) -> None:
        if not force and (self._settling or self._placement_active):
            model, data = self.sim.model._model, self.sim.data._data
            finite = _physics_state_is_valid(data)
            if not finite:
                if self._settling:
                    raise _ScenarioResetRejected("invalid_state", float(data.time))
                self._placement.invalidate((self._placement.physics_step + 1) * self.model_timestep)
                super()._update_observables(force)
                return
            left, right, opposing = _fingerpad_contacts(
                model, data, self._payload_geom_ids, self._left_pad_ids, self._right_pad_ids
            )
            if self._settling:
                self._preparation_steps += 1
                self._pregrasp_steps = self._pregrasp_steps + 1 if left and right else 0
            else:
                reference = self._goal_reference(float(data.time))
                rim_force = 0.0
                force_buffer = np.empty(6)
                for index in range(data.ncon):
                    contact = data.contact[index]
                    a, b = int(contact.geom1), int(contact.geom2)
                    other = (
                        b
                        if a in self._payload_geom_ids
                        else a
                        if b in self._payload_geom_ids
                        else -1
                    )
                    if other in self._goal_geom_ids and self._placement.terminal_time_s is None:
                        self._capture_first_contact("payload_goal", self._goal_body_id)
                    if other in self._rim_geom_ids:
                        mujoco.mj_contactForce(model, data, index, force_buffer)
                        if not np.isfinite(force_buffer).all() or force_buffer[0] < 0.0:
                            raise RuntimeError("invalid native T2 rim contact force")
                        rim_force += float(np.linalg.norm(force_buffer[:3]))
                self._placement.update(
                    time_s=float(data.time),
                    finite=finite,
                    geometry=self._geometry_facts(),
                    approach_distance_m=float(
                        np.linalg.norm(
                            data.xpos[self._payload_body_id]
                            - reference.position_world_m
                            - (0.0, 0.0, 0.022 if self._is_se3 else 0.027)
                        )
                    ),
                    gripper_contact=left or right,
                    bottom_z_m=_collision_bottom_z(model, data, self._scored_payload_geom_ids),
                    goal_y_m=float(reference.position_world_m[1]),
                    rim_force_n=rim_force,
                )
        super()._update_observables(force)

    def _post_action(self, action: npt.NDArray[np.float64]) -> tuple[float, bool, dict[str, Any]]:
        reward, _, info = super()._post_action(action)
        self.done = self._placement_active and self._placement.terminal_time_s is not None
        info.update(
            success=self._placement.success,
            failure_code=self._placement.failure_code,
            terminal_time_s=self._placement.terminal_time_s,
            events={
                {
                    "goal_approach": (
                        "goal_pose_alignment"
                        if isinstance(self._placement.spec, SE3PlacementSuccessSpec)
                        else "goal_approach"
                    ),
                    "full_containment": "containment_entry",
                    "forced_insertion": "object_goal_collision_unsafe",
                }.get(name, name): time_s
                for name, time_s in self._placement.events.items()
            },
            stage_facts=self._placement._stage_facts(),
        )
        return float(reward), bool(self.done), info

    def reward(self, action: npt.NDArray[np.float64] | None = None) -> float:
        del action
        return float(self._placement.success)

    def _check_success(self) -> bool:
        return self._placement.success

    def _recording_frame_facts(self) -> _RecordingFrameFacts:
        return (
            self._placement.success,
            self._placement.failure_code,
            self._placement.terminal_time_s,
            self._placement._stage_facts(),
        )


__all__: list[str] = []
