"""Current-state native controllers shared by feasibility and formal rollout."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import mujoco
import numpy as np
import numpy.typing as npt
from robosuite.utils.transform_utils import mat2quat

from ontimebench.control.action import BenchmarkAction, _so3_exp, _so3_log
from ontimebench.envs.base import DynamicManipulationEnv, _fingerpad_contacts
from ontimebench.envs.components.occlusion import (
    _OcclusionVisibilityAudit,
    _OcclusionVisibilityState,
    _target_visibility,
)
from ontimebench.scenarios._rejection import ScenarioRejected
from ontimebench.tasks.registry import load_task
from ontimebench.tasks.spec import GraspMotion, SE3PlacementSuccessSpec

_GRASP_CONTACT_OFFSET = (0.0, 0.0, 0.005)
_CONTACT_WORLD_OFFSETS = {
    "full_wire_passage": (0.0, 0.0, 0.01),
    "ball_capture": (0.008, 0.0, 0.0),
    "ball_capture_on_table": (0.0, 0.0, 0.0),
    "can_capture": (0.0, 0.0, -0.006),
}
_CONTACT_CAPTURE_EXTENTS = {GraspMotion.MATCHED_XYZ: 0.032, GraspMotion.MATCHED_YAW: 0.028}
_PLACEMENT_CONTACT_HEIGHT = {"moving_receptacle": 0.027, "rotating_receptacle": 0.022}


def _expert_seated(facts: Any, *, se3: bool) -> bool:
    """The placement expert releases only when seated: top below the rim, bottom at the floor,
    and for SE3 a 2 mm planar margin, floor gap <= 2 mm and tilt <= 0.05 rad."""
    seated = (
        facts.planar_clearance_m >= -0.002
        and facts.floor_gap_m >= -0.002
        and facts.entrance_gap_m >= 0.0
    )
    if se3:
        seated = (
            seated
            and facts.planar_clearance_m >= 0.002
            and facts.floor_gap_m <= 0.002
            and facts.tilt_rad <= 0.05
        )
    return bool(seated)


_REPLANNING_CONTACT_OFFSET = (0.0, -0.025, 0.0)


def _slider_capture_plane(ramp_angle_rad: float) -> float:
    return float(np.clip(0.09 - 0.004 * max(math.degrees(ramp_angle_rad) - 10.0, 0.0), 0.07, 0.09))


def _slider_contact_offset(capture_y: float, ramp_exited: bool) -> npt.NDArray[np.float64]:
    return np.array((0.0, (-0.028 if ramp_exited else -0.0387) if capture_y <= 0.078 else 0.0, 0.0))


def _grasp_axis_offset(
    motion: GraspMotion | None, *, joint_space_yaws: tuple[float, float] | None = None
) -> float:
    if motion is not GraspMotion.CONVEYOR:
        return math.pi / 2 if motion is GraspMotion.FREE else -math.pi / 2
    if joint_space_yaws is None:
        return 0.0
    # The square conveyor target has two equivalent closing axes in joint space.
    target, jaw = joint_space_yaws
    return min(
        (0.0, math.pi / 2), key=lambda offset: abs(_CaptureOracle._axis_wrap(target + offset - jaw))
    )


def _select_capture_contact_point(
    position: npt.NDArray[np.float64],
    yaw: float,
    points: tuple[npt.NDArray[np.float64], ...],
    eef: npt.NDArray[np.float64],
    yaw_rate: float,
    so3: bool,
) -> int:
    scores = []
    for i, point in enumerate(points):
        world = _CaptureOracle._point_world(position, yaw, point)
        distance = float(np.linalg.norm(world[:2] - eef[:2]))
        upstream = yaw_rate * float((world - position)[0]) if so3 and abs(yaw_rate) >= 0.20 else 0.0
        if so3:
            distance -= 0.05 * (0.04 - abs(float(point[0])))
        scores.append((upstream, distance, i))
    return min(scores)[2]


class _ContactReference:
    """One scene's expert grasp choice, fixed from scored t0 and shared by both errors."""

    def __init__(self, environment: DynamicManipulationEnv) -> None:
        self.task_id = environment._task_spec.task_id
        self._grasp_motion = environment._task_spec.grasp_motion
        if self.task_id == "wire_passage":
            raise ValueError("wire_passage has no physical contact error; report N/A")
        model, data = environment.sim.model._model, environment.sim.data._data
        site = environment.robots[0].eef_site_id["right"]
        self.initial_position_world_m = data.site_xpos[site].copy()
        self.initial_rotation_world = data.site_xmat[site].reshape(3, 3).copy()
        base = model.body(environment._robot_base_body_name).id
        self.world_from_robot_base_rotation = data.xmat[base].reshape(3, 3).copy()
        placement = self.task_id in _PLACEMENT_CONTACT_HEIGHT
        self.target_body_id = int(
            environment._goal_body_id
            if placement
            else environment._wand_body_id
            if self.task_id == "full_wire_passage"
            else environment._target_body_id
        )
        self.contact_kind = (
            "payload_goal"
            if placement
            else "fingerpad_wand"
            if self.task_id == "full_wire_passage"
            else "robot_target"
            if self.task_id == "post_impact_grasp"
            else "fingerpad_target"
        )
        self.target_body_name = model.body(self.target_body_id).name
        self._local_point = np.zeros(3)
        self._object_to_eef = np.zeros(3)
        self._object_to_eef_rotation = np.eye(3)
        target_p = data.xpos[self.target_body_id]
        target_r = data.xmat[self.target_body_id].reshape(3, 3)
        self._initial_target_yaw = math.atan2(float(target_r[1, 0]), float(target_r[0, 0]))
        left = np.mean(data.geom_xpos[sorted(environment._left_pad_ids)], axis=0)
        right = np.mean(data.geom_xpos[sorted(environment._right_pad_ids)], axis=0)
        self._initial_jaw_yaw = math.atan2(float((right - left)[1]), float((right - left)[0]))
        self._axis_offset = _grasp_axis_offset(
            self._grasp_motion,
            joint_space_yaws=(self._initial_target_yaw, self._initial_jaw_yaw)
            if environment._scenario_spec.action_mode == "J8"
            else None,
        )
        if self._grasp_motion is not None and self._grasp_motion in _CONTACT_CAPTURE_EXTENTS:
            extent = _CONTACT_CAPTURE_EXTENTS[self._grasp_motion]
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, self.target_body_id, velocity, 0
            )
            yaw_rate = float(np.clip(velocity[2], -1.5, 1.5))
            points = (np.array((extent, 0.0, 0.0)), np.array((-extent, 0.0, 0.0)))
            selected = _select_capture_contact_point(
                target_p,
                self._initial_target_yaw,
                points,
                self.initial_position_world_m,
                yaw_rate,
                self._grasp_motion is GraspMotion.MATCHED_YAW,
            )
            self._local_point = points[selected]
        self._slider_capture_y = (
            _slider_capture_plane(environment._geometry.ramp_angle_rad)
            if self.task_id == "slider_capture"
            else 0.09
        )
        self._can_max_yaw_offset = (
            _FreeDynamicsOracle(environment)._can_j8_max_yaw_offset
            if self.task_id == "can_capture" and environment._scenario_spec.action_mode == "J8"
            else None
        )
        if placement:
            payload = environment._payload_body_id
            object_r = data.xmat[payload].reshape(3, 3)
            self._object_to_eef = object_r.T @ (self.initial_position_world_m - data.xpos[payload])
            self._object_to_eef_rotation = object_r.T @ self.initial_rotation_world

    def frame(
        self,
        target_position_world_m: npt.NDArray[np.float64],
        target_rotation_world: npt.NDArray[np.float64],
        *,
        ramp_exited: bool = False,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        p = np.asarray(target_position_world_m, dtype=np.float64)
        r = np.asarray(target_rotation_world, dtype=np.float64)
        rotation = self.initial_rotation_world.copy()
        if self.task_id in _PLACEMENT_CONTACT_HEIGHT:
            return (
                p + (0, 0, _PLACEMENT_CONTACT_HEIGHT[self.task_id]) + r @ self._object_to_eef,
                r @ self._object_to_eef_rotation,
            )
        if self._grasp_motion is not None:
            yaw = (
                math.atan2(float(r[1, 0]), float(r[0, 0]))
                if self._grasp_motion.follows_yaw
                else self._initial_target_yaw
            )
            position = _CaptureOracle._point_world(p, yaw, self._local_point)
            delta = _CaptureOracle._axis_wrap(yaw + self._axis_offset - self._initial_jaw_yaw)
            rotation = _so3_exp(np.array((0.0, 0.0, delta))) @ rotation
            return position + _GRASP_CONTACT_OFFSET, rotation
        if self.task_id == "slider_capture":
            axis = r[:, 0]
            if np.linalg.norm(axis[:2]) > 1e-8:
                yaw = math.atan2(float(axis[1]), float(axis[0]))
                delta = _CaptureOracle._axis_wrap(yaw - self._initial_jaw_yaw)
                rotation = _so3_exp(np.array((0.0, 0.0, delta))) @ rotation
            return p + r @ _slider_contact_offset(self._slider_capture_y, ramp_exited), rotation
        if self.task_id == "can_capture":
            axis = np.array((-r[1, 2], r[0, 2], 0.0))
            if np.linalg.norm(axis[:2]) > 1e-8:
                yaw = math.atan2(float(axis[1]), float(axis[0]))
                delta = _CaptureOracle._axis_wrap(yaw - self._initial_jaw_yaw)
                if self._can_max_yaw_offset is not None:
                    delta += float(
                        np.clip(-delta, -self._can_max_yaw_offset, self._can_max_yaw_offset)
                    )
                rotation = _so3_exp(np.array((0.0, 0.0, delta))) @ rotation
            return p + _CONTACT_WORLD_OFFSETS[self.task_id], rotation
        if self.task_id == "post_impact_grasp":
            yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
            delta = _CaptureOracle._axis_wrap(yaw - self._initial_jaw_yaw)
            return (
                p + r @ np.asarray(_REPLANNING_CONTACT_OFFSET),
                _so3_exp(np.array((0.0, 0.0, delta))) @ rotation,
            )
        return p + _CONTACT_WORLD_OFFSETS[self.task_id], rotation

    def to_dict(self) -> dict[str, object]:
        """Serialize the reset choice so raw contact poses can be scored without an env."""
        return {
            "task_id": self.task_id,
            "target_body_id": self.target_body_id,
            "target_body_name": self.target_body_name,
            "contact_kind": self.contact_kind,
            "initial_position_world_m": self.initial_position_world_m.tolist(),
            "initial_rotation_world": self.initial_rotation_world.tolist(),
            "world_from_robot_base_rotation": self.world_from_robot_base_rotation.tolist(),
            "local_point": self._local_point.tolist(),
            "object_to_eef": self._object_to_eef.tolist(),
            "object_to_eef_rotation": self._object_to_eef_rotation.tolist(),
            "initial_target_yaw": self._initial_target_yaw,
            "initial_jaw_yaw": self._initial_jaw_yaw,
            "axis_offset": self._axis_offset,
            "slider_capture_y": self._slider_capture_y,
            "can_max_yaw_offset": self._can_max_yaw_offset,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> _ContactReference:
        result = cls.__new__(cls)
        expected = {
            "task_id",
            "target_body_id",
            "target_body_name",
            "contact_kind",
            "initial_position_world_m",
            "initial_rotation_world",
            "world_from_robot_base_rotation",
            "local_point",
            "object_to_eef",
            "object_to_eef_rotation",
            "initial_target_yaw",
            "initial_jaw_yaw",
            "axis_offset",
            "slider_capture_y",
            "can_max_yaw_offset",
        }
        if set(values) != expected:
            raise ValueError("contact reference fields do not match the serialized contract")
        task_id = values["task_id"]
        if not isinstance(task_id, str):
            raise ValueError("contact reference requires a task ID")
        task = load_task(task_id)
        if task_id == "wire_passage":
            raise ValueError("serialized contact reference requires a physical-contact task")
        result._grasp_motion = task.grasp_motion
        for key in ("task_id", "target_body_name", "contact_kind"):
            if not isinstance(values[key], str) or not values[key]:
                raise ValueError("contact reference identities must be nonempty strings")
            setattr(result, key, values[key])
        body_id = values["target_body_id"]
        if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id <= 0:
            raise ValueError("contact reference target body id must be positive")
        result.target_body_id = body_id
        arrays = {
            "initial_position_world_m": ("initial_position_world_m", (3,)),
            "initial_rotation_world": ("initial_rotation_world", (3, 3)),
            "world_from_robot_base_rotation": ("world_from_robot_base_rotation", (3, 3)),
            "local_point": ("_local_point", (3,)),
            "object_to_eef": ("_object_to_eef", (3,)),
            "object_to_eef_rotation": ("_object_to_eef_rotation", (3, 3)),
        }
        for key, (attribute, shape) in arrays.items():
            value = np.asarray(values[key], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid contact reference array: {key}")
            if shape == (3, 3):
                _so3_log(value)
            setattr(result, attribute, value.copy())
        for key in ("initial_target_yaw", "initial_jaw_yaw", "axis_offset", "slider_capture_y"):
            raw = values[key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                raise ValueError(f"invalid contact reference scalar: {key}")
            setattr(result, "_" + key, float(raw))
        can_yaw = values["can_max_yaw_offset"]
        if can_yaw is not None and (
            isinstance(can_yaw, bool)
            or not isinstance(can_yaw, (float, int))
            or not math.isfinite(can_yaw)
            or can_yaw < 0
        ):
            raise ValueError("can yaw offset must be finite non-negative or null")
        result._can_max_yaw_offset = None if can_yaw is None else float(can_yaw)
        return result

    def geometry_eligible(self, environment: DynamicManipulationEnv) -> bool:
        """Read GT scene gates for a candidate at this reference frame, never policy EEF."""
        task = self.task_id
        placement = task in _PLACEMENT_CONTACT_HEIGHT
        outcome = (
            environment._placement
            if placement
            else environment._capture
            if self._grasp_motion is not None
            else environment._outcome
        )
        if outcome.failure_code not in (None, "timeout"):
            return False
        data = environment.sim.data._data
        p = data.xpos[self.target_body_id]
        r = data.xmat[self.target_body_id].reshape(3, 3)
        free_dynamics = environment._task_spec.family == "free_dynamics_interception"
        exited = free_dynamics and "ramp_exit" in outcome.events
        candidate_p, candidate_r = self.frame(p, r, ramp_exited=exited)
        if placement:
            placement_spec = environment._task_spec.success
            if task == "rotating_receptacle" and not isinstance(
                placement_spec, SE3PlacementSuccessSpec
            ):
                raise ValueError(f"{task} requires its SE3 placement success specification")
            object_r = candidate_r @ self._object_to_eef_rotation.T
            object_p = candidate_p - object_r @ self._object_to_eef
            distance = np.linalg.norm(object_p - p - (0, 0, _PLACEMENT_CONTACT_HEIGHT[task]))
            return bool(
                distance <= (0.05 if task == "rotating_receptacle" else 0.10)
                and (
                    task != "rotating_receptacle"
                    or np.linalg.norm(_so3_log(r.T @ object_r))
                    <= cast(SE3PlacementSuccessSpec, placement_spec).maximum_orientation_error_rad
                )
            )
        if task == "full_wire_passage":
            return True  # Acquisition has contact duration, but no additional geometric gate.
        if np.linalg.norm(candidate_p - p) > 0.10:
            return False
        if free_dynamics:
            if not exited:
                return False
            contacts = environment._scored_contact_facts()
            if not outcome.supported:
                return not bool(contacts[6])
            lower, upper = environment._geometry.capture_region_bounds_world_m
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                environment.sim.model._model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                self.target_body_id,
                velocity,
                0,
            )
            return bool(
                contacts[6]
                and all(lower[i] <= p[i] <= upper[i] for i in (0, 1))
                and np.linalg.norm(velocity[3:5]) >= outcome.success_spec.minimum_contact_speed_m_s
            )
        if task == "post_impact_grasp":
            return bool(
                outcome._completed_impacts == 1
                and not outcome._previous_impact_active
                and outcome.outgoing_validation_time_s is not None
            )
        return True


def _make_contact_reference(environment: DynamicManipulationEnv) -> _ContactReference:
    if environment._episode_tick != 0:
        raise ValueError("contact reference choices must be fixed at scored t0")
    return _ContactReference(environment)


class _StaticContactIK:
    """Bounded pose search in private MjData; failure never proves unreachability."""

    def __init__(
        self,
        model: mujoco.MjModel,
        initial_data: mujoco.MjData,
        joint_names: Sequence[str],
        site_name: str,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = initial_data.qpos
        self.data.mocap_pos[:] = initial_data.mocap_pos
        self.data.mocap_quat[:] = initial_data.mocap_quat
        joint_ids = np.asarray([model.joint(name).id for name in joint_names])
        if not np.all(model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE) or not np.all(
            model.jnt_limited[joint_ids]
        ):
            raise ValueError("static contact IK requires bounded scalar arm hinge joints")
        self.qpos_addresses = model.jnt_qposadr[joint_ids].copy()
        self.dof_addresses = model.jnt_dofadr[joint_ids].copy()
        self.lower, self.upper = model.jnt_range[joint_ids].T.copy()
        self.initial_q = initial_data.qpos[self.qpos_addresses].copy()
        self.site_id = model.site(site_name).id
        self._jac_position = np.zeros((3, model.nv))
        self._jac_rotation = np.zeros((3, model.nv))

    def _forward(
        self, q: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self.data.qpos[self.qpos_addresses] = q
        mujoco.mj_kinematics(self.model, self.data)
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    @staticmethod
    def _right_log_jacobian_inverse(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x, y, z = v
        skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)
        theta = float(np.linalg.norm(v))
        coefficient = (
            1 / 12 + theta**2 / 720
            if theta < 1e-5
            else (1 - 0.5 * theta / math.tan(0.5 * theta)) / theta**2
        )
        return cast(npt.NDArray[np.float64], np.eye(3) + 0.5 * skew + coefficient * skew @ skew)

    def solve(
        self,
        position_world_m: npt.NDArray[np.float64],
        rotation_world: npt.NDArray[np.float64],
        *,
        starts: Sequence[npt.NDArray[np.float64]] | None = None,
        max_iterations: int = 160,
    ) -> dict[str, object]:
        """Search eight fixed initializations by default, at 0.1 mm / 0.001 rad."""
        goal_p = np.asarray(position_world_m, dtype=np.float64)
        goal_r = np.asarray(rotation_world, dtype=np.float64)
        if (
            goal_p.shape != (3,)
            or goal_r.shape != (3, 3)
            or not (np.isfinite(goal_p).all() and np.isfinite(goal_r).all())
        ):
            raise ValueError("static IK requires finite position and rotation")
        if starts is None:
            generator = np.random.default_rng(417)
            starts = [self.initial_q, (self.lower + self.upper) / 2]
            starts += list(generator.uniform(self.lower, self.upper, (6, len(self.initial_q))))
        if not starts or max_iterations <= 0:
            raise ValueError("static IK requires starts and a positive iteration budget")
        best: dict[str, object] | None = None
        best_objective = math.inf
        count = 0
        for start_index, start in enumerate(starts):
            q = np.asarray(start, dtype=np.float64)
            if q.shape != self.initial_q.shape or not np.isfinite(q).all():
                raise ValueError("static IK start must match the finite arm joint vector")
            q = np.clip(q, self.lower, self.upper)
            damping = 1e-2
            for iteration in range(max_iterations):
                count += 1
                position, rotation = self._forward(q)
                dp = goal_p - position
                dr = _so3_log(goal_r @ rotation.T)
                poserr, roterr = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
                error = np.concatenate((dp, 0.1 * dr))
                objective = float(error @ error)
                record: dict[str, object] = {
                    "position_error_m": poserr,
                    "rotation_error_rad": roterr,
                    "q": q.tolist(),
                    "start_index": start_index,
                    "iteration": iteration,
                    "evaluations": count,
                }
                if objective < best_objective:
                    best, best_objective = record, objective
                if poserr <= 1e-4 and roterr <= 1e-3:
                    return {
                        "status": "solution_found",
                        **record,
                        "starts_attempted": start_index + 1,
                    }
                # mj_jacSite needs COM transforms as well as joint kinematics.
                mujoco.mj_comPos(self.model, self.data)
                mujoco.mj_jacSite(
                    self.model, self.data, self._jac_position, self._jac_rotation, self.site_id
                )
                jacobian = np.vstack(
                    (
                        self._jac_position[:, self.dof_addresses],
                        0.1
                        * self._right_log_jacobian_inverse(dr)
                        @ self._jac_rotation[:, self.dof_addresses],
                    )
                )
                accepted = False
                for _ in range(8):
                    hessian = jacobian.T @ jacobian + damping**2 * np.eye(len(q))
                    gradient = jacobian.T @ error
                    free = np.ones(len(q), dtype=bool)
                    dq = np.zeros(len(q))
                    # Clipping an outward step at a joint limit leaves the other
                    # components solving for motion that cannot happen. Resolve
                    # the system with those bound components held at zero.
                    for _active in range(len(q) + 1):
                        dq[:] = 0
                        dq[free] = np.linalg.solve(hessian[np.ix_(free, free)], gradient[free])
                        blocked = free & (
                            ((q <= self.lower + 1e-12) & (dq < 0))
                            | ((q >= self.upper - 1e-12) & (dq > 0))
                        )
                        if not blocked.any():
                            break
                        free[blocked] = False
                    peak = float(np.max(np.abs(dq)))
                    if peak > 0.25:
                        dq *= 0.25 / peak
                    candidate = np.clip(q + dq, self.lower, self.upper)
                    pc, rc = self._forward(candidate)
                    ec = np.concatenate((goal_p - pc, 0.1 * _so3_log(goal_r @ rc.T)))
                    if float(ec @ ec) < objective - 1e-15:
                        q = candidate
                        damping = max(1e-5, damping * 0.5)
                        accepted = True
                        break
                    damping = min(1e4, damping * 4)
                if not accepted:
                    break
        assert best is not None
        return {
            "status": "no_solution_found",
            **best,
            "evaluations": count,
            "starts_attempted": len(starts),
        }


class _CaptureOracle:
    """Causal grasp and conveyor interception policies for the native action interface."""

    def __init__(self, environment: DynamicManipulationEnv) -> None:
        motion = environment._task_spec.grasp_motion
        if motion is None:
            raise ValueError("capture expert requires a grasp task")
        self._motion = motion
        self._surface_driven = motion is GraspMotion.CONVEYOR
        self._so3 = motion is GraspMotion.MATCHED_YAW
        self._contact_frame = self._surface_driven or self._so3
        self._phase = "track"
        self._close_steps = 0
        self._clearance_reached = False
        self._descent_active = False
        self._lift_anchor = np.zeros(3, dtype=np.float64)
        self._lift_object_anchor = np.zeros(3, dtype=np.float64)
        self._lift_jaw_angle = 0.0
        self._initial_yaw: float | None = None
        joint_space_yaws = None
        if self._surface_driven and environment._scenario_spec.action_mode == "J8":
            data = environment.sim.data._data
            address = environment._target_qpos_address
            w, x, y, z = (float(v) for v in data.qpos[address + 3 : address + 7])
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            jaw = np.mean(data.geom_xpos[sorted(environment._right_pad_ids)], axis=0)
            jaw -= np.mean(data.geom_xpos[sorted(environment._left_pad_ids)], axis=0)
            jaw_angle = math.atan2(float(jaw[1]), float(jaw[0]))
            joint_space_yaws = yaw, jaw_angle
        self._axis_offset = _grasp_axis_offset(motion, joint_space_yaws=joint_space_yaws)
        self._point_index = 0
        self._point_selected = False
        extent = _CONTACT_CAPTURE_EXTENTS.get(motion, 0.0)
        self._points = (np.array((extent, 0.0, 0.0)), np.array((-extent, 0.0, 0.0)))
        self._previous_yaw: float | None = None
        self._previous_time: float | None = None
        self._filtered_yaw_rate: float | None = None
        self._previous_wrist: float | None = None
        self._previous_yaw_action: float | None = None
        self._wrist_yaw_sign: int | None = None
        self._seen_events: set[str] = set()
        self._visibility_audit = (
            _OcclusionVisibilityAudit() if environment._task_spec.has_occluder else None
        )
        self._visibility_state: _OcclusionVisibilityState | None = None
        self._visibility_seen_events: set[str] = set()
        self._observe_visibility(environment)

    @staticmethod
    def _axis_wrap(angle: float) -> float:
        return 0.5 * math.atan2(math.sin(2.0 * angle), math.cos(2.0 * angle))

    @staticmethod
    def _point_world(
        position: npt.NDArray[np.float64], yaw: float, offset: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        c, s = math.cos(yaw), math.sin(yaw)
        return position + np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))) @ offset

    def _yaw_rate(self, yaw: float, time_s: float, instantaneous: float) -> float:
        rate = instantaneous
        if self._so3:
            if self._previous_yaw is not None and self._previous_time is not None:
                dt = time_s - self._previous_time
                if dt > 0.0:
                    delta = yaw - self._previous_yaw
                    rate = math.atan2(math.sin(delta), math.cos(delta)) / dt
            rate = float(np.clip(rate, -1.5, 1.5))
            if self._filtered_yaw_rate is not None:
                rate = 0.5 * rate + 0.5 * self._filtered_yaw_rate
            self._filtered_yaw_rate = rate
        self._previous_yaw, self._previous_time = yaw, time_s
        return rate

    def _limit_yaw(self, action: float, wrist: float) -> float:
        if (
            self._wrist_yaw_sign is None
            and self._previous_wrist is not None
            and self._previous_yaw_action is not None
            and abs(self._previous_yaw_action) >= 0.05
        ):
            delta = wrist - self._previous_wrist
            if abs(delta) >= 1e-5:
                self._wrist_yaw_sign = 1 if delta * self._previous_yaw_action > 0 else -1
        self._previous_wrist = wrist
        if self._wrist_yaw_sign is not None and abs(action) >= 1e-6:
            direction = self._wrist_yaw_sign * int(math.copysign(1.0, action))
            upper = 2.8973 - wrist < 0.15 and direction > 0
            lower = wrist + 2.8973 < 0.15 and direction < 0
            if upper or lower:
                action = math.copysign(0.5, (-1 if upper else 1) * self._wrist_yaw_sign)
        self._previous_yaw_action = action
        return action

    def _grasp_command(
        self,
        object_pose: npt.NDArray[np.float64],
        object_twist: npt.NDArray[np.float64],
        eef: npt.NDArray[np.float64],
        jaw_angle: float,
        contacts: tuple[bool, bool],
        *,
        hold_valid: bool,
        events: set[str],
        time_s: float = 0.0,
        wrist_joint_rad: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        position = object_pose[:3]
        w, x, y, z = (float(value) for value in object_pose[3:])
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_rate = self._yaw_rate(yaw, time_s, float(object_twist[5]))
        if self._initial_yaw is None:
            self._initial_yaw = yaw
        if not self._point_selected:
            self._point_index = _select_capture_contact_point(
                position, yaw, self._points, eef, yaw_rate, self._so3
            )
            self._point_selected = True

        new_events = events - self._seen_events
        self._seen_events.update(events)
        if "clearance" in new_events:
            self._clearance_reached = True
            self._lift_anchor[:2] = eef[:2]
        if self._phase == "close":
            if "bilateral_hold" in new_events or hold_valid:
                self._phase = "lift"
                self._lift_anchor = eef.copy()
                self._lift_object_anchor = position.copy()
                self._lift_jaw_angle = jaw_angle
            elif self._motion is GraspMotion.MATCHED_XYZ and self._close_steps >= 8:
                self._phase = "track"
                self._close_steps = 0
                self._descent_active = False
                self._point_index = (self._point_index + 1) % len(self._points)
        elif (
            self._phase == "descend"
            and not self._contact_frame
            and (any(contacts) or "fingerpad_contact" in new_events)
        ):
            self._phase = "close"

        contact_lookahead = 0.0 if self._surface_driven else 0.05
        lookahead = 0.0
        if self._phase == "track":
            lookahead = 0.15 if self._surface_driven else 0.30
            if self._so3:
                fraction = float(np.clip((float(eef[2] - position[2]) - 0.09) / 0.09, 0.0, 1.0))
                lookahead = 0.05 + fraction * 0.25
        elif self._phase == "descend":
            lookahead = contact_lookahead if self._contact_frame else 0.30
        elif self._phase == "close":
            lookahead = contact_lookahead
        planned_yaw = yaw + lookahead * yaw_rate if self._motion.follows_yaw else self._initial_yaw
        predicted = self._point_world(
            position + lookahead * object_twist[:3], planned_yaw, self._points[self._point_index]
        )
        horizontal_error = float(np.linalg.norm(predicted[:2] - eef[:2]))
        axis_error = abs(self._axis_wrap(planned_yaw + self._axis_offset - jaw_angle))
        close_tolerance = 0.018 if self._surface_driven else 0.020 if self._so3 else 0.025
        vertical_tolerance = 0.018 if self._surface_driven else 0.020
        if self._phase == "track":
            threshold = 0.020 if self._so3 else 0.025 if self._surface_driven else 0.040
            region_open = not self._surface_driven or position[1] >= -0.08
            staged = not self._so3 or (
                axis_error <= 0.35 and abs(float(predicted[2] + 0.09 - eef[2])) < 0.005
            )
            if region_open and horizontal_error < threshold and staged:
                self._phase = "descend"
                if self._contact_frame:
                    lookahead = contact_lookahead
                    planned_yaw = yaw + lookahead * yaw_rate if self._so3 else self._initial_yaw
                    predicted = self._point_world(
                        position + lookahead * object_twist[:3],
                        planned_yaw,
                        self._points[self._point_index],
                    )
                    horizontal_error = float(np.linalg.norm(predicted[:2] - eef[:2]))
                    axis_error = abs(self._axis_wrap(planned_yaw + self._axis_offset - jaw_angle))
                    self._descent_active = self._so3

        descent_brake = False
        if self._phase == "track":
            target, gripper = predicted + (0.0, 0.0, 0.09), -1.0
        elif self._phase == "descend":
            grasp_target = predicted + _GRASP_CONTACT_OFFSET
            descent_brake = self._so3 and eef[2] - grasp_target[2] <= 0.09
            close_aligned = horizontal_error < close_tolerance
            speed = float(np.linalg.norm(object_twist[:2]))
            if self._surface_driven and speed > np.finfo(np.float64).eps:
                error = predicted[:2] - eef[:2]
                motion_axis = object_twist[:2] / speed
                along = float(np.dot(error, motion_axis))
                close_aligned = bool(
                    np.linalg.norm(error - along * motion_axis) < 0.018 and abs(along) < 0.022
                )
            close_aligned = close_aligned and (not self._so3 or axis_error <= 0.35)
            target = grasp_target
            if self._contact_frame:
                contact_seen = any(contacts) or "fingerpad_contact" in new_events
                if contact_seen and not close_aligned:
                    self._descent_active = False
                if self._descent_active:
                    fraction = float(np.clip((eef[2] - grasp_target[2]) / 0.085, 0.0, 1.0))
                    descend_aligned = horizontal_error < (
                        0.020 + fraction * 0.015
                    ) and axis_error <= (0.35 + fraction * 0.15)
                    if descent_brake and not contact_seen and not descend_aligned:
                        target = grasp_target.copy()
                        target[2] = eef[2]
                else:
                    target = predicted + (0.0, 0.0, 0.09)
                    aligned = horizontal_error < (0.035 if self._so3 else 0.018) and (
                        not self._so3 or axis_error <= 0.50
                    )
                    if (
                        not contact_seen
                        and aligned
                        and abs(float(target[2] - eef[2])) < (0.005 if self._so3 else 0.018)
                    ):
                        self._descent_active = True
                        target = grasp_target
            gripper = -1.0
            if (
                (not self._contact_frame or self._descent_active)
                and close_aligned
                and abs(float(grasp_target[2] - eef[2])) < vertical_tolerance
            ):
                self._phase = "close"
                self._descent_active = False
        elif self._phase == "close":
            target, gripper = predicted + _GRASP_CONTACT_OFFSET, 1.0
            self._close_steps += 1
        else:
            if self._phase != "lift":
                raise RuntimeError("invalid capture phase")
            target = self._lift_anchor + (0.0, 0.0, 0.14 if self._surface_driven else 0.08)
            if not self._clearance_reached:
                target[:2] += position[:2] - self._lift_object_anchor[:2]
                target[:2] += contact_lookahead * object_twist[:2]
            gripper = 1.0

        translation = np.clip((target - eef) / 0.05, -1.0, 1.0)
        if self._phase == "descend" and descent_brake:
            translation[2] = max(translation[2], -0.60)
        if self._phase == "lift" and not self._surface_driven:
            translation[2] = np.clip(translation[2], -0.30, 0.30)
        desired_axis = (
            self._lift_jaw_angle if self._phase == "lift" else planned_yaw + self._axis_offset
        )
        yaw_action = float(np.clip(self._axis_wrap(desired_axis - jaw_angle) / 0.5, -1.0, 1.0))
        if self._so3:
            yaw_action = self._limit_yaw(yaw_action, wrist_joint_rad)
        return np.concatenate((translation * 0.05, (0.0, 0.0, yaw_action * 0.5, gripper)))

    def action(self, environment: DynamicManipulationEnv) -> BenchmarkAction:
        model, data = environment.sim.model._model, environment.sim.data._data
        robot = environment.robots[0]
        address = environment._target_qpos_address
        pose = data.qpos[address : address + 7]
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, environment._target_body_id, velocity, 0
        )
        twist = np.concatenate((velocity[3:], velocity[:3]))
        site = robot.eef_site_id["right"]
        jaw = np.mean(data.geom_xpos[sorted(environment._right_pad_ids)], axis=0)
        jaw -= np.mean(data.geom_xpos[sorted(environment._left_pad_ids)], axis=0)
        left, right, _ = _fingerpad_contacts(
            model,
            data,
            environment._target_geom_ids,
            environment._left_pad_ids,
            environment._right_pad_ids,
        )
        command = self._grasp_command(
            pose,
            twist,
            data.site_xpos[site],
            math.atan2(float(jaw[1]), float(jaw[0])),
            (left, right),
            hold_valid=environment._capture.post_hold_contact_valid,
            events=set(environment._capture.events),
            time_s=float(data.time),
            wrist_joint_rad=float(data.qpos[robot._ref_joint_pos_indexes[6]]),
        )
        mode = environment._scenario_spec.action_mode
        if mode == "E7":
            base = int(model.body(environment._robot_base_body_name).id)
            rotation = data.xmat[base].reshape(3, 3)
            scale = np.asarray((0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0))
            command = (command / scale).astype(np.float32).astype(np.float64) * scale
            motion = np.concatenate((rotation.T @ command[:3], rotation.T @ command[3:6]))
        else:
            motion = _capture_joint_motion(environment, command[:3], command[3:6], np.zeros(6))
        values = np.ascontiguousarray(
            np.concatenate((motion, command[-1:])), dtype=np.float64 if mode == "E7" else np.float32
        ).astype(np.float64)
        values.setflags(write=False)
        return BenchmarkAction(mode, "delta", values)

    def _observe_visibility(self, environment: DynamicManipulationEnv) -> None:
        if self._visibility_audit is None or (
            self._visibility_state is not None and self._visibility_state.complete
        ):
            return
        visibility = _target_visibility(
            environment.sim,
            target_body_id=environment._target_body_id,
            occluder_body_id=environment._canopy_body_id,
        )
        if visibility.silhouette_pixels == 0:
            raise ScenarioRejected(
                "visibility",
                "target_silhouette_empty",
                task_id=environment._task_spec.task_id,
                terminal_time_s=float(environment.sim.data.time),
                events=environment._capture.events,
                visibility_state=self._visibility_state,
            )
        event_names = set(environment._capture.events)
        new_events = event_names - self._visibility_seen_events
        self._visibility_seen_events = event_names
        address = environment._target_qpos_address
        self._visibility_state = self._visibility_audit.observe(
            sensor_time_s=float(environment.sim.data.time),
            target_visible_ratio=visibility.ratio,
            target_position_world_m=cast(
                tuple[float, float, float],
                tuple(float(value) for value in environment.sim.data.qpos[address : address + 3]),
            ),
            fingerpad_contact_event="fingerpad_contact" in new_events,
            bilateral_hold_event="bilateral_hold" in new_events,
        )

    def _require_valid_visibility(
        self, environment: DynamicManipulationEnv, *, terminal: bool
    ) -> None:
        state = self._visibility_state
        if state is None:
            return
        reason = None
        if state.failure_facts.initial_visibility_lost:
            reason = next(
                code for code in state.failure_codes if code.startswith("initial_visible_")
            )
        elif terminal and not state.complete:
            reason = state.failure_codes[0]
        if reason is not None:
            raise ScenarioRejected(
                "visibility",
                reason,
                task_id=environment._task_spec.task_id,
                terminal_time_s=float(environment.sim.data.time),
                events=environment._capture.events,
                visibility_state=state,
            )


def _mujoco_twist_to_spatial(
    twist_angular_linear: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if twist_angular_linear.shape != (6,) or not np.isfinite(twist_angular_linear).all():
        raise ValueError("MuJoCo object twist must be one finite angular-linear vector")
    return np.ascontiguousarray(
        np.concatenate((twist_angular_linear[3:], twist_angular_linear[:3])),
        dtype=np.float64,
    )


def _capture_joint_motion(
    environment: DynamicManipulationEnv,
    position_error: npt.NDArray[np.float64],
    rotation_error: npt.NDArray[np.float64],
    target_twist: npt.NDArray[np.float64],
    *,
    joint_position: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    model, data = environment.sim.model._model, environment.sim.data._data
    robot = environment.robots[0]
    controller = robot.part_controllers["right"]
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, robot.eef_site_id["right"])
    indices = robot._ref_joint_vel_indexes
    jacobian = np.vstack((jacobian_position[:, indices], jacobian_rotation[:, indices]))
    error = np.concatenate((position_error, rotation_error))
    inverse = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.02**2 * np.eye(6), np.eye(6))
    motion = 3.0 * (inverse @ error) + (controller.kd / controller.kp) * (inverse @ target_twist)
    if environment._task_spec.task_id in ("post_impact_grasp", "can_capture"):
        if joint_position is None:
            joint_position = data.qpos[robot._ref_joint_pos_indexes]
        joint_ids = [int(model.joint(name).id) for name in robot.robot_model.joints]
        ranges = model.jnt_range[joint_ids]
        lower = np.maximum(-0.049, ranges[:, 0] - joint_position)
        upper = np.minimum(0.049, ranges[:, 1] - joint_position)
        if np.all(motion >= lower) and np.all(motion <= upper):
            return cast(npt.NDArray[np.float64], motion)
        # A seven-joint arm has one null direction for a six-dimensional task.
        # Maximize a common task scale using that redundant joint motion, while
        # preserving the requested Cartesian direction and both native bounds.
        _, _, right = np.linalg.svd(jacobian, full_matrices=True)
        null = right[-1]
        constraints = np.vstack(
            (
                np.column_stack((motion, null)),
                np.column_stack((-motion, -null)),
                np.asarray(((1.0, 0.0), (-1.0, 0.0))),
            )
        )
        limits = np.concatenate((upper, -lower, (1.0, 0.0)))
        best: npt.NDArray[np.float64] | None = None
        for first in range(len(limits)):
            for second in range(first + 1, len(limits)):
                matrix = constraints[[first, second]]
                if abs(float(np.linalg.det(matrix))) < 1e-12:
                    continue
                candidate = np.linalg.solve(matrix, limits[[first, second]])
                if np.any(constraints @ candidate > limits + 1e-10):
                    continue
                if best is None or (candidate[0], -abs(candidate[1])) > (best[0], -abs(best[1])):
                    best = candidate
        if best is None:
            raise RuntimeError("J8 direction has no feasible native joint displacement")
        return cast(npt.NDArray[np.float64], best[0] * motion + best[1] * null)
    if np.max(np.abs(motion)) <= 0.049:
        return cast(npt.NDArray[np.float64], motion)
    if environment._task_spec.task_id in ("slider_capture", "ball_capture_on_table"):
        # Keep the level grasp correction pointing in its requested direction.
        motion *= 0.049 / np.max(np.abs(motion))
        return cast(npt.NDArray[np.float64], motion)
    weighted_jacobian = np.asarray((1.0, 1.0, 1.0, 0.1, 0.1, 0.1))[:, None] * jacobian
    hessian = weighted_jacobian.T @ weighted_jacobian + 0.02**2 * np.eye(7)
    linear = hessian @ motion
    motion = np.clip(motion, -0.049, 0.049)
    active = np.abs(motion) >= 0.049
    for _ in range(64):
        free = ~active
        candidate = motion.copy()
        if np.any(free):
            candidate[free] = np.linalg.solve(
                hessian[np.ix_(free, free)],
                linear[free] - hessian[np.ix_(free, active)] @ motion[active],
            )
        direction = candidate - motion
        step_fraction = 1.0
        bound_index = -1
        for index in np.flatnonzero(free):
            if candidate[index] > 0.049:
                fraction = (0.049 - motion[index]) / direction[index]
            elif candidate[index] < -0.049:
                fraction = (-0.049 - motion[index]) / direction[index]
            else:
                continue
            if fraction < step_fraction:
                step_fraction = float(fraction)
                bound_index = int(index)
        if bound_index >= 0:
            motion += step_fraction * direction
            motion[bound_index] = float(np.copysign(0.049, direction[bound_index]))
            active[bound_index] = True
            continue
        motion = candidate
        gradient = hessian @ motion - linear
        violation = np.where(active, np.sign(motion) * gradient, -np.inf)
        release = int(np.argmax(violation))
        if violation[release] <= 1e-12:
            return cast(npt.NDArray[np.float64], motion)
        active[release] = False
    raise RuntimeError("bounded J8 Cartesian solve did not converge")


class _ReplanningOracle:
    """The post-impact controller's causal SE(3) intercept, contact recovery, and lift planner."""

    def __init__(self, environment: DynamicManipulationEnv) -> None:
        self._joint_control = environment._scenario_spec.action_mode == "J8"
        self._joint_descending = False
        self._joint_capture_started = False
        self._phase = "pre_event"
        self._close_target: npt.NDArray[np.float64] | None = None
        self._lift_anchor: npt.NDArray[np.float64] | None = None
        self._minimum_close_lookahead = False
        self._contact_recovery = False
        self._contact_start_s: float | None = None
        self._maximum_downstream_speed = 0.0
        self._reopen_until_s: float | None = None
        self._reintercept_start_s: float | None = None
        self._seen_events: set[str] = set()
        self._planned_time_s = 0.05
        data = environment.sim.data._data
        site = environment.robots[0].eef_site_id["right"]
        self._joint_level_rotation = data.site_xmat[site].reshape(3, 3).copy()
        left = np.mean(data.geom_xpos[sorted(environment._left_pad_ids)], axis=0)
        right = np.mean(data.geom_xpos[sorted(environment._right_pad_ids)], axis=0)
        closing_axis = right - left
        self._joint_level_heading = math.atan2(float(closing_axis[1]), float(closing_axis[0]))

    @staticmethod
    def _yaw_rotation(angle: float) -> npt.NDArray[np.float64]:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))

    @staticmethod
    def _axis_delta(current: npt.NDArray[np.float64], target: npt.NDArray[np.float64]) -> float:
        difference = math.atan2(float(target[1]), float(target[0])) - math.atan2(
            float(current[1]), float(current[0])
        )
        angle = math.atan2(math.sin(difference), math.cos(difference))
        if angle > math.pi / 2:
            angle -= math.pi
        elif angle < -math.pi / 2:
            angle += math.pi
        return angle

    @staticmethod
    def _rotation_vector(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
        axis = np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            )
        )
        norm = float(np.linalg.norm(axis))
        return np.zeros(3) if angle <= 1e-9 or norm <= 1e-9 else angle * axis / norm

    def _prediction(self, time_s: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        rotation = self._yaw_rotation(float(self._twist[2]) * time_s) @ self._object_rotation
        position = self._object_position + self._twist[3:] * time_s
        if self._joint_control:
            # The target is supported before capture; contact-induced vertical
            # velocity is not a constant-height intercept trajectory.
            position[2] = self._object_position[2]
        return position + rotation @ np.asarray(_REPLANNING_CONTACT_OFFSET), rotation[:, 0]

    def _hover_height(self) -> float:
        return 0.025 if self._joint_control or self._reintercept_start_s is not None else 0.075

    def _predicted_grasp(
        self,
        forced_time_s: float | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if forced_time_s is not None:
            self._planned_time_s = forced_time_s
        elif self._phase == "close":
            self._planned_time_s = 0.05 if self._minimum_close_lookahead else 0.10
        else:
            deadline = 0.90
            downstream_speed = float(self._twist[4])
            if downstream_speed > 1e-6:
                deadline = min(
                    deadline, max(0.05, (0.34 - self._object_position[1]) / downstream_speed - 0.15)
                )
            candidates = np.arange(0.05, deadline + 0.025, 0.05)
            if not len(candidates):
                raise RuntimeError("T5 intercept horizon contains no candidate")
            self._planned_time_s = float(candidates[-1])
            for candidate in candidates:
                point, axis = self._prediction(float(candidate))
                hover = point + (0.0, 0.0, self._hover_height())
                translation_time = float(np.linalg.norm(hover - self._eef_position)) / 0.45
                rotation_time = abs(self._axis_delta(self._closing_axis, axis)) / 1.50
                if max(translation_time, rotation_time) + 0.12 <= float(candidate):
                    self._planned_time_s = float(candidate)
                    break
        return self._prediction(self._planned_time_s)

    def _recover_single_contact(self) -> None:
        if self._event_observed:
            self._maximum_downstream_speed = max(
                self._maximum_downstream_speed, float(self._twist[4])
            )
        if self._phase != "close":
            return
        if (
            "fingerpad_contact" in self._new_events or self._contact_count == 1
        ) and not self._contact_recovery:
            self._contact_start_s = (
                self._event_times["fingerpad_contact"]
                if "fingerpad_contact" in self._new_events
                else self._time_s
            )
            self._contact_recovery = True
        if not self._contact_recovery:
            return
        if self._contact_start_s is None:
            raise RuntimeError("T5 contact recovery has no start time")
        elapsed = max(0.0, self._time_s - self._contact_start_s)
        remaining = math.inf
        if self._maximum_downstream_speed > 1e-6:
            remaining = max(0.0, (0.34 - self._object_position[1]) / self._maximum_downstream_speed)
        if elapsed + 1e-12 >= 0.10 or remaining <= 0.25:
            self._phase = "track"
            self._close_target = None
            self._contact_recovery = False
            self._contact_start_s = None
            self._reintercept_start_s = self._time_s
            self._reopen_until_s = self._time_s + 0.10

    def _command(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        if self._phase == "pre_event" and self._event_observed:
            self._phase = "track"
        if self._phase in ("track", "close") and (
            "bilateral_hold" in self._new_events or self._hold_valid
        ):
            self._phase = "lift"
            self._lift_anchor = self._eef_position.copy()
        self._recover_single_contact()
        desired_rotation = self._eef_rotation
        if self._phase == "pre_event":
            target = np.asarray((0.0, 0.085, 0.845 if self._joint_control else 0.925))
            if self._joint_control:
                # Align to the currently observed incoming object frame while
                # waiting. The struck branch is not used by these actions.
                heading = math.atan2(float(self._closing_axis[1]), float(self._closing_axis[0]))
                object_heading = math.atan2(
                    float(self._object_rotation[1, 0]), float(self._object_rotation[0, 0])
                )
                yaw = math.atan2(
                    math.sin(object_heading - heading), math.cos(object_heading - heading)
                )
                desired_rotation = self._yaw_rotation(yaw) @ self._eef_rotation
            gripper = -1.0
        elif self._phase == "lift":
            if self._lift_anchor is None:
                raise RuntimeError("T5 lift has no captured EEF anchor")
            target = self._lift_anchor + (0.0, 0.0, 0.12)
            gripper = 1.0
        else:
            close_tracking = self._phase == "close" and self._contact_count == 0
            forced_time = (
                (0.05 if self._minimum_close_lookahead else 0.25) if close_tracking else None
            )
            grasp, grasp_axis = self._predicted_grasp(forced_time)
            yaw = self._axis_delta(self._closing_axis, grasp_axis)
            if self._contact_recovery or self._contact_count == 1:
                yaw = float(np.clip(yaw * 0.50, -0.04, 0.04))
            desired_rotation = self._yaw_rotation(yaw) @ self._eef_rotation
            current_axis = self._object_rotation[:, 0]
            current_grasp = self._object_position + self._object_rotation @ np.asarray(
                (0.0, -0.025, 0.0)
            )
            if self._phase == "track":
                target = grasp + (0.0, 0.0, self._hover_height())
                fraction = min(1.0, 0.025 / max(self._planned_time_s, 1e-9))
                gate_point = current_grasp + fraction * (grasp - current_grasp)
                gate_axis = (
                    self._yaw_rotation(fraction * self._axis_delta(current_axis, grasp_axis))
                    @ current_axis
                )
                distance = float(np.linalg.norm(gate_point[:2] - self._eef_position[:2]))
                axis_error = abs(self._axis_delta(self._closing_axis, gate_axis))
                close_distance = (
                    0.045
                    if self._reintercept_start_s is not None
                    and self._time_s - self._reintercept_start_s + 1e-12 >= 0.50
                    else 0.035
                )
                close_allowed = self._reopen_until_s is None or (
                    self._time_s + 1e-12 >= self._reopen_until_s
                )
                aligned = distance <= close_distance and axis_error <= 0.30 and close_allowed
                if self._joint_control and aligned:
                    self._joint_descending = True
                if self._joint_descending:
                    # Keep the fingers open until the pads surround the key.
                    target = grasp.copy()
                depth_reached = (
                    not self._joint_control
                    or abs(float(current_grasp[2] - self._eef_position[2])) <= 0.008
                )
                if aligned and depth_reached:
                    self._phase = "close"
                    self._joint_descending = False
                    self._close_target = grasp.copy()
                    self._reintercept_start_s = self._reopen_until_s = None
                    self._minimum_close_lookahead = (
                        grasp[1] - self._eef_position[1] < -max(0.0, float(self._twist[4])) * 0.025
                    )
            if self._phase == "close":
                if self._close_target is None:
                    self._close_target = grasp.copy()
                elif self._contact_count == 1 or (
                    self._contact_recovery and self._contact_count < 2
                ):
                    correction = (grasp - self._close_target) * 0.60
                    length = float(np.linalg.norm(correction[:2]))
                    if length > 0.008:
                        correction[:2] *= 0.008 / length
                    correction[2] = 0.0
                    self._close_target += correction
                elif self._contact_count == 0:
                    self._close_target = grasp.copy()
                target = self._close_target
                gripper = 1.0
            else:
                gripper = -1.0
        # This registered planner reverses commands inside the observed endpoint band.
        if gripper < 0.0 and (
            self._gripper_position[0] >= 0.0395 or self._gripper_position[1] <= -0.0395
        ):
            gripper = 1.0
        elif gripper > 0.0 and (
            self._gripper_position[0] <= 0.0005 or self._gripper_position[1] >= -0.0005
        ):
            gripper = -1.0
        translation = target - self._eef_position
        if not self._joint_control:
            translation = np.clip(translation, -0.05, 0.05)
        if self._phase == "lift":
            translation[2] = min(float(translation[2]), 0.04)
        rotation = self._rotation_vector(desired_rotation @ self._eef_rotation.T)
        if not self._joint_control:
            rotation = np.clip(rotation, -0.5, 0.5)
        return translation, rotation, gripper

    @staticmethod
    def _rotation_from_xyzw(quaternion: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        x, y, z, w = (float(value) for value in quaternion)
        return np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )

    def action(self, environment: DynamicManipulationEnv) -> BenchmarkAction:
        model, data = environment.sim.model._model, environment.sim.data._data
        robot = environment.robots[0]
        site = robot.eef_site_id["right"]
        self._eef_position = data.site_xpos[site].copy()
        self._eef_rotation = self._rotation_from_xyzw(mat2quat(data.site_xmat[site].reshape(3, 3)))
        # OnTimeBench exposes free-joint qpos at the control boundary; cached xpos still
        # describes the last pre-integration physics state. Preserve that input.
        address = environment._target_qpos_address
        pose = data.qpos[address : address + 7]
        self._object_position = pose[:3].copy()
        self._object_rotation = self._rotation_from_xyzw(pose[[4, 5, 6, 3]])
        self._twist = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, environment._target_body_id, self._twist, 0
        )
        left_center = np.mean(data.geom_xpos[sorted(environment._left_pad_ids)], axis=0)
        right_center = np.mean(data.geom_xpos[sorted(environment._right_pad_ids)], axis=0)
        self._closing_axis = right_center - left_center
        self._closing_axis /= np.linalg.norm(self._closing_axis)
        left, right, _ = _fingerpad_contacts(
            model,
            data,
            environment._target_geom_ids,
            environment._left_pad_ids,
            environment._right_pad_ids,
        )
        self._contact_count = int(left) + int(right)
        self._gripper_position = data.qpos[robot._ref_gripper_joint_pos_indexes["right"]].copy()
        self._event_observed = environment._event_fired
        self._hold_valid = environment._outcome._post_hold_contact_valid
        self._event_times = dict(environment._outcome.events)
        self._new_events = set(self._event_times) - self._seen_events
        self._seen_events.update(self._event_times)
        self._time_s = float(data.time)
        translation, rotation, gripper = self._command()
        self._joint_capture_started = self._joint_capture_started or self._phase == "close"
        suppress_contact_lift = (
            self._joint_control
            and self._joint_capture_started
            and self._phase in ("track", "close")
            and self._contact_count > 0
        )
        if suppress_contact_lift:
            # A delayed recovery command must not lift an unvalidated grasp.
            translation[2] = min(float(translation[2]), 0.0)
        mode = environment._scenario_spec.action_mode
        if mode == "E7":
            base = int(model.body(robot.robot_model.root_body).id)
            base_rotation = data.xmat[base].reshape(3, 3)
            translation = (translation / 0.05).astype(np.float32).astype(np.float64) * 0.05
            rotation = (rotation / 0.5).astype(np.float32).astype(np.float64) * 0.5
            motion = np.concatenate((base_rotation.T @ translation, base_rotation.T @ rotation))
        else:
            # Preserve the initialized approach orientation while following yaw.
            # A bounded Cartesian solve otherwise makes its roll/pitch drift
            # permanent when each new target starts from the achieved pose.
            heading = math.atan2(float(self._closing_axis[1]), float(self._closing_axis[0]))
            desired_rotation = (
                self._yaw_rotation(heading - self._joint_level_heading + float(rotation[2]))
                @ self._joint_level_rotation
            )
            # J8 bounds the joint displacement after solving the task error;
            # E7 Cartesian execution limits do not bound this internal goal.
            rotation = self._rotation_vector(desired_rotation @ self._eef_rotation.T)
            if self._phase == "track" and self._contact_count == 0:
                # A grasp axis has two equivalent jaw directions. The shortest
                # Cartesian rotation may require a wrist angle outside its
                # native range, so compare both local joint-space targets.
                jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
                mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site)
                indices = robot._ref_joint_vel_indexes
                jacobian = np.vstack((jacobian_position[:, indices], jacobian_rotation[:, indices]))
                inverse = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + 0.02**2 * np.eye(6), np.eye(6)
                )
                joint_ids = [int(model.joint(name).id) for name in robot.robot_model.joints]
                ranges = model.jnt_range[joint_ids]
                joints = data.qpos[robot._ref_joint_pos_indexes]
                yaw = self._axis_delta(self._closing_axis, desired_rotation[:, 0])
                choices = []
                for angle in (yaw, yaw - math.copysign(math.pi, yaw)):
                    goal = (
                        self._yaw_rotation(heading - self._joint_level_heading + angle)
                        @ self._joint_level_rotation
                    )
                    error = self._rotation_vector(goal @ self._eef_rotation.T)
                    joint_goal = joints + inverse @ np.concatenate((translation, error))
                    excess = np.maximum(ranges[:, 0] - joint_goal, joint_goal - ranges[:, 1])
                    choices.append(((max(0.0, float(np.max(excess))), abs(angle)), error))
                rotation = min(choices, key=lambda item: item[0])[1]
            target_twist = np.zeros(6, dtype=np.float64)
            if self._phase in ("track", "close"):
                offset = self._object_rotation @ np.asarray((0.0, -0.025, 0.0))
                target_twist[:3] = self._twist[3:] + np.cross(self._twist[:3], offset)
                # The selected flat approach orientation follows its planned
                # pose; contact-induced object roll/pitch are not its velocity.
                target_twist[2] = 0.0
            if suppress_contact_lift:
                target_twist[2] = min(float(target_twist[2]), 0.0)
            predicted_joint_position = data.qpos[robot._ref_joint_pos_indexes].copy()
            delay_steps = environment._scenario_spec.action_delay_steps
            if delay_steps:
                # The registered J8 position controller is critically damped.
                # Predict its nominal displacement while the already-issued
                # commands run, then solve against the state at application.
                controller = robot.part_controllers["right"]
                period = 1.0 / environment.control_freq
                frequency = np.sqrt(controller.kp)
                decay = np.exp(-frequency * period)
                queue = environment._action_delay_queue
                pending: list[npt.NDArray[np.float32] | npt.NDArray[np.float64]] = (
                    [] if queue is None else [item[1].value[:7] for item in queue._pending]
                )
                zero_pending: list[npt.NDArray[np.float32] | npt.NDArray[np.float64]] = [
                    np.zeros(7)
                ] * (delay_steps - len(pending))
                pending = zero_pending + pending
                velocity = data.qvel[robot._ref_joint_vel_indexes].copy()
                displacement = np.zeros(7, dtype=np.float64)
                for issued in pending:
                    displacement += (
                        1.0 - (1.0 + frequency * period) * decay
                    ) * issued + period * decay * velocity
                    velocity = (
                        frequency**2 * period * decay * issued
                        + (1.0 - frequency * period) * decay * velocity
                    )
                predicted_joint_position += displacement
                jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
                mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site)
                indices = robot._ref_joint_vel_indexes
                translation = translation - jacobian_position[:, indices] @ displacement
                rotation = rotation - jacobian_rotation[:, indices] @ displacement
            motion = _capture_joint_motion(
                environment,
                translation,
                rotation,
                target_twist,
                joint_position=predicted_joint_position,
            )
        values = np.ascontiguousarray(
            np.concatenate((motion, (gripper,))), dtype=np.float64 if mode == "E7" else np.float32
        ).astype(np.float64)
        values.setflags(write=False)
        return BenchmarkAction(mode, "delta", values)


class _FreeDynamicsOracle:
    """Fixed T4 planners inside the existing causal native action path."""

    def __init__(self, environment: DynamicManipulationEnv) -> None:
        self._task_id = environment._task_spec.task_id
        self._seen_events: set[str] = set()
        self._phase = "stage"
        angle = environment._geometry.ramp_angle_rad
        if self._task_id == "can_capture":
            self._can_init()
            self._can_j8_max_yaw_offset: float | None = None
            if environment._scenario_spec.action_mode == "J8":
                self._can_j8_rendezvous = np.asarray(
                    (
                        environment._scenario_spec.object_states[0].position_world_m[0],
                        0.07,
                        environment._geometry.support_surface_z_m
                        + 2.0
                        * float(environment.sim.model.geom_size[environment._target_geom_ids[0], 0])
                        - 0.006,
                    ),
                    dtype=np.float64,
                )
                model, data = environment.sim.model._model, environment.sim.data._data
                left = next(iter(environment._left_pad_ids))
                right = next(iter(environment._right_pad_ids))
                closing = data.geom_xpos[right] - data.geom_xpos[left]
                separation = float(np.linalg.norm(closing))
                closing /= separation
                gap = separation - sum(
                    float(
                        np.abs(data.geom_xmat[index].reshape(3, 3).T @ closing)
                        @ model.geom_size[index]
                    )
                    for index in (left, right)
                )
                radius, half_length = model.geom_size[environment._target_geom_ids[0], :2]
                self._can_j8_max_yaw_offset = math.asin(
                    gap / math.hypot(2.0 * half_length, 2.0 * radius)
                ) - math.atan2(2.0 * radius, 2.0 * half_length)
        elif self._task_id == "slider_capture":
            self._slider_init(angle)
            self._slider_grasp_depth_ready = environment._scenario_spec.action_mode != "J8"
        elif self._task_id == "ball_capture":
            self._sphere_init(angle)
        elif self._task_id == "ball_capture_on_table":
            self._tabletop_init()
        else:
            raise ValueError("T4 oracle requires a declared free-dynamics task")

    def action(self, environment: DynamicManipulationEnv) -> BenchmarkAction:
        model, data = environment.sim.model._model, environment.sim.data._data
        robot = environment.robots[0]
        site = int(robot.eef_site_id["right"])
        eef_pose = np.concatenate(
            (
                data.site_xpos[site],
                mat2quat(data.site_xmat[site].reshape(3, 3)),
            )
        ).astype(np.float64)
        address = environment._target_qpos_address
        object_pose = np.array(data.qpos[address : address + 7], dtype=np.float64)
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, environment._target_body_id, velocity, 0
        )
        object_twist = _mujoco_twist_to_spatial(velocity)
        left, right, _ = _fingerpad_contacts(
            model,
            data,
            tuple(environment._target_geom_ids),
            environment._left_pad_ids,
            environment._right_pad_ids,
        )
        flags = np.asarray((left, right), dtype=np.float64)
        pad_centers = tuple(
            np.mean(data.geom_xpos[np.asarray(tuple(sorted(ids)), dtype=np.int64)], axis=0)
            for ids in (environment._left_pad_ids, environment._right_pad_ids)
        )
        closing_axis = pad_centers[1] - pad_centers[0]
        closing_axis /= np.linalg.norm(closing_axis)
        events = set(environment._outcome.events)
        new_events = events - self._seen_events
        self._seen_events = events
        exited = "ramp_exit" in events
        hold = bool(environment._outcome.post_hold_contact_valid)
        if self._task_id == "can_capture":
            command = self._can_command(
                object_pose_wxyz=object_pose,
                object_twist_world=object_twist,
                eef_pose_xyzw=eef_pose,
                fingerpad_closing_axis_world=closing_axis,
                fingerpad_contact_flags=flags,
                ramp_exit_seen=exited,
                post_hold_contact_valid=hold,
                event_names=new_events,
            )
            self._phase = self._can_phase
        elif self._task_id == "slider_capture":
            if not self._slider_grasp_depth_ready:
                target_geom = next(iter(environment._target_geom_ids))
                target_axis = data.geom_xmat[target_geom].reshape(3, 3)[:, 2]
                target_half_height = model.geom_size[target_geom, 2]
                pads_inside = True
                for pad in environment._left_pad_ids | environment._right_pad_ids:
                    pad_rotation = data.geom_xmat[pad].reshape(3, 3)
                    pad_half_extent = np.abs(target_axis @ pad_rotation) @ model.geom_size[pad]
                    offset = abs(target_axis @ (data.geom_xpos[pad] - data.geom_xpos[target_geom]))
                    pads_inside = pads_inside and offset + pad_half_extent <= target_half_height
                self._slider_grasp_depth_ready = bool(pads_inside)
            command = self._slider_command(
                object_pose_wxyz=object_pose,
                object_twist_world=object_twist,
                eef_pose_xyzw=eef_pose,
                closing_axis_world=closing_axis,
                contact_flags=flags,
                ramp_exit_seen=exited,
                physics_hold_valid=hold,
                event_names=new_events,
            )
            if (
                environment._scenario_spec.action_mode == "J8"
                and self._slider_phase == "stage"
                and self._slider_angle_blend > 0.0
            ):
                # The faster ramp branch needs descent during scored staging:
                # J8 cannot finish the E7-height descent before the object arrives.
                pads = sorted(environment._left_pad_ids | environment._right_pad_ids)
                target = next(iter(environment._target_geom_ids))
                pad_offset = float(np.mean(data.geom_xpos[pads, 2]) - data.site_xpos[site, 2])
                seated_height = (
                    environment._geometry.support_surface_z_m
                    + float(model.geom_size[target, 2])
                    - pad_offset
                )
                height = (
                    1.0 - self._slider_angle_blend
                ) * 0.88 + self._slider_angle_blend * seated_height
                command[2] = np.clip(height - eef_pose[2], -0.05, 0.05)
            self._phase = self._slider_phase
        elif self._task_id == "ball_capture":
            command = self._sphere_command(
                object_pose[:3],
                object_twist[:3],
                eef_pose,
                np.asarray(
                    data.qpos[robot._ref_gripper_joint_pos_indexes["right"]], dtype=np.float64
                ),
                (left, right),
                ramp_exited=exited,
                ramp_coordinate_m=float(environment._ramp_coordinate()[1]),
                hold_valid=hold,
                event_names=new_events,
            )
            self._phase = self._sphere_phase
        else:
            translation, rotation, gripper = self._tabletop_command(
                eef_pose_xyzw=eef_pose,
                object_pose_wxyz=object_pose,
                object_twist_world=object_twist,
                hold_valid=hold,
                events=new_events,
            )
            command = np.concatenate((translation, rotation, [gripper]))
            self._phase = self._tabletop_phase
        mode = environment._scenario_spec.action_mode
        if mode == "E7":
            # The policy quantizes its normalized command before OSC scales
            # it. Preserve that boundary before encoding the public physical action.
            scale = np.asarray((0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0))
            command = (command / scale).astype(np.float32).astype(np.float64) * scale
            base = int(model.body(environment._robot_base_body_name).id)
            rotation_world_from_base = data.xmat[base].reshape(3, 3)
            motion = np.concatenate(
                (
                    rotation_world_from_base.T @ command[:3],
                    rotation_world_from_base.T @ command[3:6],
                )
            )
        else:
            if self._task_id == "can_capture" and "fingerpad_contact" not in events:
                command = command.copy()
                rotation_change = self._can_desired_rotation @ data.site_xmat[site].reshape(3, 3).T
                can_pads = tuple(environment._left_pad_ids) + tuple(environment._right_pad_ids)
                upstream = min(
                    can_pads,
                    key=lambda i: float((rotation_change @ (data.geom_xpos[i] - eef_pose[:3]))[1]),
                )
                finger = int(
                    model.geom(model.geom(upstream).name.replace("_pad_collision", "_collision")).id
                )
                finger_rotation = rotation_change @ data.geom_xmat[finger].reshape(3, 3)
                finger_offset = (
                    rotation_change @ (data.geom_xpos[finger] - eef_pose[:3])
                    + finger_rotation @ model.geom_aabb[finger, :3]
                )
                half_width = float(np.abs(finger_rotation[0]) @ model.geom_aabb[finger, 3:])
                axis = self._can_rotation_wxyz(object_pose[3:])[:, 2]
                target_geom = environment._target_geom_ids[0]
                target_extent_x = float(model.geom_size[target_geom, 1]) * abs(
                    float(axis[0])
                ) + float(model.geom_size[target_geom, 0]) * math.sqrt(
                    max(0.0, 1.0 - float(axis[0]) ** 2)
                )
                side = math.copysign(1.0, float(finger_offset[0]))
                position_target = self._can_j8_rendezvous.copy()
                position_target[0] = (
                    float(object_pose[0])
                    + side * (target_extent_x + half_width)
                    - float(finger_offset[0])
                )
                target_extent_y = float(model.geom_size[target_geom, 1]) * abs(
                    float(axis[1])
                ) + float(model.geom_size[target_geom, 0]) * math.sqrt(
                    max(0.0, 1.0 - float(axis[1]) ** 2)
                )
                finger_face_y = (
                    self._can_j8_rendezvous[1]
                    + float(finger_offset[1])
                    + float(np.abs(finger_rotation[1]) @ model.geom_aabb[finger, 3:])
                )
                controller = robot.part_controllers["right"]
                response_lag = float(np.mean(controller.kd / controller.kp))
                self._can_j8_center_started |= (
                    float(object_pose[1]) + response_lag * float(object_twist[1]) - target_extent_y
                    > finger_face_y
                )
                if self._can_j8_center_started:
                    position_target[0] = float(object_pose[0])
                command[:3] = np.clip((position_target - eef_pose[:3]) / 0.04, -1.0, 1.0) * 0.05
                target_twist = np.zeros(6)
            else:
                target_twist = np.zeros(6) if self._phase in ("stage", "lift") else object_twist
            if self._task_id in ("can_capture", "slider_capture", "ball_capture_on_table"):
                target_twist = target_twist.copy()
                target_twist[3:] = 0.0
            motion = _capture_joint_motion(
                environment,
                command[:3],
                command[3:6],
                target_twist,
            )
        values = np.ascontiguousarray(
            np.concatenate((motion, [command[6]])), dtype=np.float64 if mode == "E7" else np.float32
        ).astype(np.float64)
        values.setflags(write=False)
        return BenchmarkAction(mode, "delta", values)

    def _can_init(self) -> None:
        self._can_phase = "stage"
        self._can_j8_contact_rotation: npt.NDArray[np.float64] | None = None
        self._can_j8_center_started = False
        self._can_initial_closing_angle: float | None = None
        self._can_hold_rotation: npt.NDArray[np.float64] | None = None
        self._can_close_target: npt.NDArray[np.float64] | None = None
        self._can_first_contact_seen = False
        self._can_settle_remaining: int | None = None
        self._can_lift_anchor: npt.NDArray[np.float64] | None = None
        self._can_lift_object_anchor: npt.NDArray[np.float64] | None = None
        self._can_lift_rotation: npt.NDArray[np.float64] | None = None
        self._can_clearance_reached = False

    @staticmethod
    def _can_rotation_wxyz(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        w, x, y, z = (float(value) for value in quaternion)
        return np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _can_rotation_vector(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
        if angle <= 1e-09:
            return np.zeros(3, dtype=np.float64)
        axis = np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ),
            dtype=np.float64,
        )
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-09:
            axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
            axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-09:
            return np.zeros(3, dtype=np.float64)
        return angle * axis / axis_norm

    def _can_command(
        self,
        *,
        object_pose_wxyz: npt.NDArray[np.float64],
        object_twist_world: npt.NDArray[np.float64],
        eef_pose_xyzw: npt.NDArray[np.float64],
        fingerpad_closing_axis_world: npt.NDArray[np.float64],
        fingerpad_contact_flags: npt.NDArray[np.float64],
        ramp_exit_seen: bool,
        post_hold_contact_valid: bool,
        event_names: set[str],
    ) -> npt.NDArray[np.float64]:
        position = object_pose_wxyz[:3]
        velocity = object_twist_world[:3]
        eef_position = eef_pose_xyzw[:3]
        current_rotation = self._can_rotation_wxyz(eef_pose_xyzw[[6, 3, 4, 5]])
        closing_angle = math.atan2(
            float(fingerpad_closing_axis_world[1]), float(fingerpad_closing_axis_world[0])
        )
        if self._can_initial_closing_angle is None:
            self._can_initial_closing_angle = closing_angle
            self._can_hold_rotation = current_rotation.copy()
        initial_closing_angle = self._can_initial_closing_angle
        contact_count = int(np.count_nonzero(fingerpad_contact_flags > 0.5))
        if self._can_phase == "close" and (
            contact_count >= 1 or "fingerpad_contact" in event_names
        ):
            self._can_first_contact_seen = True
        effective_count = max(contact_count, int(self._can_first_contact_seen))
        if "clearance" in event_names:
            self._can_clearance_reached = True
            if self._can_lift_anchor is not None:
                self._can_lift_anchor[:2] = eef_position[:2]
        downstream_speed = max(float(velocity[1]), 0.0)
        capture_ttc = math.inf
        if downstream_speed > 0.0001:
            capture_ttc = max((0.03 - float(position[1])) / downstream_speed, 0.0)
        begin_lift = False
        if self._can_phase == "close":
            hold = "bilateral_hold" in event_names or post_hold_contact_valid
            if self._can_settle_remaining is None and (contact_count >= 2 or hold):
                self._can_settle_remaining = 2
            if self._can_settle_remaining is not None:
                if self._can_settle_remaining > 0:
                    self._can_settle_remaining -= 1
                else:
                    begin_lift = True
        if begin_lift:
            self._can_phase = "lift"
            self._can_lift_anchor = eef_position.copy()
            self._can_lift_object_anchor = position.copy()
            self._can_lift_rotation = current_rotation.copy()
        elif self._can_phase == "stage" and capture_ttc <= 0.35:
            self._can_phase = "preshape"
        if self._can_phase == "preshape" and ramp_exit_seen and (capture_ttc <= 0.2):
            self._can_phase = "intercept"
        plane_ttc = (
            math.inf
            if velocity[1] <= 1e-06
            else max(0.0, (0.03 - float(position[1])) / float(velocity[1]))
        )
        intercept = (
            position.copy() if not math.isfinite(plane_ttc) else position + plane_ttc * velocity
        )
        if math.isfinite(plane_ttc):
            intercept[1] = 0.03
        intercept[0] += 0.12 * velocity[0]
        tracking_target = position.copy()
        tracking_target[:2] += 0.035 * velocity[:2]
        tracking_target[2] += 0.12 * min(float(velocity[2]), 0.0)
        tracking_target[2] += _CONTACT_WORLD_OFFSETS["can_capture"][2]
        committed_target = position.copy()
        committed_target[:2] += 0.035 * velocity[:2]
        committed_target[2] += _CONTACT_WORLD_OFFSETS["can_capture"][2]
        if self._can_phase == "stage":
            target = np.asarray((0.0, 0.07, 0.884), dtype=np.float64)
            gripper = -1.0
        elif self._can_phase == "preshape":
            target = intercept.copy()
            target[2] += (
                0.06
                if not math.isfinite(plane_ttc)
                else 0.045 + (0.06 - 0.045) * float(np.clip(plane_ttc / 0.35, 0.0, 1.0))
            )
            gripper = -1.0
        elif self._can_phase == "intercept":
            target = tracking_target
            gripper = -1.0
            if contact_count > 0 or (
                np.linalg.norm(target - eef_position) <= 0.05
                and np.linalg.norm(velocity[:2]) <= 0.4
            ):
                self._can_phase = "close"
                self._can_close_target = committed_target.copy()
                target = self._can_close_target
                gripper = 1.0
        elif self._can_phase == "close":
            if self._can_close_target is None:
                raise RuntimeError("can close target is uninitialized")
            if ramp_exit_seen:
                if effective_count == 0:
                    self._can_close_target = tracking_target.copy()
                elif effective_count == 1:
                    correction = (committed_target - self._can_close_target) * 0.65
                    norm = float(np.linalg.norm(correction))
                    if norm > 0.006:
                        correction *= 0.006 / norm
                    self._can_close_target += correction
            target = self._can_close_target
            gripper = 1.0
        else:
            if self._can_lift_anchor is None or self._can_lift_object_anchor is None:
                raise RuntimeError("can lift anchors are uninitialized")
            target = self._can_lift_anchor + np.asarray((0.0, 0.0, 0.16), dtype=np.float64)
            if not self._can_clearance_reached:
                target[:2] += position[:2] - self._can_lift_object_anchor[:2]
            gripper = 1.0
        translation = np.clip(
            (target - eef_position) / (0.05 if self._can_phase == "lift" else 0.04), -1.0, 1.0
        )
        if self._can_phase == "lift":
            translation[2] = min(float(translation[2]), 0.35)
        if self._can_hold_rotation is None:
            raise RuntimeError("can orientation anchor is uninitialized")
        if self._can_j8_max_yaw_offset is not None and "fingerpad_contact" in event_names:
            self._can_j8_contact_rotation = current_rotation.copy()
        if self._can_phase == "lift" and self._can_lift_rotation is not None:
            desired_rotation = self._can_lift_rotation
        elif self._can_j8_contact_rotation is not None:
            desired_rotation = self._can_j8_contact_rotation
        else:
            cylinder_axis = self._can_rotation_wxyz(object_pose_wxyz[3:])[:, 2]
            grasp_axis = np.asarray((-cylinder_axis[1], cylinder_axis[0], 0.0))
            direction_reference = (
                closing_angle if self._can_j8_max_yaw_offset is not None else initial_closing_angle
            )
            desired_angle = initial_closing_angle
            if np.linalg.norm(grasp_axis[:2]) > 1e-08:
                angle = math.atan2(float(grasp_axis[1]), float(grasp_axis[0]))
                desired_angle = min(
                    (angle, angle + math.pi),
                    key=lambda value: abs(
                        math.atan2(
                            math.sin(value - direction_reference),
                            math.cos(value - direction_reference),
                        )
                    ),
                )
            if self._can_j8_max_yaw_offset is not None:
                toward_initial = math.atan2(
                    math.sin(initial_closing_angle - desired_angle),
                    math.cos(initial_closing_angle - desired_angle),
                )
                desired_angle += float(
                    np.clip(
                        toward_initial, -self._can_j8_max_yaw_offset, self._can_j8_max_yaw_offset
                    )
                )
            yaw_delta = math.atan2(
                math.sin(desired_angle - initial_closing_angle),
                math.cos(desired_angle - initial_closing_angle),
            )
            cosine, sine = (math.cos(yaw_delta), math.sin(yaw_delta))
            desired_rotation = (
                np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
                @ self._can_hold_rotation
            )
        self._can_desired_rotation = desired_rotation.copy()
        rotation = np.clip(
            self._can_rotation_vector(desired_rotation @ current_rotation.T) / 0.35, -1.0, 1.0
        )
        return np.concatenate((translation * 0.05, rotation * 0.5, [gripper]))

    def _slider_init(self, ramp_angle_rad: float) -> None:
        angle_deg = math.degrees(ramp_angle_rad)
        fraction = float(np.clip((angle_deg - 10.0) / 2.0, 0.0, 1.0))
        self._slider_angle_blend = fraction * fraction * (3.0 - 2.0 * fraction)
        self._slider_capture_y = _slider_capture_plane(ramp_angle_rad)
        self._slider_close_lead_s = min(0.29 + 0.01 * max(angle_deg - 10.0, 0.0), 0.35)
        self._slider_close_lead_s = min(
            self._slider_close_lead_s + 0.025 * self._slider_angle_blend, 0.35
        )
        self._slider_phase = "stage"
        self._slider_grasp_depth_ready = True
        self._slider_initial_closing_angle: float | None = None
        self._slider_hold_rotation: npt.NDArray[np.float64] | None = None
        self._slider_lift_rotation: npt.NDArray[np.float64] | None = None
        self._slider_lift_anchor: npt.NDArray[np.float64] | None = None
        self._slider_lift_object_anchor: npt.NDArray[np.float64] | None = None
        self._slider_close_target: npt.NDArray[np.float64] | None = None
        self._slider_first_contact_rotation: npt.NDArray[np.float64] | None = None
        self._slider_first_contact_seen = False
        self._slider_clearance_reached = False
        self._slider_static_close_target: npt.NDArray[np.float64] | None = None
        self._slider_previous_precontact_target: npt.NDArray[np.float64] | None = None
        self._slider_precontact_weight = 0.0

    @staticmethod
    def _slider_rotation_from_wxyz(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        w, x, y, z = (float(value) for value in quaternion)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _slider_rotation_vector(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        angle = math.acos(cosine)
        if angle <= 1e-09:
            return np.zeros(3, dtype=np.float64)
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-09:
            axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
            axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-09:
            return np.zeros(3, dtype=np.float64)
        return angle * axis / axis_norm

    @staticmethod
    def _slider_smoothstep(value: float, start: float, end: float) -> float:
        fraction = float(np.clip((value - start) / (end - start), 0.0, 1.0))
        return fraction * fraction * (3.0 - 2.0 * fraction)

    def _slider_precontact_target(
        self,
        static_target: npt.NDArray[np.float64],
        tracking_target: npt.NDArray[np.float64],
        time_to_plane_s: float,
        object_velocity: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64] | None:
        if self._slider_angle_blend <= 0.0:
            return None
        if self._slider_phase != "close":
            self._slider_static_close_target = static_target.copy()
            self._slider_previous_precontact_target = static_target.copy()
        elif self._slider_static_close_target is None:
            self._slider_static_close_target = (
                static_target.copy()
                if self._slider_close_target is None
                else self._slider_close_target.copy()
            )
        static_anchor = self._slider_static_close_target
        if static_anchor is None:
            raise RuntimeError("slider precontact anchor is uninitialized")
        finite_ttc = 0.1 if not math.isfinite(time_to_plane_s) else time_to_plane_s
        self._slider_precontact_weight = (
            self._slider_angle_blend
            * self._slider_smoothstep(finite_ttc, 0.1, 0.0)
            * self._slider_smoothstep(max(float(object_velocity[1]), 0.0), 0.08, 0.8)
            * self._slider_smoothstep(finite_ttc, 0.045, 0.055)
        )
        causal_target = tracking_target.copy()
        causal_target[:2] += 0.15 * object_velocity[:2]
        target = static_anchor + self._slider_precontact_weight * (causal_target - static_anchor)
        target[1] = max(float(target[1]), float(static_anchor[1]))
        previous = self._slider_previous_precontact_target
        if previous is None and self._slider_phase == "close":
            previous = self._slider_close_target
        if previous is None:
            previous = static_anchor
        advance = target[:2] - previous[:2]
        advance_norm = float(np.linalg.norm(advance))
        if advance_norm > 0.025:
            target[:2] = previous[:2] + advance * 0.025 / advance_norm
        self._slider_previous_precontact_target = target.copy()
        return target

    def _slider_command(
        self,
        *,
        object_pose_wxyz: npt.NDArray[np.float64],
        object_twist_world: npt.NDArray[np.float64],
        eef_pose_xyzw: npt.NDArray[np.float64],
        closing_axis_world: npt.NDArray[np.float64],
        contact_flags: npt.NDArray[np.float64],
        ramp_exit_seen: bool,
        physics_hold_valid: bool,
        event_names: set[str],
    ) -> npt.NDArray[np.float64]:
        """Events must contain only facts new since the preceding policy tick."""
        object_pose = np.asarray(object_pose_wxyz, dtype=np.float64)
        object_twist = np.asarray(object_twist_world, dtype=np.float64)
        eef = np.asarray(eef_pose_xyzw, dtype=np.float64)
        current_rotation = self._slider_rotation_from_wxyz(eef[[6, 3, 4, 5]])
        current_closing_angle = math.atan2(
            float(closing_axis_world[1]), float(closing_axis_world[0])
        )
        contact_count = int(np.count_nonzero(contact_flags > 0.5))
        ramp_exit_seen = ramp_exit_seen or "ramp_exit" in event_names
        offset = np.zeros(3, dtype=np.float64)
        contact_seen = (
            self._slider_first_contact_seen
            or contact_count > 0
            or "fingerpad_contact" in event_names
        )
        if self._slider_capture_y <= 0.078 and (not contact_seen):
            offset = self._slider_rotation_from_wxyz(object_pose[3:]) @ _slider_contact_offset(
                self._slider_capture_y, ramp_exit_seen
            )
        if self._slider_initial_closing_angle is None:
            self._slider_initial_closing_angle = current_closing_angle
            self._slider_hold_rotation = current_rotation.copy()
        initial_closing_angle = self._slider_initial_closing_angle
        if self._slider_phase == "close" and (
            contact_count >= 1 or "fingerpad_contact" in event_names
        ):
            self._slider_first_contact_seen = True
        effective_contact_count = max(contact_count, 1 if self._slider_first_contact_seen else 0)
        if (
            self._slider_phase == "close"
            and effective_contact_count == 1
            and (self._slider_first_contact_rotation is None)
        ):
            self._slider_first_contact_rotation = current_rotation.copy()
        if "clearance" in event_names:
            self._slider_clearance_reached = True
            if self._slider_lift_anchor is not None:
                self._slider_lift_anchor[:2] = eef[:2]
        downstream_speed = max(float(object_twist[1]), 0.0)
        supported_ttc = math.inf
        if downstream_speed > 0.0001:
            supported_ttc = max(
                (self._slider_capture_y - float(object_pose[1])) / downstream_speed, 0.0
            )
        if (
            self._slider_phase == "close"
            and self._slider_grasp_depth_ready
            and ("bilateral_hold" in event_names or physics_hold_valid)
        ):
            self._slider_phase = "lift"
            self._slider_lift_anchor = eef[:3].copy()
            self._slider_lift_object_anchor = object_pose[:3].copy()
            self._slider_lift_rotation = current_rotation.copy()
        elif self._slider_phase == "stage" and supported_ttc <= 0.35:
            self._slider_phase = "preshape"
        if self._slider_phase == "preshape" and supported_ttc <= self._slider_close_lead_s:
            self._slider_phase = "intercept"
        time_to_plane = (
            math.inf
            if float(object_twist[1]) <= 1e-06
            else max(0.0, (self._slider_capture_y - float(object_pose[1])) / float(object_twist[1]))
        )
        intercept = object_pose[:3].copy()
        if math.isfinite(time_to_plane):
            intercept = object_pose[:3] + time_to_plane * object_twist[:3]
            intercept[1] = self._slider_capture_y
        tracking_target = object_pose[:3].copy()
        tracking_target[:2] += 0.03 * object_twist[:2]
        tracking_target += offset
        static_target = intercept + offset
        precontact_target = self._slider_precontact_target(
            static_target, tracking_target, time_to_plane, object_twist[:3]
        )
        if self._slider_phase == "close" and self._slider_close_target is None:
            self._slider_close_target = static_target.copy()
        if self._slider_phase == "stage":
            target = np.array([0.0, 0.07, 0.88], dtype=np.float64)
            gripper = -1.0
        elif self._slider_phase == "preshape":
            target = static_target
            gripper = -1.0
        elif self._slider_phase == "intercept":
            tracking_supported_object = precontact_target is not None and ramp_exit_seen
            target = (
                precontact_target
                if precontact_target is not None and ramp_exit_seen
                else static_target
            )
            assert target is not None
            gripper = -1.0
            begin_close = (
                bool(np.linalg.norm(target - eef[:3]) <= 0.035)
                if tracking_supported_object
                else time_to_plane <= self._slider_close_lead_s
                or np.linalg.norm(intercept - eef[:3]) <= 0.035
            )
            if begin_close:
                self._slider_phase = "close"
                gripper = 1.0
                self._slider_close_target = target.copy()
                target = self._slider_close_target
        elif self._slider_phase == "close":
            if self._slider_close_target is None:
                raise RuntimeError("slider close target is uninitialized")
            if ramp_exit_seen:
                if effective_contact_count == 0 and precontact_target is not None:
                    self._slider_close_target = precontact_target.copy()
                elif effective_contact_count == 1:
                    correction = (tracking_target - self._slider_close_target) * 0.5
                    urgency = self._slider_smoothstep(0.2 - float(object_pose[1]), 0.12, 0.08)
                    max_step = 0.008 + urgency * (0.02 - 0.008)
                    norm = float(np.linalg.norm(correction[:2]))
                    if norm > max_step:
                        correction[:2] *= max_step / norm
                    correction[2] = 0.0
                    self._slider_close_target += correction
            target = self._slider_close_target
            gripper = 1.0
        else:
            if (
                self._slider_phase != "lift"
                or self._slider_lift_anchor is None
                or self._slider_lift_object_anchor is None
            ):
                raise RuntimeError("slider lift anchors are uninitialized")
            target = self._slider_lift_anchor + np.array([0.0, 0.0, 0.12], dtype=np.float64)
            if not self._slider_clearance_reached:
                target[:2] += object_pose[:2] - self._slider_lift_object_anchor[:2]
            gripper = 1.0
        translation_scale = 0.05
        if self._slider_phase == "close" and (not self._slider_first_contact_seen):
            translation_scale = 0.05 + self._slider_precontact_weight * (0.03 - 0.05)
        translation = np.clip((target - eef[:3]) / translation_scale, -1.0, 1.0)
        if self._slider_phase == "lift":
            translation[2] = min(float(translation[2]), 0.3)
        if self._slider_hold_rotation is None:
            raise RuntimeError("slider orientation anchor is uninitialized")
        if self._slider_phase == "lift" and self._slider_lift_rotation is not None:
            desired_rotation = self._slider_lift_rotation
        elif self._slider_phase == "close" and self._slider_first_contact_rotation is not None:
            desired_rotation = self._slider_first_contact_rotation
        else:
            object_axis = self._slider_rotation_from_wxyz(object_pose[3:])[:, 0]
            desired_closing = initial_closing_angle
            if np.linalg.norm(object_axis[:2]) > 1e-08:
                yaw = math.atan2(float(object_axis[1]), float(object_axis[0]))
                desired_closing = min(
                    (yaw, yaw + math.pi),
                    key=lambda candidate: abs(
                        math.atan2(
                            math.sin(candidate - initial_closing_angle),
                            math.cos(candidate - initial_closing_angle),
                        )
                    ),
                )
            yaw_delta = math.atan2(
                math.sin(desired_closing - initial_closing_angle),
                math.cos(desired_closing - initial_closing_angle),
            )
            cosine, sine = (math.cos(yaw_delta), math.sin(yaw_delta))
            yaw_rotation = np.array(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            desired_rotation = yaw_rotation @ self._slider_hold_rotation
        rotation = np.clip(
            self._slider_rotation_vector(desired_rotation @ current_rotation.T) / 0.5, -1.0, 1.0
        )
        return np.concatenate((translation * 0.05, rotation * 0.5, [gripper]))

    def _sphere_init(self, ramp_angle_rad: float) -> None:
        self._sphere_phase = "stage"
        staging_y = float(
            np.clip(0.04 + 0.024 * max(math.degrees(ramp_angle_rad) - 5.0, 0.0), 0.04, 0.16)
        )
        self._sphere_staging = np.asarray([0.06, staging_y, 0.961], dtype=np.float64)
        self._sphere_hold_rotation: npt.NDArray[np.float64] | None = None
        self._sphere_lift_rotation: npt.NDArray[np.float64] | None = None
        self._sphere_lift_anchor: npt.NDArray[np.float64] | None = None
        self._sphere_lift_object_anchor: npt.NDArray[np.float64] | None = None
        self._sphere_close_target: npt.NDArray[np.float64] | None = None
        self._sphere_first_contact_seen = False
        self._sphere_clearance_reached = False
        self._sphere_bilateral_verification_count = 0
        self._sphere_aperture_close_started = False

    @staticmethod
    def _sphere_rotation_xyzw(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x, y, z, w = (float(value) for value in quaternion)
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _sphere_rotation_vector(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
        if angle <= 1e-09:
            return np.zeros(3, dtype=np.float64)
        axis = np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-09:
            axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
            axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-09:
            return np.zeros(3, dtype=np.float64)
        return angle * axis / axis_norm

    @staticmethod
    def _sphere_time_to_height(
        position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64], height: float
    ) -> float:
        z_offset = float(position[2] - height)
        if z_offset <= 0.0:
            return 0.0
        discriminant = velocity[2] * velocity[2] + 2.0 * 9.81 * z_offset
        return max(0.0, (float(velocity[2]) + math.sqrt(max(discriminant, 0.0))) / 9.81)

    def _sphere_should_close(
        self,
        position: npt.NDArray[np.float64],
        velocity: npt.NDArray[np.float64],
        aperture_m: float,
    ) -> bool:
        closure_travel = max(aperture_m - 0.042, 0.0)
        dynamic_lead = float(np.clip(0.035 + closure_travel / 0.2, 0.11, 0.22))
        contact_ttc = self._sphere_time_to_height(position, velocity, 0.961 + 0.03)
        return self._sphere_aperture_close_started or contact_ttc <= dynamic_lead

    @staticmethod
    def _sphere_precontact_gripper(aperture_m: float) -> float:
        return 0.0 if aperture_m <= 0.068 else 1.0

    def _sphere_command(
        self,
        object_position: npt.NDArray[np.float64],
        object_linear_velocity: npt.NDArray[np.float64],
        eef_pose_xyzw: npt.NDArray[np.float64],
        gripper_position_m: npt.NDArray[np.float64],
        fingerpad_contacts: tuple[bool, bool],
        *,
        ramp_exited: bool,
        ramp_coordinate_m: float,
        hold_valid: bool,
        event_names: set[str],
    ) -> npt.NDArray[np.float64]:
        eef = eef_pose_xyzw[:3]
        rotation = self._sphere_rotation_xyzw(eef_pose_xyzw[3:])
        if self._sphere_hold_rotation is None:
            self._sphere_hold_rotation = rotation.copy()
        contacts = sum(fingerpad_contacts)
        if self._sphere_phase == "close" and (contacts >= 1 or "fingerpad_contact" in event_names):
            self._sphere_first_contact_seen = True
        effective_contacts = max(contacts, int(self._sphere_first_contact_seen))
        if "clearance" in event_names:
            self._sphere_clearance_reached = True
            if self._sphere_lift_anchor is not None:
                self._sphere_lift_anchor[:2] = eef[:2]
        ramp_exited = ramp_exited or "ramp_exit" in event_names
        downhill_speed = max(float(object_linear_velocity[1]), 0.0)
        preclose = (
            downhill_speed > 0.0001 and max(0.185 - ramp_coordinate_m, 0.0) / downhill_speed <= 0.15
        )
        begin_lift = False
        if self._sphere_phase == "close":
            if contacts >= 2 and (hold_valid or "bilateral_hold" in event_names):
                self._sphere_bilateral_verification_count += 1
            else:
                self._sphere_bilateral_verification_count = 0
            begin_lift = self._sphere_bilateral_verification_count > 1
        if begin_lift:
            self._sphere_phase = "lift"
            self._sphere_lift_anchor = eef.copy()
            self._sphere_lift_object_anchor = object_position.copy()
            self._sphere_lift_rotation = rotation.copy()
        elif self._sphere_phase in ("preshape", "stage") and ramp_exited:
            self._sphere_phase = "intercept"
        elif self._sphere_phase == "stage" and preclose:
            self._sphere_phase = "preshape"
        time_to_plane = self._sphere_time_to_height(object_position, object_linear_velocity, 0.961)
        intercept = (
            object_position
            + time_to_plane * object_linear_velocity
            + 0.5 * time_to_plane * time_to_plane * np.asarray([0.0, 0.0, -9.81], dtype=np.float64)
        )
        offset = np.asarray(_CONTACT_WORLD_OFFSETS["ball_capture"], dtype=np.float64)
        if self._sphere_phase == "close" and self._sphere_close_target is None:
            self._sphere_close_target = intercept + offset
        aperture = abs(float(gripper_position_m[0] - gripper_position_m[1]))
        if self._sphere_phase == "stage":
            target = self._sphere_staging.copy()
            gripper = -1.0
        elif self._sphere_phase == "preshape":
            target = intercept + offset
            if self._sphere_aperture_close_started or self._sphere_should_close(
                object_position, object_linear_velocity, aperture
            ):
                self._sphere_aperture_close_started = True
                gripper = self._sphere_precontact_gripper(aperture)
            else:
                gripper = -1.0
        elif self._sphere_phase == "intercept":
            target = intercept + offset
            gripper = -1.0
            if self._sphere_should_close(object_position, object_linear_velocity, aperture):
                self._sphere_phase = "close"
                gripper = (
                    self._sphere_precontact_gripper(aperture) if effective_contacts == 0 else 1.0
                )
                self._sphere_close_target = target.copy()
                target = self._sphere_close_target
        elif self._sphere_phase == "close":
            if self._sphere_close_target is None:
                raise RuntimeError("sphere close target is uninitialized")
            if effective_contacts == 1:
                desired = object_position + np.asarray([0.0, 0.0, -0.03], dtype=np.float64)
                correction = (desired - eef) * 0.5
                correction_norm = float(np.linalg.norm(correction))
                if correction_norm > 0.006:
                    correction *= 0.006 / correction_norm
                self._sphere_close_target = eef + correction
                target = self._sphere_close_target.copy()
            else:
                target = intercept + offset
            gripper = self._sphere_precontact_gripper(aperture) if effective_contacts == 0 else 1.0
        elif self._sphere_phase == "lift":
            if self._sphere_lift_anchor is None or self._sphere_lift_object_anchor is None:
                raise RuntimeError("sphere lift anchors are uninitialized")
            target = self._sphere_lift_anchor + np.asarray([0.0, 0.0, 0.12], dtype=np.float64)
            if not self._sphere_clearance_reached:
                target[:2] += object_position[:2] - self._sphere_lift_object_anchor[:2]
            gripper = 1.0
        else:
            raise RuntimeError(f"unknown sphere phase {self._sphere_phase!r}")
        translation = np.clip((target - eef) / 0.05, -1.0, 1.0)
        desired_rotation = (
            self._sphere_lift_rotation
            if self._sphere_phase == "lift"
            else self._sphere_hold_rotation
        )
        if desired_rotation is None:
            raise RuntimeError("sphere orientation anchor is uninitialized")
        angular = np.clip(
            self._sphere_rotation_vector(desired_rotation @ rotation.T) / 0.5, -1.0, 1.0
        )
        return np.concatenate((translation * 0.05, angular * 0.5, [gripper]))

    def _tabletop_init(self) -> None:
        self._tabletop_phase = "stage"
        self._tabletop_hold_rotation: npt.NDArray[np.float64] | None = None
        self._tabletop_lift_rotation: npt.NDArray[np.float64] | None = None
        self._tabletop_lift_anchor: npt.NDArray[np.float64] | None = None
        self._tabletop_lift_object_anchor: npt.NDArray[np.float64] | None = None
        self._tabletop_close_target: npt.NDArray[np.float64] | None = None
        self._tabletop_clearance_reached = False

    @staticmethod
    def _tabletop_rotation_xyzw(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x, y, z, w = (float(value) for value in quaternion)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _tabletop_rotation_vector(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        angle = math.acos(cosine)
        if angle <= 1e-09:
            return np.zeros(3, dtype=np.float64)
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-09:
            axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
            axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-09:
            return np.zeros(3, dtype=np.float64)
        return angle * axis / axis_norm

    def _tabletop_command(
        self,
        *,
        eef_pose_xyzw: npt.NDArray[np.float64],
        object_pose_wxyz: npt.NDArray[np.float64],
        object_twist_world: npt.NDArray[np.float64],
        hold_valid: bool,
        events: set[str],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        eef = np.asarray(eef_pose_xyzw, dtype=np.float64)
        pose = np.asarray(object_pose_wxyz, dtype=np.float64)
        twist = np.asarray(object_twist_world, dtype=np.float64)
        current_rotation = self._tabletop_rotation_xyzw(eef[3:])
        if self._tabletop_hold_rotation is None:
            self._tabletop_hold_rotation = current_rotation.copy()
        if "clearance" in events:
            self._tabletop_clearance_reached = True
            if self._tabletop_lift_anchor is not None:
                self._tabletop_lift_anchor[:2] = eef[:2]
        speed = max(float(twist[1]), 0.0)
        supported_ttc = math.inf
        if speed > 0.0001:
            supported_ttc = max((0.07 - float(pose[1])) / speed, 0.0)
        if self._tabletop_phase == "close" and ("bilateral_hold" in events or hold_valid):
            self._tabletop_phase = "lift"
            self._tabletop_lift_anchor = eef[:3].copy()
            self._tabletop_lift_object_anchor = pose[:3].copy()
            self._tabletop_lift_rotation = self._tabletop_rotation_xyzw(eef[3:])
        elif self._tabletop_phase == "stage" and supported_ttc <= 0.35:
            self._tabletop_phase = "preshape"
        if self._tabletop_phase == "preshape" and supported_ttc <= 0.28:
            self._tabletop_phase = "intercept"
        ttc = math.inf if twist[1] <= 1e-06 else max(0.0, (0.07 - float(pose[1])) / float(twist[1]))
        if not math.isfinite(ttc):
            intercept = pose[:3].copy()
        else:
            intercept = pose[:3] + ttc * twist[:3]
            intercept[1] = 0.07
        if self._tabletop_phase == "close" and self._tabletop_close_target is None:
            self._tabletop_close_target = intercept.copy()
        if self._tabletop_phase == "stage":
            target = np.array([0.0, 0.07, 0.89], dtype=np.float64)
            gripper = -1.0
        elif self._tabletop_phase == "preshape":
            target = intercept.copy()
            gripper = -1.0
        elif self._tabletop_phase == "intercept":
            target = intercept.copy()
            gripper = -1.0
            if ttc <= 0.28 or np.linalg.norm(intercept - eef[:3]) <= 0.035:
                self._tabletop_phase = "close"
                gripper = 1.0
                self._tabletop_close_target = target.copy()
                target = self._tabletop_close_target
        elif self._tabletop_phase == "close":
            if self._tabletop_close_target is None:
                raise RuntimeError("T4 tabletop sphere close target is uninitialized")
            target = self._tabletop_close_target
            gripper = 1.0
        else:
            if self._tabletop_lift_anchor is None or self._tabletop_lift_object_anchor is None:
                raise RuntimeError("T4 tabletop sphere lift anchors are uninitialized")
            target = self._tabletop_lift_anchor + np.array([0.0, 0.0, 0.12], dtype=np.float64)
            if not self._tabletop_clearance_reached:
                target[:2] += pose[:2] - self._tabletop_lift_object_anchor[:2]
            gripper = 1.0
        translation = np.clip((target - eef[:3]) / 0.05, -1.0, 1.0)
        if self._tabletop_phase == "lift":
            translation[2] = min(float(translation[2]), 0.3)
        desired_rotation = (
            self._tabletop_lift_rotation
            if self._tabletop_phase == "lift"
            else self._tabletop_hold_rotation
        )
        if desired_rotation is None:
            raise RuntimeError("T4 tabletop sphere orientation anchor is missing")
        rotation = np.clip(
            self._tabletop_rotation_vector(desired_rotation @ current_rotation.T) / 0.5, -1.0, 1.0
        )
        return (translation * 0.05, rotation * 0.5, gripper)


class _PlacementOracle:
    """Original causal OnTimeBench moving-goal placement policy through BenchmarkAction."""

    def __init__(self, environment: DynamicManipulationEnv) -> None:
        self._se3 = environment._task_spec.task_id == "rotating_receptacle"
        self._joint_control = environment._scenario_spec.action_mode == "J8"
        self._phase = "hover"
        self._object_to_eef_local: npt.NDArray[np.float64] | None = None
        self._object_to_eef_rotation: npt.NDArray[np.float64] | None = None
        self._release_aligned_steps = 0
        self._retreat_position: npt.NDArray[np.float64] | None = None
        self._retreat_rotation: npt.NDArray[np.float64] | None = None

    @staticmethod
    def _rotation_from_wxyz(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        w, x, y, z = (float(value) for value in quaternion)
        return np.array(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _yaw_rotation(yaw: float) -> npt.NDArray[np.float64]:
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))

    @staticmethod
    def _rotation_vector(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
        if angle <= 1e-9:
            return np.zeros(3, dtype=np.float64)
        axis = np.array(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
            norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            return np.zeros(3, dtype=np.float64)
        return angle * axis / norm

    def _placement_command(
        self,
        object_pose_wxyz: npt.NDArray[np.float64],
        goal_pose_wxyz: npt.NDArray[np.float64],
        goal_twist_world: npt.NDArray[np.float64],
        eef_pose_xyzw: npt.NDArray[np.float64],
        contacts: tuple[bool, bool],
        *,
        contained: bool,
        released: bool,
    ) -> npt.NDArray[np.float64]:
        object_position, eef_position = object_pose_wxyz[:3], eef_pose_xyzw[:3]
        object_rotation = self._rotation_from_wxyz(object_pose_wxyz[3:])
        eef_rotation = self._rotation_from_wxyz(eef_pose_xyzw[[6, 3, 4, 5]])
        w, x, y, z = (float(value) for value in goal_pose_wxyz[3:])
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if self._object_to_eef_local is None or (
            self._joint_control and self._se3 and self._phase in ("hover", "descend")
        ):
            # J8 keeps the held payload aligned after contact permits relative grasp motion.
            self._object_to_eef_local = object_rotation.T @ (eef_position - object_position)
            self._object_to_eef_rotation = object_rotation.T @ eef_rotation
        if self._object_to_eef_rotation is None:
            raise RuntimeError("placement grasp transform is uninitialized")

        lookahead = 0.0 if self._joint_control or self._phase == "retreat" else 0.2
        goal_rotation = self._yaw_rotation(yaw)
        desired_object_rotation = self._yaw_rotation(yaw + lookahead * float(goal_twist_world[5]))
        desired_eef_rotation = desired_object_rotation @ self._object_to_eef_rotation
        predicted_goal_position = goal_pose_wxyz[:3] + lookahead * goal_twist_world[:3]
        if self._phase == "hover":
            entrance_height = 0.004 + 2.0 * 0.020 if self._se3 else 0.005 + 2.0 * 0.025
            height = entrance_height + 0.075
        else:
            height = _PLACEMENT_CONTACT_HEIGHT[
                "rotating_receptacle" if self._se3 else "moving_receptacle"
            ]
        desired_object_position = predicted_goal_position + np.array((0.0, 0.0, height))
        desired_eef_position = (
            desired_object_position + desired_object_rotation @ self._object_to_eef_local
        )
        object_error = desired_object_position - object_position
        planar_error = float(np.linalg.norm(goal_pose_wxyz[:2] - object_position[:2]))
        orientation_error = float(
            np.linalg.norm(self._rotation_vector(goal_rotation @ object_rotation.T))
        )
        planar_tolerance = 0.002 if self._se3 else 0.005
        orientation_tolerance = 0.02 if self._se3 else 0.05
        if (
            self._phase == "hover"
            and planar_error <= planar_tolerance
            and abs(float(object_error[2])) <= 0.018
            and orientation_error <= orientation_tolerance
        ):
            self._phase = "descend"
        if self._phase == "descend":
            aligned = (
                planar_error <= (0.001 if self._se3 else 0.005)
                and abs(float(object_error[2])) <= 0.003
                and orientation_error <= (0.01 if self._se3 else 0.05)
            )
            self._release_aligned_steps = (
                self._release_aligned_steps + 1 if contained and aligned else 0
            )
            if self._release_aligned_steps >= (2 if self._se3 else 1):
                self._phase = "release"

        gripper = 1.0
        if self._phase == "release":
            desired_eef_rotation = goal_rotation @ self._object_to_eef_rotation
            gripper = -1.0
            if not any(contacts) or released:
                self._phase = "retreat"
                self._retreat_position = eef_position.copy()
                self._retreat_rotation = eef_rotation.copy()
        if self._phase == "retreat":
            if self._retreat_position is None or self._retreat_rotation is None:
                raise RuntimeError("placement retreat anchor is uninitialized")
            desired_eef_position = self._retreat_position + np.array((0.0, 0.0, 0.12))
            desired_eef_rotation = self._retreat_rotation
            gripper = -1.0

        translation = np.clip((desired_eef_position - eef_position) / 0.05, -1.0, 1.0)
        if self._joint_control and self._se3 and self._phase == "retreat":
            translation[2] = min(0.1, float(translation[2]))
        if self._phase == "descend":
            if planar_error > planar_tolerance or orientation_error > orientation_tolerance:
                translation[2] = max(0.0, float(translation[2]))
            else:
                translation[2] = max(-0.2, float(translation[2]))
        rotation = np.clip(
            self._rotation_vector(desired_eef_rotation @ eef_rotation.T) / 0.5, -1.0, 1.0
        )
        return np.concatenate((0.05 * translation, 0.5 * rotation, (gripper,)))

    def action(self, environment: DynamicManipulationEnv) -> BenchmarkAction:
        model, data = environment.sim.model._model, environment.sim.data._data
        robot = environment.robots[0]
        site = robot.eef_site_id["right"]
        address = environment._payload_qpos_address
        pose = data.qpos[address : address + 7].copy()
        goal = environment._goal_reference(float(data.time))
        goal_pose = np.concatenate(
            (data.xpos[environment._goal_body_id], data.xquat[environment._goal_body_id])
        )
        goal_twist = np.concatenate(
            (goal.linear_velocity_world_m_s, goal.angular_velocity_world_rad_s)
        )
        eef_pose = np.concatenate(
            (data.site_xpos[site], mat2quat(data.site_xmat[site].reshape(3, 3)))
        )
        left, right, _ = _fingerpad_contacts(
            model,
            data,
            environment._payload_geom_ids,
            environment._left_pad_ids,
            environment._right_pad_ids,
        )
        facts = environment._geometry_facts()
        command = self._placement_command(
            pose,
            goal_pose,
            goal_twist,
            eef_pose,
            (left, right),
            # The expert's own release gate: fully seated, below the rim and (SE3) with its
            # 2 mm planar margin. A controller precision choice, not the benchmark predicate.
            contained=_expert_seated(facts, se3=self._se3),
            released="release" in environment._placement.events,
        )
        mode = environment._scenario_spec.action_mode
        if mode == "E7":
            base = int(model.body(environment._robot_base_body_name).id)
            base_rotation = data.xmat[base].reshape(3, 3)
            scale = np.asarray((0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0))
            command = (command / scale).astype(np.float32).astype(np.float64) * scale
            motion = np.concatenate((base_rotation.T @ command[:3], base_rotation.T @ command[3:6]))
        else:
            target_twist = np.zeros(6, dtype=np.float64)
            if self._phase != "retreat":
                if self._object_to_eef_local is None:
                    raise RuntimeError("placement grasp transform is uninitialized")
                target_twist = goal_twist.copy()
                offset = self._rotation_from_wxyz(goal_pose[3:]) @ self._object_to_eef_local
                target_twist[:3] += np.cross(goal_twist[3:], offset)
            motion = _capture_joint_motion(environment, command[:3], command[3:6], target_twist)
        values = np.ascontiguousarray(
            np.concatenate((motion, command[-1:])), dtype=np.float64 if mode == "E7" else np.float32
        ).astype(np.float64)
        values.setflags(write=False)
        return BenchmarkAction(mode, "delta", values)


class _RotatingOracle:
    """OnTimeBench acquisition, upright handoff and arc-length follower through BenchmarkAction."""

    def __init__(self, environment: DynamicManipulationEnv) -> None:
        self._full = environment._task_spec.variant == "full"
        self._phase = "track" if self._full else "follow_path"
        if self._full and environment._scenario_spec.action_mode == "J8":
            self._phase = "raise_to_track"
        self._post_acquisition_steps = 0
        self._target_progress = 0.0
        self._exit_progress_m = 0.0
        self._path = environment._path_points.copy()
        self._cumulative_m = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(self._path, axis=0), axis=1)))
        )
        self._length_m = float(self._cumulative_m[-1])
        self._hold_eef_rotation: npt.NDArray[np.float64] | None = None
        self._staging_loop: npt.NDArray[np.float64] | None = None
        self._handoff_relative_rotation: npt.NDArray[np.float64] | None = None
        self._initial_jaw_angle: float | None = None
        self._initial_wand_yaw: float | None = None
        site = environment.robots[0].eef_site_id["right"]
        self._acquisition_eef_rotation = environment.sim.data.site_xmat[site].reshape(3, 3).copy()
        self._initial_eef_position = environment.sim.data.site_xpos[site].copy()

    def _point(self, progress: float) -> npt.NDArray[np.float64]:
        distance = float(np.clip(progress, 0.0, 1.0)) * self._length_m
        index = min(
            int(np.searchsorted(self._cumulative_m, distance, side="right") - 1),
            len(self._path) - 2,
        )
        edge = self._path[index + 1] - self._path[index]
        fraction = (distance - self._cumulative_m[index]) / float(np.linalg.norm(edge))
        return cast(npt.NDArray[np.float64], self._path[index] + fraction * edge)

    @staticmethod
    def _rotation_vector(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Preserve the reference controller's rotation conversion, including the pi singularity.
        angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
        if angle <= 1e-9:
            return np.zeros(3, dtype=np.float64)
        axis = np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
            norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            return np.zeros(3, dtype=np.float64)
        return angle * axis / norm

    @staticmethod
    def _action(
        environment: DynamicManipulationEnv,
        position_error: npt.NDArray[np.float64],
        rotation_error: npt.NDArray[np.float64],
        gripper: float,
        rotation_limit_rad: float = 0.5,
    ) -> BenchmarkAction:
        mode = environment._scenario_spec.action_mode
        position = np.clip(position_error / 0.05, -1.0, 1.0) * 0.05
        rotation_limit = rotation_limit_rad / 0.5
        rotation = np.clip(rotation_error / 0.5, -rotation_limit, rotation_limit) * 0.5
        representation: Literal["delta", "absolute"] = "delta"
        if mode == "E7":
            model, data = environment.sim.model._model, environment.sim.data._data
            base = int(model.body(environment._robot_base_body_name).id)
            base_rotation = data.xmat[base].reshape(3, 3)
            position = (position / 0.05).astype(np.float32).astype(np.float64) * 0.05
            rotation = (rotation / 0.5).astype(np.float32).astype(np.float64) * 0.5
            if environment._scenario_spec.action_delay_steps:
                # Queue a physical target pose so the actuator converts its
                # error against the achieved state when this command arrives.
                site = environment.robots[0].eef_site_id["right"]
                target_position = data.site_xpos[site] + np.clip(position, -0.025, 0.025)
                target_rotation = _so3_exp(np.clip(rotation, -0.25, 0.25)) @ data.site_xmat[
                    site
                ].reshape(3, 3)
                motion = np.concatenate(
                    (
                        base_rotation.T @ (target_position - data.xpos[base]),
                        _so3_log(base_rotation.T @ target_rotation),
                    )
                )
                representation = "absolute"
            else:
                motion = np.concatenate((base_rotation.T @ position, base_rotation.T @ rotation))
        else:
            motion = _capture_joint_motion(environment, position, rotation, np.zeros(6))
        values = np.ascontiguousarray(
            np.concatenate((motion, [gripper])), dtype=np.float64 if mode == "E7" else np.float32
        ).astype(np.float64)
        values.setflags(write=False)
        return BenchmarkAction(mode, representation, values)

    def action(self, environment: DynamicManipulationEnv) -> BenchmarkAction:
        model, data = environment.sim.model._model, environment.sim.data._data
        eef_site = environment.robots[0].eef_site_id["right"]
        eef = data.site_xpos[eef_site]
        eef_rotation = _ReplanningOracle._rotation_from_xyzw(
            mat2quat(data.site_xmat[eef_site].reshape(3, 3))
        )
        address = environment._wand_qpos_address
        wand_pose = data.qpos[address : address + 7]
        wand = wand_pose[:3]
        wand_rotation = _ReplanningOracle._rotation_from_xyzw(wand_pose[[4, 5, 6, 3]])
        loop_local = model.site_pos[environment._loop_site_id]
        loop = wand + wand_rotation @ loop_local
        events = environment._outcome.events
        zeros = np.zeros(3, dtype=np.float64)

        if self._phase == "raise_to_track":
            target = self._initial_eef_position.copy()
            target[2] = wand[2] + 0.12
            rotation_error = self._rotation_vector(self._acquisition_eef_rotation @ eef_rotation.T)
            if eef[2] < target[2] - 0.01:
                return self._action(environment, target - eef, rotation_error, -1.0)
            self._phase = "track"

        if self._phase in ("track", "descend", "close"):
            if "wand_acquisition" in events:
                self._phase = "settle"
            else:
                left, right, _ = _fingerpad_contacts(
                    model,
                    data,
                    environment._wand_geom_ids,
                    environment._left_pad_ids,
                    environment._right_pad_ids,
                )
                closing_axis = np.mean(data.geom_xpos[list(environment._right_pad_ids)], axis=0)
                closing_axis -= np.mean(data.geom_xpos[list(environment._left_pad_ids)], axis=0)
                jaw_angle = math.atan2(float(closing_axis[1]), float(closing_axis[0]))
                if self._initial_jaw_angle is None:
                    w, x, y, z = (float(value) for value in wand_pose[3:])
                    self._initial_wand_yaw = math.atan2(
                        2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
                    )
                    offset = jaw_angle - self._initial_wand_yaw
                    self._initial_jaw_angle = math.atan2(math.sin(offset), math.cos(offset))
                if self._phase == "descend" and (left or right):
                    self._phase = "close"
                joint_track = environment._scenario_spec.action_mode == "J8"
                if self._phase == "track" and np.linalg.norm(wand[:2] - eef[:2]) < (
                    0.006 if joint_track else 0.020
                ):
                    self._phase = "descend"
                height = (
                    (0.12 if joint_track else 0.08)
                    if self._phase == "track"
                    else _CONTACT_WORLD_OFFSETS["full_wire_passage"][2]
                )
                target = wand + np.array((0.0, 0.0, height))
                gripper = 1.0 if self._phase == "close" else -1.0
                error = target - eef
                if (
                    self._phase == "descend"
                    and np.linalg.norm(error[:2]) < 0.015
                    and abs(float(error[2])) < 0.015
                ):
                    # OnTimeBench starts closing on the following control command.
                    self._phase = "close"
                if self._initial_wand_yaw is None:
                    raise RuntimeError("T3 acquisition requires the initial wand yaw")
                angle = self._initial_wand_yaw + self._initial_jaw_angle - jaw_angle
                yaw_error = 0.5 * math.atan2(math.sin(2.0 * angle), math.cos(2.0 * angle))
                rotation_error = (
                    self._rotation_vector(self._acquisition_eef_rotation @ eef_rotation.T)
                    if joint_track or environment._scenario_spec.action_delay_steps
                    else np.array((0.0, 0.0, yaw_error))
                )
                return self._action(environment, error, rotation_error, gripper)

        if self._phase == "settle":
            if self._post_acquisition_steps < 2:
                self._post_acquisition_steps += 1
                return self._action(environment, zeros, zeros, 1.0)
            if (
                environment._scenario_spec.action_mode == "J8"
                or environment._scenario_spec.action_delay_steps
            ):
                self._phase = "clear_cradle"
                self._staging_loop = loop.copy()
                self._staging_loop[0] = (
                    environment._geometry.frame_center_world_m[0] + self._path[1, 0]
                )
                self._handoff_relative_rotation = wand_rotation.T @ eef_rotation
            else:
                self._phase = "handoff"

        if self._phase == "clear_cradle":
            if self._staging_loop is None or self._handoff_relative_rotation is None:
                raise RuntimeError("T3 cradle clearance requires the acquired pose")
            # An upright ring starts below the dock pads. Slide it forward out
            # of their footprint before lifting; upward force alone wedges it.
            desired_wand_rotation = np.eye(3, dtype=np.float64)
            relative_position = wand_rotation.T @ (eef - wand)
            desired_wand = self._staging_loop - desired_wand_rotation @ loop_local
            desired_eef = desired_wand + desired_wand_rotation @ relative_position
            desired_rotation = desired_wand_rotation @ self._handoff_relative_rotation
            position_error = desired_eef - eef
            rotation_error = self._rotation_vector(desired_rotation @ eef_rotation.T)
            if np.linalg.norm(position_error) > 0.006:
                return self._action(environment, position_error, rotation_error, 1.0)
            self._target_progress = float(self._cumulative_m[1] / self._length_m)
            self._staging_loop = None
            self._handoff_relative_rotation = None
            self._phase = "handoff"

        phase = float(data.qpos[environment._wire_qpos_address])
        phase += 0.05 * float(data.qvel[environment._wire_qvel_address])
        cosine, sine = math.cos(phase), math.sin(phase)
        frame_rotation = np.array(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))

        if self._phase == "handoff":
            if self._staging_loop is None:
                self._staging_loop = loop + np.array((0.0, 0.0, 0.012))
                self._handoff_relative_rotation = wand_rotation.T @ eef_rotation
            if self._handoff_relative_rotation is None:
                raise RuntimeError("T3 Full handoff requires the measured grasp rotation")
            # The motor turns about world X. The handoff brings the wand upright.
            desired_wand_rotation = np.eye(3, dtype=np.float64)
            relative_position = wand_rotation.T @ (eef - wand)
            desired_wand = self._staging_loop - desired_wand_rotation @ loop_local
            desired_eef = desired_wand + desired_wand_rotation @ relative_position
            desired_rotation = desired_wand_rotation @ self._handoff_relative_rotation
            position_error = desired_eef - eef
            rotation_error = 0.5 * (
                np.cross(eef_rotation[:, 0], desired_rotation[:, 0])
                + np.cross(eef_rotation[:, 1], desired_rotation[:, 1])
                + np.cross(eef_rotation[:, 2], desired_rotation[:, 2])
            )
            if np.linalg.norm(position_error) <= 0.006 and np.linalg.norm(rotation_error) <= 0.06:
                self._phase = "follow_path"
            else:
                return self._action(environment, position_error, rotation_error, 1.0)

        if self._hold_eef_rotation is None:
            self._hold_eef_rotation = eef_rotation.copy()
        point = self._point(self._target_progress)
        half_span = 0.025 / self._length_m
        tangent = self._point(self._target_progress + half_span) - self._point(
            self._target_progress - half_span
        )
        tangent /= np.linalg.norm(tangent)
        if "wire_exit" in events:
            self._phase = "release"
        if (
            "terminal_arrival" in events
            and self._target_progress >= 1.0
            and self._phase != "release"
        ):
            self._phase = "clear_exit"
        if self._phase == "clear_exit" and self._target_progress >= 1.0:
            point = self._path[-1] + self._exit_progress_m * tangent
        desired_loop = (
            np.asarray(environment._geometry.frame_center_world_m) + frame_rotation @ point
        )
        desired_eef = eef + desired_loop - loop
        position_error = desired_eef - eef
        rotation_error = self._rotation_vector(self._hold_eef_rotation @ eef_rotation.T)
        facts = environment._path_facts()
        orientation_error = math.acos(float(np.clip(wand_rotation[0, 0], -1.0, 1.0)))
        if (
            self._phase == "follow_path"
            and facts.path_distance_m <= 0.030
            and orientation_error <= 0.20
            and self._target_progress < 1.0
            and (
                not self._full
                or environment._scenario_spec.action_mode != "J8"
                or np.linalg.norm(position_error) <= 0.006
            )
        ):
            advance_m = (
                0.002 if self._full and environment._scenario_spec.action_mode == "J8" else 0.001
            )
            self._target_progress = min(1.0, self._target_progress + advance_m / self._length_m)
        exit_clearance = 0.065 if self._full else 0.050
        if (
            self._phase == "clear_exit"
            and np.linalg.norm(position_error) <= (0.025 if self._full else 0.004)
            and np.linalg.norm(rotation_error) <= 0.20
        ):
            if self._exit_progress_m < exit_clearance:
                self._exit_progress_m = min(exit_clearance, self._exit_progress_m + 0.004)
            elif (
                facts.path_distance_m >= exit_clearance
                and facts.terminal_distance_m >= exit_clearance
            ):
                self._phase = "release"
        return self._action(
            environment,
            position_error,
            rotation_error,
            -1.0 if self._phase == "release" else 1.0,
            rotation_limit_rad=0.075,
        )


def _run_native_oracle(environment: DynamicManipulationEnv) -> dict[str, Any]:
    oracle = _make_native_oracle(environment)
    for _ in range(environment._task_spec.episode_horizon_steps):
        _, _, done, info = environment.step(oracle.action())
        oracle.observe_step(done=done, success=info["success"])
        if done:
            return info
    raise RuntimeError(
        f"{environment._task_spec.family} oracle reached horizon "
        "without its native terminal outcome"
    )


class _NativeOracle:
    """Bind the existing causal servo to one already-reset native environment."""

    def __init__(
        self,
        environment: DynamicManipulationEnv,
        controller: (
            _CaptureOracle
            | _ReplanningOracle
            | _FreeDynamicsOracle
            | _PlacementOracle
            | _RotatingOracle
        ),
    ) -> None:
        self._environment = environment
        self._controller = controller
        if isinstance(controller, _CaptureOracle):
            controller._require_valid_visibility(environment, terminal=False)

    def action(self) -> BenchmarkAction:
        return self._controller.action(self._environment)

    def observe_step(self, *, done: bool, success: bool) -> None:
        # Native failure takes precedence over a target leaving the camera raster.
        if isinstance(self._controller, _CaptureOracle) and not (done and not success):
            self._controller._observe_visibility(self._environment)
            self._controller._require_valid_visibility(self._environment, terminal=done)


def _make_native_oracle(environment: DynamicManipulationEnv) -> _NativeOracle:
    family = environment._task_spec.family
    controller: (
        _CaptureOracle
        | _ReplanningOracle
        | _FreeDynamicsOracle
        | _PlacementOracle
        | _RotatingOracle
    )
    if family in ("grasp", "driven_interception"):
        controller = _CaptureOracle(environment)
    elif family == "replanning":
        controller = _ReplanningOracle(environment)
    elif family == "free_dynamics_interception":
        controller = _FreeDynamicsOracle(environment)
    elif family == "moving_placement":
        controller = _PlacementOracle(environment)
    elif family == "rotating_interaction":
        controller = _RotatingOracle(environment)
    else:
        raise ValueError(f"native oracle is not implemented for family {family!r}")
    return _NativeOracle(environment, controller)
