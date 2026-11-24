"""Native reset, solvability and export for concrete proposals."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import mujoco
import numpy as np

from ontimebench.envs.base import (
    DynamicManipulationEnv,
    _physics_state_is_valid,
    _ScenarioResetRejected,
)
from ontimebench.envs.components.occlusion import _validate_compiled_occlusion
from ontimebench.envs.factory import make_env
from ontimebench.scenarios._native_oracle import _run_native_oracle
from ontimebench.scenarios._rejection import ScenarioRejected
from ontimebench.scenarios.spec import ScenarioSpec, StrikerPulseEffectSpec, scenario_to_dict
from ontimebench.tasks.spec import TaskSpec


def _validate_capture_reset(environment: DynamicManipulationEnv) -> None:
    model, data = environment.sim.model._model, environment.sim.data._data
    if not _physics_state_is_valid(data):
        raise ScenarioRejected(
            "reset", "invalid_state", task_id=environment._task_spec.task_id
        )
    if environment._task_spec.has_occluder:
        facts = environment._compiled_occlusion_facts()
        audit = _validate_compiled_occlusion(facts)
        reason = None
        if audit.failure_facts.canopy_collision_disabled:
            reason = "canopy_collision_disabled"
        elif audit.failure_facts.lift_envelope_clearance_insufficient:
            reason = "lift_envelope_clearance_insufficient"
        elif audit.failure_facts.unsafe_canopy_contact:
            reason = "unsafe_canopy_contact"
        if reason is not None:
            raise ScenarioRejected("reset", reason, task_id=environment._task_spec.task_id)
    robot = environment.robots[0]
    for name in (*robot.robot_model.joints, *robot.gripper["right"].joints):
        joint_id = int(model.joint(name).id)
        position = float(data.qpos[model.jnt_qposadr[joint_id]])
        if not math.isfinite(position) or (
            model.jnt_limited[joint_id]
            and not (model.jnt_range[joint_id, 0] <= position <= model.jnt_range[joint_id, 1])
        ):
            rejection = ScenarioRejected(
                "reset", "joint_out_of_range", task_id=environment._task_spec.task_id
            )
            rejection.add_note(f"joint: {name}")
            raise rejection
    placement = environment._task_spec.family == "moving_placement"
    target_geoms = set(environment._payload_geom_ids if placement else environment._target_geom_ids)
    allowed_contacts = set(
        environment._gripper_geom_ids if placement else environment._support_geom_ids
    )
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & target_geoms and not pair & allowed_contacts:
            raise ScenarioRejected(
                "reset",
                "payload_collision_outside_pregrasp"
                if placement
                else "target_collision_outside_support",
                task_id=environment._task_spec.task_id,
            )
    segmentation = np.asarray(
        environment.sim.render(
            camera_name="agentview",
            width=224,
            height=224,
            segmentation=True,
        )
    )
    body_id = environment._payload_body_id if placement else environment._target_body_id
    target_visual_ids = np.flatnonzero(model.geom_bodyid == body_id)
    visible = (segmentation[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
        segmentation[:, :, 1], target_visual_ids
    )
    if not np.any(visible):
        raise ScenarioRejected(
            "reset", "target_not_visible", task_id=environment._task_spec.task_id
        )


def _validate_capture_physical_scenario(task: TaskSpec, scenario: ScenarioSpec) -> ScenarioSpec:
    environment = make_env(task.task_id, scenario)
    try:
        raw = environment.reset()
        environment._benchmark_observation(raw)
        _validate_capture_reset(environment)
        exported = environment.export_scenario()
        outcome = _run_native_oracle(environment)
        if outcome["success"] is not True or outcome["failure_code"] is not None:
            raise ScenarioRejected(
                "oracle",
                outcome["failure_code"],
                task_id=task.task_id,
                terminal_time_s=outcome["terminal_time_s"],
                events=outcome["events"],
            )
        if not isinstance(exported, ScenarioSpec):
            raise RuntimeError("capture export must return a ScenarioSpec")
        return exported
    finally:
        environment.close()


def _validate_placement_physical_scenario(task: TaskSpec, scenario: ScenarioSpec) -> ScenarioSpec:
    try:
        environment = make_env(task.task_id, scenario)
    except _ScenarioResetRejected as failure:
        raise ScenarioRejected(
            "reset",
            failure.reason,
            task_id=task.task_id,
            terminal_time_s=failure.time_s,
        ) from failure
    try:
        try:
            raw = environment.reset()
        except _ScenarioResetRejected as failure:
            raise ScenarioRejected(
                "reset",
                failure.reason,
                task_id=task.task_id,
                terminal_time_s=failure.time_s,
            ) from failure
        environment._benchmark_observation(raw)
        _validate_capture_reset(environment)
        exported = environment.export_scenario()
        outcome = _run_native_oracle(environment)
        if outcome["success"] is not True or outcome["failure_code"] is not None:
            raise ScenarioRejected(
                "oracle",
                outcome["failure_code"],
                task_id=task.task_id,
                terminal_time_s=outcome["terminal_time_s"],
                events=outcome["events"],
            )
        if not isinstance(exported, ScenarioSpec):
            raise RuntimeError("placement export must return a ScenarioSpec")
        return exported
    finally:
        environment.close()


def _validate_rotating_reset(environment: DynamicManipulationEnv) -> None:
    model, data = environment.sim.model._model, environment.sim.data._data
    task_id = environment._task_spec.task_id
    if not _physics_state_is_valid(data):
        raise ScenarioRejected("reset", "invalid_state", task_id=task_id)
    robot = environment.robots[0]
    for name in (*robot.robot_model.joints, *robot.gripper["right"].joints):
        joint_id = int(model.joint(name).id)
        position = float(data.qpos[model.jnt_qposadr[joint_id]])
        if not math.isfinite(position) or (
            model.jnt_limited[joint_id]
            and not (model.jnt_range[joint_id, 0] <= position <= model.jnt_range[joint_id, 1])
        ):
            rejection = ScenarioRejected("reset", "joint_out_of_range", task_id=task_id)
            rejection.add_note(f"joint: {name}")
            raise rejection
    if not (
        math.isclose(
            float(data.qpos[environment._wire_qpos_address]),
            environment._wire_initial_phase(),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(data.qvel[environment._wire_qvel_address]),
            environment._motion.rotation.angular_speed_rad_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ScenarioRejected("reset", "wire_motion_mismatch", task_id=task_id)
    impulse, penetration = environment._wire_contacts()
    if impulse > 0.0 or penetration > 0.0:
        raise ScenarioRejected("reset", "wand_not_threaded", task_id=task_id)
    segmentation = np.asarray(
        environment.sim.render(camera_name="agentview", width=224, height=224, segmentation=True)
    )
    visual_ids = np.flatnonzero(model.geom_bodyid == environment._wand_body_id)
    visible = (segmentation[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
        segmentation[:, :, 1], visual_ids
    )
    if not np.any(visible):
        raise ScenarioRejected("reset", "target_not_visible", task_id=task_id)


def _validate_rotating_physical_scenario(task: TaskSpec, scenario: ScenarioSpec) -> ScenarioSpec:
    try:
        environment = make_env(task.task_id, scenario)
    except _ScenarioResetRejected as failure:
        raise ScenarioRejected(
            "reset",
            failure.reason,
            task_id=task.task_id,
            terminal_time_s=failure.time_s,
        ) from failure
    try:
        try:
            raw = environment.reset()
        except _ScenarioResetRejected as failure:
            raise ScenarioRejected(
                "reset",
                failure.reason,
                task_id=task.task_id,
                terminal_time_s=failure.time_s,
            ) from failure
        environment._benchmark_observation(raw)
        _validate_rotating_reset(environment)
        exported = environment.export_scenario()
        outcome = _run_native_oracle(environment)
        if outcome["success"] is not True or outcome["failure_code"] is not None:
            raise ScenarioRejected(
                "oracle",
                outcome["failure_code"],
                task_id=task.task_id,
                terminal_time_s=outcome["terminal_time_s"],
                events=outcome["events"],
            )
        if not isinstance(exported, ScenarioSpec):
            raise RuntimeError("rotating export must return a ScenarioSpec")
        return exported
    finally:
        environment.close()


def _validate_free_dynamics_reset(environment: DynamicManipulationEnv) -> None:
    model, data = environment.sim.model._model, environment.sim.data._data
    task_id = environment._task_spec.task_id
    if not _physics_state_is_valid(data):
        raise ScenarioRejected("reset", "invalid_state", task_id=task_id)
    robot = environment.robots[0]
    for name in (*robot.robot_model.joints, *robot.gripper["right"].joints):
        joint_id = int(model.joint(name).id)
        position = float(data.qpos[model.jnt_qposadr[joint_id]])
        if not math.isfinite(position) or (
            model.jnt_limited[joint_id]
            and not (model.jnt_range[joint_id, 0] <= position <= model.jnt_range[joint_id, 1])
        ):
            rejection = ScenarioRejected("reset", "joint_out_of_range", task_id=task_id)
            rejection.add_note(f"joint: {name}")
            raise rejection
    target_velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_XBODY,
        environment._target_body_id,
        target_velocity,
        0,
    )
    if not np.allclose(target_velocity, 0.0, rtol=0.0, atol=1e-12):
        raise ScenarioRejected("reset", "release_twist_exceeded", task_id=task_id)
    if environment._preparation_steps != 300 or environment._ramp_support_steps < 25:
        raise ScenarioRejected("reset", "target_not_supported_by_ramp", task_id=task_id)
    gate_body = environment._gate_body_id
    gate_position = model.body_pos[gate_body]
    if not (
        np.array_equal(gate_position, (0.0, -0.25, 0.55)) and model.body_jntnum[gate_body] == 0
    ):
        raise ScenarioRejected("reset", "gate_not_parked", task_id=task_id)
    segmentation = np.asarray(
        environment.sim.render(camera_name="agentview", width=224, height=224, segmentation=True)
    )
    visual_ids = np.flatnonzero(model.geom_bodyid == environment._target_body_id)
    visible = (segmentation[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
        segmentation[:, :, 1], visual_ids
    )
    if not np.any(visible):
        raise ScenarioRejected("reset", "target_not_visible", task_id=task_id)


def _validate_free_dynamics_physical_scenario(
    task: TaskSpec, scenario: ScenarioSpec
) -> ScenarioSpec:
    try:
        environment = make_env(task.task_id, scenario)
    except _ScenarioResetRejected as failure:
        raise ScenarioRejected(
            "reset",
            failure.reason,
            task_id=task.task_id,
            terminal_time_s=failure.time_s,
        ) from failure
    try:
        try:
            raw = environment.reset()
        except _ScenarioResetRejected as failure:
            raise ScenarioRejected(
                "reset",
                failure.reason,
                task_id=task.task_id,
                terminal_time_s=failure.time_s,
            ) from failure
        environment._benchmark_observation(raw)
        _validate_free_dynamics_reset(environment)
        exported = environment.export_scenario()
        outcome = _run_native_oracle(environment)
        release_rejection = environment._release_dynamics_rejection()
        if release_rejection is not None:
            rejection = ScenarioRejected(
                "reset",
                release_rejection,
                task_id=task.task_id,
                terminal_time_s=environment._release_event_time_s,
            )
            rejection.add_note(
                "release linear/angular speed: "
                f"{environment._release_linear_speed_m_s}/"
                f"{environment._release_angular_speed_rad_s}"
            )
            rejection.add_note(
                "pre-robot supported along-slope velocity gain/mean acceleration: "
                f"{environment._release_along_slope_velocity_gain_m_s}/"
                f"{environment._release_mean_along_slope_acceleration_m_s2}"
            )
            raise rejection
        if outcome["success"] is not True or outcome["failure_code"] is not None:
            raise ScenarioRejected(
                "oracle",
                outcome["failure_code"],
                task_id=task.task_id,
                terminal_time_s=outcome["terminal_time_s"],
                events=outcome["events"],
            )
        if not isinstance(exported, ScenarioSpec):
            raise RuntimeError("T4 export must return a ScenarioSpec")
        return exported
    finally:
        environment.close()


def _validate_replanning_reset(environment: DynamicManipulationEnv) -> None:
    model, data = environment.sim.model._model, environment.sim.data._data
    if not _physics_state_is_valid(data):
        raise ScenarioRejected("reset", "invalid_state", task_id="post_impact_grasp")
    robot = environment.robots[0]
    for name in (*robot.robot_model.joints, *robot.gripper["right"].joints):
        joint_id = int(model.joint(name).id)
        position = float(data.qpos[model.jnt_qposadr[joint_id]])
        if not math.isfinite(position) or (
            model.jnt_limited[joint_id]
            and not (model.jnt_range[joint_id, 0] <= position <= model.jnt_range[joint_id, 1])
        ):
            rejection = ScenarioRejected("reset", "joint_out_of_range", task_id="post_impact_grasp")
            rejection.add_note(f"joint: {name}")
            raise rejection
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_XBODY,
        environment._target_body_id,
        velocity,
        0,
    )
    if not np.array_equal(velocity, np.zeros(6)):
        raise ScenarioRejected("reset", "target_twist_nonzero", task_id="post_impact_grasp")
    if (
        environment._event_fired
        or environment._last_striker_state is not None
        or environment._outcome.physics_step != 0
        or any(
            data.qpos[address] != 0.0 or data.qvel[environment._joint_qvel_addresses[branch]] != 0.0
            for branch, address in environment._joint_qpos_addresses.items()
        )
    ):
        raise ScenarioRejected("reset", "striker_not_at_rest", task_id="post_impact_grasp")
    target_ids = frozenset(environment._target_geom_ids)
    striker_ids = frozenset(
        geom_id
        for entity_id in (
            environment._bindings.left_striker_fixture_id,
            environment._bindings.right_striker_fixture_id,
        )
        for geom_id in environment._fixture_geom_ids[entity_id]
    )
    if any(
        {
            int(data.contact[index].geom1),
            int(data.contact[index].geom2),
        }
        & target_ids
        and {
            int(data.contact[index].geom1),
            int(data.contact[index].geom2),
        }
        & striker_ids
        for index in range(data.ncon)
    ):
        raise ScenarioRejected("reset", "striker_contact_at_reset", task_id="post_impact_grasp")
    segmentation = np.asarray(
        environment.sim.render(camera_name="agentview", width=224, height=224, segmentation=True)
    )
    visual_ids = np.flatnonzero(model.geom_bodyid == environment._target_body_id)
    visible = (segmentation[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
        segmentation[:, :, 1], visual_ids
    )
    if not np.any(visible):
        raise ScenarioRejected("reset", "target_not_visible", task_id="post_impact_grasp")


def _opposite_replanning_branch(scenario: ScenarioSpec) -> ScenarioSpec:
    if len(scenario.events) != 1:
        raise ValueError("T5 counterfactual requires one striker-pulse event")
    event = scenario.events[0]
    effect = event.effect
    if not isinstance(effect, StrikerPulseEffectSpec):
        raise ValueError("T5 counterfactual requires one striker-pulse event")
    if effect.striker_entity_id == "left_striker" and effect.joint_name == "left_slide":
        other = replace(effect, striker_entity_id="right_striker", joint_name="right_slide")
    elif effect.striker_entity_id == "right_striker" and effect.joint_name == "right_slide":
        other = replace(effect, striker_entity_id="left_striker", joint_name="left_slide")
    else:
        raise ValueError("T5 counterfactual event must select a bound striker")
    return replace(scenario, events=(replace(event, effect=other),))


def _validate_replanning_physical_scenario(task: TaskSpec, scenario: ScenarioSpec) -> ScenarioSpec:
    branches = (scenario, _opposite_replanning_branch(scenario))
    selected_export: ScenarioSpec | None = None
    results: list[dict[str, Any]] = []
    physical_resets: list[dict[str, object]] = []
    for branch_index, branch_scenario in enumerate(branches):
        environment = make_env(task.task_id, branch_scenario)
        try:
            raw = environment.reset()
            environment._benchmark_observation(raw)
            _validate_replanning_reset(environment)
            exported = environment.export_scenario()
            physical_reset = scenario_to_dict(exported)
            physical_reset.pop("events")
            physical_resets.append(physical_reset)
            outcome = _run_native_oracle(environment)
            if outcome["success"] is not True or outcome["failure_code"] is not None:
                rejection = ScenarioRejected(
                    "oracle",
                    outcome["failure_code"],
                    task_id=task.task_id,
                    terminal_time_s=outcome["terminal_time_s"],
                    events=outcome["events"],
                )
                rejection.add_note(
                    f"selected branch: {environment._selected_branch}; "
                    f"counterfactual index: {branch_index}"
                )
                raise rejection
            if branch_index == 0:
                selected_export = exported
            results.append(outcome)
        finally:
            environment.close()
    first, second = results
    if (
        physical_resets[0] != physical_resets[1]
        or first["events"]["side_strike_triggered"] != second["events"]["side_strike_triggered"]
        or first["outgoing_twist_world"] == second["outgoing_twist_world"]
    ):
        raise ScenarioRejected(
            "oracle", "branch_counterfactual_mismatch", task_id=task.task_id
        )
    if selected_export is None:
        raise RuntimeError("T5 physical validation did not export the selected branch")
    return selected_export


def _validate_physical_scenario(task: TaskSpec, scenario: ScenarioSpec) -> ScenarioSpec:
    if task.family == "replanning":
        return _validate_replanning_physical_scenario(task, scenario)
    if task.family == "free_dynamics_interception":
        return _validate_free_dynamics_physical_scenario(task, scenario)
    if task.family == "moving_placement":
        return _validate_placement_physical_scenario(task, scenario)
    if task.family == "rotating_interaction":
        return _validate_rotating_physical_scenario(task, scenario)
    if task.grasp_motion is not None:
        return _validate_capture_physical_scenario(task, scenario)
    return scenario
