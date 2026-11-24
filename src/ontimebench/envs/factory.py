from __future__ import annotations

from robosuite.environments.base import REGISTERED_ENVS
from robosuite.environments.base import make as make_robosuite_env

from ontimebench.envs.base import DynamicManipulationEnv
from ontimebench.envs.families.driven_interception import DrivenInterceptionEnv
from ontimebench.envs.families.free_dynamics_interception import (
    FreeDynamicsInterceptionEnv,
)
from ontimebench.envs.families.grasp import GraspEnv
from ontimebench.envs.families.moving_placement import MovingPlacementEnv
from ontimebench.envs.families.replanning import ReplanningEnv
from ontimebench.envs.families.rotating_interaction import RotatingInteractionEnv
from ontimebench.scenarios.spec import (
    DrivenInterceptionGeometrySpec,
    ReleaseGateRetractionEffectSpec,
    ScenarioSpec,
    StateTriggerSpec,
    StrikerPulseEffectSpec,
    TimeTriggerSpec,
    family_for_geometry,
)
from ontimebench.tasks.registry import TASK_FAMILIES, load_task
from ontimebench.tasks.spec import (
    DrivenInterceptionBindings,
    FreeDynamicsInterceptionBindings,
    GraspBindings,
    MovingPlacementBindings,
    ReplanningBindings,
    RotatingInteractionBindings,
    TaskSpec,
)


def _require_entities(values: tuple[str, ...], available: set[str], role: str) -> None:
    missing = set(values) - available
    if missing:
        raise ValueError(f"{role} bindings do not resolve: {sorted(missing)}")


def _validate_task_scenario(task: TaskSpec, scenario: ScenarioSpec) -> None:
    if family_for_geometry(scenario.geometry) != task.family:
        raise ValueError("scenario GeometrySpec does not match task family")
    if scenario.action_mode not in task.allowed_action_modes:
        raise ValueError("scenario action mode is not allowed by TaskSpec")
    if scenario.observation_profile not in task.allowed_observation_profiles:
        raise ValueError("scenario observation profile is not allowed by TaskSpec")
    object_ids = {item.instance_id for item in scenario.object_states}
    fixture_ids = {item.instance_id for item in scenario.fixture_states}
    bindings = task.bindings
    if isinstance(bindings, GraspBindings):
        _require_entities((bindings.target_object_id,), object_ids, "object")
    elif isinstance(bindings, DrivenInterceptionBindings):
        _require_entities((bindings.target_object_id,), object_ids, "object")
        _require_entities((bindings.driver_entity_id,), object_ids | fixture_ids, "driver entity")
        _require_entities(bindings.occluder_fixture_ids, fixture_ids, "occluder fixture")
        assert isinstance(scenario.geometry, DrivenInterceptionGeometrySpec)
        if not task.has_occluder and scenario.geometry.occlusion_target_clearance_m != 0:
            raise ValueError("non-zero occlusion clearance requires a bound occluder")
    elif isinstance(bindings, MovingPlacementBindings):
        _require_entities((bindings.payload_object_id,), object_ids, "payload object")
        _require_entities((bindings.goal_fixture_id,), fixture_ids, "goal fixture")
    elif isinstance(bindings, RotatingInteractionBindings):
        _require_entities((bindings.wand_object_id,), object_ids, "wand object")
        _require_entities((bindings.wire_fixture_id,), fixture_ids, "wire fixture")
        _require_entities(bindings.cradle_fixture_ids, fixture_ids, "cradle fixture")
    elif isinstance(bindings, FreeDynamicsInterceptionBindings):
        _require_entities((bindings.target_object_id,), object_ids, "target object")
        _require_entities(
            (
                bindings.ramp_fixture_id,
                bindings.release_gate_fixture_id,
                bindings.downstream_surface_fixture_id,
            ),
            fixture_ids,
            "free-dynamics fixture",
        )
        if len(scenario.events) != 1:
            raise ValueError("T4 requires exactly one release-gate event")
        event = scenario.events[0]
        effect = event.effect
        if (
            isinstance(effect, ReleaseGateRetractionEffectSpec)
            and effect.gate_entity_id != bindings.release_gate_fixture_id
        ):
            raise ValueError("T4 release event must target its bound release gate")
        if (
            event.event_id != "release_gate"
            or not isinstance(event.trigger, TimeTriggerSpec)
            or event.trigger.control_tick != 0
            or not isinstance(effect, ReleaseGateRetractionEffectSpec)
        ):
            raise ValueError("T4 requires its exact tick-0 release-gate event")
    elif isinstance(bindings, ReplanningBindings):
        _require_entities((bindings.target_object_id,), object_ids, "target object")
        _require_entities(
            (
                bindings.belt_fixture_id,
                bindings.left_striker_fixture_id,
                bindings.right_striker_fixture_id,
            ),
            fixture_ids,
            "replanning fixture",
        )
        if len(scenario.events) != 1:
            raise ValueError("T5 requires exactly one selected-striker event")
        event = scenario.events[0]
        effect = event.effect
        if (
            event.event_id != "selected_strike"
            or not isinstance(event.trigger, StateTriggerSpec)
            or not isinstance(effect, StrikerPulseEffectSpec)
            or effect.striker_entity_id
            not in {
                bindings.left_striker_fixture_id,
                bindings.right_striker_fixture_id,
            }
        ):
            raise ValueError("T5 event must select exactly one bound striker")


_FAMILY_ENV_NAMES = {
    "grasp": GraspEnv.__name__,
    "driven_interception": DrivenInterceptionEnv.__name__,
    "moving_placement": MovingPlacementEnv.__name__,
    "rotating_interaction": RotatingInteractionEnv.__name__,
    "free_dynamics_interception": FreeDynamicsInterceptionEnv.__name__,
    "replanning": ReplanningEnv.__name__,
}


def make_env(
    task_id: str,
    scenario_spec: ScenarioSpec,
    *,
    execution_translation_limit_m: float = 0.05,
) -> DynamicManipulationEnv:
    if not isinstance(task_id, str) or not task_id or task_id not in TASK_FAMILIES:
        raise ValueError(f"unknown task: {task_id}")
    if not isinstance(scenario_spec, ScenarioSpec):
        raise ValueError("scenario_spec must be a ScenarioSpec")
    task = load_task(task_id)
    _validate_task_scenario(task, scenario_spec)
    env_name = _FAMILY_ENV_NAMES[task.family]
    if env_name not in REGISTERED_ENVS:
        raise NotImplementedError(f"Robosuite environment is not registered: {env_name}")
    environment = make_robosuite_env(
        env_name,
        task_spec=task,
        scenario_spec=scenario_spec,
        execution_translation_limit_m=execution_translation_limit_m,
    )
    if not isinstance(environment, DynamicManipulationEnv):
        raise RuntimeError("registered Family Env must derive from DynamicManipulationEnv")
    environment.set_ep_meta(scenario_spec)
    return environment


__all__ = ["make_env"]
