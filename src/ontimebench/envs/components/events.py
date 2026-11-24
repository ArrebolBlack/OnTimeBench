"""Trigger each scene event once on the simulation clock."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import mujoco
import numpy as np

from ontimebench.scenarios.spec import (
    ContactTriggerSpec,
    EventSpec,
    ReleaseGateRetractionEffectSpec,
    StateTriggerSpec,
    StrikerPulseEffectSpec,
    TimeTriggerSpec,
    Vector3,
)

_TIME_EPSILON_S = 1e-12


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True, slots=True)
class _ContactSample:
    entity_a_id: str
    entity_b_id: str
    normal_force_n: float

    def __post_init__(self) -> None:
        _identity(self.entity_a_id, "contact entity_a_id")
        _identity(self.entity_b_id, "contact entity_b_id")
        if self.entity_a_id == self.entity_b_id:
            raise ValueError("contact sample entities must differ")
        if _finite(self.normal_force_n, "contact normal_force_n") < 0.0:
            raise ValueError("contact normal_force_n must be non-negative")


@dataclass(frozen=True, slots=True)
class _JointState:
    position_m: float
    velocity_m_s: float

    def __post_init__(self) -> None:
        _finite(self.position_m, "joint position_m")
        _finite(self.velocity_m_s, "joint velocity_m_s")


@dataclass(frozen=True, slots=True)
class _ReleaseGateCommand:
    event_id: str
    gate_entity_id: str
    parked_position_world_m: Vector3


@dataclass(frozen=True, slots=True)
class _StrikerCommand:
    event_id: str
    striker_entity_id: str
    joint_name: str
    target_position_m: float
    force_n: float


_ActuatorCommand: TypeAlias = _ReleaseGateCommand | _StrikerCommand


@dataclass(frozen=True, slots=True)
class _StrikerPulseSnapshot:
    event_id: str
    trigger_time_s: float
    contact_cutoff_reached: bool
    impact_group_count: int
    impact_active: bool
    group_contact_duration_s: float
    separation_duration_s: float
    maximum_contact_duration_exceeded: bool


@dataclass(frozen=True, slots=True)
class _EventStepResult:
    fired_event_ids: tuple[str, ...]
    event_flags: tuple[bool, ...]
    actuator_commands: tuple[_ActuatorCommand, ...]
    striker_states: tuple[_StrikerPulseSnapshot, ...]


@dataclass(slots=True)
class _StrikerRuntimeState:
    rest_position_m: float
    trigger_time_s: float | None = None
    contact_cutoff_reached: bool = False
    continuous_contact_duration_s: float = 0.0
    impact_group_count: int = 0
    impact_active: bool = False
    group_contact_duration_s: float = 0.0
    separation_duration_s: float = 0.0
    maximum_contact_duration_exceeded: bool = False


def _joint_key(entity_id: str, joint_name: str) -> tuple[str, str]:
    return entity_id, joint_name


def _contact_key(entity_a_id: str, entity_b_id: str) -> tuple[str, str]:
    return (entity_a_id, entity_b_id) if entity_a_id <= entity_b_id else (entity_b_id, entity_a_id)


def _requested_contact_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise ValueError("requested_entity_pairs must be a tuple")
    result: list[tuple[str, str]] = []
    identities: set[tuple[str, str]] = set()
    for pair in value:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("each requested entity pair must be a two-item tuple")
        entity_a_id = _identity(pair[0], "requested pair entity_a_id")
        entity_b_id = _identity(pair[1], "requested pair entity_b_id")
        if entity_a_id == entity_b_id:
            raise ValueError("requested contact pair entities must differ")
        key = _contact_key(entity_a_id, entity_b_id)
        if key in identities:
            raise ValueError("requested entity pairs must be unique as unordered pairs")
        identities.add(key)
        result.append((entity_a_id, entity_b_id))
    return tuple(result)


def _contact_geom_owners(
    model: mujoco.MjModel,
    requested_pairs: tuple[tuple[str, str], ...],
    entity_geom_ids: Mapping[str, tuple[int, ...]],
) -> dict[int, str]:
    if not isinstance(entity_geom_ids, Mapping) or not all(
        isinstance(entity_id, str) for entity_id in entity_geom_ids
    ):
        raise ValueError("entity_geom_ids must be a string-keyed mapping")
    requested_entities = {entity_id for pair in requested_pairs for entity_id in pair}
    actual_entities = set(entity_geom_ids)
    missing = requested_entities - actual_entities
    unknown = actual_entities - requested_entities
    if missing:
        raise ValueError(f"entity_geom_ids is missing requested entities: {sorted(missing)}")
    if unknown:
        raise ValueError(f"entity_geom_ids contains unrequested entities: {sorted(unknown)}")

    owners: dict[int, str] = {}
    for entity_id, raw_geom_ids in entity_geom_ids.items():
        _identity(entity_id, "entity_geom_ids entity_id")
        if not isinstance(raw_geom_ids, tuple) or not raw_geom_ids:
            raise ValueError(f"geom IDs for {entity_id!r} must be a non-empty tuple")
        if any(
            isinstance(geom_id, bool)
            or not isinstance(geom_id, int)
            or not 0 <= geom_id < model.ngeom
            for geom_id in raw_geom_ids
        ):
            raise ValueError(f"geom IDs for {entity_id!r} must be valid compiled geom IDs")
        if len(set(raw_geom_ids)) != len(raw_geom_ids):
            raise ValueError(f"geom IDs for {entity_id!r} must not contain duplicates")
        for geom_id in raw_geom_ids:
            previous_owner = owners.get(geom_id)
            if previous_owner is not None:
                raise ValueError(
                    f"compiled geom {geom_id} has conflicting entity ownership: "
                    f"{previous_owner!r} and {entity_id!r}"
                )
            owners[geom_id] = entity_id
    return owners


def _mujoco_contact_samples(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    requested_entity_pairs: tuple[tuple[str, str], ...],
    entity_geom_ids: Mapping[str, tuple[int, ...]],
) -> tuple[_ContactSample, ...]:
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise ValueError("model and data must be MuJoCo MjModel and MjData")
    if data.model is not model:
        raise ValueError("MjData must belong to the supplied MjModel")
    requested_pairs = _requested_contact_pairs(requested_entity_pairs)
    geom_owners = _contact_geom_owners(model, requested_pairs, entity_geom_ids)
    requested_keys = {_contact_key(*pair) for pair in requested_pairs}
    aggregated = {key: 0.0 for key in requested_keys}
    active_keys: set[tuple[str, str]] = set()
    force_torque = np.empty(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        entity_a_id = geom_owners.get(int(contact.geom1))
        entity_b_id = geom_owners.get(int(contact.geom2))
        if entity_a_id is None or entity_b_id is None or entity_a_id == entity_b_id:
            continue
        key = _contact_key(entity_a_id, entity_b_id)
        if key not in requested_keys:
            continue
        mujoco.mj_contactForce(model, data, contact_index, force_torque)
        normal_force_n = float(force_torque[0])
        if not math.isfinite(normal_force_n) or normal_force_n < 0.0:
            raise ValueError("MuJoCo contact normal force must be finite and non-negative")
        total_force_n = aggregated[key] + normal_force_n
        if not math.isfinite(total_force_n):
            raise ValueError("aggregated contact normal force must be finite")
        aggregated[key] = total_force_n
        active_keys.add(key)

    return tuple(
        _ContactSample(entity_a_id, entity_b_id, aggregated[_contact_key(*pair)])
        for pair in requested_pairs
        for entity_a_id, entity_b_id in (pair,)
        if _contact_key(*pair) in active_keys
    )


class _EventRuntime:
    """Physics-rate execution state for one episode's frozen EventSpec tuple."""

    def __init__(
        self,
        events: tuple[EventSpec, ...],
        *,
        state_trigger_target_entity_id: str | None = None,
        observation_boundary_y_m: float | None = None,
    ) -> None:
        if not isinstance(events, tuple) or not all(
            isinstance(event, EventSpec) for event in events
        ):
            raise ValueError("events must be a tuple of EventSpec")
        event_ids = tuple(event.event_id for event in events)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("event IDs must be unique")
        needs_state_context = any(
            isinstance(event.trigger, StateTriggerSpec)
            or isinstance(event.effect, StrikerPulseEffectSpec)
            for event in events
        )
        self._target_entity_id: str | None
        self._observation_boundary_y_m: float | None
        if needs_state_context:
            self._target_entity_id = _identity(
                state_trigger_target_entity_id, "state trigger target entity_id"
            )
            self._observation_boundary_y_m = _finite(
                observation_boundary_y_m, "observation_boundary_y_m"
            )
        elif state_trigger_target_entity_id is not None or observation_boundary_y_m is not None:
            raise ValueError("state trigger context is only valid for state/striker events")
        else:
            self._target_entity_id = None
            self._observation_boundary_y_m = None
        contact_pairs: list[tuple[str, str]] = []
        contact_pair_keys: set[tuple[str, str]] = set()
        for event in events:
            candidates: list[tuple[str, str]] = []
            if isinstance(event.trigger, ContactTriggerSpec):
                candidates.append((event.trigger.entity_a_id, event.trigger.entity_b_id))
            if isinstance(event.effect, StrikerPulseEffectSpec):
                assert self._target_entity_id is not None
                candidates.append((event.effect.striker_entity_id, self._target_entity_id))
            for pair in candidates:
                key = _contact_key(*pair)
                if key not in contact_pair_keys:
                    contact_pair_keys.add(key)
                    contact_pairs.append(pair)
        self._events = events
        self._contact_pairs = tuple(contact_pairs)
        self._fired = [False] * len(events)
        self._striker_states: list[_StrikerRuntimeState | None] = [None] * len(events)
        self._previous_target_y_m: float | None = None
        self._last_simulator_time_s: float | None = None
        self._last_control_tick: int | None = None
        self._has_stepped = False
        self._is_reset = False

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self._events)

    @property
    def contact_pairs(self) -> tuple[tuple[str, str], ...]:
        return self._contact_pairs

    def reset(
        self,
        *,
        simulator_time_s: float,
        joint_states: Mapping[tuple[str, str], _JointState],
        target_body_y_m: float | None = None,
    ) -> None:
        time_s = _finite(simulator_time_s, "simulator_time_s")
        if time_s < 0.0:
            raise ValueError("simulator_time_s must be non-negative")
        if not isinstance(joint_states, Mapping):
            raise ValueError("joint_states must be a mapping")
        if self._target_entity_id is not None:
            target_y = _finite(target_body_y_m, "target_body_y_m")
        elif target_body_y_m is not None:
            raise ValueError("target_body_y_m is only valid for state/striker events")
        else:
            target_y = None

        striker_states: list[_StrikerRuntimeState | None] = []
        for event in self._events:
            if not isinstance(event.effect, StrikerPulseEffectSpec):
                striker_states.append(None)
                continue
            key = _joint_key(event.effect.striker_entity_id, event.effect.joint_name)
            try:
                joint_state = joint_states[key]
            except KeyError:
                raise ValueError(f"joint_states is missing selected striker joint {key}") from None
            if not isinstance(joint_state, _JointState):
                raise ValueError(f"joint_states entry {key} must be _JointState")
            striker_states.append(_StrikerRuntimeState(joint_state.position_m))

        self._fired = [False] * len(self._events)
        self._striker_states = striker_states
        self._previous_target_y_m = target_y
        self._last_simulator_time_s = time_s
        self._last_control_tick = None
        self._has_stepped = False
        self._is_reset = True

    def _contact_lookup(self, contacts: tuple[_ContactSample, ...]) -> dict[tuple[str, str], float]:
        if not isinstance(contacts, tuple) or not all(
            isinstance(contact, _ContactSample) for contact in contacts
        ):
            raise ValueError("contacts must be a tuple of _ContactSample")
        result: dict[tuple[str, str], float] = {}
        for contact in contacts:
            key = _contact_key(contact.entity_a_id, contact.entity_b_id)
            if key in result:
                raise ValueError("contacts must contain one aggregated sample per entity pair")
            result[key] = contact.normal_force_n
        return result

    def _triggered(
        self,
        event: EventSpec,
        *,
        control_tick: int,
        contact_forces: Mapping[tuple[str, str], float],
        target_body_y_m: float | None,
    ) -> bool:
        trigger = event.trigger
        if isinstance(trigger, TimeTriggerSpec):
            return control_tick == trigger.control_tick
        if isinstance(trigger, ContactTriggerSpec):
            key = _contact_key(trigger.entity_a_id, trigger.entity_b_id)
            force = contact_forces.get(key)
            return force is not None and force >= trigger.minimum_normal_force_n
        assert isinstance(trigger, StateTriggerSpec)
        assert self._previous_target_y_m is not None
        assert self._observation_boundary_y_m is not None
        assert target_body_y_m is not None
        return self._previous_target_y_m < self._observation_boundary_y_m <= target_body_y_m

    @staticmethod
    def _update_striker_contact(
        state: _StrikerRuntimeState,
        effect: StrikerPulseEffectSpec,
        *,
        selected_contact: bool,
        elapsed_s: float,
    ) -> None:
        if selected_contact:
            state.continuous_contact_duration_s += elapsed_s
            if not state.impact_active:
                state.impact_active = True
                state.impact_group_count += 1
                state.group_contact_duration_s = 0.0
            state.group_contact_duration_s += elapsed_s
            state.separation_duration_s = 0.0
            if (
                state.continuous_contact_duration_s + _TIME_EPSILON_S
                >= effect.contact_cutoff_delay_s
            ):
                state.contact_cutoff_reached = True
            if state.group_contact_duration_s > effect.maximum_contact_duration_s + _TIME_EPSILON_S:
                state.maximum_contact_duration_exceeded = True
            return

        if not state.contact_cutoff_reached:
            state.continuous_contact_duration_s = 0.0
        if state.impact_active:
            state.separation_duration_s += elapsed_s
            if (
                state.separation_duration_s + _TIME_EPSILON_S
                >= effect.contact_grouping_hysteresis_s
            ):
                state.impact_active = False
        else:
            state.separation_duration_s = 0.0

    def _striker_command(
        self,
        event: EventSpec,
        effect: StrikerPulseEffectSpec,
        state: _StrikerRuntimeState,
        *,
        simulator_time_s: float,
        elapsed_s: float,
        contact_forces: Mapping[tuple[str, str], float],
        joint_states: Mapping[tuple[str, str], _JointState],
    ) -> tuple[_StrikerCommand, _StrikerPulseSnapshot]:
        assert state.trigger_time_s is not None
        assert self._target_entity_id is not None
        contact_key = _contact_key(effect.striker_entity_id, self._target_entity_id)
        self._update_striker_contact(
            state,
            effect,
            selected_contact=contact_key in contact_forces,
            elapsed_s=elapsed_s,
        )
        joint_key = _joint_key(effect.striker_entity_id, effect.joint_name)
        try:
            joint_state = joint_states[joint_key]
        except KeyError:
            raise ValueError(
                f"joint_states is missing selected striker joint {joint_key}"
            ) from None
        if not isinstance(joint_state, _JointState):
            raise ValueError(f"joint_states entry {joint_key} must be _JointState")
        pulse_elapsed_s = simulator_time_s - state.trigger_time_s
        within_pulse = pulse_elapsed_s <= effect.pulse_duration_s
        target = (
            effect.strike_position_m
            if within_pulse and not state.contact_cutoff_reached
            else state.rest_position_m
        )
        force = (
            effect.kp_n_m * (target - joint_state.position_m)
            - effect.kd_n_s_m * joint_state.velocity_m_s
        )
        force *= effect.actuator_scale
        bounded_force = min(max(force, -effect.maximum_force_n), effect.maximum_force_n)
        return (
            _StrikerCommand(
                event_id=event.event_id,
                striker_entity_id=effect.striker_entity_id,
                joint_name=effect.joint_name,
                target_position_m=target,
                force_n=bounded_force,
            ),
            _StrikerPulseSnapshot(
                event_id=event.event_id,
                trigger_time_s=state.trigger_time_s,
                contact_cutoff_reached=state.contact_cutoff_reached,
                impact_group_count=state.impact_group_count,
                impact_active=state.impact_active,
                group_contact_duration_s=state.group_contact_duration_s,
                separation_duration_s=state.separation_duration_s,
                maximum_contact_duration_exceeded=(state.maximum_contact_duration_exceeded),
            ),
        )

    def step(
        self,
        *,
        simulator_time_s: float,
        control_tick: int,
        contacts: tuple[_ContactSample, ...],
        joint_states: Mapping[tuple[str, str], _JointState],
        target_body_y_m: float | None = None,
    ) -> _EventStepResult:
        if not self._is_reset or self._last_simulator_time_s is None:
            raise RuntimeError("event runtime must be reset before step")
        time_s = _finite(simulator_time_s, "simulator_time_s")
        if time_s < 0.0:
            raise ValueError("simulator_time_s must be non-negative")
        elapsed_s = time_s - self._last_simulator_time_s
        if elapsed_s < 0.0 or (self._has_stepped and elapsed_s <= 0.0):
            raise ValueError("simulator_time_s must strictly increase after the first step")
        if isinstance(control_tick, bool) or not isinstance(control_tick, int) or control_tick < 0:
            raise ValueError("control_tick must be a non-negative integer")
        if self._last_control_tick is None:
            if control_tick != 0:
                raise ValueError("the first control_tick after reset must be zero")
        elif not self._last_control_tick <= control_tick <= self._last_control_tick + 1:
            raise ValueError("control_tick must stay constant or increase by exactly one")
        if not isinstance(joint_states, Mapping):
            raise ValueError("joint_states must be a mapping")
        contact_forces = self._contact_lookup(contacts)
        if self._target_entity_id is not None:
            target_y = _finite(target_body_y_m, "target_body_y_m")
        elif target_body_y_m is not None:
            raise ValueError("target_body_y_m is only valid for state/striker events")
        else:
            target_y = None

        fired_event_ids: list[str] = []
        commands: list[_ActuatorCommand] = []
        striker_snapshots: list[_StrikerPulseSnapshot] = []
        for index, event in enumerate(self._events):
            if not self._fired[index] and self._triggered(
                event,
                control_tick=control_tick,
                contact_forces=contact_forces,
                target_body_y_m=target_y,
            ):
                self._fired[index] = True
                fired_event_ids.append(event.event_id)
                striker_state = self._striker_states[index]
                if striker_state is not None:
                    striker_state.trigger_time_s = time_s
            if not self._fired[index]:
                continue
            effect = event.effect
            if isinstance(effect, ReleaseGateRetractionEffectSpec):
                if event.event_id not in fired_event_ids:
                    continue
                commands.append(
                    _ReleaseGateCommand(
                        event_id=event.event_id,
                        gate_entity_id=effect.gate_entity_id,
                        parked_position_world_m=effect.parked_position_world_m,
                    )
                )
                continue
            assert isinstance(effect, StrikerPulseEffectSpec)
            striker_state = self._striker_states[index]
            assert striker_state is not None
            command, snapshot = self._striker_command(
                event,
                effect,
                striker_state,
                simulator_time_s=time_s,
                elapsed_s=elapsed_s,
                contact_forces=contact_forces,
                joint_states=joint_states,
            )
            commands.append(command)
            striker_snapshots.append(snapshot)

        self._previous_target_y_m = target_y
        self._last_simulator_time_s = time_s
        self._last_control_tick = control_tick
        self._has_stepped = True
        return _EventStepResult(
            fired_event_ids=tuple(fired_event_ids),
            event_flags=tuple(self._fired),
            actuator_commands=tuple(commands),
            striker_states=tuple(striker_snapshots),
        )


__all__: list[str] = []
