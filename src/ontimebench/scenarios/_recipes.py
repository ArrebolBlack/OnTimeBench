"""Recipe validation and atomic sampling into concrete scenario proposals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import mujoco
import numpy as np
import numpy.typing as npt

from ontimebench.envs.components.motion import (
    _axis_angle_quaternion,
    _motion_state_at_time,
    _quaternion_product_xyzw,
    _validate_motion_initial_pose,
)
from ontimebench.envs.factory import _validate_task_scenario
from ontimebench.scenarios.spec import (
    CameraSpec,
    CircularTranslation,
    ConstantAngularVelocityRotation,
    ConstantVelocityTranslation,
    HarmonicRotation,
    HarmonicTranslation,
    ScenarioSpec,
    _parse_motion,
    scenario_from_dict,
)
from ontimebench.tasks.spec import (
    DrivenInterceptionBindings,
    FreeDynamicsInterceptionBindings,
    GraspBindings,
    GraspMotion,
    MovingPlacementBindings,
    ReplanningBindings,
    RotatingInteractionBindings,
    TaskSpec,
)

_GRASP_ASSET_IDS = {
    "obj.cuboid_grasp_v1",
    "obj.asym_t_t1_v1",
}

_GRASP_CANONICAL_LIGHT = {
    "light_id": "table_key",
    "position_world_m": (1.0, 1.0, 1.5),
    "direction_world": (
        -0.19245008972987526,
        -0.19245008972987526,
        -0.9622504486493763,
    ),
    "ambient_rgb": (0.0, 0.0, 0.0),
    "diffuse_rgb": (0.7, 0.7, 0.7),
    "specular_rgb": (0.3, 0.3, 0.3),
    "attenuation": (1.0, 0.0, 0.0),
    "cutoff_deg": 45.0,
    "active": True,
}

_T4_ASSETS = {
    "ball_capture": ("arena.unsupported_ramp_v1", "obj.sphere_r030_v1"),
    "ball_capture_on_table": ("arena.supported_ramp_v1", "obj.sphere_r030_v1"),
    "slider_capture": ("arena.supported_ramp_v1", "obj.slider_v1"),
    "can_capture": ("arena.supported_ramp_v1", "obj.can_v1"),
}

_T4_TARGET_SUPPORT_EXTENT_M = {
    "ball_capture": 0.030,
    "ball_capture_on_table": 0.030,
    "slider_capture": 0.020,
    "can_capture": 0.024,
}

_RECIPE_FIELD_ORDER = (
    "action_mode",
    "observation_profile",
    "physics_timestep_s",
    "control_frequency_hz",
    "sensor_frequency_hz",
    "action_delay",
    "robot_initial_state",
    "object_states",
    "fixture_states",
    "geometry",
    "motion",
    "physics",
    "cameras",
    "lighting",
    "events",
)


_RECIPE_FIELDS = set(_RECIPE_FIELD_ORDER)


_ACTION_DELAY_FIELDS = {"enabled", "delay_steps"}


_SAMPLE_KEYS = {"min", "max", "choices", "uniform_intervals"}


_OBJECT_STATE_FIELDS = {
    "instance_id",
    "asset_id",
    "position_world_m",
    "quaternion_world_xyzw",
    "linear_velocity_world_m_s",
    "angular_velocity_world_rad_s",
}


_MOTION_FIELDS = {"entity_id", "translation", "rotation"}


def _exact_fields(value: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} unknown fields: {sorted(unknown)}")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _sample_range(value: Mapping[str, object], rng: np.random.Generator) -> float | list[float]:
    lower_value = value["min"]
    upper_value = value["max"]
    if isinstance(lower_value, Sequence) and not isinstance(lower_value, (str, bytes)):
        if not isinstance(upper_value, Sequence) or isinstance(upper_value, (str, bytes)):
            raise ValueError("range min and max must have matching scalar or vector shapes")
        lower = np.asarray([_number(item, "range min") for item in lower_value])
        upper = np.asarray([_number(item, "range max") for item in upper_value])
        if lower.shape != upper.shape or lower.ndim != 1 or lower.size == 0:
            raise ValueError("range min and max vectors must be non-empty and equal-length")
        if np.any(lower >= upper):
            raise ValueError("every vector range min must be strictly less than max")
        return cast(list[float], rng.uniform(lower, upper).tolist())
    lower_scalar = _number(lower_value, "range min")
    upper_scalar = _number(upper_value, "range max")
    if lower_scalar >= upper_scalar:
        raise ValueError("range min must be strictly less than max")
    return float(rng.uniform(lower_scalar, upper_scalar))


def _sample_recipe_value(value: object, rng: np.random.Generator) -> object:
    if isinstance(value, Mapping):
        mapping = _mapping(value, "recipe value")
        keys = set(mapping)
        if keys == {"degrees"}:
            return math.radians(_sample_recipe_number(mapping["degrees"], rng, "angle degrees"))
        if keys == {"yaw_rad"}:
            yaw = _sample_recipe_number(mapping["yaw_rad"], rng, "initial yaw")
            return [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]
        if keys & _SAMPLE_KEYS:
            if keys == {"min", "max"}:
                return _sample_range(mapping, rng)
            if keys == {"choices"}:
                choices = mapping["choices"]
                if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
                    raise ValueError("choices must be a sequence")
                if len(choices) < 2:
                    raise ValueError("choices must contain at least two values")
                selected = choices[int(rng.integers(0, len(choices)))]
                return _sample_recipe_value(selected, rng)
            if keys == {"uniform_intervals"}:
                intervals = _uniform_intervals(mapping["uniform_intervals"], "uniform intervals")
                lengths = intervals[:, 1] - intervals[:, 0]
                index = int(rng.choice(len(intervals), p=lengths / lengths.sum()))
                return float(rng.uniform(intervals[index, 0], intervals[index, 1]))
            raise ValueError("sampling records must be exactly min/max or choices")
        return {key: _sample_recipe_value(item, rng) for key, item in mapping.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_sample_recipe_value(item, rng) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("recipe values must be finite")
        return value
    raise ValueError(f"unsupported recipe value type: {type(value).__name__}")


def _recipe_sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _sample_recipe_number(value: object, rng: np.random.Generator, name: str) -> float:
    return _number(_sample_recipe_value(value, rng), name)


def _sample_recipe_vector(
    value: object, rng: np.random.Generator, name: str
) -> npt.NDArray[np.float64]:
    sampled = _sample_recipe_value(value, rng)
    vector = _recipe_sequence(sampled, name)
    if len(vector) != 3:
        raise ValueError(f"{name} must have length 3")
    return np.asarray(
        [_number(component, f"{name} component") for component in vector],
        dtype=np.float64,
    )


def _vector3_tuple(value: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _vector4_tuple(
    value: npt.NDArray[np.float64],
) -> tuple[float, float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _choice_values(value: Mapping[str, object], name: str) -> Sequence[object]:
    _exact_fields(value, {"choices"}, name)
    choices = _recipe_sequence(value["choices"], f"{name}.choices")
    if len(choices) < 2:
        raise ValueError(f"{name}.choices must contain at least two values")
    return choices


def _uniform_intervals(value: object, name: str) -> npt.NDArray[np.float64]:
    raw = _recipe_sequence(value, name)
    if len(raw) < 2:
        raise ValueError(f"{name} must contain at least two disjoint intervals")
    intervals: list[tuple[float, float]] = []
    for item in raw:
        pair = _recipe_sequence(item, name)
        if len(pair) != 2:
            raise ValueError(f"{name} intervals must contain lower and upper bounds")
        lower, upper = (_number(component, name) for component in pair)
        if lower >= upper or (intervals and lower <= intervals[-1][1]):
            raise ValueError(f"{name} intervals must be ordered, disjoint and non-empty")
        intervals.append((lower, upper))
    return np.asarray(intervals, dtype=np.float64)


def _radians_recipe(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _radians_recipe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_radians_recipe(item) for item in value]
    return math.radians(_number(value, "angle degrees"))


def _validate_number_recipe(
    value: object,
    name: str,
    *,
    domain: Literal[
        "finite",
        "nonnegative",
        "positive",
        "zero",
        "nonzero",
        "unit_interval",
        "camera_fov",
        "light_cutoff",
    ] = "finite",
) -> None:
    if isinstance(value, Mapping):
        raw = _mapping(value, name)
        if set(raw) == {"degrees"}:
            _validate_number_recipe(raw["degrees"], name, domain=domain)
            return
        if set(raw) == {"min", "max"}:
            lower = _number(raw["min"], f"{name}.min")
            upper = _number(raw["max"], f"{name}.max")
            if lower >= upper:
                raise ValueError(f"{name} range min must be strictly less than max")
            if domain == "nonzero" and lower <= 0.0 <= upper:
                raise ValueError(f"{name} support must exclude zero")
            candidates: Sequence[object] = (lower, upper)
        elif set(raw) == {"choices"}:
            candidates = _choice_values(raw, name)
        elif set(raw) == {"uniform_intervals"}:
            intervals = _uniform_intervals(raw["uniform_intervals"], name)
            for lower, upper in intervals:
                _validate_number_recipe(
                    {"min": float(lower), "max": float(upper)}, name, domain=domain
                )
            return
        else:
            raise ValueError("sampling records must be exactly min/max or choices")
        for candidate in candidates:
            _validate_number_recipe(candidate, name, domain=domain)
        return
    number = _number(value, name)
    if domain == "nonnegative" and number < 0.0:
        raise ValueError(f"{name} must be non-negative throughout its support")
    if domain == "positive" and number <= 0.0:
        raise ValueError(f"{name} must be positive throughout its support")
    if domain == "zero" and number != 0.0:
        raise ValueError(f"{name} must be zero throughout its support")
    if domain == "nonzero" and number == 0.0:
        raise ValueError(f"{name} support must exclude zero")
    if domain == "unit_interval" and not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must remain in [0, 1] throughout its support")
    if domain == "camera_fov" and not 0.0 < number < 180.0:
        raise ValueError(f"{name} must remain in (0, 180) throughout its support")
    if domain == "light_cutoff" and not 0.0 < number <= 180.0:
        raise ValueError(f"{name} must remain in (0, 180] throughout its support")


def _validate_vector_recipe(
    value: object,
    name: str,
    *,
    length: int = 3,
    domain: Literal["finite", "nonnegative", "positive", "zero", "unit_interval"] = "finite",
) -> None:
    if isinstance(value, Mapping):
        raw = _mapping(value, name)
        if set(raw) == {"min", "max"}:
            lower = _recipe_sequence(raw["min"], f"{name}.min")
            upper = _recipe_sequence(raw["max"], f"{name}.max")
            if len(lower) != length or len(upper) != length:
                raise ValueError(f"{name} range vectors must have length {length}")
            for low, high in zip(lower, upper, strict=True):
                _validate_number_recipe({"min": low, "max": high}, name, domain=domain)
        elif set(raw) == {"choices"}:
            for candidate in _choice_values(raw, name):
                _validate_vector_recipe(candidate, name, length=length, domain=domain)
        else:
            raise ValueError("sampling records must be exactly min/max or choices")
        return
    vector = _recipe_sequence(value, name)
    if len(vector) != length:
        raise ValueError(f"{name} must have length {length}")
    for component in vector:
        _validate_number_recipe(component, name, domain=domain)


def _unit_vector_recipe_values(value: object, name: str) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, Mapping):
        return tuple(
            vector
            for choice in _choice_values(_mapping(value, name), name)
            for vector in _unit_vector_recipe_values(choice, name)
        )
    raw = _recipe_sequence(value, name)
    if len(raw) != 3:
        raise ValueError(f"{name} must have length 3")
    vector = (_number(raw[0], name), _number(raw[1], name), _number(raw[2], name))
    ConstantAngularVelocityRotation(vector, 0.0)
    return (vector,)


def _record_recipe_choices(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    raw = _mapping(value, name)
    if set(raw) == {"choices"}:
        return tuple(_mapping(choice, name) for choice in _choice_values(raw, name))
    if set(raw) & _SAMPLE_KEYS:
        raise ValueError(f"{name} record selector must be exactly choices")
    return (raw,)


def _choose_record(
    choices: tuple[Mapping[str, object], ...], rng: np.random.Generator
) -> Mapping[str, object]:
    return choices[0] if len(choices) == 1 else choices[int(rng.integers(0, len(choices)))]


def _validate_grasp_asset_recipe(value: object) -> None:
    if isinstance(value, Mapping):
        for candidate in _choice_values(_mapping(value, "grasp asset_id"), "grasp asset_id"):
            _validate_grasp_asset_recipe(candidate)
    elif not isinstance(value, str) or value not in _GRASP_ASSET_IDS:
        raise ValueError(f"unknown grasp asset_id in recipe: {value}")


def _validate_initial_position_recipe(value: object, name: str) -> None:
    if isinstance(value, Mapping) and "harmonic_center_world_m" in value:
        center = _mapping(value, name)
        _exact_fields(center, {"harmonic_center_world_m"}, name)
        _validate_vector_recipe(center["harmonic_center_world_m"], name)
    else:
        _validate_vector_recipe(value, name)


def _validate_initial_velocity_recipe(value: object, name: str) -> None:
    if isinstance(value, Mapping) and "from_motion" in value:
        marker = _mapping(value, name)
        _exact_fields(marker, {"from_motion"}, name)
        if marker["from_motion"] is not True:
            raise ValueError(f"{name}.from_motion must be true")
    else:
        _validate_vector_recipe(value, name)


def _validate_grasp_object_recipe(value: Mapping[str, object]) -> None:
    _exact_fields(value, _OBJECT_STATE_FIELDS, "grasp target object_state")
    if not isinstance(value["instance_id"], str) or not value["instance_id"]:
        raise ValueError("grasp object_state requires a concrete instance_id")
    _validate_grasp_asset_recipe(value["asset_id"])
    _validate_initial_position_recipe(value["position_world_m"], "grasp target position_world_m")
    yaw = _mapping(value["quaternion_world_xyzw"], "grasp target quaternion_world_xyzw")
    _exact_fields(yaw, {"yaw_rad"}, "grasp target quaternion_world_xyzw")
    _validate_number_recipe(yaw["yaw_rad"], "grasp target yaw_rad")
    _validate_initial_velocity_recipe(
        value["linear_velocity_world_m_s"], "grasp initial linear velocity"
    )
    _validate_initial_velocity_recipe(
        value["angular_velocity_world_rad_s"], "grasp initial angular velocity"
    )


def _validate_grasp_translation_recipe(value: object) -> Mapping[str, object]:
    translation = _mapping(value, "grasp target translation")
    kind = translation.get("type")
    if kind == "fixed":
        _exact_fields(translation, {"type"}, "grasp target translation")
    elif kind == "constant_velocity":
        _exact_fields(
            translation,
            {"type", "velocity_world_m_s"},
            "grasp target translation",
        )
        _validate_vector_recipe(translation["velocity_world_m_s"], "grasp constant velocity")
    elif kind == "harmonic":
        _exact_fields(
            translation,
            {"type", "drift_velocity_world_m_s", "amplitude_m", "frequency_hz", "phase_rad"},
            "grasp target translation",
        )
        _validate_vector_recipe(
            translation["drift_velocity_world_m_s"],
            "grasp harmonic drift_velocity_world_m_s",
        )
        _validate_vector_recipe(
            translation["amplitude_m"], "grasp harmonic amplitude", domain="nonnegative"
        )
        _validate_vector_recipe(
            translation["frequency_hz"], "grasp harmonic frequency", domain="nonnegative"
        )
        _validate_vector_recipe(translation["phase_rad"], "grasp harmonic phase")
    elif kind == "circular":
        _exact_fields(
            translation,
            {
                "type",
                "axis_world",
                "radial_direction_world",
                "radius_m",
                "angular_speed_rad_s",
                "phase_rad",
            },
            "grasp target translation",
        )
        axes = _unit_vector_recipe_values(translation["axis_world"], "grasp circular axis")
        radials = _unit_vector_recipe_values(
            translation["radial_direction_world"], "grasp circular radial direction"
        )
        for axis in axes:
            for radial in radials:
                CircularTranslation((0.0, 0.0, 0.0), axis, radial, 1.0, 0.0, 0.0)
        _validate_number_recipe(
            translation["radius_m"], "grasp circular radius_m", domain="positive"
        )
        _validate_number_recipe(translation["angular_speed_rad_s"], "grasp circular angular speed")
        _validate_number_recipe(translation["phase_rad"], "grasp circular phase")
    else:
        raise ValueError(f"unknown grasp target translation type: {kind}")
    return translation


def _validate_grasp_rotation_recipe(value: object) -> Mapping[str, object]:
    rotation = _mapping(value, "grasp target rotation")
    kind = rotation.get("type")
    if kind == "fixed":
        _exact_fields(rotation, {"type"}, "grasp target rotation")
    elif kind == "constant_angular_velocity":
        _exact_fields(
            rotation,
            {"type", "axis_world", "angular_speed_rad_s"},
            "grasp target rotation",
        )
        _unit_vector_recipe_values(rotation["axis_world"], "grasp constant angular velocity axis")
        _validate_number_recipe(rotation["angular_speed_rad_s"], "grasp constant angular speed")
    elif kind == "harmonic":
        _exact_fields(
            rotation,
            {"type", "axis_world", "amplitude_rad", "frequency_hz", "phase_rad"},
            "grasp target rotation",
        )
        _unit_vector_recipe_values(rotation["axis_world"], "grasp harmonic rotation axis")
        _validate_number_recipe(
            rotation["amplitude_rad"], "grasp harmonic rotation amplitude", domain="nonnegative"
        )
        _validate_number_recipe(
            rotation["frequency_hz"], "grasp harmonic rotation frequency", domain="nonnegative"
        )
        _validate_number_recipe(rotation["phase_rad"], "grasp harmonic rotation phase")
    else:
        raise ValueError(f"unknown grasp target rotation type: {kind}")
    return rotation


def _select_grasp_law_recipe(
    value: object,
    rng: np.random.Generator,
    *,
    name: str,
    translation: bool,
) -> Mapping[str, object]:
    choices = _record_recipe_choices(value, name)
    for choice in choices:
        if translation:
            _validate_grasp_translation_recipe(choice)
        else:
            _validate_grasp_rotation_recipe(choice)
    return _choose_record(choices, rng)


def _validate_grasp_motion_recipe(value: Mapping[str, object]) -> None:
    _exact_fields(value, _MOTION_FIELDS, "grasp target motion")
    if not isinstance(value["entity_id"], str) or not value["entity_id"]:
        raise ValueError("grasp motion requires a concrete entity_id")
    for translation in _record_recipe_choices(value["translation"], "grasp target translation"):
        _validate_grasp_translation_recipe(translation)
    for rotation in _record_recipe_choices(value["rotation"], "grasp target rotation"):
        _validate_grasp_rotation_recipe(rotation)


def _validate_body_physics_recipes(
    value: object, *, assets_by_entity: Mapping[str, frozenset[str]] | None = None
) -> None:
    physics = _mapping(value, "physics")
    _exact_fields(physics, {"gravity_world_m_s2", "bodies", "contacts", "joints"}, "physics")
    for entry in _recipe_sequence(physics["bodies"], "body physics"):
        choices = _record_recipe_choices(entry, "body physics")
        tagged = any("asset_id" in body for body in choices)
        allowed = {"entity_id", "mass_kg", "diagonal_inertia_kg_m2"}
        if tagged:
            allowed.add("asset_id")
        associations: set[tuple[str, str]] = set()
        for body in choices:
            _exact_fields(body, allowed, "body physics")
            if not isinstance(body["entity_id"], str) or not body["entity_id"]:
                raise ValueError("body physics requires a concrete entity_id")
            if tagged:
                entity, asset = body["entity_id"], body["asset_id"]
                if (
                    assets_by_entity is None
                    or not isinstance(asset, str)
                    or entity not in assets_by_entity
                    or asset not in assets_by_entity[entity]
                    or (entity, asset) in associations
                ):
                    raise ValueError("body asset association must name one unique declared asset")
                associations.add((entity, asset))
            _validate_number_recipe(body["mass_kg"], "mass_kg", domain="positive")
            _validate_vector_recipe(
                body["diagonal_inertia_kg_m2"], "diagonal inertia", domain="positive"
            )
        if tagged:
            assert assets_by_entity is not None
            entities = {entity for entity, _ in associations}
            if len(entities) != 1 or associations != {
                (entity, asset) for entity in entities for asset in assets_by_entity[entity]
            }:
                raise ValueError("body asset association must cover every asset of one entity")


_JOINT_PHYSICS_FIELDS = {
    "entity_id",
    "joint_name",
    "damping_per_dof",
    "friction_loss_per_dof",
}


def _validate_joint_physics_recipes(
    value: object,
    *,
    allowed_dofs: Mapping[tuple[str, str], int],
    required: frozenset[tuple[str, str]] = frozenset(),
    name: str,
) -> None:
    keys: list[tuple[str, str]] = []
    for entry in _recipe_sequence(value, name):
        alternatives = _record_recipe_choices(entry, name)
        first = alternatives[0]
        _exact_fields(first, _JOINT_PHYSICS_FIELDS, name)
        entity_id = first["entity_id"]
        joint_name = first["joint_name"]
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError(f"{name} requires a concrete entity_id")
        if not isinstance(joint_name, str) or not joint_name:
            raise ValueError(f"{name} requires a concrete joint_name")
        first_key = (entity_id, joint_name)
        if first_key not in allowed_dofs:
            raise ValueError(f"{name} contains an unknown entity/joint binding")
        keys.append(first_key)
        for joint in alternatives:
            _exact_fields(joint, _JOINT_PHYSICS_FIELDS, name)
            key = (joint["entity_id"], joint["joint_name"])
            if key != first_key:
                raise ValueError(f"{name} alternatives must preserve entity/joint binding")
            dof_count = allowed_dofs[first_key]
            _validate_vector_recipe(
                joint["damping_per_dof"],
                f"{name} damping",
                length=dof_count,
                domain="nonnegative",
            )
            _validate_vector_recipe(
                joint["friction_loss_per_dof"],
                f"{name} friction loss",
                length=dof_count,
                domain="nonnegative",
            )
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} must bind each entity/joint at most once")
    if not required.issubset(set(keys)):
        raise ValueError(f"{name} is missing a required entity/joint binding")


def _sample_planar_initial_pose(
    target: Mapping[str, object], rng: np.random.Generator, *, name: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    position_recipe = target["position_world_m"]
    if isinstance(position_recipe, Mapping) and "harmonic_center_world_m" in position_recipe:
        position_recipe = position_recipe["harmonic_center_world_m"]
    initial_position = _sample_recipe_vector(
        position_recipe, rng, f"{name} target position_world_m"
    )
    yaw_record = _mapping(target["quaternion_world_xyzw"], f"{name} target quaternion_world_xyzw")
    _exact_fields(yaw_record, {"yaw_rad"}, f"{name} target quaternion_world_xyzw")
    yaw = _sample_recipe_number(yaw_record["yaw_rad"], rng, f"{name} target yaw_rad")
    half_yaw = 0.5 * yaw
    initial_quaternion = np.asarray(
        [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)], dtype=np.float64
    )
    return initial_position, initial_quaternion


def _materialize_translation_recipe(
    raw: Mapping[str, object],
    initial_position: npt.NDArray[np.float64],
    rng: np.random.Generator,
    *,
    name: str,
    harmonic_center: bool = False,
) -> dict[str, object]:
    kind = raw["type"]
    if harmonic_center and kind != "harmonic":
        raise ValueError("harmonic_center_world_m requires a harmonic translation")
    if kind == "fixed":
        return {"type": "fixed"}
    if kind == "constant_velocity":
        velocity = _sample_recipe_vector(
            raw["velocity_world_m_s"], rng, f"{name} constant velocity"
        )
        constant_law = ConstantVelocityTranslation(_vector3_tuple(velocity))
        return {
            "type": "constant_velocity",
            "velocity_world_m_s": list(constant_law.velocity_world_m_s),
        }
    if kind == "harmonic":
        drift = _sample_recipe_vector(
            raw["drift_velocity_world_m_s"], rng, f"{name} harmonic drift_velocity_world_m_s"
        )
        amplitude = _sample_recipe_vector(raw["amplitude_m"], rng, f"{name} harmonic amplitude")
        frequency = _sample_recipe_vector(raw["frequency_hz"], rng, f"{name} harmonic frequency")
        phase_vector = _sample_recipe_vector(raw["phase_rad"], rng, f"{name} harmonic phase")
        center = (
            initial_position
            if harmonic_center
            else initial_position - amplitude * np.sin(phase_vector)
        )
        harmonic_law = HarmonicTranslation(
            _vector3_tuple(center),
            _vector3_tuple(drift),
            _vector3_tuple(amplitude),
            _vector3_tuple(frequency),
            _vector3_tuple(phase_vector),
        )
        return {
            "type": "harmonic",
            "center_world_m": list(harmonic_law.center_world_m),
            "drift_velocity_world_m_s": list(harmonic_law.drift_velocity_world_m_s),
            "amplitude_m": list(harmonic_law.amplitude_m),
            "frequency_hz": list(harmonic_law.frequency_hz),
            "phase_rad": list(harmonic_law.phase_rad),
        }
    if kind != "circular":
        raise AssertionError(f"validated {name} translation kind became invalid")
    axis = _sample_recipe_vector(raw["axis_world"], rng, f"{name} circular axis")
    radial = _sample_recipe_vector(
        raw["radial_direction_world"], rng, f"{name} circular radial direction"
    )
    radius = _sample_recipe_number(raw["radius_m"], rng, f"{name} circular radius_m")
    angular_speed = _sample_recipe_number(
        raw["angular_speed_rad_s"], rng, f"{name} circular angular_speed_rad_s"
    )
    circular_phase = _sample_recipe_number(raw["phase_rad"], rng, f"{name} circular phase_rad")
    probe = CircularTranslation(
        _vector3_tuple(initial_position),
        _vector3_tuple(axis),
        _vector3_tuple(radial),
        radius,
        angular_speed,
        circular_phase,
    )
    tangent = np.cross(axis, radial)
    center = initial_position - radius * (
        math.cos(circular_phase) * radial + math.sin(circular_phase) * tangent
    )
    circular_law = CircularTranslation(
        _vector3_tuple(center),
        probe.axis_world,
        probe.radial_direction_world,
        probe.radius_m,
        probe.angular_speed_rad_s,
        probe.phase_rad,
    )
    return {
        "type": "circular",
        "center_world_m": list(circular_law.center_world_m),
        "axis_world": list(circular_law.axis_world),
        "radial_direction_world": list(circular_law.radial_direction_world),
        "radius_m": circular_law.radius_m,
        "angular_speed_rad_s": circular_law.angular_speed_rad_s,
        "phase_rad": circular_law.phase_rad,
    }


def _materialize_rotation_recipe(
    raw: Mapping[str, object],
    initial_quaternion: npt.NDArray[np.float64],
    rng: np.random.Generator,
    *,
    name: str,
) -> dict[str, object]:
    kind = raw["type"]
    if kind == "fixed":
        return {"type": "fixed"}
    if kind == "constant_angular_velocity":
        axis = _sample_recipe_vector(
            raw["axis_world"], rng, f"{name} constant angular velocity axis"
        )
        angular_speed = _sample_recipe_number(
            raw["angular_speed_rad_s"],
            rng,
            f"{name} constant angular velocity angular_speed_rad_s",
        )
        constant_law = ConstantAngularVelocityRotation(_vector3_tuple(axis), angular_speed)
        return {
            "type": "constant_angular_velocity",
            "axis_world": list(constant_law.axis_world),
            "angular_speed_rad_s": constant_law.angular_speed_rad_s,
        }
    if kind != "harmonic":
        raise AssertionError(f"validated {name} rotation kind became invalid")
    axis = _sample_recipe_vector(raw["axis_world"], rng, f"{name} harmonic rotation axis")
    amplitude = _sample_recipe_number(
        raw["amplitude_rad"], rng, f"{name} harmonic rotation amplitude_rad"
    )
    frequency = _sample_recipe_number(
        raw["frequency_hz"], rng, f"{name} harmonic rotation frequency_hz"
    )
    phase = _sample_recipe_number(raw["phase_rad"], rng, f"{name} harmonic rotation phase_rad")
    probe = HarmonicRotation(
        _vector4_tuple(initial_quaternion),
        _vector3_tuple(axis),
        amplitude,
        frequency,
        phase,
    )
    initial_angle = probe.amplitude_rad * math.sin(probe.phase_rad)
    base_quaternion = _quaternion_product_xyzw(
        _axis_angle_quaternion(axis, -initial_angle), initial_quaternion
    )
    harmonic_law = HarmonicRotation(
        _vector4_tuple(base_quaternion),
        probe.axis_world,
        probe.amplitude_rad,
        probe.frequency_hz,
        probe.phase_rad,
    )
    return {
        "type": "harmonic",
        "base_quaternion_world_xyzw": list(harmonic_law.base_quaternion_world_xyzw),
        "axis_world": list(harmonic_law.axis_world),
        "amplitude_rad": harmonic_law.amplitude_rad,
        "frequency_hz": harmonic_law.frequency_hz,
        "phase_rad": harmonic_law.phase_rad,
    }


def _validated_grasp_object_recipes(
    recipe: Mapping[str, object], target_id: str
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[tuple[Mapping[str, object], ...], ...],
    int,
]:
    object_records = tuple(
        _mapping(item, "grasp object_state")
        for item in _recipe_sequence(recipe["object_states"], "grasp object_states")
    )
    object_choices = tuple(
        _record_recipe_choices(item, "grasp object_state") for item in object_records
    )
    for choices in object_choices:
        for choice in choices:
            _validate_grasp_object_recipe(choice)
        if len({choice["instance_id"] for choice in choices}) != 1:
            raise ValueError("grasp object choices must preserve one instance_id")
    target_object_indexes = tuple(
        index
        for index, choices in enumerate(object_choices)
        if choices[0]["instance_id"] == target_id
    )
    if len(target_object_indexes) != 1:
        raise ValueError("grasp recipe requires exactly one bound target object_state")
    return object_records, object_choices, target_object_indexes[0]


def _validated_grasp_motion_recipes(
    recipe: Mapping[str, object], target_id: str
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[tuple[Mapping[str, object], ...], ...],
    int,
]:
    motion_records = tuple(
        _mapping(item, "grasp driven_motion")
        for item in _recipe_sequence(recipe["motion"], "grasp motion")
    )
    motion_choices = tuple(
        _record_recipe_choices(item, "grasp driven_motion") for item in motion_records
    )
    for choices in motion_choices:
        for choice in choices:
            _validate_grasp_motion_recipe(choice)
        if len({choice["entity_id"] for choice in choices}) != 1:
            raise ValueError("grasp motion choices must preserve one entity_id")
    target_motion_indexes = tuple(
        index
        for index, choices in enumerate(motion_choices)
        if choices[0]["entity_id"] == target_id
    )
    if len(target_motion_indexes) != 1:
        raise ValueError("grasp recipe requires exactly one bound target motion")
    return motion_records, motion_choices, target_motion_indexes[0]


def _materialize_grasp_target(
    *,
    target_id: str,
    object_choices: tuple[Mapping[str, object], ...],
    motion_choices: tuple[Mapping[str, object], ...],
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, object]]:
    target_object = _choose_record(object_choices, rng)
    initial_position, initial_quaternion = _sample_planar_initial_pose(
        target_object, rng, name="grasp"
    )
    target_motion = _choose_record(motion_choices, rng)
    translation = _select_grasp_law_recipe(
        target_motion["translation"],
        rng,
        name="grasp target translation",
        translation=True,
    )
    rotation = _select_grasp_law_recipe(
        target_motion["rotation"],
        rng,
        name="grasp target rotation",
        translation=False,
    )
    concrete_target_motion: dict[str, object] = {
        "entity_id": target_id,
        "translation": _materialize_translation_recipe(
            translation,
            initial_position,
            rng,
            name="grasp",
            harmonic_center=_uses_harmonic_center(target_object),
        ),
        "rotation": _materialize_rotation_recipe(rotation, initial_quaternion, rng, name="grasp"),
    }
    concrete_target_object: dict[str, object] = {
        "instance_id": target_id,
        "asset_id": _sample_recipe_value(target_object["asset_id"], rng),
        "quaternion_world_xyzw": initial_quaternion.tolist(),
        **_materialize_initial_motion_fields(
            target_object, concrete_target_motion, initial_position, initial_quaternion, rng
        ),
    }
    return concrete_target_object, concrete_target_motion


def _uses_harmonic_center(target: Mapping[str, object]) -> bool:
    position = target["position_world_m"]
    return isinstance(position, Mapping) and "harmonic_center_world_m" in position


def _materialize_initial_motion_fields(
    target: Mapping[str, object],
    motion: dict[str, object],
    position: npt.NDArray[np.float64],
    quaternion: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> dict[str, object]:
    state = _motion_state_at_time(
        _parse_motion([motion])[0],
        initial_position_world_m=position,
        initial_quaternion_world_xyzw=quaternion,
        simulator_time_s=0.0,
    )
    result: dict[str, object] = {
        "position_world_m": (
            state.position_world_m if _uses_harmonic_center(target) else position
        ).tolist()
    }
    for field, derivative in (
        ("linear_velocity_world_m_s", state.linear_velocity_world_m_s),
        ("angular_velocity_world_rad_s", state.angular_velocity_world_rad_s),
    ):
        value = target[field]
        result[field] = (
            derivative.tolist()
            if isinstance(value, Mapping) and value.get("from_motion") is True
            else _sample_recipe_value(value, rng)
        )
    return result


def _materialize_grasp_recipe(
    task: TaskSpec, recipe: Mapping[str, object], rng: np.random.Generator
) -> dict[str, object]:
    _exact_fields(recipe, _RECIPE_FIELDS, "grasp recipe")
    if not isinstance(task.bindings, GraspBindings):
        raise ValueError("grasp task requires GraspBindings")
    target_id = task.bindings.target_object_id
    object_records, object_choices, target_object_index = _validated_grasp_object_recipes(
        recipe, target_id
    )
    motion_records, motion_choices, target_motion_index = _validated_grasp_motion_recipes(
        recipe, target_id
    )

    def asset_values(value: object) -> frozenset[str]:
        if isinstance(value, str):
            return frozenset((value,))
        return frozenset().union(
            *(
                asset_values(item)
                for item in _choice_values(_mapping(value, "asset_id"), "asset_id")
            )
        )

    assets = frozenset().union(
        *(asset_values(item["asset_id"]) for item in object_choices[target_object_index])
    )
    _validate_body_physics_recipes(recipe["physics"], assets_by_entity={target_id: assets})
    _validate_joint_physics_recipes(
        _mapping(recipe["physics"], "grasp physics")["joints"],
        allowed_dofs={(target_id, f"{target_id}_joint0"): 6},
        name="grasp joint physics",
    )
    concrete_target_object, concrete_target_motion = _materialize_grasp_target(
        target_id=target_id,
        object_choices=object_choices[target_object_index],
        motion_choices=motion_choices[target_motion_index],
        rng=rng,
    )
    concrete_objects = [
        concrete_target_object if index == target_object_index else _sample_recipe_value(item, rng)
        for index, item in enumerate(object_records)
    ]
    concrete_motion = [
        concrete_target_motion if index == target_motion_index else _sample_recipe_value(item, rng)
        for index, item in enumerate(motion_records)
    ]
    physics_recipe = dict(_mapping(recipe["physics"], "grasp physics"))
    bodies: list[object] = []
    for entry in _recipe_sequence(physics_recipe["bodies"], "grasp body physics"):
        choices = _record_recipe_choices(entry, "grasp body physics")
        if "asset_id" in choices[0]:
            body = dict(
                next(
                    item
                    for item in choices
                    if item["entity_id"] == target_id
                    and item["asset_id"] == concrete_target_object["asset_id"]
                )
            )
            body.pop("asset_id")
            bodies.append(body)
        else:
            bodies.append(entry)
    physics_recipe["bodies"] = bodies

    concrete: dict[str, object] = {}
    for field in _RECIPE_FIELD_ORDER:
        if field == "object_states":
            concrete[field] = concrete_objects
        elif field == "motion":
            concrete[field] = concrete_motion
        elif field == "physics":
            concrete[field] = _sample_recipe_value(physics_recipe, rng)
        else:
            concrete[field] = _sample_recipe_value(recipe[field], rng)
    return concrete


def _validate_surface_entity_recipes(
    recipe: Mapping[str, object],
    *,
    target_id: str,
    expected_assets: Mapping[str, str],
) -> None:
    for field in ("object_states", "fixture_states", "motion"):
        expected_count = len(expected_assets) if field == "fixture_states" else 1
        if len(_recipe_sequence(recipe[field], f"surface-driven T1 {field}")) != expected_count:
            raise ValueError(f"surface-driven T1 requires exactly {expected_count} {field} records")
    for entry in _recipe_sequence(recipe["object_states"], "surface-driven T1 objects"):
        for target in _record_recipe_choices(entry, "surface-driven T1 target"):
            _exact_fields(target, _OBJECT_STATE_FIELDS, "surface-driven T1 target")
            if target["instance_id"] != target_id:
                raise ValueError(
                    "surface-driven T1 target instance_id must resolve the task binding"
                )
            asset = target["asset_id"]
            assets = (
                _choice_values(_mapping(asset, "Belt asset_id"), "Belt asset_id")
                if isinstance(asset, Mapping)
                else (asset,)
            )
            if any(value != "obj.cube_44_v1" for value in assets):
                raise ValueError("surface-driven T1 asset choices require the fixed cube asset")
            _validate_vector_recipe(
                target["position_world_m"], "surface-driven T1 initial position"
            )
            _validate_camera_quaternion_recipe(target["quaternion_world_xyzw"], "T1 target")
            _validate_vector_recipe(
                target["linear_velocity_world_m_s"], "Belt reset velocity", domain="zero"
            )
            _validate_vector_recipe(
                target["angular_velocity_world_rad_s"], "Belt reset angular velocity", domain="zero"
            )
    fixture_ids: list[str] = []
    for entry in _recipe_sequence(recipe["fixture_states"], "surface-driven T1 fixtures"):
        ids = set()
        for fixture in _record_recipe_choices(entry, "surface-driven T1 fixture"):
            _exact_fields(
                fixture,
                {"instance_id", "asset_id", "position_world_m", "quaternion_world_xyzw"},
                "surface-driven T1 fixture",
            )
            instance_id = fixture["instance_id"]
            if not isinstance(instance_id, str) or instance_id not in expected_assets:
                raise ValueError("surface-driven T1 fixture must reference its bound entity")
            if fixture["asset_id"] != expected_assets[instance_id]:
                raise ValueError("surface-driven T1 fixture must reference its canonical asset")
            ids.add(instance_id)
            _validate_vector_recipe(
                fixture["position_world_m"], "surface-driven T1 fixture position"
            )
            _validate_camera_quaternion_recipe(fixture["quaternion_world_xyzw"], "T1 fixture")
        if len(ids) != 1:
            raise ValueError("fixture alternatives must preserve their entity binding")
        fixture_ids.extend(ids)
    if len(set(fixture_ids)) != len(fixture_ids) or set(fixture_ids) != set(expected_assets):
        raise ValueError("surface-driven T1 fixtures must resolve each binding exactly once")


def _validate_surface_motion_recipes(recipe: Mapping[str, object], belt_id: str) -> None:
    for entry in _recipe_sequence(recipe["motion"], "surface-driven T1 motion"):
        for motion in _record_recipe_choices(entry, "surface-driven T1 motion"):
            _exact_fields(motion, _MOTION_FIELDS, "surface-driven T1 motion")
            if motion["entity_id"] != belt_id:
                raise ValueError("surface-driven T1 motion must reference the belt, not target")
            for law in _record_recipe_choices(
                motion["translation"], "surface-driven T1 translation"
            ):
                _exact_fields(law, {"type", "velocity_world_m_s"}, "surface-driven T1 translation")
                if law["type"] != "constant_velocity":
                    raise ValueError("surface-driven T1 translation requires constant_velocity")
                _validate_vector_recipe(
                    law["velocity_world_m_s"], "surface-driven T1 surface velocity"
                )
            for law in _record_recipe_choices(motion["rotation"], "surface-driven T1 rotation"):
                _exact_fields(law, {"type"}, "surface-driven T1 rotation")
                if law["type"] != "fixed":
                    raise ValueError("surface-driven T1 fixture rotation requires fixed")


def _validate_surface_physics_recipes(
    recipe: Mapping[str, object], *, target_id: str, expected_assets: Mapping[str, str]
) -> None:
    _validate_body_physics_recipes(recipe["physics"])
    physics = _mapping(recipe["physics"], "surface-driven T1 physics")
    _validate_joint_physics_recipes(
        physics["joints"],
        allowed_dofs={(target_id, f"{target_id}_joint0"): 6},
        name="surface-driven T1 joint physics",
    )
    bodies = _recipe_sequence(physics["bodies"], "surface-driven T1 bodies")
    if len(bodies) != 1:
        raise ValueError("surface-driven T1 requires exactly one target body physics record")
    for body in _record_recipe_choices(bodies[0], "surface-driven T1 body"):
        if body["entity_id"] != target_id:
            raise ValueError("surface-driven T1 body physics must reference the target")
    contact_entries = _recipe_sequence(physics["contacts"], "surface-driven T1 contacts")
    expected_contact_ids = {target_id, *expected_assets}
    if len(contact_entries) != len(expected_contact_ids):
        raise ValueError("surface-driven T1 contacts must resolve each binding exactly once")
    contact_ids: list[str] = []
    for entry in contact_entries:
        ids = set()
        for contact in _record_recipe_choices(entry, "surface-driven T1 contact"):
            _exact_fields(
                contact,
                {"entity_id", "friction_slide_spin_roll", "solref", "solimp"},
                "surface-driven T1 contact",
            )
            entity_id = contact["entity_id"]
            if not isinstance(entity_id, str) or entity_id not in expected_contact_ids:
                raise ValueError("surface-driven T1 contact requires a bound entity_id")
            ids.add(entity_id)
            _validate_vector_recipe(
                contact["friction_slide_spin_roll"],
                "surface-driven T1 friction",
                domain="nonnegative",
            )
            _validate_vector_recipe(contact["solref"], "surface-driven T1 solref", length=2)
            _validate_vector_recipe(contact["solimp"], "surface-driven T1 solimp", length=5)
        if len(ids) != 1:
            raise ValueError("contact alternatives must preserve their entity binding")
        contact_ids.extend(ids)
    if len(set(contact_ids)) != len(contact_ids) or set(contact_ids) != expected_contact_ids:
        raise ValueError("surface-driven T1 contacts must resolve each binding exactly once")


def _validate_surface_recipe(task: TaskSpec, recipe: Mapping[str, object]) -> None:
    """Check every declared closed-record choice before consuming the proposal RNG."""
    if not isinstance(task.bindings, DrivenInterceptionBindings):
        raise ValueError("surface-driven T1 requires DrivenInterceptionBindings")
    target_id, belt_id = task.bindings.target_object_id, task.bindings.driver_entity_id
    occluder_ids = task.bindings.occluder_fixture_ids
    expected_assets = {belt_id: "arena.conveyor_v1"}
    if task.has_occluder:
        if len(occluder_ids) != 1:
            raise ValueError("T1-OCC requires exactly one bound canopy")
        expected_assets[occluder_ids[0]] = "fixture.occlusion_canopy_v1"
    _validate_surface_entity_recipes(recipe, target_id=target_id, expected_assets=expected_assets)
    _validate_surface_motion_recipes(recipe, belt_id)
    _validate_surface_physics_recipes(recipe, target_id=target_id, expected_assets=expected_assets)
    _validate_capture_sensor_recipes(recipe, light_id="conveyor_key", name="surface-driven T1")


def _validate_placement_preparation_recipe(
    recipe: Mapping[str, object],
    *,
    payload_id: str,
    goal_id: str,
    assets: tuple[str, str],
) -> None:
    for field, instance_id, asset_id in zip(
        ("object_states", "fixture_states"), (payload_id, goal_id), assets, strict=True
    ):
        entries = _recipe_sequence(recipe[field], f"T2 {field}")
        if len(entries) != 1:
            raise ValueError(f"T2 requires exactly one {field} record")
        for record in _record_recipe_choices(entries[0], f"T2 {field}"):
            fields = (
                _OBJECT_STATE_FIELDS
                if field == "object_states"
                else {"instance_id", "asset_id", "position_world_m", "quaternion_world_xyzw"}
            )
            _exact_fields(record, fields, f"T2 {field}")
            if record["instance_id"] != instance_id or record["asset_id"] != asset_id:
                raise ValueError("T2 preparation must use the exact bound functional asset")
            _validate_vector_recipe(record["position_world_m"], "T2 preparation position")
            _validate_initial_quaternion_recipe(record["quaternion_world_xyzw"], "T2 preparation")
            if field == "object_states":
                for name in ("linear_velocity_world_m_s", "angular_velocity_world_rad_s"):
                    _validate_vector_recipe(record[name], f"T2 {name}")


def _validate_placement_geometry_recipe(recipe: Mapping[str, object]) -> None:
    geometry = _mapping(recipe["geometry"], "T2 geometry")
    _exact_fields(
        geometry,
        {
            "support_surface_z_m",
            "goal_planar_clearance_m",
            "goal_bottom_center_z_m",
            "goal_window_exit_y_m",
        },
        "T2 geometry",
    )
    for name, value in geometry.items():
        _validate_number_recipe(
            value,
            f"T2 {name}",
            domain="positive"
            if name in {"goal_planar_clearance_m", "goal_bottom_center_z_m"}
            else "finite",
        )


def _validate_placement_motion_recipe(
    task: TaskSpec, recipe: Mapping[str, object], goal_id: str
) -> None:
    motions = _recipe_sequence(recipe["motion"], "T2 motion")
    if len(motions) != 1:
        raise ValueError("T2 requires exactly one driven goal motion")
    for motion in _record_recipe_choices(motions[0], "T2 motion"):
        _exact_fields(motion, _MOTION_FIELDS, "T2 motion")
        if motion["entity_id"] != goal_id:
            raise ValueError("T2 motion must reference the goal fixture")
        for law in _record_recipe_choices(motion["translation"], "T2 translation"):
            _exact_fields(law, {"type", "velocity_world_m_s"}, "T2 translation")
            if law["type"] != "constant_velocity":
                raise ValueError("T2 requires constant world +Y goal translation")
            _validate_axis_vector_recipe(
                law["velocity_world_m_s"], axis=1, domain="positive", name="T2 translation"
            )
        for law in _record_recipe_choices(motion["rotation"], "T2 rotation"):
            if task.variant == "se3":
                _exact_fields(law, {"type", "axis_world", "angular_speed_rad_s"}, "T2 rotation")
                if law["type"] != "constant_angular_velocity" or _unit_vector_recipe_values(
                    law["axis_world"], "T2 rotation axis"
                ) != ((0.0, 0.0, 1.0),):
                    raise ValueError("T2-SE3 requires world-Z constant angular velocity")
                _validate_number_recipe(law["angular_speed_rad_s"], "T2 angular speed")
            else:
                _exact_fields(law, {"type"}, "T2 rotation")
                if law["type"] != "fixed":
                    raise ValueError("T2-TRANS requires fixed rotation")


def _validate_placement_physics_recipe(
    recipe: Mapping[str, object], *, payload_id: str, goal_id: str
) -> None:
    _validate_body_physics_recipes(recipe["physics"])
    physics = _mapping(recipe["physics"], "T2 physics")
    _validate_vector_recipe(physics["gravity_world_m_s2"], "T2 gravity")
    _validate_joint_physics_recipes(
        physics["joints"],
        allowed_dofs={
            (payload_id, f"{payload_id}_joint0"): 6,
            (goal_id, f"{goal_id}_travel"): 1,
            (goal_id, f"{goal_id}_x"): 1,
            (goal_id, f"{goal_id}_z"): 1,
            (goal_id, f"{goal_id}_yaw"): 1,
        },
        name="T2 joint physics",
    )
    for field in ("bodies", "contacts"):
        entries = _recipe_sequence(physics[field], f"T2 {field}")
        if len(entries) != 2:
            raise ValueError(f"T2 requires exactly two {field} records")
        ids: list[str] = []
        for entry in entries:
            alternatives = _record_recipe_choices(entry, f"T2 {field}")
            raw_entity_id = alternatives[0].get("entity_id")
            if not isinstance(raw_entity_id, str) or raw_entity_id not in (payload_id, goal_id):
                raise ValueError("T2 physics requires its payload and goal entity references")
            entity_id = raw_entity_id
            ids.append(entity_id)
            for record in alternatives:
                if record.get("entity_id") != entity_id:
                    raise ValueError("T2 alternatives must preserve their entity reference")
                if field == "contacts":
                    _exact_fields(
                        record,
                        {"entity_id", "friction_slide_spin_roll", "solref", "solimp"},
                        "T2 contact",
                    )
                    _validate_vector_recipe(
                        record["friction_slide_spin_roll"], "T2 friction", domain="nonnegative"
                    )
                    _validate_vector_recipe(record["solref"], "T2 solref", length=2)
                    _validate_vector_recipe(record["solimp"], "T2 solimp", length=5)
        if set(ids) != {payload_id, goal_id}:
            raise ValueError("T2 physics requires each bound entity exactly once")


def _validate_placement_camera_recipe(recipe: Mapping[str, object]) -> None:
    _validate_capture_sensor_recipes(recipe, light_id="conveyor_key", name="T2")
    cameras = _mapping(recipe["cameras"], "T2 cameras")
    if (cameras["image_height_px"], cameras["image_width_px"]) != (224, 224):
        raise ValueError("T2 camera raster must be concrete 224x224")
    views = _recipe_sequence(cameras["views"], "T2 camera views")
    names = (
        ("agentview", "wrist")
        if recipe["observation_profile"] == "agentview_wrist"
        else ("agentview",)
    )
    if len(views) != len(names) or any(
        camera["view_name"] != expected
        for entry, expected in zip(views, names, strict=True)
        for camera in _record_recipe_choices(entry, "T2 camera")
    ):
        raise ValueError("T2 camera view order must match its explicit observation profile")


def _validate_placement_recipe(task: TaskSpec, recipe: Mapping[str, object]) -> None:
    """Reject all malformed declared alternatives, including branches not sampled."""
    _validate_frozen_recipe_clock(recipe, "T2")
    if not isinstance(task.bindings, MovingPlacementBindings):
        raise ValueError("T2 requires MovingPlacementBindings")
    payload_id, goal_id = task.bindings.payload_object_id, task.bindings.goal_fixture_id
    assets = (
        ("obj.asym_t_t2_handle_v1", "fixture.t_slot_v1")
        if task.variant == "se3"
        else ("obj.cuboid_t2_handle_v1", "fixture.open_tray_v1")
    )
    _validate_placement_preparation_recipe(
        recipe, payload_id=payload_id, goal_id=goal_id, assets=assets
    )
    _validate_family_robot_recipe(recipe, "T2")
    _validate_placement_geometry_recipe(recipe)
    _validate_placement_motion_recipe(task, recipe, goal_id)
    _validate_placement_physics_recipe(recipe, payload_id=payload_id, goal_id=goal_id)
    if _recipe_sequence(recipe["events"], "T2 events"):
        raise ValueError("T2 does not define discrete ScenarioSpec events")
    _validate_placement_camera_recipe(recipe)


def _wire_phase_recipe(value: object, name: str) -> Mapping[str, object]:
    phase = _mapping(value, name)
    _exact_fields(phase, {"phase_x_rad"}, name)
    _validate_number_recipe(phase["phase_x_rad"], f"{name}.phase_x_rad")
    return phase


def _validate_rotating_entity_recipes(
    recipe: Mapping[str, object],
    *,
    wand_id: str,
    expected_fixture_ids: tuple[str, ...],
) -> None:
    objects = _recipe_sequence(recipe["object_states"], "T3 object states")
    if len(objects) != 1:
        raise ValueError("T3 requires exactly one wand object state")
    wand = _mapping(objects[0], "T3 wand object state")
    _exact_fields(wand, _OBJECT_STATE_FIELDS, "T3 wand object state")
    if wand["instance_id"] != wand_id or wand["asset_id"] != "obj.wand_loop_v1":
        raise ValueError("T3 wand state must resolve its fixed functional asset binding")
    _validate_vector_recipe(wand["position_world_m"], "T3 wand position")
    _validate_initial_quaternion_recipe(wand["quaternion_world_xyzw"], "T3 wand orientation")
    _validate_vector_recipe(wand["linear_velocity_world_m_s"], "T3 wand linear velocity")
    _validate_vector_recipe(wand["angular_velocity_world_rad_s"], "T3 wand angular velocity")

    fixtures = _recipe_sequence(recipe["fixture_states"], "T3 fixture states")
    if len(fixtures) != len(expected_fixture_ids):
        raise ValueError("T3 fixture states must resolve wire and its optional cradle exactly once")
    for index, (entry, expected_id) in enumerate(zip(fixtures, expected_fixture_ids, strict=True)):
        fixture = _mapping(entry, "T3 fixture state")
        _exact_fields(
            fixture,
            {"instance_id", "asset_id", "position_world_m", "quaternion_world_xyzw"},
            "T3 fixture state",
        )
        asset_id = "fixture.rotating_wire_v1" if index == 0 else "fixture.wand_cradle_v1"
        if fixture["instance_id"] != expected_id or fixture["asset_id"] != asset_id:
            raise ValueError("T3 fixture state must resolve its fixed functional asset binding")
        _validate_vector_recipe(fixture["position_world_m"], "T3 fixture position")
        if index == 0:
            _wire_phase_recipe(fixture["quaternion_world_xyzw"], "T3 wire phase")
        else:
            _validate_camera_quaternion_recipe(
                fixture["quaternion_world_xyzw"], "T3 cradle orientation"
            )


def _validate_rotating_geometry_recipe(recipe: Mapping[str, object]) -> None:
    geometry = _mapping(recipe["geometry"], "T3 geometry")
    _exact_fields(
        geometry,
        {
            "frame_center_world_m",
            "rotation_axis_world",
            "wire_radius_m",
            "wire_path_points_frame_m",
        },
        "T3 geometry",
    )
    _validate_vector_recipe(geometry["frame_center_world_m"], "T3 frame center")
    if _unit_vector_recipe_values(geometry["rotation_axis_world"], "T3 rotation axis") != (
        (1.0, 0.0, 0.0),
    ):
        raise ValueError("T3 GeometrySpec axis must be world X")
    _validate_number_recipe(geometry["wire_radius_m"], "T3 wire radius", domain="positive")
    points = _recipe_sequence(geometry["wire_path_points_frame_m"], "T3 wire path")
    if len(points) < 2:
        raise ValueError("T3 requires a finite polyline with at least two points")
    for point in points:
        _validate_vector_recipe(point, "T3 wire path point")


def _validate_rotating_motion_recipe(recipe: Mapping[str, object], wire_id: str) -> None:
    motions = _recipe_sequence(recipe["motion"], "T3 motion")
    if len(motions) != 1:
        raise ValueError("T3 requires exactly one wire motion")
    motion = _mapping(motions[0], "T3 wire motion")
    _exact_fields(motion, _MOTION_FIELDS, "T3 wire motion")
    if motion["entity_id"] != wire_id:
        raise ValueError("T3 motion must resolve the wire fixture binding")
    translation = _mapping(motion["translation"], "T3 translation")
    _exact_fields(translation, {"type"}, "T3 translation")
    if translation["type"] != "fixed":
        raise ValueError("T3 wire translation must be fixed")
    rotation = _mapping(motion["rotation"], "T3 rotation")
    _exact_fields(rotation, {"type", "axis_world", "angular_speed_rad_s"}, "T3 rotation")
    if rotation["type"] != "constant_angular_velocity" or _unit_vector_recipe_values(
        rotation["axis_world"], "T3 motion axis"
    ) != ((1.0, 0.0, 0.0),):
        raise ValueError("T3 motion must be constant angular velocity around world X")
    _validate_number_recipe(rotation["angular_speed_rad_s"], "T3 angular speed", domain="nonzero")


def _validate_rotating_physics_recipe(
    recipe: Mapping[str, object], *, wand_id: str, expected_fixture_ids: tuple[str, ...]
) -> None:
    _validate_body_physics_recipes(recipe["physics"])
    physics = _mapping(recipe["physics"], "T3 physics")
    _validate_vector_recipe(physics["gravity_world_m_s2"], "T3 gravity")
    wire_id = expected_fixture_ids[0]
    _validate_joint_physics_recipes(
        physics["joints"],
        allowed_dofs={
            (wand_id, f"{wand_id}_joint0"): 6,
            (wire_id, f"{wire_id}_hinge"): 1,
        },
        name="T3 joint physics",
    )
    expected_entities = {wand_id, *expected_fixture_ids}
    for field in ("bodies", "contacts"):
        entries = _recipe_sequence(physics[field], f"T3 {field}")
        if len(entries) != len(expected_entities):
            raise ValueError(f"T3 requires each bound entity once in {field}")
        ids: list[str] = []
        for entry in entries:
            alternatives = _record_recipe_choices(entry, f"T3 {field}")
            entity_id = alternatives[0].get("entity_id")
            if not isinstance(entity_id, str) or entity_id not in expected_entities:
                raise ValueError(f"T3 {field} contains an unknown entity binding")
            ids.append(entity_id)
            for record in alternatives:
                if record.get("entity_id") != entity_id:
                    raise ValueError(f"T3 {field} alternatives must preserve entity binding")
                if field == "contacts":
                    _exact_fields(
                        record,
                        {"entity_id", "friction_slide_spin_roll", "solref", "solimp"},
                        "T3 contact",
                    )
                    _validate_vector_recipe(
                        record["friction_slide_spin_roll"], "T3 friction", domain="nonnegative"
                    )
                    _validate_vector_recipe(record["solref"], "T3 solref", length=2)
                    _validate_vector_recipe(record["solimp"], "T3 solimp", length=5)
        if len(set(ids)) != len(ids) or set(ids) != expected_entities:
            raise ValueError(f"T3 {field} must bind each entity exactly once")


def _validate_rotating_camera_recipe(recipe: Mapping[str, object]) -> None:
    _validate_capture_sensor_recipes(recipe, light_id="rotating_wire_key", name="T3")
    cameras = _mapping(recipe["cameras"], "T3 cameras")
    if (cameras["image_height_px"], cameras["image_width_px"]) != (224, 224):
        raise ValueError("T3 camera raster must be concrete 224x224")
    expected_views = (
        ("agentview", "wrist")
        if recipe["observation_profile"] == "agentview_wrist"
        else ("agentview",)
    )
    views = _recipe_sequence(cameras["views"], "T3 camera views")
    if len(views) != len(expected_views) or any(
        camera["view_name"] != expected
        for entry, expected in zip(views, expected_views, strict=True)
        for camera in _record_recipe_choices(entry, "T3 camera")
    ):
        raise ValueError("T3 camera order must match its explicit observation profile")


def _validate_rotating_recipe(task: TaskSpec, recipe: Mapping[str, object]) -> None:
    """Validate every T3 branch before the single proposal RNG is consumed."""
    _validate_frozen_recipe_clock(recipe, "T3")
    if not isinstance(task.bindings, RotatingInteractionBindings):
        raise ValueError("T3 requires RotatingInteractionBindings")
    wand_id, wire_id = task.bindings.wand_object_id, task.bindings.wire_fixture_id
    full = task.variant == "full"
    expected_fixture_ids = (wire_id, *task.bindings.cradle_fixture_ids)
    if full != (len(task.bindings.cradle_fixture_ids) == 1):
        raise ValueError("T3-Full alone requires exactly one cradle binding")
    _validate_rotating_entity_recipes(
        recipe, wand_id=wand_id, expected_fixture_ids=expected_fixture_ids
    )
    _validate_family_robot_recipe(recipe, "T3")
    _validate_rotating_geometry_recipe(recipe)
    _validate_rotating_motion_recipe(recipe, wire_id)
    _validate_rotating_physics_recipe(
        recipe, wand_id=wand_id, expected_fixture_ids=expected_fixture_ids
    )
    if _recipe_sequence(recipe["events"], "T3 events"):
        raise ValueError("T3 does not define discrete ScenarioSpec events")
    _validate_rotating_camera_recipe(recipe)


def _materialize_rotating_recipe(
    task: TaskSpec, recipe: Mapping[str, object], rng: np.random.Generator
) -> Mapping[str, object]:
    _validate_rotating_recipe(task, recipe)
    motion = _sample_recipe_value(_ordered_recipe_value(recipe["motion"]), rng)
    objects = _recipe_sequence(recipe["object_states"], "T3 object states")
    fixtures = _recipe_sequence(recipe["fixture_states"], "T3 fixture states")
    wire = _mapping(fixtures[0], "T3 wire fixture state")
    phase_recipe = _wire_phase_recipe(wire["quaternion_world_xyzw"], "T3 wire phase")
    phase = _sample_recipe_number(phase_recipe["phase_x_rad"], rng, "T3 phase_x_rad")
    quaternion = [math.sin(0.5 * phase), 0.0, 0.0, math.cos(0.5 * phase)]

    concrete_objects = [
        {
            field: _sample_recipe_value(_ordered_recipe_value(record[field]), rng)
            for field in (
                "instance_id",
                "asset_id",
                "position_world_m",
                "quaternion_world_xyzw",
                "linear_velocity_world_m_s",
                "angular_velocity_world_rad_s",
            )
        }
        for record in (_mapping(item, "T3 object state") for item in objects)
    ]
    concrete_fixtures = []
    for index, item in enumerate(fixtures):
        record = _mapping(item, "T3 fixture state")
        concrete_fixtures.append(
            {
                field: quaternion
                if index == 0 and field == "quaternion_world_xyzw"
                else _sample_recipe_value(_ordered_recipe_value(record[field]), rng)
                for field in (
                    "instance_id",
                    "asset_id",
                    "position_world_m",
                    "quaternion_world_xyzw",
                )
            }
        )
    sampled: dict[str, object] = {
        "motion": motion,
        "object_states": concrete_objects,
        "fixture_states": concrete_fixtures,
    }
    for field in (
        "robot_initial_state",
        "geometry",
        "physics",
        "cameras",
        "lighting",
        "action_delay",
        "action_mode",
        "observation_profile",
        "physics_timestep_s",
        "control_frequency_hz",
        "sensor_frequency_hz",
        "events",
    ):
        sampled[field] = _sample_recipe_value(_ordered_recipe_value(recipe[field]), rng)
    return sampled


def _validate_t4_bounds_recipe(value: object, *, dimension: int, name: str) -> None:
    bounds = _recipe_sequence(value, name)
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain lower and upper vectors")
    for endpoint in bounds:
        _validate_vector_recipe(endpoint, name, length=dimension)


def _validate_frozen_recipe_clock(recipe: Mapping[str, object], name: str) -> None:
    for field, required in (
        ("physics_timestep_s", 0.002),
        ("control_frequency_hz", 20),
        ("sensor_frequency_hz", 20),
    ):
        if isinstance(recipe[field], bool) or recipe[field] != required:
            raise ValueError(f"{name} {field} must be the concrete frozen clock value")


def _validate_family_robot_recipe(recipe: Mapping[str, object], name: str) -> None:
    robot = _mapping(recipe["robot_initial_state"], f"{name} robot initial state")
    _exact_fields(
        robot,
        {"joint_position_rad", "gripper_position_m"},
        f"{name} robot initial state",
    )
    _validate_vector_recipe(robot["joint_position_rad"], f"{name} arm position", length=7)
    _validate_vector_recipe(robot["gripper_position_m"], f"{name} finger position", length=2)


def _validate_free_dynamics_entities(
    recipe: Mapping[str, object],
    bindings: FreeDynamicsInterceptionBindings,
    *,
    arena_asset: str,
    target_asset: str,
) -> tuple[str, str, str]:

    objects = _recipe_sequence(recipe["object_states"], "T4 object states")
    if len(objects) != 1:
        raise ValueError("T4 requires exactly one target object state")
    for target in _record_recipe_choices(objects[0], "T4 target object state"):
        _exact_fields(target, _OBJECT_STATE_FIELDS, "T4 target object state")
        if target["instance_id"] != bindings.target_object_id or target["asset_id"] != target_asset:
            raise ValueError("T4 target must resolve its fixed functional asset binding")
        position = _mapping(target["position_world_m"], "T4 atomic target position")
        if set(position) == {"ramp_center_offset_m"}:
            offset = _recipe_sequence(position["ramp_center_offset_m"], "T4 ramp center offset")
            if len(offset) != 3:
                raise ValueError("T4 ramp center offset must have length 3")
            _validate_number_recipe(offset[0], "T4 lateral offset")
            if isinstance(offset[1], Mapping):
                along = _mapping(offset[1], "T4 along-ramp offset")
                if set(along) == {"from_release_gate"}:
                    if target_asset != "obj.can_v1" or along["from_release_gate"] is not True:
                        raise ValueError("only the can derives its support extent from the gate")
                else:
                    _validate_number_recipe(along, "T4 along-ramp offset")
            else:
                _validate_number_recipe(offset[1], "T4 along-ramp offset")
            _validate_number_recipe(offset[2], "T4 support height", domain="positive")
        else:
            _exact_fields(
                position, {"lateral_offset_m", "along_ramp_offset_m"}, "T4 target position"
            )
            _validate_number_recipe(position["lateral_offset_m"], "T4 lateral offset")
            _validate_number_recipe(position["along_ramp_offset_m"], "T4 along-ramp offset")
        orientation = _mapping(target["quaternion_world_xyzw"], "T4 atomic target orientation")
        _exact_fields(orientation, {"axis_yaw_rad"}, "T4 atomic target orientation")
        _validate_number_recipe(orientation["axis_yaw_rad"], "T4 target axis yaw")
        for field in ("linear_velocity_world_m_s", "angular_velocity_world_rad_s"):
            _validate_vector_recipe(target[field], f"T4 {field}", domain="zero")

    preparation = _recipe_sequence(recipe["preparation_object_states"], "T4 preparation states")
    if len(preparation) != 1:
        raise ValueError("T4 requires one explicit temporary target state")
    for state in _record_recipe_choices(preparation[0], "T4 preparation target"):
        _exact_fields(state, _OBJECT_STATE_FIELDS, "T4 preparation target")
        if state["instance_id"] != bindings.target_object_id or state["asset_id"] != target_asset:
            raise ValueError("T4 preparation must name the same target and asset")
        _validate_vector_recipe(state["position_world_m"], "T4 preparation position")
        preparation_orientation = state["quaternion_world_xyzw"]
        if isinstance(preparation_orientation, Mapping) and set(preparation_orientation) == {
            "yaw_rad"
        }:
            _validate_number_recipe(preparation_orientation["yaw_rad"], "T4 preparation yaw")
        else:
            _validate_camera_quaternion_recipe(preparation_orientation, "T4 preparation quaternion")
        for field in ("linear_velocity_world_m_s", "angular_velocity_world_rad_s"):
            _validate_vector_recipe(state[field], f"T4 preparation {field}", domain="zero")

    fixture_ids = (
        bindings.ramp_fixture_id,
        bindings.release_gate_fixture_id,
        bindings.downstream_surface_fixture_id,
    )
    fixtures = _recipe_sequence(recipe["fixture_states"], "T4 fixture states")
    if len(fixtures) != 3:
        raise ValueError("T4 requires exactly ramp, gate, and downstream fixture states")
    for entry, expected_id in zip(fixtures, fixture_ids, strict=True):
        alternatives = _record_recipe_choices(entry, "T4 fixture state")
        for fixture in alternatives:
            _exact_fields(
                fixture,
                {"instance_id", "asset_id", "position_world_m", "quaternion_world_xyzw"},
                "T4 fixture state",
            )
            if fixture["instance_id"] != expected_id or fixture["asset_id"] != arena_asset:
                raise ValueError("T4 fixture state must preserve its composite-arena binding")
            fixture_position = fixture["position_world_m"]
            if isinstance(fixture_position, Mapping) and set(fixture_position) == {
                "ramp_center_offset_m"
            }:
                if expected_id != bindings.release_gate_fixture_id:
                    raise ValueError("only the reset gate uses a ramp center fixture offset")
                _validate_vector_recipe(
                    fixture_position["ramp_center_offset_m"], "T4 gate center offset"
                )
            else:
                _validate_vector_recipe(fixture_position, "T4 fixture position")
            _validate_camera_quaternion_recipe(fixture["quaternion_world_xyzw"], "T4 fixture")
    return fixture_ids


def _validate_free_dynamics_geometry(recipe: Mapping[str, object]) -> None:
    geometry = _mapping(recipe["geometry"], "T4 geometry")
    _exact_fields(
        geometry,
        {
            "support_surface_z_m",
            "ramp_angle_rad",
            "ramp_downstream_lip_world_m",
            "ramp_half_size_m",
            "capture_region_bounds_world_m",
            "exit_region_bounds_xy_m",
        },
        "T4 geometry",
    )
    _validate_number_recipe(geometry["support_surface_z_m"], "T4 support surface")
    _validate_number_recipe(geometry["ramp_angle_rad"], "T4 ramp angle", domain="positive")
    _validate_vector_recipe(geometry["ramp_downstream_lip_world_m"], "T4 ramp lip")
    _validate_vector_recipe(geometry["ramp_half_size_m"], "T4 ramp half size", domain="positive")
    _validate_t4_bounds_recipe(
        geometry["capture_region_bounds_world_m"], dimension=3, name="T4 capture bounds"
    )
    _validate_t4_bounds_recipe(
        geometry["exit_region_bounds_xy_m"], dimension=2, name="T4 exit bounds"
    )
    if _recipe_sequence(recipe["motion"], "T4 motion"):
        raise ValueError("T4 target becomes free after release and accepts no driven motion")


def _validate_free_dynamics_physics(
    recipe: Mapping[str, object],
    bindings: FreeDynamicsInterceptionBindings,
    fixture_ids: tuple[str, str, str],
) -> None:
    _validate_body_physics_recipes(recipe["physics"])
    physics = _mapping(recipe["physics"], "T4 physics")
    _validate_vector_recipe(physics["gravity_world_m_s2"], "T4 gravity")
    bodies = _recipe_sequence(physics["bodies"], "T4 body physics")
    if len(bodies) != 1:
        raise ValueError("T4 requires exactly one target body physics record")
    for body in _record_recipe_choices(bodies[0], "T4 target body physics"):
        if body["entity_id"] != bindings.target_object_id:
            raise ValueError("T4 body physics must resolve the target")
    expected_contact_ids = set((bindings.target_object_id, *fixture_ids))
    contacts = _recipe_sequence(physics["contacts"], "T4 contact physics")
    if len(contacts) != 4:
        raise ValueError("T4 requires target, ramp, gate, and downstream contact records")
    contact_ids: list[str] = []
    for entry in contacts:
        alternatives = _record_recipe_choices(entry, "T4 contact physics")
        entity_id = alternatives[0].get("entity_id")
        if not isinstance(entity_id, str) or entity_id not in expected_contact_ids:
            raise ValueError("T4 contact physics contains an unknown entity binding")
        contact_ids.append(entity_id)
        for contact in alternatives:
            _exact_fields(
                contact,
                {"entity_id", "friction_slide_spin_roll", "solref", "solimp"},
                "T4 contact physics",
            )
            if contact["entity_id"] != entity_id:
                raise ValueError("T4 contact alternatives must preserve entity binding")
            friction = contact["friction_slide_spin_roll"]
            if isinstance(friction, Mapping) and set(friction) == {"from_entity_id"}:
                if (
                    friction["from_entity_id"] not in expected_contact_ids
                    or friction["from_entity_id"] == entity_id
                ):
                    raise ValueError("T4 friction must reference another declared contact entity")
            else:
                _validate_vector_recipe(friction, "T4 friction", domain="nonnegative")
            _validate_vector_recipe(contact["solref"], "T4 solref", length=2)
            _validate_vector_recipe(contact["solimp"], "T4 solimp", length=5)
    if len(set(contact_ids)) != len(contact_ids) or set(contact_ids) != expected_contact_ids:
        raise ValueError("T4 contact physics must bind each entity exactly once")
    for entry in contacts:
        for contact in _record_recipe_choices(entry, "T4 contact physics"):
            friction = contact["friction_slide_spin_roll"]
            if isinstance(friction, Mapping) and set(friction) == {"from_entity_id"}:
                source_index = contact_ids.index(str(friction["from_entity_id"]))
                for source in _record_recipe_choices(contacts[source_index], "T4 source contact"):
                    _validate_vector_recipe(
                        source["friction_slide_spin_roll"],
                        "T4 shared friction source",
                        domain="nonnegative",
                    )
    _validate_joint_physics_recipes(
        physics["joints"],
        allowed_dofs={
            (bindings.target_object_id, f"{bindings.target_object_id}_joint0"): 6,
        },
        required=frozenset(),
        name="T4 joint physics",
    )


def _validate_free_dynamics_event(
    recipe: Mapping[str, object], bindings: FreeDynamicsInterceptionBindings
) -> None:
    events = _recipe_sequence(recipe["events"], "T4 events")
    if len(events) != 1:
        raise ValueError("T4 requires exactly one release event")
    for event in _record_recipe_choices(events[0], "T4 release event"):
        _exact_fields(event, {"event_id", "trigger", "effect"}, "T4 release event")
        if event["event_id"] != "release_gate":
            raise ValueError("T4 release event requires the canonical event_id")
        trigger = _mapping(event["trigger"], "T4 release trigger")
        _exact_fields(trigger, {"type", "control_tick"}, "T4 release trigger")
        if trigger != {"type": "time", "control_tick": 0}:
            raise ValueError("T4 release event must fire exactly at scored control tick 0")
        effect = _mapping(event["effect"], "T4 release effect")
        _exact_fields(
            effect,
            {"type", "gate_entity_id", "parked_position_world_m"},
            "T4 release effect",
        )
        if (
            effect["type"] != "release_gate_retraction"
            or effect["gate_entity_id"] != bindings.release_gate_fixture_id
        ):
            raise ValueError("T4 release effect must resolve the bound native gate")
        _validate_vector_recipe(effect["parked_position_world_m"], "T4 gate parked position")


def _validate_family_camera_profile(
    recipe: Mapping[str, object], *, light_id: str, name: str
) -> None:
    _validate_capture_sensor_recipes(recipe, light_id=light_id, name=name)
    cameras = _mapping(recipe["cameras"], f"{name} cameras")
    if (cameras["image_height_px"], cameras["image_width_px"]) != (224, 224):
        raise ValueError(f"{name} camera raster must be concrete 224x224")
    expected_views = (
        ("agentview", "wrist")
        if recipe["observation_profile"] == "agentview_wrist"
        else ("agentview",)
    )
    views = _recipe_sequence(cameras["views"], f"{name} camera views")
    if len(views) != len(expected_views) or any(
        camera["view_name"] != expected
        for entry, expected in zip(views, expected_views, strict=True)
        for camera in _record_recipe_choices(entry, f"{name} camera")
    ):
        raise ValueError(f"{name} camera order must match its explicit observation profile")


def _validate_free_dynamics_recipe(task: TaskSpec, recipe: Mapping[str, object]) -> None:
    """Validate all T4 alternatives before the single proposal RNG is consumed."""
    _exact_fields(recipe, _RECIPE_FIELDS | {"preparation_object_states"}, "T4 recipe")
    _validate_frozen_recipe_clock(recipe, "T4")
    if task.task_id not in _T4_ASSETS or not isinstance(
        task.bindings, FreeDynamicsInterceptionBindings
    ):
        raise ValueError("T4 recipe requires a registered FreeDynamicsInterception task")
    bindings = task.bindings
    arena_asset, target_asset = _T4_ASSETS[task.task_id]
    fixture_ids = _validate_free_dynamics_entities(
        recipe,
        bindings,
        arena_asset=arena_asset,
        target_asset=target_asset,
    )
    _validate_family_robot_recipe(recipe, "T4")
    _validate_free_dynamics_geometry(recipe)
    _validate_free_dynamics_physics(recipe, bindings, fixture_ids)
    _validate_free_dynamics_event(recipe, bindings)
    _validate_family_camera_profile(recipe, light_id="free_dynamics_key", name="T4")


def _t4_ramp_rotation(angle_rad: float) -> npt.NDArray[np.float64]:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, sine), (0.0, -sine, cosine)),
        dtype=np.float64,
    )


def _t4_target_quaternion(
    task_id: str, angle_rad: float, axis_yaw_rad: float
) -> npt.NDArray[np.float64]:
    ramp = _t4_ramp_rotation(angle_rad)
    if task_id != "can_capture":
        if axis_yaw_rad != 0.0:
            raise ValueError("only the T4 can has an initial in-plane axis yaw")
        if task_id in ("ball_capture", "ball_capture_on_table"):
            return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        rotation = ramp
    else:
        transverse, downhill, normal = ramp[:, 0], ramp[:, 1], ramp[:, 2]
        cosine, sine = math.cos(axis_yaw_rad), math.sin(axis_yaw_rad)
        rotation = np.column_stack(
            (cosine * downhill - sine * transverse, normal, cosine * transverse + sine * downhill)
        )
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return np.ascontiguousarray(quaternion[[1, 2, 3, 0]])


def _t4_ramp_center(
    task_id: str, geometry: Mapping[str, object], rotation: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    half = np.asarray(geometry["ramp_half_size_m"], dtype=np.float64)
    offset = np.asarray((0.0, half[1], 0.0 if task_id == "ball_capture" else half[2]))
    return cast(
        npt.NDArray[np.float64],
        np.asarray(geometry["ramp_downstream_lip_world_m"], dtype=np.float64) - rotation @ offset,
    )


def _materialize_recipe_fields(
    sampled: dict[str, object],
    recipe: Mapping[str, object],
    rng: np.random.Generator,
    fields: Sequence[str],
) -> None:
    for field in fields:
        sampled[field] = _sample_recipe_value(_ordered_recipe_value(recipe[field]), rng)


def _materialize_free_dynamics_state(
    task: TaskSpec,
    target: Mapping[str, object],
    geometry: Mapping[str, object],
    rng: np.random.Generator,
    gate_position_recipe: object,
) -> tuple[dict[str, object], dict[str, object]]:
    # These four atomic draws own the ramp frame, support pose, and can axis together.
    angle = _sample_recipe_number(geometry["ramp_angle_rad"], rng, "T4 ramp angle")
    raw_position = _mapping(target["position_world_m"], "T4 atomic target position")
    raw_orientation = _mapping(target["quaternion_world_xyzw"], "T4 atomic target orientation")
    axis_yaw = _sample_recipe_number(raw_orientation["axis_yaw_rad"], rng, "T4 axis yaw")
    lip = _sample_recipe_vector(geometry["ramp_downstream_lip_world_m"], rng, "T4 ramp lip")
    ramp_rotation = _t4_ramp_rotation(angle)
    if set(raw_position) == {"ramp_center_offset_m"}:
        raw_offset = _recipe_sequence(
            raw_position["ramp_center_offset_m"], "T4 target center offset"
        )
        lateral = _sample_recipe_number(raw_offset[0], rng, "T4 lateral offset")
        height = _sample_recipe_number(raw_offset[2], rng, "T4 center support height")
        center = _t4_ramp_center(task.task_id, geometry, ramp_rotation)
        if isinstance(raw_offset[1], Mapping) and set(raw_offset[1]) == {"from_release_gate"}:
            gate = _mapping(gate_position_recipe, "T4 can release gate position")
            _exact_fields(gate, {"ramp_center_offset_m"}, "T4 can release gate position")
            gate_offset = _sample_recipe_vector(gate["ramp_center_offset_m"], rng, "T4 gate offset")
            gate_world = center + ramp_rotation @ gate_offset
            gate_local = ramp_rotation.T @ (gate_world - center)
            extent = 0.045 * abs(math.sin(axis_yaw)) + 0.024 * abs(math.cos(axis_yaw))
            along = float(gate_local[1]) - 0.004 - extent
        else:
            along = _sample_recipe_number(raw_offset[1], rng, "T4 along-ramp center offset")
        position = center + ramp_rotation @ np.asarray((lateral, along, height))
    else:
        lateral = _sample_recipe_number(raw_position["lateral_offset_m"], rng, "T4 lateral offset")
        along = _sample_recipe_number(
            raw_position["along_ramp_offset_m"], rng, "T4 along-ramp offset"
        )
        position = lip + ramp_rotation @ np.asarray(
            (lateral, along, _T4_TARGET_SUPPORT_EXTENT_M[task.task_id]), dtype=np.float64
        )
    quaternion = _t4_target_quaternion(task.task_id, angle, axis_yaw)
    concrete_geometry = {
        "support_surface_z_m": _sample_recipe_value(geometry["support_surface_z_m"], rng),
        "ramp_angle_rad": angle,
        "ramp_downstream_lip_world_m": lip.tolist(),
        "ramp_half_size_m": _sample_recipe_value(
            _ordered_recipe_value(geometry["ramp_half_size_m"]), rng
        ),
        "capture_region_bounds_world_m": _sample_recipe_value(
            _ordered_recipe_value(geometry["capture_region_bounds_world_m"]), rng
        ),
        "exit_region_bounds_xy_m": _sample_recipe_value(
            _ordered_recipe_value(geometry["exit_region_bounds_xy_m"]), rng
        ),
    }
    concrete_target = {
        "instance_id": target["instance_id"],
        "asset_id": target["asset_id"],
        "position_world_m": position.tolist(),
        "quaternion_world_xyzw": quaternion.tolist(),
        "linear_velocity_world_m_s": _sample_recipe_value(
            _ordered_recipe_value(target["linear_velocity_world_m_s"]), rng
        ),
        "angular_velocity_world_rad_s": _sample_recipe_value(
            _ordered_recipe_value(target["angular_velocity_world_rad_s"]), rng
        ),
    }
    return concrete_geometry, concrete_target


def _materialize_free_dynamics_recipe(
    task: TaskSpec, recipe: Mapping[str, object], rng: np.random.Generator
) -> Mapping[str, object]:
    _validate_free_dynamics_recipe(task, recipe)
    assert isinstance(task.bindings, FreeDynamicsInterceptionBindings)
    target_choices = _record_recipe_choices(
        _recipe_sequence(recipe["object_states"], "T4 object states")[0],
        "T4 target object state",
    )
    target = dict(_choose_record(target_choices, rng))
    geometry = dict(_mapping(recipe["geometry"], "T4 geometry"))
    # The source manifest draws slope and lateral offset before friction, then
    # can yaw. Derived fields reuse those draws without consuming the RNG again.
    geometry["ramp_angle_rad"] = _sample_recipe_number(geometry["ramp_angle_rad"], rng, "T4 slope")
    position_recipe = dict(_mapping(target["position_world_m"], "T4 target position"))
    if set(position_recipe) == {"ramp_center_offset_m"}:
        offset = list(_recipe_sequence(position_recipe["ramp_center_offset_m"], "T4 center offset"))
        offset[0] = _sample_recipe_number(offset[0], rng, "T4 lateral offset")
        position_recipe["ramp_center_offset_m"] = offset
    else:
        position_recipe["lateral_offset_m"] = _sample_recipe_number(
            position_recipe["lateral_offset_m"], rng, "T4 lateral offset"
        )
    target["position_world_m"] = position_recipe
    concrete_physics = dict(
        _mapping(
            _sample_recipe_value(_ordered_recipe_value(recipe["physics"]), rng),
            "T4 concrete physics",
        )
    )
    contacts = [
        dict(_mapping(entry, "T4 concrete contact"))
        for entry in _recipe_sequence(concrete_physics["contacts"], "T4 concrete contacts")
    ]
    contact_by_id = {str(contact["entity_id"]): contact for contact in contacts}
    for contact in contacts:
        friction = contact["friction_slide_spin_roll"]
        if isinstance(friction, Mapping) and set(friction) == {"from_entity_id"}:
            source = contact_by_id[str(friction["from_entity_id"])]["friction_slide_spin_roll"]
            contact["friction_slide_spin_roll"] = list(
                _recipe_sequence(source, "T4 shared friction")
            )
    concrete_physics["contacts"] = contacts
    geometry = dict(
        _mapping(_sample_recipe_value(_ordered_recipe_value(geometry), rng), "T4 geometry")
    )
    fixtures = [
        dict(_mapping(_sample_recipe_value(_ordered_recipe_value(entry), rng), "T4 fixture"))
        for entry in _recipe_sequence(recipe["fixture_states"], "T4 fixtures")
    ]
    gate = next(
        fixture
        for fixture in fixtures
        if fixture["instance_id"] == task.bindings.release_gate_fixture_id
    )
    concrete_geometry, concrete_target = _materialize_free_dynamics_state(
        task, target, geometry, rng, gate["position_world_m"]
    )
    rotation = _t4_ramp_rotation(_number(concrete_geometry["ramp_angle_rad"], "T4 slope"))
    center = _t4_ramp_center(task.task_id, concrete_geometry, rotation)
    for fixture in fixtures:
        position = fixture["position_world_m"]
        if isinstance(position, Mapping) and set(position) == {"ramp_center_offset_m"}:
            fixture["position_world_m"] = (
                center + rotation @ np.asarray(position["ramp_center_offset_m"], dtype=np.float64)
            ).tolist()
    preparation_recipe = _choose_record(
        _record_recipe_choices(
            _recipe_sequence(recipe["preparation_object_states"], "T4 preparation states")[0],
            "T4 preparation target",
        ),
        rng,
    )
    # Temporary x, y, then yaw are explicit input draws after the existing episode factors.
    preparation = {
        field: _sample_recipe_value(_ordered_recipe_value(preparation_recipe[field]), rng)
        for field in (
            "instance_id",
            "asset_id",
            "position_world_m",
            "quaternion_world_xyzw",
            "linear_velocity_world_m_s",
            "angular_velocity_world_rad_s",
        )
    }
    sampled: dict[str, object] = {
        "geometry": concrete_geometry,
        "object_states": [concrete_target],
        "preparation_object_states": [preparation],
        "motion": [],
        "physics": concrete_physics,
        "fixture_states": fixtures,
    }
    _materialize_recipe_fields(
        sampled,
        recipe,
        rng,
        (
            "robot_initial_state",
            "events",
            "cameras",
            "lighting",
            "action_delay",
            "action_mode",
            "observation_profile",
            "physics_timestep_s",
            "control_frequency_hz",
            "sensor_frequency_hz",
        ),
    )
    return sampled


def _validate_replanning_object_recipe(
    recipe: Mapping[str, object], bindings: ReplanningBindings
) -> None:
    objects = _recipe_sequence(recipe["object_states"], "T5 object states")
    if len(objects) != 1:
        raise ValueError("T5 requires exactly one keyed-puck object state")
    for target in _record_recipe_choices(objects[0], "T5 target object state"):
        _exact_fields(target, _OBJECT_STATE_FIELDS, "T5 target object state")
        if (
            target["instance_id"] != bindings.target_object_id
            or target["asset_id"] != "obj.keyed_puck_v1"
        ):
            raise ValueError("T5 target must resolve the frozen keyed-puck binding")
        position = _mapping(target["position_world_m"], "T5 atomic target position")
        _exact_fields(position, {"x_m"}, "T5 atomic target position")
        _validate_number_recipe(position["x_m"], "T5 target x")
        orientation = _mapping(target["quaternion_world_xyzw"], "T5 atomic target orientation")
        _exact_fields(orientation, {"yaw_rad"}, "T5 atomic target orientation")
        _validate_number_recipe(orientation["yaw_rad"], "T5 target yaw")
        for field in ("linear_velocity_world_m_s", "angular_velocity_world_rad_s"):
            _validate_vector_recipe(target[field], f"T5 {field}", domain="zero")


def _validate_replanning_fixture_recipes(
    recipe: Mapping[str, object], bindings: ReplanningBindings
) -> tuple[str, str, str]:
    fixture_ids = (
        bindings.belt_fixture_id,
        bindings.left_striker_fixture_id,
        bindings.right_striker_fixture_id,
    )
    fixtures = _recipe_sequence(recipe["fixture_states"], "T5 fixture states")
    if len(fixtures) != 3:
        raise ValueError("T5 requires its belt and two striker fixture states")
    for entry, expected_id in zip(fixtures, fixture_ids, strict=True):
        for fixture in _record_recipe_choices(entry, "T5 fixture state"):
            _exact_fields(
                fixture,
                {"instance_id", "asset_id", "position_world_m", "quaternion_world_xyzw"},
                "T5 fixture state",
            )
            if (
                fixture["instance_id"] != expected_id
                or fixture["asset_id"] != "arena.replanning_conveyor_v1"
            ):
                raise ValueError("T5 fixtures must preserve the composite-arena bindings")
            _validate_vector_recipe(fixture["position_world_m"], "T5 fixture position")
            _validate_camera_quaternion_recipe(fixture["quaternion_world_xyzw"], "T5 fixture")
    return fixture_ids


def _validate_replanning_geometry_recipe(recipe: Mapping[str, object]) -> None:
    geometry = _mapping(recipe["geometry"], "T5 geometry")
    _exact_fields(
        geometry,
        {
            "support_surface_z_m",
            "observation_boundary_y_m",
            "workspace_bounds_xy_m",
            "striker_travel_m",
        },
        "T5 geometry",
    )
    _validate_number_recipe(geometry["support_surface_z_m"], "T5 support surface")
    _validate_number_recipe(geometry["observation_boundary_y_m"], "T5 observation boundary")
    _validate_t4_bounds_recipe(
        geometry["workspace_bounds_xy_m"], dimension=2, name="T5 workspace bounds"
    )
    _validate_number_recipe(geometry["striker_travel_m"], "T5 striker travel", domain="positive")


def _validate_replanning_motion_recipe(
    recipe: Mapping[str, object], bindings: ReplanningBindings
) -> None:
    motions = _recipe_sequence(recipe["motion"], "T5 motion")
    if len(motions) != 1:
        raise ValueError("T5 requires exactly one belt motion")
    for motion in _record_recipe_choices(motions[0], "T5 belt motion"):
        _exact_fields(motion, _MOTION_FIELDS, "T5 belt motion")
        if motion["entity_id"] != bindings.belt_fixture_id:
            raise ValueError("T5 motion must bind the belt")
        translation = _mapping(motion["translation"], "T5 belt translation")
        _exact_fields(translation, {"type", "velocity_world_m_s"}, "T5 belt translation")
        if translation["type"] != "constant_velocity":
            raise ValueError("T5 belt requires constant_velocity translation")
        _validate_axis_vector_recipe(
            translation["velocity_world_m_s"],
            axis=1,
            domain="positive",
            name="T5 belt velocity",
        )
        rotation = _mapping(motion["rotation"], "T5 belt rotation")
        if rotation != {"type": "fixed"}:
            raise ValueError("T5 belt rotation must be fixed")


def _validate_replanning_physics_recipe(
    recipe: Mapping[str, object],
    bindings: ReplanningBindings,
    fixture_ids: tuple[str, str, str],
) -> None:
    _validate_body_physics_recipes(recipe["physics"])
    physics = _mapping(recipe["physics"], "T5 physics")
    _validate_vector_recipe(physics["gravity_world_m_s2"], "T5 gravity")
    bodies = _recipe_sequence(physics["bodies"], "T5 body physics")
    if len(bodies) != 1:
        raise ValueError("T5 requires exactly one target body-physics record")
    for body in _record_recipe_choices(bodies[0], "T5 target body physics"):
        if body["entity_id"] != bindings.target_object_id:
            raise ValueError("T5 body physics must resolve the target")
    contacts = _recipe_sequence(physics["contacts"], "T5 contact physics")
    expected_contacts = set((bindings.target_object_id, *fixture_ids))
    if len(contacts) != 4:
        raise ValueError("T5 requires target, belt, and both striker contact records")
    contact_ids: list[str] = []
    for entry in contacts:
        alternatives = _record_recipe_choices(entry, "T5 contact physics")
        entity_id = alternatives[0].get("entity_id")
        if not isinstance(entity_id, str) or entity_id not in expected_contacts:
            raise ValueError("T5 contact physics contains an unknown entity binding")
        contact_ids.append(entity_id)
        for contact in alternatives:
            _exact_fields(
                contact,
                {"entity_id", "friction_slide_spin_roll", "solref", "solimp"},
                "T5 contact physics",
            )
            if contact["entity_id"] != entity_id:
                raise ValueError("T5 contact alternatives must preserve entity binding")
            _validate_vector_recipe(
                contact["friction_slide_spin_roll"], "T5 friction", domain="nonnegative"
            )
            _validate_vector_recipe(contact["solref"], "T5 solref", length=2)
            _validate_vector_recipe(contact["solimp"], "T5 solimp", length=5)
    if len(set(contact_ids)) != 4 or set(contact_ids) != expected_contacts:
        raise ValueError("T5 contact physics must bind each entity exactly once")
    expected_joints = frozenset(
        {
            (bindings.left_striker_fixture_id, "left_slide"),
            (bindings.right_striker_fixture_id, "right_slide"),
        }
    )
    _validate_joint_physics_recipes(
        physics["joints"],
        allowed_dofs={
            (bindings.target_object_id, f"{bindings.target_object_id}_joint0"): 6,
            **{key: 1 for key in expected_joints},
        },
        required=expected_joints,
        name="T5 joint physics",
    )


def _validate_replanning_event_recipe(
    recipe: Mapping[str, object], bindings: ReplanningBindings
) -> None:
    events = _recipe_sequence(recipe["events"], "T5 events")
    if len(events) != 1:
        raise ValueError("T5 requires exactly one selected-striker event")
    for event in _record_recipe_choices(events[0], "T5 selected-striker event"):
        _exact_fields(event, {"event_id", "trigger", "effect"}, "T5 selected-striker event")
        if event["event_id"] != "selected_strike":
            raise ValueError("T5 requires the canonical selected_strike event_id")
        if _mapping(event["trigger"], "T5 trigger") != {"type": "state"}:
            raise ValueError("T5 selected strike requires the state trigger")
        effect = _mapping(event["effect"], "T5 striker effect")
        _exact_fields(
            effect,
            {
                "type",
                "striker_entity_id",
                "joint_name",
                "strike_position_m",
                "pulse_duration_s",
                "kp_n_m",
                "kd_n_s_m",
                "maximum_force_n",
                "actuator_scale",
                "contact_cutoff_delay_s",
                "contact_grouping_hysteresis_s",
                "maximum_contact_duration_s",
            },
            "T5 striker effect",
        )
        branch = (
            "left"
            if effect["striker_entity_id"] == bindings.left_striker_fixture_id
            else "right"
            if effect["striker_entity_id"] == bindings.right_striker_fixture_id
            else ""
        )
        if (
            effect["type"] != "striker_pulse"
            or not branch
            or effect["joint_name"] != f"{branch}_slide"
            or effect["strike_position_m"] != 0.098
            or effect["pulse_duration_s"] != 0.350
            or effect["kp_n_m"] != 100.0
            or effect["kd_n_s_m"] != 1.0
            or effect["maximum_force_n"] != 4.0
            or effect["contact_cutoff_delay_s"] != 0.020
            or effect["contact_grouping_hysteresis_s"] != 0.012
            or effect["maximum_contact_duration_s"] != 0.100
        ):
            raise ValueError("T5 event must preserve its frozen selected-striker pulse")
        _validate_number_recipe(effect["actuator_scale"], "T5 actuator scale", domain="positive")


def _validate_replanning_recipe(task: TaskSpec, recipe: Mapping[str, object]) -> None:
    """Validate every T5 branch and sample record before consuming its RNG."""

    _validate_frozen_recipe_clock(recipe, "T5")
    if task.task_id != "post_impact_grasp" or not isinstance(task.bindings, ReplanningBindings):
        raise ValueError("T5 recipe requires the registered Replanning task")
    bindings = task.bindings
    _validate_replanning_object_recipe(recipe, bindings)
    fixture_ids = _validate_replanning_fixture_recipes(recipe, bindings)
    _validate_family_robot_recipe(recipe, "T5")
    _validate_replanning_geometry_recipe(recipe)
    _validate_replanning_motion_recipe(recipe, bindings)
    _validate_replanning_physics_recipe(recipe, bindings, fixture_ids)
    _validate_replanning_event_recipe(recipe, bindings)
    _validate_family_camera_profile(recipe, light_id="replanning_key", name="T5")


def _materialize_replanning_recipe(
    task: TaskSpec, recipe: Mapping[str, object], rng: np.random.Generator
) -> Mapping[str, object]:
    _validate_replanning_recipe(task, recipe)
    target = _choose_record(
        _record_recipe_choices(
            _recipe_sequence(recipe["object_states"], "T5 object states")[0],
            "T5 target object state",
        ),
        rng,
    )
    raw_position = _mapping(target["position_world_m"], "T5 atomic target position")
    raw_orientation = _mapping(target["quaternion_world_xyzw"], "T5 atomic target orientation")
    concrete_physics = _sample_recipe_value(_ordered_recipe_value(recipe["physics"]), rng)
    initial_x = _sample_recipe_number(raw_position["x_m"], rng, "T5 initial x")
    yaw = _sample_recipe_number(raw_orientation["yaw_rad"], rng, "T5 initial yaw")
    motion = _choose_record(
        _record_recipe_choices(
            _recipe_sequence(recipe["motion"], "T5 motion")[0], "T5 belt motion"
        ),
        rng,
    )
    concrete_motion = cast(
        dict[str, object], _sample_recipe_value(_ordered_recipe_value(motion), rng)
    )
    event = _choose_record(
        _record_recipe_choices(
            _recipe_sequence(recipe["events"], "T5 events")[0],
            "T5 selected-striker event",
        ),
        rng,
    )
    concrete_event = cast(
        dict[str, object], _sample_recipe_value(_ordered_recipe_value(event), rng)
    )
    half_yaw = 0.5 * yaw
    concrete_target = {
        "instance_id": target["instance_id"],
        "asset_id": target["asset_id"],
        "position_world_m": [initial_x, -0.29, 0.812],
        "quaternion_world_xyzw": [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)],
        "linear_velocity_world_m_s": _sample_recipe_value(
            _ordered_recipe_value(target["linear_velocity_world_m_s"]), rng
        ),
        "angular_velocity_world_rad_s": _sample_recipe_value(
            _ordered_recipe_value(target["angular_velocity_world_rad_s"]), rng
        ),
    }
    sampled: dict[str, object] = {
        "object_states": [concrete_target],
        "motion": [concrete_motion],
        "events": [concrete_event],
        "physics": concrete_physics,
    }
    _materialize_recipe_fields(
        sampled,
        recipe,
        rng,
        (
            "robot_initial_state",
            "fixture_states",
            "geometry",
            "cameras",
            "lighting",
            "action_delay",
            "action_mode",
            "observation_profile",
            "physics_timestep_s",
            "control_frequency_hz",
            "sensor_frequency_hz",
        ),
    )
    return sampled


def _validate_capture_sensor_recipes(
    recipe: Mapping[str, object], *, light_id: str, name: str
) -> None:
    cameras = _mapping(recipe["cameras"], f"{name} cameras")
    _exact_fields(cameras, {"image_height_px", "image_width_px", "views"}, f"{name} cameras")
    for entry in _recipe_sequence(cameras["views"], f"{name} camera views"):
        for camera in _record_recipe_choices(entry, f"{name} camera"):
            _exact_fields(
                camera,
                {"view_name", "position_parent_m", "quaternion_parent_xyzw", "vertical_fov_deg"},
                f"{name} camera",
            )
            if camera["view_name"] not in ("agentview", "wrist"):
                raise ValueError(f"{name} camera view_name must be agentview or wrist")
            _validate_vector_recipe(camera["position_parent_m"], f"{name} camera position")
            _validate_camera_quaternion_recipe(camera["quaternion_parent_xyzw"], name)
            _validate_number_recipe(
                camera["vertical_fov_deg"], f"{name} camera FOV", domain="camera_fov"
            )
    lighting = _mapping(recipe["lighting"], f"{name} lighting")
    _exact_fields(lighting, {"lights"}, f"{name} lighting")
    for entry in _recipe_sequence(lighting["lights"], f"{name} lights"):
        for light in _record_recipe_choices(entry, f"{name} light"):
            _exact_fields(light, set(_GRASP_CANONICAL_LIGHT), f"{name} light")
            if light["light_id"] != light_id or not isinstance(light["active"], bool):
                raise ValueError(f"{name} requires an explicit {light_id} light")
            _validate_vector_recipe(light["position_world_m"], f"{name} light position")
            _unit_vector_recipe_values(light["direction_world"], f"{name} light direction")
            for field in ("ambient_rgb", "diffuse_rgb", "specular_rgb"):
                _validate_vector_recipe(
                    light[field], f"{name} light {field}", domain="unit_interval"
                )
            _validate_vector_recipe(
                light["attenuation"], f"{name} light attenuation", domain="nonnegative"
            )
            _validate_number_recipe(
                light["cutoff_deg"], f"{name} light cutoff", domain="light_cutoff"
            )


def _validate_initial_quaternion_recipe(value: object, name: str) -> None:
    if isinstance(value, Mapping) and set(value) == {"yaw_rad"}:
        _validate_number_recipe(value["yaw_rad"], f"{name} yaw")
        return
    _validate_camera_quaternion_recipe(value, name)


def _validate_camera_quaternion_recipe(value: object, name: str) -> None:
    if isinstance(value, Mapping):
        for choice in _choice_values(_mapping(value, name), name):
            _validate_camera_quaternion_recipe(choice, name)
        return
    raw = _recipe_sequence(value, f"{name} camera quaternion")
    if len(raw) != 4:
        raise ValueError(f"{name} camera quaternion must have length 4")
    quaternion = tuple(_number(component, f"{name} camera quaternion") for component in raw)
    CameraSpec(
        "agentview", (0.0, 0.0, 0.0), cast(tuple[float, float, float, float], quaternion), 40.0
    )


def _materialize_recipe(
    task: TaskSpec, recipe: Mapping[str, object], rng: np.random.Generator
) -> ScenarioSpec:
    fields = _RECIPE_FIELDS | (
        {"preparation_object_states"} if "preparation_object_states" in recipe else set()
    )
    _exact_fields(recipe, fields, "training recipe")
    if recipe["action_mode"] != "E7" or recipe["observation_profile"] != "agentview":
        raise ValueError("training recipes require E7 actions and the agentview camera")
    if task.family != "free_dynamics_interception" and "preparation_object_states" in recipe:
        raise ValueError("only T4 recipes declare a temporary preparation target")
    if task.grasp_motion is GraspMotion.CONVEYOR:
        _validate_surface_recipe(task, recipe)
        # Added delay/view interventions cannot consume draws before the physical proposal.
        recipe = {
            field: recipe[field]
            for field in (
                *(field for field in _RECIPE_FIELD_ORDER if field != "action_delay"),
                "action_delay",
            )
        }
    sampled: object
    if task.family == "grasp":
        sampled = _materialize_grasp_recipe(task, recipe, rng)
    elif task.grasp_motion in (GraspMotion.MATCHED_XYZ, GraspMotion.MATCHED_YAW):
        sampled = _materialize_matched_recipe(task, recipe, rng)
    elif task.family == "moving_placement":
        _validate_placement_recipe(task, recipe)
        # Motion is sampled before any intervention on preparation or sensors.
        # The fixed order also makes YAML key order immaterial. No extra RNG exists.
        order = (
            "motion",
            "robot_initial_state",
            "object_states",
            "fixture_states",
            "geometry",
            "physics",
            "cameras",
            "lighting",
            "action_delay",
            "action_mode",
            "observation_profile",
            "physics_timestep_s",
            "control_frequency_hz",
            "sensor_frequency_hz",
            "events",
        )
        sampled = {
            field: _sample_recipe_value(_ordered_recipe_value(recipe[field]), rng)
            for field in order
        }
    elif task.family == "rotating_interaction":
        sampled = _materialize_rotating_recipe(task, recipe, rng)
    elif task.family == "free_dynamics_interception":
        sampled = _materialize_free_dynamics_recipe(task, recipe, rng)
    elif task.family == "replanning":
        sampled = _materialize_replanning_recipe(task, recipe, rng)
    else:
        sampled = _sample_recipe_value(recipe, rng)
    concrete = dict(_mapping(sampled, "materialized recipe"))
    delay = _mapping(concrete.pop("action_delay"), "materialized action_delay")
    _exact_fields(delay, _ACTION_DELAY_FIELDS, "materialized action_delay")
    enabled = delay["enabled"]
    delay_steps = delay["delay_steps"]
    if (
        not isinstance(enabled, bool)
        or isinstance(delay_steps, bool)
        or not isinstance(delay_steps, int)
    ):
        raise ValueError("materialized action_delay has invalid types")
    if (enabled and delay_steps <= 0) or (not enabled and delay_steps != 0):
        raise ValueError("materialized action_delay enabled/delay_steps combination is invalid")
    concrete["action_delay_steps"] = delay_steps
    scenario = scenario_from_dict(concrete, family=task.family)
    _validate_task_scenario(task, scenario)

    initial_poses = {
        state.instance_id: (state.position_world_m, state.quaternion_world_xyzw)
        for state in scenario.object_states + scenario.fixture_states
    }
    for motion in scenario.motion:
        position, quaternion = initial_poses[motion.entity_id]
        _validate_motion_initial_pose(
            motion,
            initial_position_world_m=position,
            initial_quaternion_world_xyzw=quaternion,
        )
    return scenario


def _validate_axis_vector_recipe(
    value: object,
    *,
    axis: int,
    domain: Literal["finite", "nonnegative", "positive"],
    name: str,
) -> None:
    if isinstance(value, Mapping):
        for candidate in _choice_values(_mapping(value, name), name):
            _validate_axis_vector_recipe(candidate, axis=axis, domain=domain, name=name)
        return
    vector = _recipe_sequence(value, name)
    if len(vector) != 3:
        raise ValueError(f"{name} must have length 3")
    for index, component in enumerate(vector):
        _validate_number_recipe(component, name, domain=domain if index == axis else "zero")


def _validate_matched_translation_recipe(value: Mapping[str, object]) -> None:
    name = "T1 matched translation"
    if value.get("type") == "constant_velocity":
        _exact_fields(value, {"type", "velocity_world_m_s"}, name)
        _validate_axis_vector_recipe(
            value["velocity_world_m_s"], axis=1, domain="positive", name=name
        )
    elif value.get("type") == "harmonic":
        _exact_fields(
            value,
            {"type", "drift_velocity_world_m_s", "amplitude_m", "frequency_hz", "phase_rad"},
            name,
        )
        _validate_axis_vector_recipe(
            value["drift_velocity_world_m_s"], axis=1, domain="positive", name=f"{name} drift"
        )
        for field in ("amplitude_m", "frequency_hz", "phase_rad"):
            _validate_axis_vector_recipe(
                value[field],
                axis=0,
                domain="finite" if field == "phase_rad" else "nonnegative",
                name=f"{name} {field}",
            )
    else:
        raise ValueError("T1 matched translation requires harmonic or constant_velocity")


def _number_from_uniform(value: object, unit: float) -> float:
    """Map one RNG draw through a validated scalar/min-max/choices recipe."""
    if not isinstance(value, Mapping):
        return _number(value, "T1 matched scalar")
    raw = _mapping(value, "T1 matched scalar")
    if set(raw) == {"min", "max"}:
        lower = _number(raw["min"], "T1 matched min")
        return lower + unit * (_number(raw["max"], "T1 matched max") - lower)
    choices = _choice_values(raw, "T1 matched scalar")
    interval = unit * len(choices)
    index = int(interval)
    return _number_from_uniform(choices[index], interval - index)


def _axis_value_from_uniform(value: object, axis: int, unit: float) -> float:
    if isinstance(value, Mapping):
        choices = _choice_values(_mapping(value, "T1 matched vector"), "T1 matched vector")
        interval = unit * len(choices)
        index = int(interval)
        return _axis_value_from_uniform(choices[index], axis, interval - index)
    vector = _recipe_sequence(value, "T1 matched vector")
    return _number_from_uniform(vector[axis], unit)


def _ordered_recipe_value(value: object) -> object:
    if isinstance(value, Mapping):
        raw = _mapping(value, "T1 matched recipe")
        return {key: _ordered_recipe_value(raw[key]) for key in sorted(raw)}
    if isinstance(value, list):
        return [_ordered_recipe_value(item) for item in value]
    return value


def _validate_matched_recipe(
    task: TaskSpec, recipe: Mapping[str, object]
) -> tuple[
    str,
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    if not isinstance(task.bindings, DrivenInterceptionBindings):
        raise ValueError("T1 matched recipe requires DrivenInterceptionBindings")
    target_id = task.bindings.target_object_id
    if task.bindings.driver_entity_id != target_id:
        raise ValueError("T1 matched driver must be the target free body")
    objects = _recipe_sequence(recipe["object_states"], "T1 matched objects")
    motions = _recipe_sequence(recipe["motion"], "T1 matched motion")
    if len(objects) != 1 or len(motions) != 1 or recipe["fixture_states"] != []:
        raise ValueError("T1 matched recipe requires one target, one motion and no fixtures")
    object_choices = _record_recipe_choices(objects[0], "T1 matched object")
    for target in object_choices:
        _exact_fields(target, _OBJECT_STATE_FIELDS, "T1 matched object")
        if target["instance_id"] != target_id or target["asset_id"] != "obj.asym_t_t1_v1":
            raise ValueError(
                "T1 matched object requires its bound target and fixed asymmetric T asset"
            )
        _validate_initial_position_recipe(target["position_world_m"], "T1 matched initial position")
        yaw = _mapping(target["quaternion_world_xyzw"], "T1 matched initial quaternion")
        _exact_fields(yaw, {"yaw_rad"}, "T1 matched initial quaternion")
        _validate_number_recipe(yaw["yaw_rad"], "T1 matched initial yaw")
        for field in ("linear_velocity_world_m_s", "angular_velocity_world_rad_s"):
            _validate_initial_velocity_recipe(target[field], f"T1 matched initial {field}")
    motion_choices = _record_recipe_choices(motions[0], "T1 matched motion")
    for motion in motion_choices:
        _exact_fields(motion, _MOTION_FIELDS, "T1 matched motion")
        if motion["entity_id"] != target_id:
            raise ValueError("T1 matched motion must reference the target")
        for law in _record_recipe_choices(motion["translation"], "T1 matched translation"):
            _validate_matched_translation_recipe(law)
        for law in _record_recipe_choices(motion["rotation"], "T1 matched rotation"):
            _exact_fields(law, {"type", "axis_world", "angular_speed_rad_s"}, "T1 matched rotation")
            if law["type"] != "constant_angular_velocity" or law["axis_world"] != [0.0, 0.0, 1.0]:
                raise ValueError("T1 matched rotation requires constant world +Z angular velocity")
            _validate_number_recipe(
                law["angular_speed_rad_s"],
                "T1 matched angular speed",
                domain="zero" if task.grasp_motion is GraspMotion.MATCHED_XYZ else "nonzero",
            )
    _validate_body_physics_recipes(recipe["physics"])
    physics = _mapping(recipe["physics"], "T1 matched physics")
    _validate_vector_recipe(physics["gravity_world_m_s2"], "T1 matched gravity")
    bodies = _recipe_sequence(physics["bodies"], "T1 matched bodies")
    contacts = _recipe_sequence(physics["contacts"], "T1 matched contacts")
    _validate_joint_physics_recipes(
        physics["joints"],
        allowed_dofs={(target_id, f"{target_id}_joint0"): 6},
        name="T1 matched joint physics",
    )
    if len(bodies) != 1 or len(contacts) != 1:
        raise ValueError("T1 matched physics requires one target body/contact")
    for body in _record_recipe_choices(bodies[0], "T1 matched body"):
        if body["entity_id"] != target_id:
            raise ValueError("T1 matched body physics must reference the target")
    for contact in _record_recipe_choices(contacts[0], "T1 matched contact"):
        _exact_fields(
            contact,
            {"entity_id", "friction_slide_spin_roll", "solref", "solimp"},
            "T1 matched contact",
        )
        if contact["entity_id"] != target_id:
            raise ValueError("T1 matched contact must reference the target")
        _validate_vector_recipe(
            contact["friction_slide_spin_roll"], "T1 matched friction", domain="nonnegative"
        )
        _validate_vector_recipe(contact["solref"], "T1 matched solref", length=2)
        _validate_vector_recipe(contact["solimp"], "T1 matched solimp", length=5)
    _validate_capture_sensor_recipes(recipe, light_id="table_key", name="T1 matched")
    return target_id, object_choices, motion_choices


def _sample_matched_motion_recipe(
    motion_choices: tuple[Mapping[str, object], ...], rng: np.random.Generator
) -> tuple[dict[str, object], dict[str, object]]:
    # Every law consumes these same eight draws, including explicit zero yaw and
    # absent harmonic parameters. Non-motion interventions cannot change the law.
    law_units = rng.random(8)
    motion = motion_choices[int(law_units[0] * len(motion_choices))]
    translations = _record_recipe_choices(motion["translation"], "T1 matched translation")
    translation = translations[int(law_units[1] * len(translations))]
    velocity_field = (
        "drift_velocity_world_m_s" if translation["type"] == "harmonic" else "velocity_world_m_s"
    )
    sampled_translation: dict[str, object] = {
        "type": translation["type"],
        velocity_field: [
            0.0,
            _axis_value_from_uniform(translation[velocity_field], 1, float(law_units[2])),
            0.0,
        ],
    }
    if translation["type"] == "harmonic":
        for index, field in enumerate(("amplitude_m", "frequency_hz", "phase_rad"), start=3):
            sampled_translation[field] = [
                _axis_value_from_uniform(translation[field], 0, float(law_units[index])),
                0.0,
                0.0,
            ]
    rotations = _record_recipe_choices(motion["rotation"], "T1 matched rotation")
    rotation = rotations[int(law_units[6] * len(rotations))]
    sampled_rotation = {
        "type": rotation["type"],
        "axis_world": rotation["axis_world"],
        "angular_speed_rad_s": _number_from_uniform(
            rotation["angular_speed_rad_s"], float(law_units[7])
        ),
    }
    return sampled_translation, sampled_rotation


def _materialize_matched_recipe(
    task: TaskSpec, recipe: Mapping[str, object], rng: np.random.Generator
) -> dict[str, object]:
    """Use fixed law draws, then sample pose and compile the common COM reference."""
    target_id, object_choices, motion_choices = _validate_matched_recipe(task, recipe)
    sampled_translation, sampled_rotation = _sample_matched_motion_recipe(motion_choices, rng)
    target = _choose_record(object_choices, rng)
    position, quaternion = _sample_planar_initial_pose(target, rng, name="T1 matched")
    concrete_translation = _materialize_translation_recipe(
        sampled_translation,
        position,
        rng,
        name="T1 matched",
        harmonic_center=_uses_harmonic_center(target),
    )
    concrete_motion: dict[str, object] = {
        "entity_id": target_id,
        "translation": concrete_translation,
        "rotation": _materialize_rotation_recipe(
            sampled_rotation, quaternion, rng, name="T1 matched"
        ),
    }
    concrete: dict[str, object] = {}
    for field in _RECIPE_FIELD_ORDER:
        if field == "object_states":
            concrete[field] = [
                {
                    "instance_id": target_id,
                    "asset_id": target["asset_id"],
                    "quaternion_world_xyzw": quaternion.tolist(),
                    **_materialize_initial_motion_fields(
                        target, concrete_motion, position, quaternion, rng
                    ),
                }
            ]
        elif field != "motion":
            concrete[field] = _sample_recipe_value(_ordered_recipe_value(recipe[field]), rng)
    concrete["motion"] = [concrete_motion]
    return concrete
