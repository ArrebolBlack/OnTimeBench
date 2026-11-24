from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Literal, TypeAlias

from ontimebench.control.action import ActionMode

ObservationProfile: TypeAlias = Literal["agentview", "agentview_wrist"]
Family: TypeAlias = Literal[
    "grasp",
    "driven_interception",
    "moving_placement",
    "rotating_interaction",
    "free_dynamics_interception",
    "replanning",
]


@dataclass(frozen=True, slots=True)
class GraspBindings:
    target_object_id: str

    def __post_init__(self) -> None:
        _text(self.target_object_id, "target_object_id")


@dataclass(frozen=True, slots=True)
class DrivenInterceptionBindings:
    target_object_id: str
    driver_entity_id: str
    occluder_fixture_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.target_object_id, "target_object_id")
        _text(self.driver_entity_id, "driver_entity_id")
        _record_string_tuple(self.occluder_fixture_ids, "occluder_fixture_ids")
        occupied = {self.target_object_id, self.driver_entity_id}
        if any(item in occupied for item in self.occluder_fixture_ids):
            raise ValueError("occluder bindings must differ from target and driver")


@dataclass(frozen=True, slots=True)
class MovingPlacementBindings:
    payload_object_id: str
    goal_fixture_id: str

    def __post_init__(self) -> None:
        _distinct_text_fields(
            (self.payload_object_id, self.goal_fixture_id), "moving-placement binding entity IDs"
        )


@dataclass(frozen=True, slots=True)
class RotatingInteractionBindings:
    wand_object_id: str
    wire_fixture_id: str
    cradle_fixture_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _distinct_text_fields(
            (self.wand_object_id, self.wire_fixture_id), "rotating binding entity IDs"
        )
        _record_string_tuple(self.cradle_fixture_ids, "cradle_fixture_ids")
        occupied = {self.wand_object_id, self.wire_fixture_id}
        if any(item in occupied for item in self.cradle_fixture_ids):
            raise ValueError("cradle bindings must differ from wand and wire")


@dataclass(frozen=True, slots=True)
class FreeDynamicsInterceptionBindings:
    target_object_id: str
    ramp_fixture_id: str
    release_gate_fixture_id: str
    downstream_surface_fixture_id: str

    def __post_init__(self) -> None:
        _distinct_text_fields(
            (
                self.target_object_id,
                self.ramp_fixture_id,
                self.release_gate_fixture_id,
                self.downstream_surface_fixture_id,
            ),
            "free-dynamics binding entity IDs",
        )


@dataclass(frozen=True, slots=True)
class ReplanningBindings:
    target_object_id: str
    belt_fixture_id: str
    left_striker_fixture_id: str
    right_striker_fixture_id: str

    def __post_init__(self) -> None:
        _distinct_text_fields(
            (
                self.target_object_id,
                self.belt_fixture_id,
                self.left_striker_fixture_id,
                self.right_striker_fixture_id,
            ),
            "replanning binding entity IDs",
        )


TaskBindings: TypeAlias = (
    GraspBindings
    | DrivenInterceptionBindings
    | MovingPlacementBindings
    | RotatingInteractionBindings
    | FreeDynamicsInterceptionBindings
    | ReplanningBindings
)


@dataclass(frozen=True, slots=True)
class CaptureSuccessSpec:
    bilateral_contact_duration_s: float
    minimum_lift_or_clearance_m: float
    stable_hold_duration_s: float

    def __post_init__(self) -> None:
        _positive_fields(self)


@dataclass(frozen=True, slots=True)
class TranslatingPlacementSuccessSpec:
    minimum_insertion_depth_m: float
    planar_overlap_tolerance_m: float
    placement_hold_duration_s: float
    maximum_held_rim_force_n: float
    held_rim_force_window_s: float

    def __post_init__(self) -> None:
        _positive_fields(self)


@dataclass(frozen=True, slots=True)
class SE3PlacementSuccessSpec:
    minimum_insertion_depth_m: float
    planar_overlap_tolerance_m: float
    placement_hold_duration_s: float
    maximum_held_rim_force_n: float
    held_rim_force_window_s: float
    maximum_tilt_rad: float
    # Only the goal-approach stage metric uses this; success needs the T seated in its slot.
    maximum_orientation_error_rad: float

    def __post_init__(self) -> None:
        _positive_fields(self)


@dataclass(frozen=True, slots=True)
class PhaseSuccessSpec:
    phase_alignment_distance_m: float
    loop_entry_distance_m: float
    path_progress_threshold: float
    terminal_progress_threshold: float
    terminal_distance_m: float

    def __post_init__(self) -> None:
        _validate_rotating_success(self)


@dataclass(frozen=True, slots=True)
class FullRotatingSuccessSpec:
    phase_alignment_distance_m: float
    loop_entry_distance_m: float
    path_progress_threshold: float
    terminal_progress_threshold: float
    terminal_distance_m: float
    wire_exit_clearance_m: float
    require_wand_release: bool

    def __post_init__(self) -> None:
        _validate_rotating_success(self)
        if self.require_wand_release is not True:
            raise ValueError("full rotating success requires wand release")


@dataclass(frozen=True, slots=True)
class UnsupportedCaptureSuccessSpec:
    require_contact_after_ramp_exit: bool
    require_contact_before_tray: bool
    bilateral_contact_duration_s: float
    minimum_lift_or_clearance_m: float
    stable_hold_duration_s: float

    def __post_init__(self) -> None:
        if self.require_contact_after_ramp_exit is not True:
            raise ValueError("unsupported capture requires contact after ramp exit")
        if self.require_contact_before_tray is not True:
            raise ValueError("unsupported capture requires contact before tray")
        _positive_fields(
            self,
            exclude={"require_contact_after_ramp_exit", "require_contact_before_tray"},
        )


@dataclass(frozen=True, slots=True)
class SupportedCaptureSuccessSpec:
    require_supported_contact: bool
    minimum_contact_speed_m_s: float
    require_transverse_closing_axis: bool
    bilateral_contact_duration_s: float
    minimum_lift_or_clearance_m: float
    stable_hold_duration_s: float

    def __post_init__(self) -> None:
        if self.require_supported_contact is not True:
            raise ValueError("supported capture requires supported contact")
        _boolean(self.require_transverse_closing_axis, "require_transverse_closing_axis")
        _positive_fields(
            self,
            exclude={"require_supported_contact", "require_transverse_closing_axis"},
        )


@dataclass(frozen=True, slots=True)
class ReplanningSuccessSpec:
    require_exactly_one_selected_impact: bool
    require_contact_after_impact: bool
    maximum_grasp_axis_error_deg: float
    bilateral_contact_duration_s: float
    minimum_lift_or_clearance_m: float
    stable_hold_duration_s: float

    def __post_init__(self) -> None:
        if self.require_exactly_one_selected_impact is not True:
            raise ValueError("replanning requires exactly one selected impact")
        if self.require_contact_after_impact is not True:
            raise ValueError("replanning requires robot contact after impact")
        _positive_fields(
            self,
            exclude={"require_exactly_one_selected_impact", "require_contact_after_impact"},
        )


TaskSuccessSpec: TypeAlias = (
    CaptureSuccessSpec
    | TranslatingPlacementSuccessSpec
    | SE3PlacementSuccessSpec
    | PhaseSuccessSpec
    | FullRotatingSuccessSpec
    | UnsupportedCaptureSuccessSpec
    | SupportedCaptureSuccessSpec
    | ReplanningSuccessSpec
)


class GraspMotion(Enum):
    FREE = "free"
    CONVEYOR = "conveyor"
    MATCHED_XYZ = "matched_xyz"
    MATCHED_YAW = "matched_yaw"

    @property
    def follows_yaw(self) -> bool:
        return self in (GraspMotion.FREE, GraspMotion.MATCHED_YAW)


_DRIVEN_MOTION = {
    "surface_driven": GraspMotion.CONVEYOR,
    "physical_occlusion": GraspMotion.CONVEYOR,
    "matched_xyz": GraspMotion.MATCHED_XYZ,
    "matched_yaw": GraspMotion.MATCHED_YAW,
}


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    family: Family
    variant: str
    bindings: TaskBindings
    allowed_observation_profiles: tuple[ObservationProfile, ...]
    allowed_action_modes: tuple[ActionMode, ...]
    episode_horizon_steps: int
    success: TaskSuccessSpec
    metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_task_spec(self)

    @property
    def grasp_motion(self) -> GraspMotion | None:
        if self.family == "grasp":
            return GraspMotion.FREE
        if self.family == "driven_interception":
            return _DRIVEN_MOTION[self.variant]
        return None

    @property
    def has_occluder(self) -> bool:
        return isinstance(self.bindings, DrivenInterceptionBindings) and bool(
            self.bindings.occluder_fixture_ids
        )


_TASK_FIELDS = {
    "schema_version",
    "variant",
    "bindings",
    "allowed_observation_profiles",
    "allowed_action_modes",
    "episode_horizon_steps",
    "success",
    "metrics",
}

_KNOWN_METRICS = (
    "success_rate",
    "task_completion",
    "completion_time_s",
    "safety_failure_rate",
    "failure_rate",
)

FAMILY_VARIANTS: dict[Family, frozenset[str]] = {
    "grasp": frozenset({"m0_m2"}),
    "driven_interception": frozenset(_DRIVEN_MOTION),
    "moving_placement": frozenset({"translation", "se3"}),
    "rotating_interaction": frozenset({"phase", "full"}),
    "free_dynamics_interception": frozenset(
        {"unsupported_sphere", "supported_sphere", "supported_slider", "supported_can"}
    ),
    "replanning": frozenset({"single_side_strike"}),
}


def _record_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    result = tuple(_text(item, name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _distinct_text_fields(values: tuple[object, ...], name: str) -> None:
    checked = tuple(_text(value, name) for value in values)
    if len(checked) != len(set(checked)):
        raise ValueError(f"{name} must be distinct")


def _positive_fields(record: Any, *, exclude: set[str] | None = None) -> None:
    omitted = exclude or set()
    for field in fields(record):
        if field.name not in omitted:
            _positive_float(getattr(record, field.name), field.name)


def _validate_rotating_success(value: PhaseSuccessSpec | FullRotatingSuccessSpec) -> None:
    _positive_fields(value, exclude={"require_wand_release"})
    if value.path_progress_threshold > 1 or value.terminal_progress_threshold > 1:
        raise ValueError("path progress thresholds must be at most one")
    if value.terminal_progress_threshold <= value.path_progress_threshold:
        raise ValueError("terminal progress must exceed path progress")


def _validate_task_spec(task: TaskSpec) -> None:
    _text(task.task_id, "task_id")
    if not isinstance(task.family, str) or task.family not in FAMILY_VARIANTS:
        raise ValueError(f"invalid task family: {task.family}")
    _text(task.variant, "variant")
    if task.variant not in FAMILY_VARIANTS[task.family]:
        raise ValueError(f"invalid {task.family} variant: {task.variant}")
    expected_bindings: dict[Family, type[object]] = {
        "grasp": GraspBindings,
        "driven_interception": DrivenInterceptionBindings,
        "moving_placement": MovingPlacementBindings,
        "rotating_interaction": RotatingInteractionBindings,
        "free_dynamics_interception": FreeDynamicsInterceptionBindings,
        "replanning": ReplanningBindings,
    }
    if not isinstance(task.bindings, expected_bindings[task.family]):
        raise ValueError("bindings record does not match task family")
    if isinstance(task.bindings, DrivenInterceptionBindings):
        bindings = task.bindings
        if len(bindings.occluder_fixture_ids) != int(task.variant == "physical_occlusion"):
            raise ValueError("only the occluded conveyor requires a single canopy fixture")
        if (bindings.target_object_id != bindings.driver_entity_id) != (
            task.grasp_motion is GraspMotion.CONVEYOR
        ):
            raise ValueError(
                "a conveyor drives a separate target; matched targets drive themselves"
            )
    expected_success: type[object]
    if task.grasp_motion is not None:
        expected_success = CaptureSuccessSpec
    elif task.variant == "translation":
        expected_success = TranslatingPlacementSuccessSpec
    elif task.variant == "se3":
        expected_success = SE3PlacementSuccessSpec
    elif task.variant == "phase":
        expected_success = PhaseSuccessSpec
    elif task.variant == "full":
        expected_success = FullRotatingSuccessSpec
    elif task.variant == "unsupported_sphere":
        expected_success = UnsupportedCaptureSuccessSpec
    elif task.variant in {"supported_sphere", "supported_slider", "supported_can"}:
        expected_success = SupportedCaptureSuccessSpec
    else:
        expected_success = ReplanningSuccessSpec
    if not isinstance(task.success, expected_success):
        raise ValueError("success record does not match task variant")
    if isinstance(task.success, SupportedCaptureSuccessSpec):
        if task.success.require_transverse_closing_axis:
            raise ValueError("OnTimeBench supported capture uses bilateral fingerpad contact")
    if task.allowed_observation_profiles != ("agentview", "agentview_wrist"):
        raise ValueError("allowed observation profiles must equal the frozen ordered pair")
    if task.allowed_action_modes != ("E7", "J8"):
        raise ValueError("allowed action modes must equal the frozen ordered pair")
    if (
        isinstance(task.episode_horizon_steps, bool)
        or not isinstance(task.episode_horizon_steps, int)
        or task.episode_horizon_steps <= 0
    ):
        raise ValueError("episode_horizon_steps must be a positive integer")
    if task.metrics != _KNOWN_METRICS:
        raise ValueError("metrics must equal the frozen ordered metric tuple")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _exact(mapping: Mapping[str, object], fields: set[str], name: str) -> None:
    missing = fields - mapping.keys()
    unknown = mapping.keys() - fields
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} unknown fields: {sorted(unknown)}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_float(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = tuple(_text(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _parse_bindings(value: object, family: Family) -> TaskBindings:
    if family not in FAMILY_VARIANTS:
        raise ValueError(f"invalid task family: {family}")
    data = _mapping(value, "bindings")
    if family == "grasp":
        fields = {"target_object_id"}
        _exact(data, fields, "bindings")
        return GraspBindings(_text(data["target_object_id"], "target_object_id"))
    if family == "driven_interception":
        fields = {"target_object_id", "driver_entity_id", "occluder_fixture_ids"}
        _exact(data, fields, "bindings")
        return DrivenInterceptionBindings(
            _text(data["target_object_id"], "target_object_id"),
            _text(data["driver_entity_id"], "driver_entity_id"),
            _string_tuple(data["occluder_fixture_ids"], "occluder_fixture_ids"),
        )
    if family == "moving_placement":
        fields = {"payload_object_id", "goal_fixture_id"}
        _exact(data, fields, "bindings")
        return MovingPlacementBindings(
            _text(data["payload_object_id"], "payload_object_id"),
            _text(data["goal_fixture_id"], "goal_fixture_id"),
        )
    if family == "rotating_interaction":
        fields = {"wand_object_id", "wire_fixture_id", "cradle_fixture_ids"}
        _exact(data, fields, "bindings")
        return RotatingInteractionBindings(
            _text(data["wand_object_id"], "wand_object_id"),
            _text(data["wire_fixture_id"], "wire_fixture_id"),
            _string_tuple(data["cradle_fixture_ids"], "cradle_fixture_ids"),
        )
    if family == "free_dynamics_interception":
        fields = {
            "target_object_id",
            "ramp_fixture_id",
            "release_gate_fixture_id",
            "downstream_surface_fixture_id",
        }
        _exact(data, fields, "bindings")
        return FreeDynamicsInterceptionBindings(
            _text(data["target_object_id"], "target_object_id"),
            _text(data["ramp_fixture_id"], "ramp_fixture_id"),
            _text(data["release_gate_fixture_id"], "release_gate_fixture_id"),
            _text(data["downstream_surface_fixture_id"], "downstream_surface_fixture_id"),
        )
    if family != "replanning":
        raise ValueError(f"invalid task family: {family}")
    fields = {
        "target_object_id",
        "belt_fixture_id",
        "left_striker_fixture_id",
        "right_striker_fixture_id",
    }
    _exact(data, fields, "bindings")
    left = _text(data["left_striker_fixture_id"], "left_striker_fixture_id")
    right = _text(data["right_striker_fixture_id"], "right_striker_fixture_id")
    return ReplanningBindings(
        _text(data["target_object_id"], "target_object_id"),
        _text(data["belt_fixture_id"], "belt_fixture_id"),
        left,
        right,
    )


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _parse_success(value: object, family: Family, variant: str) -> TaskSuccessSpec:
    if family not in FAMILY_VARIANTS:
        raise ValueError(f"invalid task family: {family}")
    if variant not in FAMILY_VARIANTS[family]:
        raise ValueError(f"invalid {family} variant: {variant}")
    data = _mapping(value, "success")
    if family in {"grasp", "driven_interception"}:
        fields = {
            "bilateral_contact_duration_s",
            "minimum_lift_or_clearance_m",
            "stable_hold_duration_s",
        }
        _exact(data, fields, "success")
        return CaptureSuccessSpec(
            bilateral_contact_duration_s=_positive_float(
                data["bilateral_contact_duration_s"], "bilateral_contact_duration_s"
            ),
            minimum_lift_or_clearance_m=_positive_float(
                data["minimum_lift_or_clearance_m"], "minimum_lift_or_clearance_m"
            ),
            stable_hold_duration_s=_positive_float(
                data["stable_hold_duration_s"], "stable_hold_duration_s"
            ),
        )
    if family == "moving_placement":
        placement_fields = (
            "minimum_insertion_depth_m",
            "planar_overlap_tolerance_m",
            "placement_hold_duration_s",
            "maximum_held_rim_force_n",
            "held_rim_force_window_s",
        )
        extra = ("maximum_tilt_rad", "maximum_orientation_error_rad") if variant == "se3" else ()
        _exact(data, {*placement_fields, *extra}, "success")
        values = {name: _positive_float(data[name], name) for name in (*placement_fields, *extra)}
        if variant == "se3":
            return SE3PlacementSuccessSpec(**values)
        return TranslatingPlacementSuccessSpec(**values)
    if family == "rotating_interaction":
        fields = {
            "phase_alignment_distance_m",
            "loop_entry_distance_m",
            "path_progress_threshold",
            "terminal_progress_threshold",
            "terminal_distance_m",
        }
        if variant == "full":
            fields |= {"wire_exit_clearance_m", "require_wand_release"}
        _exact(data, fields, "success")
        path_progress = _positive_float(data["path_progress_threshold"], "path_progress_threshold")
        terminal_progress = _positive_float(
            data["terminal_progress_threshold"], "terminal_progress_threshold"
        )
        common = dict(
            phase_alignment_distance_m=_positive_float(
                data["phase_alignment_distance_m"], "phase_alignment_distance_m"
            ),
            loop_entry_distance_m=_positive_float(
                data["loop_entry_distance_m"], "loop_entry_distance_m"
            ),
            path_progress_threshold=path_progress,
            terminal_progress_threshold=terminal_progress,
            terminal_distance_m=_positive_float(data["terminal_distance_m"], "terminal_distance_m"),
        )
        if variant == "phase":
            return PhaseSuccessSpec(**common)
        if variant == "full":
            return FullRotatingSuccessSpec(
                **common,
                wire_exit_clearance_m=_positive_float(
                    data["wire_exit_clearance_m"], "wire_exit_clearance_m"
                ),
                require_wand_release=_boolean(data["require_wand_release"], "require_wand_release"),
            )
        raise ValueError(f"invalid rotating_interaction variant: {variant}")
    if family == "free_dynamics_interception" and variant == "unsupported_sphere":
        fields = {
            "require_contact_after_ramp_exit",
            "require_contact_before_tray",
            "bilateral_contact_duration_s",
            "minimum_lift_or_clearance_m",
            "stable_hold_duration_s",
        }
        _exact(data, fields, "success")
        return UnsupportedCaptureSuccessSpec(
            require_contact_after_ramp_exit=_boolean(
                data["require_contact_after_ramp_exit"], "require_contact_after_ramp_exit"
            ),
            require_contact_before_tray=_boolean(
                data["require_contact_before_tray"], "require_contact_before_tray"
            ),
            bilateral_contact_duration_s=_positive_float(
                data["bilateral_contact_duration_s"], "bilateral_contact_duration_s"
            ),
            minimum_lift_or_clearance_m=_positive_float(
                data["minimum_lift_or_clearance_m"], "minimum_lift_or_clearance_m"
            ),
            stable_hold_duration_s=_positive_float(
                data["stable_hold_duration_s"], "stable_hold_duration_s"
            ),
        )
    if family == "free_dynamics_interception" and variant in {
        "supported_sphere",
        "supported_slider",
        "supported_can",
    }:
        fields = {
            "require_supported_contact",
            "minimum_contact_speed_m_s",
            "require_transverse_closing_axis",
            "bilateral_contact_duration_s",
            "minimum_lift_or_clearance_m",
            "stable_hold_duration_s",
        }
        _exact(data, fields, "success")
        transverse = _boolean(
            data["require_transverse_closing_axis"], "require_transverse_closing_axis"
        )
        return SupportedCaptureSuccessSpec(
            require_supported_contact=_boolean(
                data["require_supported_contact"], "require_supported_contact"
            ),
            minimum_contact_speed_m_s=_positive_float(
                data["minimum_contact_speed_m_s"], "minimum_contact_speed_m_s"
            ),
            require_transverse_closing_axis=transverse,
            bilateral_contact_duration_s=_positive_float(
                data["bilateral_contact_duration_s"], "bilateral_contact_duration_s"
            ),
            minimum_lift_or_clearance_m=_positive_float(
                data["minimum_lift_or_clearance_m"], "minimum_lift_or_clearance_m"
            ),
            stable_hold_duration_s=_positive_float(
                data["stable_hold_duration_s"], "stable_hold_duration_s"
            ),
        )
    if family != "replanning":
        raise ValueError(f"invalid success schema for {family}/{variant}")
    fields = {
        "require_exactly_one_selected_impact",
        "require_contact_after_impact",
        "maximum_grasp_axis_error_deg",
        "bilateral_contact_duration_s",
        "minimum_lift_or_clearance_m",
        "stable_hold_duration_s",
    }
    _exact(data, fields, "success")
    return ReplanningSuccessSpec(
        _boolean(
            data["require_exactly_one_selected_impact"], "require_exactly_one_selected_impact"
        ),
        _boolean(data["require_contact_after_impact"], "require_contact_after_impact"),
        _positive_float(data["maximum_grasp_axis_error_deg"], "maximum_grasp_axis_error_deg"),
        _positive_float(data["bilateral_contact_duration_s"], "bilateral_contact_duration_s"),
        _positive_float(data["minimum_lift_or_clearance_m"], "minimum_lift_or_clearance_m"),
        _positive_float(data["stable_hold_duration_s"], "stable_hold_duration_s"),
    )


def parse_task_spec(data: Mapping[str, object], *, task_id: str, family: Family) -> TaskSpec:
    if family not in FAMILY_VARIANTS:
        raise ValueError(f"invalid task family: {family}")
    _exact(data, _TASK_FIELDS, "task")
    if (
        isinstance(data["schema_version"], bool)
        or not isinstance(data["schema_version"], int)
        or data["schema_version"] != 1
    ):
        raise ValueError("task schema_version must be 1")
    observations = _string_tuple(
        data["allowed_observation_profiles"], "allowed_observation_profiles"
    )
    actions = _string_tuple(data["allowed_action_modes"], "allowed_action_modes")
    horizon = data["episode_horizon_steps"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("episode_horizon_steps must be a positive integer")
    metrics = _string_tuple(data["metrics"], "metrics")
    return TaskSpec(
        task_id=_text(task_id, "task_id"),
        family=family,
        variant=_text(data["variant"], "variant"),
        bindings=_parse_bindings(data["bindings"], family),
        allowed_observation_profiles=observations,  # type: ignore[arg-type]
        allowed_action_modes=actions,  # type: ignore[arg-type]
        episode_horizon_steps=horizon,
        success=_parse_success(data["success"], family, _text(data["variant"], "variant")),
        metrics=metrics,
    )


__all__ = ["TaskSpec", "parse_task_spec"]
