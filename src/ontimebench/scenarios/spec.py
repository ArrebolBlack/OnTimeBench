from __future__ import annotations

import dataclasses
import math
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from ontimebench.control.action import ActionMode
from ontimebench.tasks.spec import Family

Vector2: TypeAlias = tuple[float, float]
Vector3: TypeAlias = tuple[float, float, float]
Vector4: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class RobotInitialStateSpec:
    joint_position_rad: tuple[float, float, float, float, float, float, float]
    gripper_position_m: Vector2

    def __post_init__(self) -> None:
        _record_vector(self.joint_position_rad, 7, "joint_position_rad")
        _record_vector(self.gripper_position_m, 2, "gripper_position_m")


@dataclass(frozen=True, slots=True)
class ObjectStateSpec:
    instance_id: str
    asset_id: str
    position_world_m: Vector3
    quaternion_world_xyzw: Vector4
    linear_velocity_world_m_s: Vector3
    angular_velocity_world_rad_s: Vector3

    def __post_init__(self) -> None:
        _text(self.instance_id, "instance_id")
        _text(self.asset_id, "asset_id")
        _record_vector(self.position_world_m, 3, "position_world_m")
        _record_quaternion(self.quaternion_world_xyzw, "quaternion_world_xyzw")
        _record_vector(self.linear_velocity_world_m_s, 3, "linear_velocity_world_m_s")
        _record_vector(self.angular_velocity_world_rad_s, 3, "angular_velocity_world_rad_s")


@dataclass(frozen=True, slots=True)
class FixtureStateSpec:
    instance_id: str
    asset_id: str
    position_world_m: Vector3
    quaternion_world_xyzw: Vector4

    def __post_init__(self) -> None:
        _text(self.instance_id, "instance_id")
        _text(self.asset_id, "asset_id")
        _record_vector(self.position_world_m, 3, "position_world_m")
        _record_quaternion(self.quaternion_world_xyzw, "quaternion_world_xyzw")


@dataclass(frozen=True, slots=True)
class GraspGeometrySpec:
    support_surface_z_m: float
    workspace_bounds_xy_m: tuple[Vector2, Vector2]
    downstream_exit_y_m: float

    def __post_init__(self) -> None:
        _number(self.support_surface_z_m, "support_surface_z_m")
        _record_bounds(self.workspace_bounds_xy_m, 2, "workspace_bounds_xy_m")
        _number(self.downstream_exit_y_m, "downstream_exit_y_m")


@dataclass(frozen=True, slots=True)
class DrivenInterceptionGeometrySpec:
    support_surface_z_m: float
    workspace_bounds_xy_m: tuple[Vector2, Vector2]
    capture_corridor_bounds_xy_m: tuple[Vector2, Vector2]
    downstream_exit_y_m: float
    occlusion_target_clearance_m: float

    def __post_init__(self) -> None:
        _number(self.support_surface_z_m, "support_surface_z_m")
        workspace = _record_bounds(self.workspace_bounds_xy_m, 2, "workspace_bounds_xy_m")
        corridor = _record_bounds(
            self.capture_corridor_bounds_xy_m, 2, "capture_corridor_bounds_xy_m"
        )
        if any(
            corridor[0][axis] < workspace[0][axis] or corridor[1][axis] > workspace[1][axis]
            for axis in range(2)
        ):
            raise ValueError("capture corridor must be contained in workspace bounds")
        _number(self.downstream_exit_y_m, "downstream_exit_y_m")
        clearance = _number(self.occlusion_target_clearance_m, "occlusion_target_clearance_m")
        if clearance < 0:
            raise ValueError("occlusion_target_clearance_m must be non-negative")


@dataclass(frozen=True, slots=True)
class MovingPlacementGeometrySpec:
    support_surface_z_m: float
    goal_planar_clearance_m: float
    goal_bottom_center_z_m: float
    goal_window_exit_y_m: float

    def __post_init__(self) -> None:
        _number(self.support_surface_z_m, "support_surface_z_m")
        if _number(self.goal_planar_clearance_m, "goal_planar_clearance_m") <= 0:
            raise ValueError("goal_planar_clearance_m must be positive")
        if _number(self.goal_bottom_center_z_m, "goal_bottom_center_z_m") <= 0:
            raise ValueError("goal_bottom_center_z_m must be positive")
        _number(self.goal_window_exit_y_m, "goal_window_exit_y_m")


@dataclass(frozen=True, slots=True)
class RotatingInteractionGeometrySpec:
    frame_center_world_m: Vector3
    rotation_axis_world: Vector3
    wire_radius_m: float
    wire_path_points_frame_m: tuple[Vector3, ...]

    def __post_init__(self) -> None:
        _record_vector(self.frame_center_world_m, 3, "frame_center_world_m")
        _record_unit_vector(self.rotation_axis_world, "rotation_axis_world")
        if _number(self.wire_radius_m, "wire_radius_m") <= 0:
            raise ValueError("wire_radius_m must be positive")
        if not isinstance(self.wire_path_points_frame_m, tuple):
            raise ValueError("wire_path_points_frame_m must be a tuple")
        points = tuple(
            _record_vector(point, 3, "wire_path_point") for point in self.wire_path_points_frame_m
        )
        if len(points) < 2 or any(left == right for left, right in zip(points, points[1:])):
            raise ValueError("wire path requires at least two distinct adjacent points")


@dataclass(frozen=True, slots=True)
class FreeDynamicsInterceptionGeometrySpec:
    support_surface_z_m: float
    ramp_angle_rad: float
    ramp_downstream_lip_world_m: Vector3
    ramp_half_size_m: Vector3
    capture_region_bounds_world_m: tuple[Vector3, Vector3]
    exit_region_bounds_xy_m: tuple[Vector2, Vector2]

    def __post_init__(self) -> None:
        _number(self.support_surface_z_m, "support_surface_z_m")
        if _number(self.ramp_angle_rad, "ramp_angle_rad") <= 0:
            raise ValueError("ramp_angle_rad must be positive")
        _record_vector(self.ramp_downstream_lip_world_m, 3, "ramp_downstream_lip_world_m")
        half = _record_vector(self.ramp_half_size_m, 3, "ramp_half_size_m")
        if any(component <= 0 for component in half):
            raise ValueError("ramp_half_size_m must be positive")
        _record_bounds(self.capture_region_bounds_world_m, 3, "capture_region_bounds_world_m")
        _record_bounds(self.exit_region_bounds_xy_m, 2, "exit_region_bounds_xy_m")


@dataclass(frozen=True, slots=True)
class ReplanningGeometrySpec:
    support_surface_z_m: float
    observation_boundary_y_m: float
    workspace_bounds_xy_m: tuple[Vector2, Vector2]
    striker_travel_m: float

    def __post_init__(self) -> None:
        _number(self.support_surface_z_m, "support_surface_z_m")
        _number(self.observation_boundary_y_m, "observation_boundary_y_m")
        _record_bounds(self.workspace_bounds_xy_m, 2, "workspace_bounds_xy_m")
        if _number(self.striker_travel_m, "striker_travel_m") <= 0:
            raise ValueError("striker_travel_m must be positive")


GeometrySpec: TypeAlias = (
    GraspGeometrySpec
    | DrivenInterceptionGeometrySpec
    | MovingPlacementGeometrySpec
    | RotatingInteractionGeometrySpec
    | FreeDynamicsInterceptionGeometrySpec
    | ReplanningGeometrySpec
)


@dataclass(frozen=True, slots=True)
class FixedTranslation:
    pass


@dataclass(frozen=True, slots=True)
class ConstantVelocityTranslation:
    velocity_world_m_s: Vector3

    def __post_init__(self) -> None:
        _record_vector(self.velocity_world_m_s, 3, "velocity_world_m_s")


@dataclass(frozen=True, slots=True)
class HarmonicTranslation:
    center_world_m: Vector3
    drift_velocity_world_m_s: Vector3
    amplitude_m: Vector3
    frequency_hz: Vector3
    phase_rad: Vector3

    def __post_init__(self) -> None:
        _record_vector(self.center_world_m, 3, "center_world_m")
        _record_vector(self.drift_velocity_world_m_s, 3, "drift_velocity_world_m_s")
        amplitude = _record_vector(self.amplitude_m, 3, "amplitude_m")
        frequency = _record_vector(self.frequency_hz, 3, "frequency_hz")
        _record_vector(self.phase_rad, 3, "phase_rad")
        if any(component < 0 for component in amplitude + frequency):
            raise ValueError("harmonic amplitude and frequency must be non-negative")


@dataclass(frozen=True, slots=True)
class CircularTranslation:
    center_world_m: Vector3
    axis_world: Vector3
    radial_direction_world: Vector3
    radius_m: float
    angular_speed_rad_s: float
    phase_rad: float

    def __post_init__(self) -> None:
        _record_vector(self.center_world_m, 3, "center_world_m")
        axis = _record_unit_vector(self.axis_world, "axis_world")
        radial = _record_unit_vector(self.radial_direction_world, "radial_direction_world")
        if not math.isclose(
            sum(a * b for a, b in zip(axis, radial, strict=True)), 0.0, abs_tol=1e-6
        ):
            raise ValueError("circular axis and radial direction must be orthogonal")
        if _number(self.radius_m, "radius_m") <= 0:
            raise ValueError("circular radius_m must be positive")
        _number(self.angular_speed_rad_s, "angular_speed_rad_s")
        _number(self.phase_rad, "phase_rad")


TranslationSpec: TypeAlias = (
    FixedTranslation | ConstantVelocityTranslation | HarmonicTranslation | CircularTranslation
)


@dataclass(frozen=True, slots=True)
class FixedRotation:
    pass


@dataclass(frozen=True, slots=True)
class ConstantAngularVelocityRotation:
    axis_world: Vector3
    angular_speed_rad_s: float

    def __post_init__(self) -> None:
        _record_unit_vector(self.axis_world, "axis_world")
        _number(self.angular_speed_rad_s, "angular_speed_rad_s")


@dataclass(frozen=True, slots=True)
class HarmonicRotation:
    base_quaternion_world_xyzw: Vector4
    axis_world: Vector3
    amplitude_rad: float
    frequency_hz: float
    phase_rad: float

    def __post_init__(self) -> None:
        _record_quaternion(self.base_quaternion_world_xyzw, "base_quaternion_world_xyzw")
        _record_unit_vector(self.axis_world, "axis_world")
        if _number(self.amplitude_rad, "amplitude_rad") < 0:
            raise ValueError("harmonic rotation amplitude must be non-negative")
        if _number(self.frequency_hz, "frequency_hz") < 0:
            raise ValueError("harmonic rotation frequency must be non-negative")
        _number(self.phase_rad, "phase_rad")


RotationSpec: TypeAlias = FixedRotation | ConstantAngularVelocityRotation | HarmonicRotation


@dataclass(frozen=True, slots=True)
class DrivenMotionSpec:
    entity_id: str
    translation: TranslationSpec
    rotation: RotationSpec

    def __post_init__(self) -> None:
        _text(self.entity_id, "entity_id")
        if not isinstance(
            self.translation,
            (
                FixedTranslation,
                ConstantVelocityTranslation,
                HarmonicTranslation,
                CircularTranslation,
            ),
        ):
            raise ValueError("invalid translation record")
        if not isinstance(
            self.rotation, (FixedRotation, ConstantAngularVelocityRotation, HarmonicRotation)
        ):
            raise ValueError("invalid rotation record")


@dataclass(frozen=True, slots=True)
class BodyPhysicsSpec:
    entity_id: str
    mass_kg: float
    diagonal_inertia_kg_m2: Vector3

    def __post_init__(self) -> None:
        _text(self.entity_id, "entity_id")
        if _number(self.mass_kg, "mass_kg") <= 0:
            raise ValueError("mass_kg must be positive")
        inertia = _record_vector(self.diagonal_inertia_kg_m2, 3, "diagonal_inertia_kg_m2")
        if any(component <= 0 for component in inertia):
            raise ValueError("diagonal inertia must be positive")


@dataclass(frozen=True, slots=True)
class ContactPhysicsSpec:
    entity_id: str
    friction_slide_spin_roll: Vector3
    solref: Vector2
    solimp: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        _text(self.entity_id, "entity_id")
        friction = _record_vector(self.friction_slide_spin_roll, 3, "friction_slide_spin_roll")
        if any(component < 0 for component in friction):
            raise ValueError("friction must be non-negative")
        _record_vector(self.solref, 2, "solref")
        _record_vector(self.solimp, 5, "solimp")


@dataclass(frozen=True, slots=True)
class JointPhysicsSpec:
    entity_id: str
    joint_name: str
    damping_per_dof: tuple[float, ...]
    friction_loss_per_dof: tuple[float, ...]

    def __post_init__(self) -> None:
        _text(self.entity_id, "entity_id")
        _text(self.joint_name, "joint_name")
        damping = _record_float_tuple(self.damping_per_dof, "damping_per_dof")
        friction = _record_float_tuple(self.friction_loss_per_dof, "friction_loss_per_dof")
        if not damping or len(damping) != len(friction) or any(v < 0 for v in damping + friction):
            raise ValueError(
                "joint per-DOF sequences must be non-empty, equal-length, and non-negative"
            )


@dataclass(frozen=True, slots=True)
class PhysicsSpec:
    gravity_world_m_s2: Vector3
    bodies: tuple[BodyPhysicsSpec, ...]
    contacts: tuple[ContactPhysicsSpec, ...]
    joints: tuple[JointPhysicsSpec, ...]

    def __post_init__(self) -> None:
        _record_vector(self.gravity_world_m_s2, 3, "gravity_world_m_s2")
        _record_typed_tuple(self.bodies, BodyPhysicsSpec, "bodies")
        _record_typed_tuple(self.contacts, ContactPhysicsSpec, "contacts")
        _record_typed_tuple(self.joints, JointPhysicsSpec, "joints")
        _unique_values((item.entity_id for item in self.bodies), "body physics entity IDs")
        _unique_values((item.entity_id for item in self.contacts), "contact physics entity IDs")
        _unique_values(
            ((item.entity_id, item.joint_name) for item in self.joints),
            "joint physics identities",
        )


@dataclass(frozen=True, slots=True)
class CameraSpec:
    view_name: Literal["agentview", "wrist"]
    position_parent_m: Vector3
    quaternion_parent_xyzw: Vector4
    vertical_fov_deg: float

    def __post_init__(self) -> None:
        if self.view_name not in ("agentview", "wrist"):
            raise ValueError("invalid camera view_name")
        _record_vector(self.position_parent_m, 3, "position_parent_m")
        _record_quaternion(self.quaternion_parent_xyzw, "quaternion_parent_xyzw")
        fov = _number(self.vertical_fov_deg, "vertical_fov_deg")
        if not 0 < fov < 180:
            raise ValueError("vertical_fov_deg must be between 0 and 180")


@dataclass(frozen=True, slots=True)
class CameraSetSpec:
    image_height_px: int
    image_width_px: int
    views: tuple[CameraSpec, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.image_height_px, bool)
            or isinstance(self.image_width_px, bool)
            or not isinstance(self.image_height_px, int)
            or not isinstance(self.image_width_px, int)
            or (self.image_height_px, self.image_width_px) != (224, 224)
        ):
            raise ValueError("v1 camera raster must be 224x224")
        _record_typed_tuple(self.views, CameraSpec, "views")
        names = tuple(view.view_name for view in self.views)
        if names not in (("agentview",), ("agentview", "wrist")):
            raise ValueError("camera views must match a frozen observation profile")


@dataclass(frozen=True, slots=True)
class LightSpec:
    light_id: str
    position_world_m: Vector3
    direction_world: Vector3
    ambient_rgb: Vector3
    diffuse_rgb: Vector3
    specular_rgb: Vector3
    attenuation: Vector3
    cutoff_deg: float
    active: bool

    def __post_init__(self) -> None:
        _text(self.light_id, "light_id")
        _record_vector(self.position_world_m, 3, "position_world_m")
        _record_unit_vector(self.direction_world, "direction_world")
        colors = (
            _record_vector(self.ambient_rgb, 3, "ambient_rgb"),
            _record_vector(self.diffuse_rgb, 3, "diffuse_rgb"),
            _record_vector(self.specular_rgb, 3, "specular_rgb"),
        )
        if any(component < 0 or component > 1 for color in colors for component in color):
            raise ValueError("light RGB values must be in [0, 1]")
        attenuation = _record_vector(self.attenuation, 3, "attenuation")
        if any(component < 0 for component in attenuation):
            raise ValueError("light attenuation must be non-negative")
        cutoff = _number(self.cutoff_deg, "cutoff_deg")
        if not 0 < cutoff <= 180:
            raise ValueError("cutoff_deg must be in (0, 180]")
        if not isinstance(self.active, bool):
            raise ValueError("light active must be bool")


@dataclass(frozen=True, slots=True)
class LightingSpec:
    lights: tuple[LightSpec, ...]

    def __post_init__(self) -> None:
        _record_typed_tuple(self.lights, LightSpec, "lights")
        _unique_values((light.light_id for light in self.lights), "light IDs")


@dataclass(frozen=True, slots=True)
class TimeTriggerSpec:
    control_tick: int

    def __post_init__(self) -> None:
        _integer(self.control_tick, "control_tick")


@dataclass(frozen=True, slots=True)
class ContactTriggerSpec:
    entity_a_id: str
    entity_b_id: str
    minimum_normal_force_n: float

    def __post_init__(self) -> None:
        _text(self.entity_a_id, "entity_a_id")
        _text(self.entity_b_id, "entity_b_id")
        if self.entity_a_id == self.entity_b_id:
            raise ValueError("contact trigger entities must differ")
        if _number(self.minimum_normal_force_n, "minimum_normal_force_n") < 0:
            raise ValueError("minimum_normal_force_n must be non-negative")


@dataclass(frozen=True, slots=True)
class StateTriggerSpec:
    pass


TriggerSpec: TypeAlias = TimeTriggerSpec | ContactTriggerSpec | StateTriggerSpec


@dataclass(frozen=True, slots=True)
class ReleaseGateRetractionEffectSpec:
    gate_entity_id: str
    parked_position_world_m: Vector3

    def __post_init__(self) -> None:
        _text(self.gate_entity_id, "gate_entity_id")
        _record_vector(self.parked_position_world_m, 3, "parked_position_world_m")


@dataclass(frozen=True, slots=True)
class StrikerPulseEffectSpec:
    striker_entity_id: str
    joint_name: str
    strike_position_m: float
    pulse_duration_s: float
    kp_n_m: float
    kd_n_s_m: float
    maximum_force_n: float
    actuator_scale: float
    contact_cutoff_delay_s: float
    contact_grouping_hysteresis_s: float
    maximum_contact_duration_s: float

    def __post_init__(self) -> None:
        _text(self.striker_entity_id, "striker_entity_id")
        _text(self.joint_name, "joint_name")
        _number(self.strike_position_m, "strike_position_m")
        for name in (
            "pulse_duration_s",
            "kp_n_m",
            "kd_n_s_m",
            "maximum_force_n",
            "actuator_scale",
            "contact_cutoff_delay_s",
            "contact_grouping_hysteresis_s",
            "maximum_contact_duration_s",
        ):
            if _number(getattr(self, name), name) <= 0:
                raise ValueError(
                    "striker pulse durations, gains, force, and scale must be positive"
                )


EffectSpec: TypeAlias = ReleaseGateRetractionEffectSpec | StrikerPulseEffectSpec


@dataclass(frozen=True, slots=True)
class EventSpec:
    event_id: str
    trigger: TriggerSpec
    effect: EffectSpec

    def __post_init__(self) -> None:
        _text(self.event_id, "event_id")
        if not isinstance(self.trigger, (TimeTriggerSpec, ContactTriggerSpec, StateTriggerSpec)):
            raise ValueError("invalid event trigger record")
        if not isinstance(self.effect, (ReleaseGateRetractionEffectSpec, StrikerPulseEffectSpec)):
            raise ValueError("invalid event effect record")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    action_mode: ActionMode
    observation_profile: Literal["agentview", "agentview_wrist"]
    physics_timestep_s: float
    control_frequency_hz: int
    sensor_frequency_hz: int
    action_delay_steps: int
    robot_initial_state: RobotInitialStateSpec
    object_states: tuple[ObjectStateSpec, ...]
    fixture_states: tuple[FixtureStateSpec, ...]
    geometry: GeometrySpec
    motion: tuple[DrivenMotionSpec, ...]
    physics: PhysicsSpec
    cameras: CameraSetSpec
    lighting: LightingSpec
    events: tuple[EventSpec, ...]
    preparation_object_states: tuple[ObjectStateSpec, ...] = ()

    def __post_init__(self) -> None:
        _validate_scenario(self)
        _record_typed_tuple(
            self.preparation_object_states, ObjectStateSpec, "preparation_object_states"
        )
        if isinstance(self.geometry, FreeDynamicsInterceptionGeometrySpec):
            if len(self.preparation_object_states) != 1 or len(self.object_states) != 1:
                raise ValueError("T4 requires one explicit preparation target state")
            preparation, released = self.preparation_object_states[0], self.object_states[0]
            if (preparation.instance_id, preparation.asset_id) != (
                released.instance_id,
                released.asset_id,
            ):
                raise ValueError("T4 preparation and release must name the same target")
            if preparation.linear_velocity_world_m_s != (0.0, 0.0, 0.0) or (
                preparation.angular_velocity_world_rad_s != (0.0, 0.0, 0.0)
            ):
                raise ValueError("T4 temporary placement must declare zero initial twist")
        elif self.preparation_object_states:
            raise ValueError("only T4 declares the temporary preparation target")


_SCENARIO_FIELDS = {
    "action_mode",
    "observation_profile",
    "physics_timestep_s",
    "control_frequency_hz",
    "sensor_frequency_hz",
    "action_delay_steps",
    "robot_initial_state",
    "object_states",
    "fixture_states",
    "geometry",
    "motion",
    "physics",
    "cameras",
    "lighting",
    "events",
}


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _exact(data: Mapping[str, object], fields: set[str], name: str) -> None:
    missing = fields - data.keys()
    unknown = data.keys() - fields
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} unknown fields: {sorted(unknown)}")


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _vector(value: object, length: int, name: str) -> tuple[float, ...]:
    values = _sequence(value, name)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}")
    return tuple(_number(item, name) for item in values)


def _unit_vector(value: object, name: str) -> Vector3:
    vector = _vector(value, 3, name)
    if not math.isclose(math.sqrt(sum(item * item for item in vector)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must be unit length")
    return vector  # type: ignore[return-value]


def _quaternion(value: object, name: str) -> Vector4:
    quaternion = _vector(value, 4, name)
    if not math.isclose(math.sqrt(sum(item * item for item in quaternion)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must be unit length")
    return quaternion  # type: ignore[return-value]


def _bounds(value: object, width: int, name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rows = _sequence(value, name)
    if len(rows) != 2:
        raise ValueError(f"{name} must have two rows")
    lower = _vector(rows[0], width, name)
    upper = _vector(rows[1], width, name)
    if any(left >= right for left, right in zip(lower, upper, strict=True)):
        raise ValueError(f"{name} bounds must be strictly increasing")
    return lower, upper


def _record_float_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    return tuple(_number(item, name) for item in value)


def _record_vector(value: object, length: int, name: str) -> tuple[float, ...]:
    result = _record_float_tuple(value, name)
    if len(result) != length:
        raise ValueError(f"{name} must have length {length}")
    return result


def _record_unit_vector(value: object, name: str) -> Vector3:
    vector = _record_vector(value, 3, name)
    if not math.isclose(math.sqrt(sum(item * item for item in vector)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must be unit length")
    return vector  # type: ignore[return-value]


def _record_quaternion(value: object, name: str) -> Vector4:
    quaternion = _record_vector(value, 4, name)
    if not math.isclose(math.sqrt(sum(item * item for item in quaternion)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must be unit length")
    return quaternion  # type: ignore[return-value]


def _record_bounds(
    value: object, width: int, name: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a two-row tuple")
    lower = _record_vector(value[0], width, name)
    upper = _record_vector(value[1], width, name)
    if any(left >= right for left, right in zip(lower, upper, strict=True)):
        raise ValueError(f"{name} bounds must be strictly increasing")
    return lower, upper


def _record_typed_tuple(value: object, item_type: type[object], name: str) -> tuple[object, ...]:
    if not isinstance(value, tuple) or not all(isinstance(item, item_type) for item in value):
        raise ValueError(f"{name} must be a tuple of {item_type.__name__}")
    return value


def _unique_values(values: Iterable[Hashable], name: str) -> None:
    sequence: tuple[Hashable, ...] = tuple(values)
    if len(sequence) != len(set(sequence)):
        raise ValueError(f"{name} must be unique")


_GEOMETRY_FAMILIES: dict[type[object], Family] = {
    GraspGeometrySpec: "grasp",
    DrivenInterceptionGeometrySpec: "driven_interception",
    MovingPlacementGeometrySpec: "moving_placement",
    RotatingInteractionGeometrySpec: "rotating_interaction",
    FreeDynamicsInterceptionGeometrySpec: "free_dynamics_interception",
    ReplanningGeometrySpec: "replanning",
}


def family_for_geometry(geometry: GeometrySpec) -> Family:
    try:
        return _GEOMETRY_FAMILIES[type(geometry)]
    except KeyError:
        raise ValueError("unknown GeometrySpec record") from None


def _validate_scenario(scenario: ScenarioSpec) -> None:
    if scenario.action_mode not in ("E7", "J8"):
        raise ValueError("scenario action_mode must be E7 or J8")
    if scenario.observation_profile not in ("agentview", "agentview_wrist"):
        raise ValueError("invalid scenario observation_profile")
    timestep = _number(scenario.physics_timestep_s, "physics_timestep_s")
    control_frequency = _integer(scenario.control_frequency_hz, "control_frequency_hz", minimum=1)
    sensor_frequency = _integer(scenario.sensor_frequency_hz, "sensor_frequency_hz", minimum=1)
    if timestep != 0.002 or control_frequency != 20 or sensor_frequency != 20:
        raise ValueError("v1 clock requires 0.002 s physics, 20 Hz control, and 20 Hz sensors")
    if round(1.0 / (timestep * control_frequency)) != 25:
        raise ValueError("v1 clock must execute 25 physics steps per control tick")
    _integer(scenario.action_delay_steps, "action_delay_steps")
    if not isinstance(scenario.robot_initial_state, RobotInitialStateSpec):
        raise ValueError("robot_initial_state must be RobotInitialStateSpec")
    _record_typed_tuple(scenario.object_states, ObjectStateSpec, "object_states")
    _record_typed_tuple(scenario.fixture_states, FixtureStateSpec, "fixture_states")
    identities = tuple(
        item.instance_id for item in scenario.object_states + scenario.fixture_states
    )
    _unique_values(identities, "object and fixture instance IDs")
    family = family_for_geometry(scenario.geometry)
    _record_typed_tuple(scenario.motion, DrivenMotionSpec, "motion")
    _unique_values((motion.entity_id for motion in scenario.motion), "motion entity IDs")
    if family == "free_dynamics_interception" and scenario.motion:
        raise ValueError("free-dynamics scenarios require an empty motion sequence")
    if not isinstance(scenario.physics, PhysicsSpec):
        raise ValueError("physics must be PhysicsSpec")
    if not isinstance(scenario.cameras, CameraSetSpec):
        raise ValueError("cameras must be CameraSetSpec")
    expected_views = (
        ("agentview",) if scenario.observation_profile == "agentview" else ("agentview", "wrist")
    )
    if tuple(view.view_name for view in scenario.cameras.views) != expected_views:
        raise ValueError("camera view order does not match observation_profile")
    if not isinstance(scenario.lighting, LightingSpec):
        raise ValueError("lighting must be LightingSpec")
    _record_typed_tuple(scenario.events, EventSpec, "events")
    _unique_values((event.event_id for event in scenario.events), "event IDs")
    if family == "free_dynamics_interception":
        if len(scenario.events) != 1:
            raise ValueError("T4 requires exactly one release-gate event")
        event = scenario.events[0]
        if not isinstance(event.trigger, TimeTriggerSpec) or event.trigger.control_tick != 0:
            raise ValueError("T4 release gate requires a tick-0 TimeTriggerSpec")
        if not isinstance(event.effect, ReleaseGateRetractionEffectSpec):
            raise ValueError("T4 requires ReleaseGateRetractionEffectSpec")
    elif family == "replanning":
        if len(scenario.events) != 1:
            raise ValueError("T5 requires exactly one selected-striker event")
        event = scenario.events[0]
        if not isinstance(event.trigger, StateTriggerSpec):
            raise ValueError("T5 requires StateTriggerSpec")
        if not isinstance(event.effect, StrikerPulseEffectSpec):
            raise ValueError("T5 requires StrikerPulseEffectSpec")
    elif scenario.events:
        raise ValueError(f"{family} scenarios do not accept events")
    entity_ids = set(identities)
    referenced = {item.entity_id for item in scenario.motion}
    referenced |= {item.entity_id for item in scenario.physics.bodies}
    referenced |= {item.entity_id for item in scenario.physics.contacts}
    referenced |= {item.entity_id for item in scenario.physics.joints}
    for event in scenario.events:
        if isinstance(event.trigger, ContactTriggerSpec):
            referenced |= {event.trigger.entity_a_id, event.trigger.entity_b_id}
        if isinstance(event.effect, ReleaseGateRetractionEffectSpec):
            referenced.add(event.effect.gate_entity_id)
        elif isinstance(event.effect, StrikerPulseEffectSpec):
            referenced.add(event.effect.striker_entity_id)
    unknown_references = referenced - entity_ids
    if unknown_references:
        raise ValueError(
            f"scenario contains unknown entity references: {sorted(unknown_references)}"
        )


def _parse_robot(value: object) -> RobotInitialStateSpec:
    data = _mapping(value, "robot_initial_state")
    _exact(data, {"joint_position_rad", "gripper_position_m"}, "robot_initial_state")
    return RobotInitialStateSpec(
        _vector(data["joint_position_rad"], 7, "joint_position_rad"),  # type: ignore[arg-type]
        _vector(data["gripper_position_m"], 2, "gripper_position_m"),  # type: ignore[arg-type]
    )


def _parse_object(value: object) -> ObjectStateSpec:
    data = _mapping(value, "object_state")
    fields = {
        "instance_id",
        "asset_id",
        "position_world_m",
        "quaternion_world_xyzw",
        "linear_velocity_world_m_s",
        "angular_velocity_world_rad_s",
    }
    _exact(data, fields, "object_state")
    return ObjectStateSpec(
        _text(data["instance_id"], "instance_id"),
        _text(data["asset_id"], "asset_id"),
        _vector(data["position_world_m"], 3, "position_world_m"),  # type: ignore[arg-type]
        _quaternion(data["quaternion_world_xyzw"], "quaternion_world_xyzw"),
        _vector(data["linear_velocity_world_m_s"], 3, "linear_velocity_world_m_s"),  # type: ignore[arg-type]
        _vector(data["angular_velocity_world_rad_s"], 3, "angular_velocity_world_rad_s"),  # type: ignore[arg-type]
    )


def _parse_fixture(value: object) -> FixtureStateSpec:
    data = _mapping(value, "fixture_state")
    fields = {"instance_id", "asset_id", "position_world_m", "quaternion_world_xyzw"}
    _exact(data, fields, "fixture_state")
    return FixtureStateSpec(
        _text(data["instance_id"], "instance_id"),
        _text(data["asset_id"], "asset_id"),
        _vector(data["position_world_m"], 3, "position_world_m"),  # type: ignore[arg-type]
        _quaternion(data["quaternion_world_xyzw"], "quaternion_world_xyzw"),
    )


def _parse_geometry(value: object, family: Family) -> GeometrySpec:
    if family not in {
        "grasp",
        "driven_interception",
        "moving_placement",
        "rotating_interaction",
        "free_dynamics_interception",
        "replanning",
    }:
        raise ValueError(f"invalid scenario family: {family}")
    data = _mapping(value, "geometry")
    if family == "grasp":
        fields = {"support_surface_z_m", "workspace_bounds_xy_m", "downstream_exit_y_m"}
        _exact(data, fields, "geometry")
        return GraspGeometrySpec(
            _number(data["support_surface_z_m"], "support_surface_z_m"),
            _bounds(data["workspace_bounds_xy_m"], 2, "workspace_bounds_xy_m"),  # type: ignore[arg-type]
            _number(data["downstream_exit_y_m"], "downstream_exit_y_m"),
        )
    if family == "driven_interception":
        fields = {
            "support_surface_z_m",
            "workspace_bounds_xy_m",
            "capture_corridor_bounds_xy_m",
            "downstream_exit_y_m",
            "occlusion_target_clearance_m",
        }
        _exact(data, fields, "geometry")
        clearance = _number(data["occlusion_target_clearance_m"], "occlusion_target_clearance_m")
        if clearance < 0:
            raise ValueError("occlusion_target_clearance_m must be non-negative")
        return DrivenInterceptionGeometrySpec(
            _number(data["support_surface_z_m"], "support_surface_z_m"),
            _bounds(data["workspace_bounds_xy_m"], 2, "workspace_bounds_xy_m"),  # type: ignore[arg-type]
            _bounds(data["capture_corridor_bounds_xy_m"], 2, "capture_corridor_bounds_xy_m"),  # type: ignore[arg-type]
            _number(data["downstream_exit_y_m"], "downstream_exit_y_m"),
            clearance,
        )
    if family == "moving_placement":
        fields = {
            "support_surface_z_m",
            "goal_planar_clearance_m",
            "goal_bottom_center_z_m",
            "goal_window_exit_y_m",
        }
        _exact(data, fields, "geometry")
        clearance = _number(data["goal_planar_clearance_m"], "goal_planar_clearance_m")
        if clearance <= 0:
            raise ValueError("goal_planar_clearance_m must be positive")
        return MovingPlacementGeometrySpec(
            _number(data["support_surface_z_m"], "support_surface_z_m"),
            clearance,
            _number(data["goal_bottom_center_z_m"], "goal_bottom_center_z_m"),
            _number(data["goal_window_exit_y_m"], "goal_window_exit_y_m"),
        )
    if family == "rotating_interaction":
        fields = {
            "frame_center_world_m",
            "rotation_axis_world",
            "wire_radius_m",
            "wire_path_points_frame_m",
        }
        _exact(data, fields, "geometry")
        points = tuple(
            _vector(point, 3, "wire_path_point")
            for point in _sequence(data["wire_path_points_frame_m"], "wire_path_points_frame_m")
        )
        if len(points) < 2 or any(left == right for left, right in zip(points, points[1:])):
            raise ValueError("wire path requires at least two distinct adjacent points")
        radius = _number(data["wire_radius_m"], "wire_radius_m")
        if radius <= 0:
            raise ValueError("wire_radius_m must be positive")
        return RotatingInteractionGeometrySpec(
            _vector(data["frame_center_world_m"], 3, "frame_center_world_m"),  # type: ignore[arg-type]
            _unit_vector(data["rotation_axis_world"], "rotation_axis_world"),
            radius,
            points,  # type: ignore[arg-type]
        )
    if family == "free_dynamics_interception":
        fields = {
            "support_surface_z_m",
            "ramp_angle_rad",
            "ramp_downstream_lip_world_m",
            "ramp_half_size_m",
            "capture_region_bounds_world_m",
            "exit_region_bounds_xy_m",
        }
        _exact(data, fields, "geometry")
        half = _vector(data["ramp_half_size_m"], 3, "ramp_half_size_m")
        if any(item <= 0 for item in half):
            raise ValueError("ramp_half_size_m must be positive")
        return FreeDynamicsInterceptionGeometrySpec(
            _number(data["support_surface_z_m"], "support_surface_z_m"),
            _number(data["ramp_angle_rad"], "ramp_angle_rad"),
            _vector(data["ramp_downstream_lip_world_m"], 3, "ramp_downstream_lip_world_m"),  # type: ignore[arg-type]
            half,  # type: ignore[arg-type]
            _bounds(data["capture_region_bounds_world_m"], 3, "capture_region_bounds_world_m"),  # type: ignore[arg-type]
            _bounds(data["exit_region_bounds_xy_m"], 2, "exit_region_bounds_xy_m"),  # type: ignore[arg-type]
        )
    if family != "replanning":
        raise ValueError(f"invalid scenario family: {family}")
    fields = {
        "support_surface_z_m",
        "observation_boundary_y_m",
        "workspace_bounds_xy_m",
        "striker_travel_m",
    }
    _exact(data, fields, "geometry")
    travel = _number(data["striker_travel_m"], "striker_travel_m")
    if travel <= 0:
        raise ValueError("striker_travel_m must be positive")
    return ReplanningGeometrySpec(
        _number(data["support_surface_z_m"], "support_surface_z_m"),
        _number(data["observation_boundary_y_m"], "observation_boundary_y_m"),
        _bounds(data["workspace_bounds_xy_m"], 2, "workspace_bounds_xy_m"),  # type: ignore[arg-type]
        travel,
    )


def _parse_physics(value: object) -> PhysicsSpec:
    data = _mapping(value, "physics")
    _exact(data, {"gravity_world_m_s2", "bodies", "contacts", "joints"}, "physics")
    bodies: list[BodyPhysicsSpec] = []
    for value in _sequence(data["bodies"], "bodies"):
        item = _mapping(value, "body_physics")
        _exact(item, {"entity_id", "mass_kg", "diagonal_inertia_kg_m2"}, "body_physics")
        mass = _number(item["mass_kg"], "mass_kg")
        inertia = _vector(item["diagonal_inertia_kg_m2"], 3, "diagonal_inertia_kg_m2")
        if mass <= 0 or any(component <= 0 for component in inertia):
            raise ValueError("body mass and inertia must be positive")
        bodies.append(BodyPhysicsSpec(_text(item["entity_id"], "entity_id"), mass, inertia))  # type: ignore[arg-type]
    contacts: list[ContactPhysicsSpec] = []
    for value in _sequence(data["contacts"], "contacts"):
        item = _mapping(value, "contact_physics")
        fields = {"entity_id", "friction_slide_spin_roll", "solref", "solimp"}
        _exact(item, fields, "contact_physics")
        friction = _vector(item["friction_slide_spin_roll"], 3, "friction_slide_spin_roll")
        if any(component < 0 for component in friction):
            raise ValueError("friction must be non-negative")
        contacts.append(
            ContactPhysicsSpec(
                _text(item["entity_id"], "entity_id"),
                friction,  # type: ignore[arg-type]
                _vector(item["solref"], 2, "solref"),  # type: ignore[arg-type]
                _vector(item["solimp"], 5, "solimp"),  # type: ignore[arg-type]
            )
        )
    joints: list[JointPhysicsSpec] = []
    for value in _sequence(data["joints"], "joints"):
        item = _mapping(value, "joint_physics")
        fields = {"entity_id", "joint_name", "damping_per_dof", "friction_loss_per_dof"}
        _exact(item, fields, "joint_physics")
        damping = tuple(
            _number(v, "damping_per_dof")
            for v in _sequence(item["damping_per_dof"], "damping_per_dof")
        )
        friction = tuple(
            _number(v, "friction_loss_per_dof")
            for v in _sequence(item["friction_loss_per_dof"], "friction_loss_per_dof")
        )
        if not damping or len(damping) != len(friction) or any(v < 0 for v in damping + friction):
            raise ValueError(
                "joint per-DOF sequences must be non-empty, equal-length, and non-negative"
            )
        joints.append(
            JointPhysicsSpec(
                _text(item["entity_id"], "entity_id"),
                _text(item["joint_name"], "joint_name"),
                damping,
                friction,
            )
        )
    return PhysicsSpec(
        _vector(data["gravity_world_m_s2"], 3, "gravity_world_m_s2"),  # type: ignore[arg-type]
        tuple(bodies),
        tuple(contacts),
        tuple(joints),
    )


def _parse_cameras(value: object, observation_profile: str) -> CameraSetSpec:
    data = _mapping(value, "cameras")
    _exact(data, {"image_height_px", "image_width_px", "views"}, "cameras")
    height = data["image_height_px"]
    width = data["image_width_px"]
    if (
        isinstance(height, bool)
        or not isinstance(height, int)
        or isinstance(width, bool)
        or not isinstance(width, int)
        or (height, width) != (224, 224)
    ):
        raise ValueError("v1 camera raster must be 224x224")
    views: list[CameraSpec] = []
    for value in _sequence(data["views"], "views"):
        item = _mapping(value, "camera")
        fields = {"view_name", "position_parent_m", "quaternion_parent_xyzw", "vertical_fov_deg"}
        _exact(item, fields, "camera")
        name = item["view_name"]
        if name != "agentview" and name != "wrist":
            raise ValueError("invalid camera view_name")
        fov = _number(item["vertical_fov_deg"], "vertical_fov_deg")
        if not 0 < fov < 180:
            raise ValueError("vertical_fov_deg must be between 0 and 180")
        views.append(
            CameraSpec(
                name,
                _vector(item["position_parent_m"], 3, "position_parent_m"),  # type: ignore[arg-type]
                _quaternion(item["quaternion_parent_xyzw"], "quaternion_parent_xyzw"),
                fov,
            )
        )
    expected = ("agentview",) if observation_profile == "agentview" else ("agentview", "wrist")
    if tuple(view.view_name for view in views) != expected:
        raise ValueError("camera view order does not match observation_profile")
    return CameraSetSpec(224, 224, tuple(views))


def _parse_lighting(value: object) -> LightingSpec:
    data = _mapping(value, "lighting")
    _exact(data, {"lights"}, "lighting")
    lights: list[LightSpec] = []
    for value in _sequence(data["lights"], "lights"):
        item = _mapping(value, "light")
        fields = {
            "light_id",
            "position_world_m",
            "direction_world",
            "ambient_rgb",
            "diffuse_rgb",
            "specular_rgb",
            "attenuation",
            "cutoff_deg",
            "active",
        }
        _exact(item, fields, "light")
        if not isinstance(item["active"], bool):
            raise ValueError("light active must be bool")
        colors = [
            _vector(item[name], 3, name) for name in ("ambient_rgb", "diffuse_rgb", "specular_rgb")
        ]
        if any(component < 0 or component > 1 for color in colors for component in color):
            raise ValueError("light RGB values must be in [0, 1]")
        attenuation = _vector(item["attenuation"], 3, "attenuation")
        if any(component < 0 for component in attenuation):
            raise ValueError("light attenuation must be non-negative")
        lights.append(
            LightSpec(
                _text(item["light_id"], "light_id"),
                _vector(item["position_world_m"], 3, "position_world_m"),  # type: ignore[arg-type]
                _unit_vector(item["direction_world"], "direction_world"),
                colors[0],  # type: ignore[arg-type]
                colors[1],  # type: ignore[arg-type]
                colors[2],  # type: ignore[arg-type]
                attenuation,  # type: ignore[arg-type]
                _number(item["cutoff_deg"], "cutoff_deg"),
                item["active"],
            )
        )
    return LightingSpec(tuple(lights))


def _parse_translation(value: object) -> TranslationSpec:
    data = _mapping(value, "translation")
    kind = data.get("type")
    if kind == "fixed":
        _exact(data, {"type"}, "translation")
        return FixedTranslation()
    if kind == "constant_velocity":
        _exact(data, {"type", "velocity_world_m_s"}, "translation")
        return ConstantVelocityTranslation(
            _vector(data["velocity_world_m_s"], 3, "velocity_world_m_s")  # type: ignore[arg-type]
        )
    if kind == "harmonic":
        fields = {
            "type",
            "center_world_m",
            "drift_velocity_world_m_s",
            "amplitude_m",
            "frequency_hz",
            "phase_rad",
        }
        _exact(data, fields, "translation")
        amplitude = _vector(data["amplitude_m"], 3, "amplitude_m")
        frequency = _vector(data["frequency_hz"], 3, "frequency_hz")
        if any(component < 0 for component in amplitude + frequency):
            raise ValueError("harmonic amplitude and frequency must be non-negative")
        return HarmonicTranslation(
            _vector(data["center_world_m"], 3, "center_world_m"),  # type: ignore[arg-type]
            _vector(data["drift_velocity_world_m_s"], 3, "drift_velocity_world_m_s"),  # type: ignore[arg-type]
            amplitude,  # type: ignore[arg-type]
            frequency,  # type: ignore[arg-type]
            _vector(data["phase_rad"], 3, "phase_rad"),  # type: ignore[arg-type]
        )
    if kind == "circular":
        fields = {
            "type",
            "center_world_m",
            "axis_world",
            "radial_direction_world",
            "radius_m",
            "angular_speed_rad_s",
            "phase_rad",
        }
        _exact(data, fields, "translation")
        axis = _unit_vector(data["axis_world"], "axis_world")
        radial = _unit_vector(data["radial_direction_world"], "radial_direction_world")
        if not math.isclose(
            sum(a * b for a, b in zip(axis, radial, strict=True)), 0.0, abs_tol=1e-6
        ):
            raise ValueError("circular axis and radial direction must be orthogonal")
        radius = _number(data["radius_m"], "radius_m")
        if radius <= 0:
            raise ValueError("circular radius_m must be positive")
        return CircularTranslation(
            _vector(data["center_world_m"], 3, "center_world_m"),  # type: ignore[arg-type]
            axis,
            radial,
            radius,
            _number(data["angular_speed_rad_s"], "angular_speed_rad_s"),
            _number(data["phase_rad"], "phase_rad"),
        )
    raise ValueError(f"unknown translation type: {kind}")


def _parse_rotation(value: object) -> RotationSpec:
    data = _mapping(value, "rotation")
    kind = data.get("type")
    if kind == "fixed":
        _exact(data, {"type"}, "rotation")
        return FixedRotation()
    if kind == "constant_angular_velocity":
        _exact(data, {"type", "axis_world", "angular_speed_rad_s"}, "rotation")
        return ConstantAngularVelocityRotation(
            _unit_vector(data["axis_world"], "axis_world"),
            _number(data["angular_speed_rad_s"], "angular_speed_rad_s"),
        )
    if kind == "harmonic":
        fields = {
            "type",
            "base_quaternion_world_xyzw",
            "axis_world",
            "amplitude_rad",
            "frequency_hz",
            "phase_rad",
        }
        _exact(data, fields, "rotation")
        amplitude = _number(data["amplitude_rad"], "amplitude_rad")
        frequency = _number(data["frequency_hz"], "frequency_hz")
        if amplitude < 0 or frequency < 0:
            raise ValueError("harmonic rotation amplitude and frequency must be non-negative")
        return HarmonicRotation(
            _quaternion(data["base_quaternion_world_xyzw"], "base_quaternion_world_xyzw"),
            _unit_vector(data["axis_world"], "axis_world"),
            amplitude,
            frequency,
            _number(data["phase_rad"], "phase_rad"),
        )
    raise ValueError(f"unknown rotation type: {kind}")


def _parse_motion(value: object) -> tuple[DrivenMotionSpec, ...]:
    result: list[DrivenMotionSpec] = []
    for raw in _sequence(value, "motion"):
        data = _mapping(raw, "driven_motion")
        _exact(data, {"entity_id", "translation", "rotation"}, "driven_motion")
        result.append(
            DrivenMotionSpec(
                _text(data["entity_id"], "entity_id"),
                _parse_translation(data["translation"]),
                _parse_rotation(data["rotation"]),
            )
        )
    identities = tuple(item.entity_id for item in result)
    if len(set(identities)) != len(identities):
        raise ValueError("motion may contain at most one record per entity")
    return tuple(result)


def _parse_trigger(value: object) -> TriggerSpec:
    data = _mapping(value, "event trigger")
    kind = data.get("type")
    if kind == "time":
        _exact(data, {"type", "control_tick"}, "event trigger")
        return TimeTriggerSpec(_integer(data["control_tick"], "control_tick"))
    if kind == "contact":
        fields = {"type", "entity_a_id", "entity_b_id", "minimum_normal_force_n"}
        _exact(data, fields, "event trigger")
        force = _number(data["minimum_normal_force_n"], "minimum_normal_force_n")
        if force < 0:
            raise ValueError("minimum_normal_force_n must be non-negative")
        return ContactTriggerSpec(
            _text(data["entity_a_id"], "entity_a_id"),
            _text(data["entity_b_id"], "entity_b_id"),
            force,
        )
    if kind == "state":
        _exact(data, {"type"}, "event trigger")
        return StateTriggerSpec()
    raise ValueError(f"unknown event trigger type: {kind}")


def _parse_effect(value: object) -> EffectSpec:
    data = _mapping(value, "event effect")
    kind = data.get("type")
    if kind == "release_gate_retraction":
        fields = {"type", "gate_entity_id", "parked_position_world_m"}
        _exact(data, fields, "event effect")
        return ReleaseGateRetractionEffectSpec(
            _text(data["gate_entity_id"], "gate_entity_id"),
            cast(Vector3, _vector(data["parked_position_world_m"], 3, "parked_position_world_m")),
        )
    if kind == "striker_pulse":
        fields = {
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
        }
        _exact(data, fields, "event effect")
        positive = {
            name: _number(data[name], name)
            for name in fields
            if name not in {"type", "striker_entity_id", "joint_name", "strike_position_m"}
        }
        if any(number <= 0 for number in positive.values()):
            raise ValueError("striker pulse durations, gains, force, and scale must be positive")
        return StrikerPulseEffectSpec(
            _text(data["striker_entity_id"], "striker_entity_id"),
            _text(data["joint_name"], "joint_name"),
            _number(data["strike_position_m"], "strike_position_m"),
            positive["pulse_duration_s"],
            positive["kp_n_m"],
            positive["kd_n_s_m"],
            positive["maximum_force_n"],
            positive["actuator_scale"],
            positive["contact_cutoff_delay_s"],
            positive["contact_grouping_hysteresis_s"],
            positive["maximum_contact_duration_s"],
        )
    raise ValueError(f"unknown event effect type: {kind}")


def _parse_events(value: object) -> tuple[EventSpec, ...]:
    result: list[EventSpec] = []
    for raw in _sequence(value, "events"):
        data = _mapping(raw, "event")
        _exact(data, {"event_id", "trigger", "effect"}, "event")
        result.append(
            EventSpec(
                _text(data["event_id"], "event_id"),
                _parse_trigger(data["trigger"]),
                _parse_effect(data["effect"]),
            )
        )
    identities = tuple(event.event_id for event in result)
    if len(set(identities)) != len(identities):
        raise ValueError("event_id values must be unique")
    return tuple(result)


def scenario_from_dict(data: Mapping[str, object], *, family: Family) -> ScenarioSpec:
    if not isinstance(family, str) or family not in {
        "grasp",
        "driven_interception",
        "moving_placement",
        "rotating_interaction",
        "free_dynamics_interception",
        "replanning",
    }:
        raise ValueError(f"invalid scenario family: {family}")
    fields = _SCENARIO_FIELDS | (
        {"preparation_object_states"} if family == "free_dynamics_interception" else set()
    )
    _exact(data, fields, "scenario")
    timestep = _number(data["physics_timestep_s"], "physics_timestep_s")
    control_frequency = _integer(data["control_frequency_hz"], "control_frequency_hz", minimum=1)
    sensor_frequency = _integer(data["sensor_frequency_hz"], "sensor_frequency_hz", minimum=1)
    objects = tuple(
        _parse_object(value) for value in _sequence(data["object_states"], "object_states")
    )
    fixtures = tuple(
        _parse_fixture(value) for value in _sequence(data["fixture_states"], "fixture_states")
    )
    action_mode = cast(ActionMode, data["action_mode"])
    observation_profile = cast(str, data["observation_profile"])
    motion = _parse_motion(data["motion"])
    physics = _parse_physics(data["physics"])
    events = _parse_events(data["events"])
    return ScenarioSpec(
        action_mode=action_mode,
        observation_profile=observation_profile,  # type: ignore[arg-type]
        physics_timestep_s=timestep,
        control_frequency_hz=control_frequency,
        sensor_frequency_hz=sensor_frequency,
        action_delay_steps=_integer(data["action_delay_steps"], "action_delay_steps"),
        robot_initial_state=_parse_robot(data["robot_initial_state"]),
        object_states=objects,
        fixture_states=fixtures,
        geometry=_parse_geometry(data["geometry"], family),
        motion=motion,
        physics=physics,
        cameras=_parse_cameras(data["cameras"], observation_profile),
        lighting=_parse_lighting(data["lighting"]),
        events=events,
        preparation_object_states=(
            tuple(
                _parse_object(value)
                for value in _sequence(
                    data["preparation_object_states"], "preparation_object_states"
                )
            )
            if family == "free_dynamics_interception"
            else ()
        ),
    )


def _json_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _translation_to_dict(spec: TranslationSpec) -> dict[str, object]:
    if isinstance(spec, FixedTranslation):
        return {"type": "fixed"}
    if isinstance(spec, ConstantVelocityTranslation):
        return {"type": "constant_velocity", "velocity_world_m_s": list(spec.velocity_world_m_s)}
    if isinstance(spec, HarmonicTranslation):
        return {
            "type": "harmonic",
            "center_world_m": list(spec.center_world_m),
            "drift_velocity_world_m_s": list(spec.drift_velocity_world_m_s),
            "amplitude_m": list(spec.amplitude_m),
            "frequency_hz": list(spec.frequency_hz),
            "phase_rad": list(spec.phase_rad),
        }
    return {
        "type": "circular",
        "center_world_m": list(spec.center_world_m),
        "axis_world": list(spec.axis_world),
        "radial_direction_world": list(spec.radial_direction_world),
        "radius_m": spec.radius_m,
        "angular_speed_rad_s": spec.angular_speed_rad_s,
        "phase_rad": spec.phase_rad,
    }


def _rotation_to_dict(spec: RotationSpec) -> dict[str, object]:
    if isinstance(spec, FixedRotation):
        return {"type": "fixed"}
    if isinstance(spec, ConstantAngularVelocityRotation):
        return {
            "type": "constant_angular_velocity",
            "axis_world": list(spec.axis_world),
            "angular_speed_rad_s": spec.angular_speed_rad_s,
        }
    return {
        "type": "harmonic",
        "base_quaternion_world_xyzw": list(spec.base_quaternion_world_xyzw),
        "axis_world": list(spec.axis_world),
        "amplitude_rad": spec.amplitude_rad,
        "frequency_hz": spec.frequency_hz,
        "phase_rad": spec.phase_rad,
    }


def _trigger_to_dict(spec: TriggerSpec) -> dict[str, object]:
    if isinstance(spec, TimeTriggerSpec):
        return {"type": "time", "control_tick": spec.control_tick}
    if isinstance(spec, ContactTriggerSpec):
        return {
            "type": "contact",
            "entity_a_id": spec.entity_a_id,
            "entity_b_id": spec.entity_b_id,
            "minimum_normal_force_n": spec.minimum_normal_force_n,
        }
    return {"type": "state"}


def _effect_to_dict(spec: EffectSpec) -> dict[str, object]:
    if isinstance(spec, ReleaseGateRetractionEffectSpec):
        return {
            "type": "release_gate_retraction",
            "gate_entity_id": spec.gate_entity_id,
            "parked_position_world_m": list(spec.parked_position_world_m),
        }
    return {
        "type": "striker_pulse",
        "striker_entity_id": spec.striker_entity_id,
        "joint_name": spec.joint_name,
        "strike_position_m": spec.strike_position_m,
        "pulse_duration_s": spec.pulse_duration_s,
        "kp_n_m": spec.kp_n_m,
        "kd_n_s_m": spec.kd_n_s_m,
        "maximum_force_n": spec.maximum_force_n,
        "actuator_scale": spec.actuator_scale,
        "contact_cutoff_delay_s": spec.contact_cutoff_delay_s,
        "contact_grouping_hysteresis_s": spec.contact_grouping_hysteresis_s,
        "maximum_contact_duration_s": spec.maximum_contact_duration_s,
    }


def scenario_to_dict(scenario: ScenarioSpec) -> dict[str, object]:
    encoded = _json_value(scenario)
    assert isinstance(encoded, dict)
    if not scenario.preparation_object_states:
        encoded.pop("preparation_object_states")
    encoded["motion"] = [
        {
            "entity_id": motion.entity_id,
            "translation": _translation_to_dict(motion.translation),
            "rotation": _rotation_to_dict(motion.rotation),
        }
        for motion in scenario.motion
    ]
    encoded["events"] = [
        {
            "event_id": event.event_id,
            "trigger": _trigger_to_dict(event.trigger),
            "effect": _effect_to_dict(event.effect),
        }
        for event in scenario.events
    ]
    return encoded


__all__ = ["ScenarioSpec", "scenario_from_dict", "scenario_to_dict"]
