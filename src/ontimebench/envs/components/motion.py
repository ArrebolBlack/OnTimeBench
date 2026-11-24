"""Prescribed translation and rotation of moving objects."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ontimebench.scenarios.spec import (
    CircularTranslation,
    ConstantAngularVelocityRotation,
    ConstantVelocityTranslation,
    DrivenMotionSpec,
    FixedRotation,
    FixedTranslation,
    HarmonicRotation,
    HarmonicTranslation,
)


@dataclass(frozen=True, slots=True)
class _MotionState:
    position_world_m: npt.NDArray[np.float64]
    quaternion_world_xyzw: npt.NDArray[np.float64]
    yaw_world_rad: float
    linear_velocity_world_m_s: npt.NDArray[np.float64]
    angular_velocity_world_rad_s: npt.NDArray[np.float64]
    linear_acceleration_world_m_s2: npt.NDArray[np.float64]


def _vector(value: npt.ArrayLike, length: int, name: str) -> npt.NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with length {length}")
    return result


def _unit_quaternion(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    result = _vector(value, 4, name)
    if not math.isclose(float(np.linalg.norm(result)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must be unit length")
    return result / np.linalg.norm(result)


def _cross3(
    left: Sequence[float] | npt.NDArray[np.float64],
    right: Sequence[float] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Cross product for the fixed three-component vectors in the motion laws."""
    return np.array(
        (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ),
        dtype=np.float64,
    )


def _quaternion_product_xyzw(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    left_vector, left_w = left[:3], left[3]
    right_vector, right_w = right[:3], right[3]
    result = np.empty(4, dtype=np.float64)
    result[:3] = left_w * right_vector + right_w * left_vector + _cross3(left_vector, right_vector)
    result[3] = left_w * right_w - float(np.dot(left_vector, right_vector))
    return result / np.linalg.norm(result)


def _axis_angle_quaternion(axis_world: npt.ArrayLike, angle_rad: float) -> npt.NDArray[np.float64]:
    axis = _vector(axis_world, 3, "rotation axis")
    half_angle = 0.5 * angle_rad
    return np.concatenate((axis * math.sin(half_angle), np.array([math.cos(half_angle)])))


def _translation_state(
    law: FixedTranslation | ConstantVelocityTranslation | HarmonicTranslation | CircularTranslation,
    initial_position: npt.NDArray[np.float64],
    time_s: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if isinstance(law, FixedTranslation):
        return initial_position.copy(), np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    if isinstance(law, ConstantVelocityTranslation):
        velocity = np.asarray(law.velocity_world_m_s, dtype=np.float64)
        return initial_position + velocity * time_s, velocity, np.zeros(3, dtype=np.float64)
    if isinstance(law, HarmonicTranslation):
        center = np.asarray(law.center_world_m, dtype=np.float64)
        drift = np.asarray(law.drift_velocity_world_m_s, dtype=np.float64)
        amplitude = np.asarray(law.amplitude_m, dtype=np.float64)
        angular_frequency = 2.0 * math.pi * np.asarray(law.frequency_hz, dtype=np.float64)
        phase_vector = angular_frequency * time_s + np.asarray(law.phase_rad, dtype=np.float64)
        return (
            center + drift * time_s + amplitude * np.sin(phase_vector),
            drift + amplitude * angular_frequency * np.cos(phase_vector),
            -amplitude * angular_frequency**2 * np.sin(phase_vector),
        )
    center = np.asarray(law.center_world_m, dtype=np.float64)
    axis = np.asarray(law.axis_world, dtype=np.float64)
    radial = np.asarray(law.radial_direction_world, dtype=np.float64)
    tangent = _cross3(axis, radial)
    angle = law.angular_speed_rad_s * time_s + law.phase_rad
    direction = math.cos(angle) * radial + math.sin(angle) * tangent
    derivative = law.angular_speed_rad_s * (-math.sin(angle) * radial + math.cos(angle) * tangent)
    return (
        center + law.radius_m * direction,
        law.radius_m * derivative,
        -law.radius_m * law.angular_speed_rad_s**2 * direction,
    )


def _rotation_state(
    law: FixedRotation | ConstantAngularVelocityRotation | HarmonicRotation,
    initial_quaternion: npt.NDArray[np.float64],
    time_s: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if isinstance(law, FixedRotation):
        return initial_quaternion.copy(), np.zeros(3, dtype=np.float64)
    if isinstance(law, ConstantAngularVelocityRotation):
        axis = np.asarray(law.axis_world, dtype=np.float64)
        angle = law.angular_speed_rad_s * time_s
        quaternion = _quaternion_product_xyzw(
            _axis_angle_quaternion(axis, angle), initial_quaternion
        )
        return quaternion, axis * law.angular_speed_rad_s
    axis = np.asarray(law.axis_world, dtype=np.float64)
    angular_frequency = 2.0 * math.pi * law.frequency_hz
    phase = angular_frequency * time_s + law.phase_rad
    angle = law.amplitude_rad * math.sin(phase)
    base = np.asarray(law.base_quaternion_world_xyzw, dtype=np.float64)
    quaternion = _quaternion_product_xyzw(_axis_angle_quaternion(axis, angle), base)
    angular_velocity = axis * law.amplitude_rad * angular_frequency * math.cos(phase)
    return quaternion, angular_velocity


def _motion_state_at_time(
    motion: DrivenMotionSpec,
    *,
    initial_position_world_m: npt.ArrayLike,
    initial_quaternion_world_xyzw: npt.ArrayLike,
    simulator_time_s: float,
) -> _MotionState:
    if (
        isinstance(simulator_time_s, bool)
        or not isinstance(simulator_time_s, (int, float))
        or not math.isfinite(simulator_time_s)
        or simulator_time_s < 0.0
    ):
        raise ValueError("simulator_time_s must be finite and non-negative")
    initial_position = _vector(initial_position_world_m, 3, "initial position")
    raw_initial_quaternion = np.asarray(initial_quaternion_world_xyzw, dtype=np.float64)
    initial_quaternion = _unit_quaternion(raw_initial_quaternion, "initial quaternion")
    position, linear_velocity, linear_acceleration = _translation_state(
        motion.translation, initial_position, float(simulator_time_s)
    )
    quaternion, angular_velocity = _rotation_state(
        motion.rotation, initial_quaternion, float(simulator_time_s)
    )
    x, y, z, w = quaternion
    yaw = (
        2.0 * math.atan2(z, w)
        if x == 0.0 and y == 0.0
        else math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )
    if (
        isinstance(motion.rotation, ConstantAngularVelocityRotation)
        and motion.rotation.axis_world[:2] == (0.0, 0.0)
        and initial_quaternion[0] == 0.0
        and initial_quaternion[1] == 0.0
    ):
        # The old planar driver advances its scalar angle directly each physics step.
        yaw = 2.0 * math.atan2(raw_initial_quaternion[2], raw_initial_quaternion[3]) + (
            motion.rotation.angular_speed_rad_s
            * float(simulator_time_s)
            * motion.rotation.axis_world[2]
        )
    return _MotionState(
        position_world_m=position,
        quaternion_world_xyzw=quaternion,
        yaw_world_rad=yaw,
        linear_velocity_world_m_s=linear_velocity,
        angular_velocity_world_rad_s=angular_velocity,
        linear_acceleration_world_m_s2=linear_acceleration,
    )


def _bounded_planar_wrench(
    reference: _MotionState,
    *,
    position_world_m: npt.NDArray[np.float64],
    quaternion_world_wxyz: npt.NDArray[np.float64],
    linear_velocity_world_m_s: npt.NDArray[np.float64],
    angular_velocity_world_rad_s: npt.NDArray[np.float64],
    mass_kg: float,
    translation_kp_n_m: float,
    translation_kd_n_s_m: float,
    friction_compensation_acceleration_m_s2: float,
    yaw_kp_nm_rad: float,
    yaw_kd_nm_s_rad: float,
    tilt_kp_nm_rad: float,
    tilt_kd_nm_s_rad: float,
    force_moment_arm_m: float,
    maximum_force_n: float,
    maximum_yaw_torque_nm: float,
    maximum_tilt_torque_nm: float,
) -> npt.NDArray[np.float64]:
    """Evaluate a planar PD/contact driver; Family hooks apply the bounded COM wrench."""
    quaternion = _unit_quaternion(quaternion_world_wxyz, "body quaternion")
    body_w, body_x, body_y, body_z = quaternion
    body_yaw = math.atan2(
        2.0 * (body_w * body_z + body_x * body_y),
        1.0 - 2.0 * (body_y * body_y + body_z * body_z),
    )
    body_up_world = np.array(
        [
            2 * (body_x * body_z + body_y * body_w),
            2 * (body_y * body_z - body_x * body_w),
            1 - 2 * (body_x * body_x + body_y * body_y),
        ],
        dtype=np.float64,
    )
    force = (
        mass_kg * reference.linear_acceleration_world_m_s2[:2]
        + translation_kp_n_m * (reference.position_world_m[:2] - position_world_m[:2])
        + translation_kd_n_s_m
        * (reference.linear_velocity_world_m_s[:2] - linear_velocity_world_m_s[:2])
    )
    speed = float(np.linalg.norm(reference.linear_velocity_world_m_s[:2]))
    if speed > np.finfo(np.float64).eps:
        force += (
            mass_kg
            * friction_compensation_acceleration_m_s2
            * reference.linear_velocity_world_m_s[:2]
            / speed
        )
    force_norm = float(np.linalg.norm(force))
    if force_norm > maximum_force_n:
        force *= maximum_force_n / force_norm
    yaw_error = (reference.yaw_world_rad - body_yaw + math.pi) % (2.0 * math.pi) - math.pi
    yaw_torque = yaw_kp_nm_rad * yaw_error + yaw_kd_nm_s_rad * (
        reference.angular_velocity_world_rad_s[2] - angular_velocity_world_rad_s[2]
    )
    yaw_torque = float(np.clip(yaw_torque, -maximum_yaw_torque_nm, maximum_yaw_torque_nm))
    application_torque = _cross3(-force_moment_arm_m * body_up_world, [*force, 0.0])[:2]
    tilt = (
        application_torque
        + tilt_kp_nm_rad * _cross3(body_up_world, [0.0, 0.0, 1.0])[:2]
        - tilt_kd_nm_s_rad * angular_velocity_world_rad_s[:2]
    )
    tilt_norm = float(np.linalg.norm(tilt))
    if tilt_norm > maximum_tilt_torque_nm:
        tilt *= maximum_tilt_torque_nm / tilt_norm
    return np.asarray((*force, 0.0, *tilt, yaw_torque), dtype=np.float64)


def _validate_motion_initial_pose(
    motion: DrivenMotionSpec,
    *,
    initial_position_world_m: npt.ArrayLike,
    initial_quaternion_world_xyzw: npt.ArrayLike,
) -> None:
    initial_position = _vector(initial_position_world_m, 3, "initial position")
    initial_quaternion = _unit_quaternion(initial_quaternion_world_xyzw, "initial quaternion")
    state = _motion_state_at_time(
        motion,
        initial_position_world_m=initial_position,
        initial_quaternion_world_xyzw=initial_quaternion,
        simulator_time_s=0.0,
    )
    if not np.allclose(state.position_world_m, initial_position, rtol=0.0, atol=1e-9):
        raise ValueError("entity initial position must equal its motion pose at t=0")
    if not math.isclose(
        abs(float(np.dot(state.quaternion_world_xyzw, initial_quaternion))),
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError("entity initial quaternion must equal its motion pose at t=0")


def _conveyor_surface_velocity(
    motion: DrivenMotionSpec, *, world_from_geom_rotation: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Compile constant world translation to MuJoCo's geom-local surface field."""
    if not isinstance(motion.translation, ConstantVelocityTranslation) or not isinstance(
        motion.rotation, FixedRotation
    ):
        raise ValueError("conveyor surface motion requires constant translation and fixed rotation")
    rotation = np.asarray(world_from_geom_rotation, dtype=np.float64)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("world_from_geom_rotation must be a finite proper rotation")
    surface_velocity = np.zeros(6, dtype=np.float64)
    surface_velocity[:3] = rotation.T @ np.asarray(motion.translation.velocity_world_m_s)
    return surface_velocity


def _moving_goal_effort(
    reference: _MotionState,
    *,
    position_world_m: npt.NDArray[np.float64],
    rotation_world_from_body: npt.NDArray[np.float64],
    velocity_world: npt.NDArray[np.float64],
    bias_force: npt.NDArray[np.float64],
    mass_kg: float,
    yaw_inertia_kg_m2: float,
) -> npt.NDArray[np.float64]:
    """Drive the existing world-Y rail and world-Z yaw joints through native motors."""
    x, y, z, w = reference.quaternion_world_xyzw
    desired_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_error = desired_yaw - math.atan2(
        rotation_world_from_body[1, 0], rotation_world_from_body[0, 0]
    )
    yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
    force = 2000.0 * (reference.position_world_m[1] - position_world_m[1])
    force += (
        2.0
        * math.sqrt(2000.0 * mass_kg)
        * (reference.linear_velocity_world_m_s[1] - velocity_world[4])
    )
    torque = 10.0 * yaw_error + 2.0 * math.sqrt(10.0 * yaw_inertia_kg_m2) * (
        reference.angular_velocity_world_rad_s[2] - velocity_world[2]
    )
    return np.asarray(
        (np.clip(force + bias_force[0], -100.0, 100.0), np.clip(torque + bias_force[1], -5.0, 5.0)),
        dtype=np.float64,
    )


__all__: list[str] = []
